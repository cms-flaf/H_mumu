import argparse
import os
import pickle as pkl
import tomllib
from datetime import datetime
from pprint import pprint
from uuid import uuid1 as uuid

import numpy as np
import pandas as pd
import torch

from model_generation.test import Tester
from model_generation.dataloader import DataLoader
from model_generation.network import Network
from model_generation.preprocess import Preprocessor
from model_generation.train import Trainer

"""
This is the high-level script describing a full train/test cycle.
Usual workflow is:
    1) load config
    2) create Dataloader and read in .root samples
    3) create Preprocessor and set/transform training weights
    3) create Trainer and run training. Plot losses.
    4) create Tester and run inference. Produce all plots.
    5) save a copy of the model and parameters used. Results location set in config. 
"""


def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser(
        prog="NN_Generator",
        description="For a given dataset and config file, creates a network, trains it, and runs testing",
    )
    parser.add_argument("-c", "--config", required=True, help="the .toml config file")
    parser.add_argument(
        "-r",
        "--rootfile",
        required=True,
        help="the .root file to use for testing and training events",
    )
    parser.add_argument(
        "-l",
        "--label",
        required=False,
        help="some string to append to the output folder name",
    )
    args = parser.parse_args()
    return args


def write_parameters(start, end, config, dataset, variables_used):
    """
    Writes the parameters used in this model generation run to a text file.
    """
    frmt = "%Y-%m-%d %H:%M:%S"
    with open("used_params.txt", "w") as f:
        f.write(f"Started training: {start.strftime(frmt)}\n")
        f.write(f"Finished training: {end.strftime(frmt)}\n")
        f.write(f"Dataset used: {dataset}\n")
        f.write("\n")
        pprint(config, stream=f)
        f.write("\n")
        f.write("Variables used for network input vector:")
        pprint(variables_used, stream=f)


if __name__ == "__main__":
    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    # Load in all train/valid/test entries as a big DF
    # Sets initial things like class weight, sample_name, label
    dataloader = DataLoader(**config["dataloader"])
    train_df, valid_df, test_df = dataloader.generate_dataframes(args.rootfile)

    # Add a Training_Weight column and apply any needed renorms
    config["preprocess"]["classification"] = config["dataloader"]["classification"]
    config["preprocess"]["signal_types"] = config["dataloader"]["signal_types"]
    preprocessor = Preprocessor(**config["preprocess"])
    pprint(vars(preprocessor))
    train_df = preprocessor.add_train_weights(train_df)
    valid_df = preprocessor.add_train_weights(valid_df)

    # Renorm sets to m=0 s=1 separately.
    # Don't want to leak info from test into train
    train_df = dataloader._dispatch_input_renorm(train_df)
    valid_df = dataloader._dispatch_input_renorm(valid_df)
    test_df = dataloader._dispatch_input_renorm(test_df)

    # Split into (x,y), w tuples
    train_data = dataloader.df_to_dataset(train_df)
    valid_data = dataloader.df_to_dataset(valid_df)
    test_data = dataloader.df_to_dataset(test_df)

    print("*** Running with the following parameters: ***")
    pprint(config)

    # Set device for training (cpu or cuda)
    if config["meta"]["use_cuda"] and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Moving to CUDA...")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        device = None

    # Modify layer_list to have input and output layers
    layer_list = config["network"]["layer_list"]
    # Look at the number of data columns
    input_size = len(dataloader.data_columns)
    # Look at first y (label) element shape
    output_size = len(train_data[0][1][0])
    config["network"]["layer_list"] = [input_size] + layer_list + [output_size]

    # Init objects
    model = Network(device=device, **config["network"])
    trainer = Trainer(
        train_data, valid_data, config["optimizer"], **config["training"], device=device
    )
    tester = Tester(
        test_data, test_df, device=device, **config["testing"] | config["dataloader"]
    )

    # Run the traininig and testing!
    start = datetime.now()
    model = trainer.train(model)
    end = datetime.now()
    tester.test(model)

    # Prepare the output directory to save files
    os.chdir(config["meta"]["results_dir"])
    run_name = str(uuid())
    if args.label:
        run_name += f"_{args.label}"
    os.mkdir(run_name)
    os.chdir(run_name)
    print("Saving outputs to", run_name)

    # Save output files
    trainer.plot_losses()
    trainer.plot_losses(valid=True)
    trainer.write_loss_data()
    tester.make_hist(log=False, weight=True, norm=True)
    tester.make_multihist(log=True, weight=True)
    tester.make_stackplot(log=True)
    tester.make_transformed_stackplot()
    tester.make_roc_plot(log=False)
    tester.make_roc_plot(log=True)
    if config["dataloader"]["classification"] == "multiclass":
        tester.plot_multiclass_probs()
    tester.testing_df.to_pickle("evaluated_testing_df.pkl")
    train_df.to_pickle("used_training_df.pkl")
    with open("trained_model.torch", "wb") as f:
        torch.save(model, f)
    write_parameters(start, end, config, args.rootfile, dataloader.data_columns)

    # Save Combine outputs too
    tester.make_thist()
    # tester._run_combine()
