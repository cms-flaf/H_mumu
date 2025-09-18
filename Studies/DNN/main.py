import argparse
import os
import pickle as pkl
from datetime import datetime
from pprint import pprint
from uuid import uuid1 as uuid

import numpy as np
import pandas as pd
import tomllib
import torch
from model_generation.dataloader import DataLoader
from model_generation.kfold import KFolder
from model_generation.network import Network
from model_generation.preprocess import Preprocessor
from model_generation.test import Tester
from model_generation.train import Trainer

"""
This is the high-level script describing a full train/test cycle.
Usual workflow is:
    - load config
    - create Dataloader and read in .root samples
    - create Preprocessor and set/transform training weights
    - create Trainer and run training. Plot losses.
    - create Tester and run inference. Produce all plots.
    - save a copy of the model and parameters used. Results location set in config. 
This (since k-fold) repeats the train process for each subflod, 
then puts all the inference results together at the end.
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
        f.write("Variables used for network input vector:\n")
        pprint(variables_used, stream=f)


def build_layer_list(config, dataloader, p):
    # Modify layer_list to have input and output layers
    layer_list = config["network"]["layer_list"]
    # Look at the number of data columns
    input_size = len(dataloader.data_columns)
    # Look at first y (label) element shape
    if dataloader.classification == "binary":
        output_size = 1
    else:
        output_size = len(p)
    config["network"]["layer_list"] = [input_size] + layer_list + [output_size]
    return config


if __name__ == "__main__":
    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    # Alter the dataloader config to set test_size to zero
    # Testing size determined by fold
    config["dataloader"]["test_size"] = 0

    # Init the output directory
    run_name = str(uuid())
    if args.label:
        run_name += f"_{args.label}"
    os.chdir(config["meta"]["results_dir"])
    os.mkdir(run_name)
    os.chdir(run_name)
    base_dir = os.getcwd()

    # Load in all train/valid/test entries as a big DF
    # Sets initial things like class weight, sample_name, label
    dataloader = DataLoader(**config["dataloader"])
    df = dataloader.build_master_df(args.rootfile)
    df = dataloader._add_labels(df)
    if dataloader.classification == "multiclass":
        df = dataloader._add_multiclass_labels(df)
    df = dataloader._add_class_weights(df)

    # Add the correct layer-list to the config (instead of just hidden)
    p = pd.unique(df.sample_name)
    config = build_layer_list(config, dataloader, p)
    # Init our other objects
    preprocessor = Preprocessor(**config["preprocess"] | config["dataloader"])
    kfold = KFolder(**config["kfold"])

    # Set device for training (cpu or cuda)
    if config["meta"]["use_cuda"] and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Moving to CUDA...")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        device = None

    # Start the k-fold training loop
    models = {}
    start = datetime.now()
    test_results = None
    for i, (test_df, everything_else) in enumerate(
        kfold.split(df, k=config["kfold"]["k"])
    ):
        train_df, valid_df, empty_df = dataloader._split_dataframe(everything_else)
        assert len(empty_df) == 0
        # Add a Training_Weight column and apply any needed renorms
        train_df = preprocessor.add_train_weights(train_df)
        valid_df = preprocessor.add_train_weights(valid_df)

        # Renorm sets to m=0 s=1 separately.
        # Don't want to leak info from test into train
        train_df, (m, s) = dataloader._dispatch_input_renorm(train_df)
        valid_df, _ = dataloader._dispatch_input_renorm(valid_df)
        test_df, _ = dataloader._dispatch_input_renorm(test_df)

        # Parse into (x,y), w tuples (aka "datasets")
        train_data = dataloader.df_to_dataset(train_df)
        valid_data = dataloader.df_to_dataset(valid_df)
        test_data = dataloader.df_to_dataset(test_df)

        # Init training run specific objects
        model = Network(device=device, **config["network"])
        trainer = Trainer(
            train_data,
            valid_data,
            config["optimizer"],
            **config["training"],
            device=device,
        )
        tester = Tester(
            test_data,
            test_df,
            device=device,
            **config["testing"] | config["dataloader"],
        )

        # Run the traininig!
        model = trainer.train(model)
        tester.test(model)

        # Stash the output of the testing inference
        cols = ["FullEventId", "NN_Output"]
        if tester.classification == "multiclass":
            cols += [f"Prob_{p}" for p in tester.processes]
        results = tester.testing_df[cols]
        if test_results is None:
            test_results = results
        else:
            test_results = pd.concat([test_results, results])

        # Prepare an output sub-directory to save files
        run_name = f"{i}_fold"
        os.mkdir(run_name)
        os.chdir(run_name)
        print("Saving outputs to", run_name)

        # Save the input renorm variables
        if m is not None and s is not None:
            with open(f"renorm_variables_{i}.pkl", "wb") as f:
                pkl.dump((m, s), f)

        # Save output files
        trainer.plot_losses()
        trainer.plot_losses(valid=True)
        trainer.write_loss_data()
        outname = f"trained_model_{i}"

        # Save model
        with open(outname + ".torch", "wb") as f:
            torch.save(model, f)
        (x_data, _), _ = train_data
        if device is None:
            x = torch.tensor(x_data, device=device)
        else:
            x = torch.tensor(x_data, device=device, dtype=torch.double)
        torch.onnx.export(
            model=model,
            args=(x[0:3]),
            f=outname + ".onnx",
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": [0], "y": [0]},
        )
        # Back to the main kfold dir
        os.chdir(base_dir)

    # Done!
    end = datetime.now()
    write_parameters(start, end, config, args.rootfile, dataloader.data_columns)

    # Combine the inference results with the main dataframe
    df = pd.merge(df, test_results)
    df.to_pickle("evaluated_testing_df.pkl")

    # Plot/save
    tester.testing_df = df
    tester.make_hist(log=False, weight=True, norm=True)
    tester.make_hist(log=True, weight=True, norm=False)
    tester.make_multihist(log=True, weight=True)
    tester.make_stackplot(log=True)
    tester.make_transformed_stackplot()
    tester.make_roc_plot(log=True)
    tester.make_roc_plot(log=False)
    tester.make_thist()
