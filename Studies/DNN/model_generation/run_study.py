import argparse
import os
import pickle as pkl
import tomllib
from datetime import datetime
from itertools import product
from pprint import pprint
from test import Tester
from uuid import uuid1 as uuid

import numpy as np
import pandas as pd
import torch
from dataloader import DataLoader
from network import Network
from train import Trainer


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


def write_parameters(start, end, config, dataset):
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


def main(config, df):
    dataloader = DataLoader(**config["dataloader"])

    df = dataloader.label_and_reweight(df)
    train_df, valid_df, test_df = dataloader._split_dataframe(df)
    if dataloader.downsample_upweight:
        train_df = dataloader._downsample_and_upweight(train_df)
        valid_df = dataloader._downsample_and_upweight(valid_df)
    train_data = dataloader._df_to_dataset(train_df)
    valid_data = dataloader._df_to_dataset(valid_df)
    test_data = dataloader._df_to_dataset(test_df)

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

    # Save output files
    trainer.plot_losses()
    trainer.write_loss_data()
    tester.make_hist(log=False, weight=True, norm=True)
    tester.make_hist(log=True, weight=True)
    tester.make_stackplot(log=True, weight=True)
    tester.make_roc_plot()
    tester.testing_df.to_pickle("evaluated_testing_df.pkl")
    with open("trained_model.torch", "wb") as f:
        torch.save(model, f)
    write_parameters(start, end, config, args.rootfile)


if __name__ == "__main__":
    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    dataloader = DataLoader(**config["dataloader"])
    base_df = dataloader.build_master_df(args.rootfile)
    base_df = dataloader._apply_gauss_renorm(base_df)

    batch_sizes = [1]
    # batch_sizes = [10, 50, 100, 500]
    target_ratios = [40, 20, 10, 5, 1]

    run_name = str(uuid())
    os.mkdir(run_name)
    os.chdir(run_name)
    base_dir = os.getcwd()

    # Modify layer_list to have input and output layers
    layer_list = config["network"]["layer_list"]
    # Look at the number of data columns
    input_size = len(dataloader.data_columns)
    # Look at first y (label) element shape
    output_size = len(pd.unique(base_df.sample_name))
    config["network"]["layer_list"] = [input_size] + layer_list + [output_size]

    for t, b in product(target_ratios, batch_sizes):
        config["dataloader"]["target_ratio"] = t
        config["training"]["batch_size"] = b
        dirname = f"{b}batch_{t}target"
        os.mkdir(dirname)
        os.chdir(dirname)
        main(config, base_df.copy())
        os.chdir(base_dir)
