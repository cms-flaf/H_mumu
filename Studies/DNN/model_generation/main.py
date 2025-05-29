import argparse
from datetime import datetime
import os
import pickle as pkl
import tomllib
from pprint import pprint
from uuid import uuid1 as uuid

import numpy as np
import pandas as pd
import torch

from network import Network
from test import Tester
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
        "-d",
        "--dataset",
        required=True,
        help="the directory containing the training, testing, and validation datasets",
    )
    args = parser.parse_args()
    return args


def load_datasets(directory):
    """
    Reads in the dataset files from the supplied directory.
    """
    with open(directory + "training_events.pkl", "rb") as f:
        training_data = pkl.load(f)
    with open(directory + "validation_events.pkl", "rb") as f:
        validation_data = pkl.load(f)
    with open(directory + "testing_events.pkl", "rb") as f:
        testing_data = pkl.load(f)
    testing_df = pd.read_pickle(directory + "testing_dataframe.pkl")
    return training_data, validation_data, testing_data, testing_df


def write_parameters(start, end, config, dataset, trainer):
    """
    Writes the parameters used in this model generation run to a text file.
    """
    frmt = "%Y-%m-%d %H:%M:%S"
    with open("used_params.txt", "w") as f:
        f.write(f"Started training: {start.strftime(frmt)}\n")
        f.write(f"Finished training: {end.strftime(frmt)}\n")
        f.write(f"Dataset used: {dataset}\n")
        f.write(f"Positive weight: {trainer.weight}\n")
        f.write("\n")
        pprint(config, stream=f)


if __name__ == "__main__":

    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    train_data, valid_data, test_data, test_df = load_datasets(args.dataset)
    print("*** Running with the following parameters: ***")
    pprint(config)

    # Set device for training (cpu or cuda)
    if config['meta']['use_cuda'] and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Moving to CUDA...")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        device = None

    # Init objects
    model = Network(device=device, **config["network"])
    trainer = Trainer(train_data, valid_data, config["optimizer"], **config["training"], device=device)
    tester = Tester(test_data, test_df, device=device)

    # Run the traininig and testing!
    start = datetime.now()
    model = trainer.train(model)
    end = datetime.now()
    tester.test(model)

    # Prepare the output directory to save files
    os.chdir(config["meta"]["results_dir"])
    run_name = str(uuid())
    os.mkdir(run_name)
    os.chdir(run_name)
    print("Saving outputs to", run_name)

    # Save output files
    trainer.plot_losses()
    trainer.plot_losses(valid=False)
    tester.make_hist(log=True)
    tester.make_hist(log=False)
    tester.testing_df.to_pickle("evaluated_testing_df.pkl")
    with open("trained_model.torch", "wb") as f:
        torch.save(model, f)
    write_parameters(start, end, config, args.dataset, trainer)
