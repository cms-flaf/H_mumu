import argparse
import os
import pickle as pkl
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
from uuid import uuid1 as uuid
from statistics import mean

import numpy as np
import optuna
import pandas as pd
import tomllib
import torch
from model_generation.dataloader import DataLoader
from model_generation.network import Network
from model_generation.preprocess import Preprocessor
from model_generation.train import Trainer
from model_generation.test import Tester


@dataclass
class Args:
    config: str
    rootfile: str


args = Args(
    "/afs/cern.ch/user/a/ayeagle/H_mumu/Studies/DNN/configs/config.toml",
    "/eos/user/a/ayeagle/H_mumu/root_files/hlrp_v1p1/Run3_2022",
)


def objective(trial, config, train_data, valid_data, test_data, test_df, device=None):
    # Declare the study parameters
    nodes_per_layer = trial.suggest_int("nodes_per_layer", 10, 150)
    hidden_layers = trial.suggest_int("hidden_layers", 1, 6)
    batch_size = trial.suggest_int("batch_size", 10, 3000)
    lr = trial.suggest_float("lr", 1e-6, 1e-2)
    epochs = trial.suggest_int("epochs", 50, 1000)
    dropout = trial.suggest_float("dropout", 0, 0.4)
    # weight_decay = trial.suggest_float("weight_decay", 0, 0.2)

    # Update config dict with the study params
    input_size = config["network"]["input_size"]
    output_size = config["network"]["output_size"]

    config["network"]["layer_list"] = (
        [input_size] + hidden_layers * [nodes_per_layer] + [output_size]
    )
    config["training"]["batch_size"] = batch_size
    config["optimizer"]["lr"] = lr
    config["training"]["epochs"] = epochs
    config["network"]["dropout"] = dropout
    # config["optimizer"]["weight_decay"] = weight_decay

    # Init objects
    model = Network(device=device, **config["network"])
    trainer = Trainer(
        train_data, valid_data, config["optimizer"], **config["training"], device=device
    )

    # Run the traininig and testing!
    model = trainer.train(model)

    tester = Tester(
        test_data,
        test_df,
        device=device,
        **config["testing"] | config["dataloader"],
    )
    tester.test(model)

    # Evaluate
    return tester.get_roc_auc()


# Read in config and datasets from args
with open(args.config, "rb") as f:
    config = tomllib.load(f)

dataloader = DataLoader(**config["dataloader"])

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
train_df, _ = dataloader._dispatch_input_renorm(train_df)
valid_df, _ = dataloader._dispatch_input_renorm(valid_df)
test_df, _ = dataloader._dispatch_input_renorm(test_df)

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

config["network"]["input_size"] = len(dataloader.data_columns)
config["network"]["output_size"] = len(train_data[0][1][0])

# Do the dang thing!
study_name = "hyper_p_test1"
study = optuna.create_study(
    study_name=study_name,
    storage=f"sqlite:///study_{study_name}.db",
    load_if_exists=True,
)
f = lambda x: objective(x, config, train_data, valid_data, test_data, test_df, device)
study.optimize(f, n_trials=500)

pprint(study.best_params)
