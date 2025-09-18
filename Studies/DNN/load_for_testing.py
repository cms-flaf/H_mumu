import argparse
import os
import pickle as pkl
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tomllib
from model_generation.dataloader import DataLoader

"""
Quick n' dirty lil script to load all the parsed anaTuples into a Pandas df.
Quick way to do counts/sums for events/weights and check acceptance. 
"""


@dataclass
class ARGS:
    config: str
    rootfile: str


args = ARGS(
    "configs/kfold_config.toml",
    "/eos/user/a/ayeagle/H_mumu/root_files/v2vbfselnew/Run3_2022",
)

# Read in config and datasets from args
with open(args.config, "rb") as f:
    config = tomllib.load(f)

config["dataloader"]["selection_cut"] = ""

dataloader = DataLoader(**config["dataloader"])
df = dataloader.build_master_df(args.rootfile)
df = dataloader._add_labels(df)
if dataloader.classification == "multiclass":
    df = dataloader._add_multiclass_labels(df)
df = dataloader._add_class_weights(df)
