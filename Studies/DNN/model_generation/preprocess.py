import warnings
from math import floor
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import toml
import yaml

warnings.simplefilter(action="ignore", category=FutureWarning)


class Preprocessor:
    """
    General workflow goes: Dataloader -> Preprocessor -> Trainer.
    Contains functions for modifying a loaded Pandas Df.
    Sets Train_Weight with any needed transforms.
    """

    def __init__(
        self,
        signal_types,
        classification,
        uniform_train_weight,
        zero_negative_weights,
        equalize_class_weights,
        downsample_upweight,
        target_ratio,
        use_mass_resolution,
        **kwargs,
    ):
        self.signal_types = signal_types
        self.classification = classification
        self.uniform_train_weight = uniform_train_weight
        self.zero_negative_weights = zero_negative_weights
        self.equalize_class_weights = equalize_class_weights
        self.downsample_upweight = downsample_upweight
        self.target_ratio = target_ratio
        self.use_mass_resolution = use_mass_resolution

    ### Functions for modifying the loaded dataframe ###

    def _equalize_train_weights(self, df):
        if self.classification == "binary":
            df = self._equalize_train_weights_binary(df)
        elif self.classification == "multiclass":
            df = self._equalize_train_weights_multiclass(df)
        return df

    def _equalize_train_weights_multiclass(self, df):
        total = np.sum(df.Training_Weight)
        all_proc = sorted(pd.unique(df.sample_name))
        new_train_weights = np.zeros(len(df))
        for process in all_proc:
            mask = df.sample_name == process
            subtotal = df[mask].Training_Weight.sum()
            factor = (1 / len(all_proc)) * (total / subtotal)
            new_train_weights[mask] = df[mask].Training_Weight * factor
        df["Training_Weight"] = new_train_weights
        return df

    def _equalize_train_weights_binary(self, df):
        total = np.sum(df.Training_Weight)
        weights = df.Training_Weight.values.copy()
        for label in [0, 1]:
            mask = df.Label == label
            subtotal = weights[mask].sum()
            factor = (1 / 2) * (total / subtotal)
            weights[mask] = weights[mask] * factor
        df["Training_Weight"] = weights
        return df

    def _apply_zero_neg_weight(self, df):
        # Stash the pre-zeroing weights
        cols = ["sample_name", "Training_Weight"]
        total_initial = df[cols].groupby("sample_name").sum()
        total_initial.columns = ["initial"]
        # Do the zero
        df["Training_Weight"] = np.clip(df.Training_Weight.values, a_min=0, a_max=None)
        # Get post-zeroing weights
        total_final = df[cols].groupby("sample_name").sum()
        total_final.columns = ["final"]
        totals = pd.merge(total_initial, total_final, left_index=True, right_index=True)
        # Now correct so the pre and post are equal
        for p, (i, f) in totals.iterrows():
            mask = df.sample_name == p
            scale = np.ones(len(df))
            scale[mask] = i / f
            df["Training_Weight"] = df.Training_Weight * scale
        return df

    def _downsample_and_upweight(self, df):
        print("Dispatching a downsample and reweigh...")
        counts = df.value_counts("sample_name")
        minority = counts.idxmin()
        n_min = counts[minority]
        for process in pd.unique(df.sample_name):
            n = counts[process]
            current_ratio = n / n_min
            print("Process:", process)
            print("Current ratio:", current_ratio, "Target ratio:", self.target_ratio)
            if current_ratio > self.target_ratio:
                mask = df.sample_name == process
                # Downsample
                selected = df[mask]
                other = df[~mask]
                factor = (self.target_ratio * n_min) / n
                print("Factor:", factor)
                df = pd.concat([other, selected.sample(frac=factor)])
                # Upweigh
                mask = df.sample_name == process
                weights = df.Training_Weight.values.copy()
                weights[mask] *= 1 / factor
                df["Training_Weight"] = weights
                # Done!
        return df

    def _apply_mass_resolution(self, df):
        new_weight = df.Training_Weight.values.copy() / df.m_mumu_resolution
        factor = sum(df.Training_Weight) / sum(new_weight)
        df["Training_Weight"] = factor * new_weight
        return df

    ### Main runner function ###

    def add_train_weights(self, df):
        """
        Used in training.
        """
        # Init the weights
        if self.uniform_train_weight:
            train_weights = np.ones(len(df), dtype="float")
        else:
            train_weights = df.Class_Weight.values.copy()
        # Scale weights to number of events
        # Do this to keep learning rate the same when using uniform weights
        train_weights = train_weights * len(train_weights) / sum(train_weights)
        df["Training_Weight"] = train_weights
        # Now apply any df transforms
        if self.zero_negative_weights:
            df = self._apply_zero_neg_weight(df)
        if self.use_mass_resolution:
            df = self._apply_mass_resolution(df)
        if self.downsample_upweight:
            df = self._downsample_and_upweight(df)
        if self.equalize_class_weights:
            df = self._equalize_train_weights(df)
        return df
