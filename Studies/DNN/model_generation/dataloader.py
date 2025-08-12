import warnings
from glob import glob
from math import floor
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import toml
import uproot
import yaml
from parse_column_names import parse_column_names
from sample_type_lookup import lookup

warnings.simplefilter(action="ignore", category=FutureWarning)


class DataLoader:

    def __init__(
        self,
        columns_config,
        signal_types,
        valid_size,
        test_size,
        selection_cut,
        classification,
        renorm_inputs,
        file_stitching,
        **kwargs,
    ):
        self._get_column_info(columns_config)
        self.signal_types = signal_types
        self.valid_size = valid_size
        self.test_size = test_size
        self.selection_cut = selection_cut
        self.classification = classification
        self.renorm_inputs = renorm_inputs
        if file_stitching is not None:
            with open(file_stitching, "r") as f:
                file_stitching = toml.load(f)
        self.file_stitching = file_stitching

    def _get_column_info(self, columns_config):
        with open(columns_config, "r") as file:
            config = yaml.safe_load(file)
        config = config["vars_to_save"]
        self.data_columns = parse_column_names(config, column_type="data")
        self.header_columns = parse_column_names(config, column_type="header")
        self.all_columns = parse_column_names(config, column_type="all")
        print("Header columns:")
        pprint(self.header_columns)
        print("Data columns for network input:")
        pprint(self.data_columns)

    ### Functions for modifying the loaded dataframe ###
    ### Everything in this section takes a df and returns a df ###

    def _ensure_float(self, df):
        df[self.data_columns] = df[self.data_columns].astype("float")
        return df

    def _add_sample_names(self, df):
        df["sample_name"] = df.sample_type.apply(lambda x: lookup[x])
        return df

    def _add_labels(self, df):
        df["Label"] = df.sample_name.apply(
            lambda x: 1 if x in self.signal_types else 0
        ).astype(float)
        return df

    def _add_multiclass_labels(self, df):
        all_process = sorted(pd.unique(df.sample_name))
        self.label_cols = []
        for p in all_process:
            labels = np.zeros(len(df))
            labels[df.sample_name == p] = 1
            col_name = f"Label_{p}"
            self.label_cols.append(col_name)
            df[col_name] = labels
        return df

    def _add_class_weights(self, df):
        df["Class_Weight"] = df.weight_MC_Lumi_pu.values.copy()
        if self.file_stitching is not None:
            df = self._renorm_class_weight(df)
        return df

    def _apply_linear_renorm(self, df):
        """
        Takes a data column and maps all values
        to the range [-1, 1]
        """
        print("Applying linear renorm...")
        for col in self.data_columns:
            data = df[col].values
            M = max(data)
            m = min(data)
            print(f"{col} - Min: {m}, Max: {M}")
            df[col] = 2 * (data - m) / (M - m) - 1
        return dflabel

    def _apply_gauss_renorm(self, df):
        """
        Takes a data column and maps all values
        to average = 0 and std = 1
        """
        print("Applying gaussian renorm...")
        for col in self.data_columns:
            data = df[col].values.copy()
            m = np.mean(data)
            s = np.std(data)
            print(f"{col} - Mean: {m}, StDev: {s}")
            df[col] = (data - m) / s
        return df

    def _renorm_class_weight(self, df):
        print("Applying class renorms per input file...")
        for source_file, factors in self.file_stitching.items():
            # Factors is currently a list, in case there are multiple causes of adjustment 
            factor = 1
            for entry in factors:
                factor *= entry
            mask = df.source_file == source_file
            scale_factor = np.ones(len(df))
            print(source_file, 1 / factor)
            scale_factor[mask] = 1 / factor
            df["Class_Weight"] = df.Class_Weight * scale_factor
        return df

    def _dispatch_input_renorm(self, df):
        # Apply variables renorm
        if self.renorm_inputs == "no":
            return df
        elif self.renorm_inputs == "linear":
            return self._apply_linear_renorm(df)
        elif self.renorm_inputs == "gauss":
            return self._apply_gauss_renorm(df)

    ### Primary worker functions ###

    def _root_to_dataframe(self, filename):
        """
        A single .root file into a Pandas Df.
        """
        cols = self.all_columns
        with uproot.open(filename) as f:
            tree = f["Events"]
            df = tree.arrays(cols, cut=self.selection_cut, library="pd")
        df["source_file"] = Path(filename).stem
        return df

    def _split_dataframe(self, data):
        """
        Turns the given (whole) Df into three Dfs,
        each containing a relative portion of each process' events.
        valid_size and traing_size dictate the fractional size of each new Df.
        """
        training_df = pd.DataFrame(columns=data.columns)
        valid_df = pd.DataFrame(columns=data.columns)
        testing_df = pd.DataFrame(columns=data.columns)
        # Add each category to each dataframe
        for category in pd.unique(data.sample_name):
            selected = (
                data[data.sample_name == category].sample(frac=1).reset_index(drop=True)
            )
            number = len(selected)
            valid_size = floor(number * self.valid_size)
            test_size = floor(number * self.test_size)
            # Add a size number of rows to the df
            valid_df = pd.concat([valid_df, selected[:valid_size]])
            selected = selected[valid_size:]
            testing_df = pd.concat([testing_df, selected[:test_size]])
            selected = selected[test_size:]
            training_df = pd.concat([training_df, selected])
        # Shuffle and return
        training_df = training_df.sample(frac=1).reset_index(drop=True)
        valid_df = valid_df.sample(frac=1).reset_index(drop=True)
        testing_df = testing_df.sample(frac=1).reset_index(drop=True)
        return training_df, valid_df, testing_df

    def build_master_df(self, directory):
        df = None
        for filename in glob(f"{directory}/*.root"):
            print(f"Generating testing/training samples sets from {filename}")
            if df is None:
                df = self._root_to_dataframe(filename)
            else:
                df = pd.concat([df, self._root_to_dataframe(filename)])
        df = self._add_sample_names(df)
        return df

    ### Main runner function ###

    def generate_dataframes(self, directory):
        # Build a single dataframe from root files
        df = self.build_master_df(directory)
        # Modify the main dataframe
        df = self._add_labels(df)
        if self.classification == "multiclass":
            df = self._add_multiclass_labels(df)
        df = self._add_class_weights(df)
        # Split and return
        train_df, valid_df, test_df = self._split_dataframe(df)
        return train_df, valid_df, test_df

    def df_to_dataset(self, df):
        # Get labels
        if self.classification == "binary":
            y = df.Label.values
            y = y.reshape([len(y), 1])
        elif self.classification == "multiclass":
            y = df[self.label_cols].values
        # Get the input vectors
        x = df[self.data_columns].values
        # Class weights
        try:
            w = df.Training_Weight.values
        except AttributeError:
            w = df.Class_Weight.values
        w = w.reshape([len(w), 1])
        # Return (x,y) tuple
        return (x, y), w
