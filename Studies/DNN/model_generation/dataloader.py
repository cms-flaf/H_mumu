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
from model_generation.parse_column_names import parse_column_names
from model_generation.sample_type_lookup import lookup

warnings.simplefilter(action="ignore", category=FutureWarning)


class DataLoader:
    """
    Class for doing the inital data reading and processing.
    This sets the Labels and Class_Weights.
    Additional Train_Weight preprocessing is done in Preprocessor.
    This is usually the first workflow step (after reading configs and boilerplate)
    """

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
        if selection_cut == "":
            selection_cut = None
        self.selection_cut = selection_cut
        self.classification = classification
        self.renorm_inputs = renorm_inputs
        # File stitching defines the MC_Lumi_pu -> Class_Weight corrections
        if file_stitching == "":
            file_stitching = None
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
        """
        Quick wrapper to make sure everything fed to the net is float (no ints!)
        """
        df[self.data_columns] = df[self.data_columns].astype("float")
        return df

    def _add_sample_names(self, df):
        """
        Map sample_type to human friendly sample_names
        """
        df["sample_name"] = df.sample_type.apply(lambda x: lookup[x])
        return df

    def _add_labels(self, df):
        """
        Adds the training labels [0, 1] for bkg and sig (resp.)
        """
        df["Label"] = df.sample_name.apply(
            lambda x: 1 if x in self.signal_types else 0
        ).astype(float)
        return df

    def _add_multiclass_labels(self, df):
        """
        One-hot encoded label for multiclass
        """
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
        """
        Class_Weight is the corrected MC_Lumi_pu applied for all plotting
        """
        df["Class_Weight"] = df.weight_MC_Lumi_pu.values.copy()
        if self.file_stitching is not None:
            df = self._renorm_class_weight(df)
        return df

    def _apply_gauss_renorm(self, df):
        """
        Takes a data column and maps all values
        to average = 0 and std = 1
        """
        print("Applying gaussian renorm...")
        means = np.zeros(len(self.data_columns))
        stds = np.zeros(len(self.data_columns))
        for i, col in enumerate(self.data_columns):
            data = df[col].values.copy()
            m = np.mean(data)
            s = np.std(data)
            means[i] = m
            stds[i] = s
            print(f"{col} - Mean: {m}, StDev: {s}")
            df[col] = (data - m) / s
        return df, (means, stds)

    def _renorm_class_weight(self, df):
        """
        Does the actual rescaling of MC_Lumi_pu to Class_Weight
        Can include other factors, but mostly for "source degeneracy"
        """
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
        """
        Simple switch for applying input renorms
        """
        # Apply variables renorm
        if self.renorm_inputs == "no":
            return df, (None, None)
        elif self.renorm_inputs == "gauss":
            return self._apply_gauss_renorm(df)
        else:
            raise ValueError("no or gauss only options for input renorming")

    ### Primary worker functions ###

    def _root_to_dataframe(self, filename):
        """
        Turns a single .root file into a Pandas Df.
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
        each containing a relative portion of each process' events (i.e., striated).
        valid_size and train_size dictate the fractional size of each new Df.
        test_size gets whatever is left in the Df.
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
        """
        Goes over each input root file and builds one big Pandas Df.
        """
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
        """
        The main function that should be called externally.
        Takes a directory containing the input root files
        and returns split Pandas Dfs for training, testing, validation
        """
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
        """
        Takes a Pandas Df and returns a "dataset"
        A dataset is what Trainer and Validator expect.
        It's a tuple of Numpy arrays of the form:
        (x_data, y_data), weights
        """
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
