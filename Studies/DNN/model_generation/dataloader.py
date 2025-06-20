import warnings
from glob import glob
from math import floor
from pprint import pprint

import numpy as np
import pandas as pd
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
        additional_cut,
        **kwargs,
    ):
        self._get_column_info(columns_config)
        self.signal_types = signal_types
        self.valid_size = valid_size
        self.test_size = test_size
        self.additional_cut = additional_cut

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

    ### Worker functions ###

    def _ensure_float(self, df):
        df[self.data_columns] = df[self.data_columns].astype("float")
        df["Label"] = df.Label.astype("float")
        return df
    
    def _add_sample_names(self, df):
        df['sample_name'] = df.sample_type.apply(lambda x: lookup[x])
        return df
        

    def _add_class_weights(self, df, dummy=False):
        """
        Used for plotting (hists)
        dummy should always be False. 
        This is just for testing
        (count hist is the same as weight hist when dummy=True)
        """
        if dummy:
            w = np.ones(len(df))
        else:
            w = df.weight_MC_Lumi_pu
        df["Class_Weight"] = w
        return df

    def _add_train_weights(self, df):
        """
        Used in training.
        Uses Class_Weight (run _add_class_weights first)
        """
        sig = df[df.Label == 1]
        bkg = df[df.Label != 1]
        # First, scale signal to have equal training weight to background
        w = sum(bkg.Class_Weight) / sum(sig.Class_Weight)
        print("Pos. weight:", w)
        weights = np.ones(len(df))
        weights[df.Label == 1] = w
        train_weights = df.Class_Weight * weights
        # Then normalize to the size of the training set
        # Without normalizing, this is akin to down-scaling the learning rate
        # e.g.: sum(weight) << len(weight) -> smaller losses, smaller optim steps
        norm = len(train_weights)/sum(train_weights)
        df['Training_Weight'] = train_weights * norm
        return df

    def _root_to_dataframe(self, filename):
        """
        A single .root file into a Pandas Df.
        """
        cols = self.all_columns
        with uproot.open(filename) as f:
            tree = f["Events"]
            df = tree.arrays(cols, cut=self.additional_cut, library="pd")
        df["source_file"] = filename
        df["Label"] = df.sample_type.apply(lambda x: 1 if x in self.signal_types else 0)
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
        for category in pd.unique(data.source_file):
            selected = (
                data[data.source_file == category].sample(frac=1).reset_index(drop=True)
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

    def _df_to_dataset(self, df):
        # Get labels
        y = df.Label.values
        y = y.reshape([len(y), 1])
        # Get the input vectors
        x = df[self.data_columns].values
        # Class weights
        w = df.Training_Weight.values
        w = w.reshape([len(w), 1])
        # Return (x,y) tuple
        return (x, y), w

    ### Main runner function ###
    def gen_datasets(self, directory):
        df = None
        for filename in glob(f"{directory}/*.root"):
            print(f"Generating testing/training samples sets from {filename}")
            if df is None:
                df = self._root_to_dataframe(filename)
            else:
                df = pd.concat([df, self._root_to_dataframe(filename)])
        df = self._ensure_float(df)
        df = self._add_sample_names(df)
        df = self._add_class_weights(df)
        df = self._add_train_weights(df)
        print(df.value_counts("sample_type"))
        train_df, valid_df, test_df = self._split_dataframe(df)
        print("Dataset sizes:")
        print(
            f"Training: {len(train_df)},\tValidation: {len(valid_df)},\tTesting:{len(test_df)}"
        )
        train_set = self._df_to_dataset(train_df)
        valid_set = self._df_to_dataset(valid_df)
        test_set = self._df_to_dataset(test_df)
        return train_set, valid_set, test_set, test_df
