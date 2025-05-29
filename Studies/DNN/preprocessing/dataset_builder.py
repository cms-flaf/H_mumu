import argparse
import os
import pickle as pkl
import warnings
from math import floor

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)


class DatasetBuilder:

    ### Init ###

    def __init__(self, input_df, run_name, columns_to_use, dataset_directory, valid_size, test_size, **kwargs):
        self.run_name = run_name
        self.columns_to_use = columns_to_use
        self.input_df = self._ensure_float(input_df)
        self.dataset_directory = dataset_directory
        self.valid_size = valid_size
        self.test_size = test_size

    def _ensure_float(self, df):
        for col in self.columns_to_use + ["label"]:
            df[col] = df[col].astype("float")
        return df

    ### Main functions ###

    def split_into_frames(self):
        """
        Creates three new data frames from the supplied data:
        training, validation, and testing
        Each contains N events per category, as dictated by cuts dict
        """
        # Init
        data = self.input_df
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

    def frame_to_datafile(self, df, filename):
        """
        Turns the supplied dataframe into numpy arrays that can feed into the network.
        Network input vectors only includes elements specified in cols.
        Saved as a (input_vector, labels) tuple via pickle
        """
        # Make labels
        y = df.label.values
        y = y.reshape([len(y), 1])
        # Get the input vectors
        x = df[self.columns_to_use].values
        # Ship 'em out
        with open(filename, "wb") as f:
            pkl.dump((x, y), f)
        print("Data saved to", filename)

    ### Runner ###

    def build(self):
        """
        Do the dirt.
        """
        training_df, valid_df, testing_df = self.split_into_frames()

        os.chdir(self.dataset_directory)
        os.mkdir(self.run_name)
        os.chdir(self.run_name)

        self.frame_to_datafile(training_df, "training_events.pkl")
        self.frame_to_datafile(valid_df, "validation_events.pkl")
        self.frame_to_datafile(testing_df, "testing_events.pkl")

        testing_cols = ["source_file", "sample_type"] + self.columns_to_use + ["label"]
        testing_df = testing_df[testing_cols]
        testing_df.to_pickle("testing_dataframe.pkl")
