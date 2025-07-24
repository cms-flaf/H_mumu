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
        classification,
        uniform_train_weight,
        zero_negative_weights,
        equalize_class_weights,
        downsample_upweight, 
        target_ratio,
        renorm_inputs = 'no',
        **kwargs,
    ):
        self._get_column_info(columns_config)
        self.signal_types = signal_types
        self.valid_size = valid_size
        self.test_size = test_size
        self.additional_cut = additional_cut
        self.classification = classification
        self.uniform_train_weight = uniform_train_weight
        self.renorm_inputs = renorm_inputs
        self.zero_negative_weights = zero_negative_weights
        self.equalize_class_weights = equalize_class_weights
        self.downsample_upweight = downsample_upweight
        self.target_ratio = target_ratio

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

    def _ensure_float(self, df):
        df[self.data_columns] = df[self.data_columns].astype("float")
        return df
    
    def _add_sample_names(self, df):
        df['sample_name'] = df.sample_type.apply(lambda x: lookup[x])
        return df


    def _add_labels(self, df):
        df['Label'] = df.sample_name.apply(lambda x: 1 if x in self.signal_types else 0).astype(float)
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
        """
        Used for plotting (hists)
        dummy should always be False. 
        This is just for testing
        (count hist is the same as weight hist when dummy=True)
        """
        df["Class_Weight"] = df.weight_MC_Lumi_pu
        return df

    def _add_train_weights(self, df):
        """
        Used in training.
        Uses Class_Weight (run _add_class_weights first)
        """
        # Zero negative weights
        # Negative weights give a loss outside the domain of BCE
        # Alternatively, use batch_size = 0 (all the events)
        if self.uniform_train_weight:
            train_weights = np.ones(len(df), dtype='float')
        else:
            train_weights = df.Class_Weight.values.copy()
        if self.zero_negative_weights:
            mask = train_weights < 0
            train_weights[mask] = 0
        # Scale weights to number of events
        # Do this to keep learning rate the same when using uniform weights
        train_weights = train_weights * len(train_weights)/sum(train_weights)
        # Add and return
        df['Training_Weight'] = train_weights 
        return df

    def _equalize_train_weights(self, df):
        if self.classification == 'binary':
            df = self._equalize_train_weights_binary(df)
        elif self.classification == 'multiclass':
            df = self._equalize_train_weights_multiclass(df)
        return df


    def _equalize_train_weights_multiclass(self, df):
        total = np.sum(df.Training_Weight)
        all_proc = sorted(pd.unique(df.sample_name))
        new_train_weights = np.zeros(len(df))
        for process in all_proc:
            mask = df.sample_name == process
            subtotal = df[mask].Training_Weight.sum()
            factor = (1/len(all_proc)) * (total/subtotal) 
            new_train_weights[mask] = df[mask].Training_Weight * factor
        df['Training_Weight'] = new_train_weights
        return df

    def _equalize_train_weights_binary(self, df):
        total = np.sum(df.Training_Weight)
        weights = df.Training_Weight.values.copy()
        for label in [0,1]:
            mask = df.Label == label
            subtotal = weights[mask].sum()
            factor = (1/2) * (total/subtotal) 
            weights[mask] = weights[mask] * factor
        df['Training_Weight'] = weights
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
            df[col] = 2*(data - m)/(M- m) - 1
        return df


    def _apply_gauss_renorm(self, df):
        """
        Takes a data column and maps all values
        to average = 0 and std = 1
        """
        print("Applying gaussian renorm...")
        for col in self.data_columns:
            data = df[col].values
            m = np.mean(data)
            s = np.std(data)
            print(f"{col} - Mean: {m}, StDev: {s}")
            df[col] = (data - m)/s * 2
        return df

    def _downsample_and_upweight(self, df):
        print("Dispatching a downsample and reweigh...")
        counts = df.value_counts('sample_name')
        minority = counts.idxmin()
        n_min = counts[minority]
        for process in pd.unique(df.sample_name):
            n = counts[process]
            current_ratio = n/n_min
            print("Process:", process)
            print("Current ratio:", current_ratio, "Target ratio:", self.target_ratio)
            if current_ratio > self.target_ratio:
                mask = df.sample_name == process
                # Downsample
                selected = df[mask]
                other = df[~mask]
                factor = (self.target_ratio * n_min)/n
                print("Factor:", factor)
                df = pd.concat([other, selected.sample(frac=factor)])
                # Reweigh
                mask = df.sample_name == process
                weights = df.Training_Weight.values.copy()
                weights[mask] *= 1/factor
                df['Training_Weight'] = weights
                # Done!
        return df


    ### Primary worker functions ###

    def _root_to_dataframe(self, filename):
        """
        A single .root file into a Pandas Df.
        """
        cols = self.all_columns
        with uproot.open(filename) as f:
            tree = f["Events"]
            df = tree.arrays(cols, cut=self.additional_cut, library="pd")
        df["sample_name"] = filename
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

    def _df_to_dataset(self, df):
        # Get labels
        if self.classification == 'binary':
            y = df.Label.values
            y = y.reshape([len(y), 1])
        elif self.classification == 'multiclass':
            y = df[self.label_cols].values
        # Get the input vectors
        x = df[self.data_columns].values
        # Class weights
        w = df.Training_Weight.values
        w = w.reshape([len(w), 1])
        # Return (x,y) tuple
        return (x, y), w

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

    def label_and_reweight(self, df):
        df = self._add_labels(df)
        if self.classification == 'multiclass':
            df = self._add_multiclass_labels(df)
        #df = self._ensure_float(df)
        df = self._add_class_weights(df)
        df = self._add_train_weights(df)
        if self.equalize_class_weights:
            df = self._equalize_train_weights(df)
        return df

    ### Main runner function ###

    def gen_datasets(self, directory):
        # Build a single dataframe from root files
        df = self.build_master_df(directory)
        # Modify the main dataframe
        # Add any weights, labels, values, etc. 
        df = self.label_and_reweight(df)
        # Apply variables renorm
        if self.renorm_inputs == 'no':
            pass
        elif self.renorm_inputs == 'linear':
            df = self._apply_linear_renorm(df)
        elif self.renorm_inputs == 'gauss':
            df = self._apply_gauss_renorm(df)
        else:
            raise ValueError("renorm_inputs should only be gauss or linear, or 'no' for none.")
        # Split into train/valid/test and ship
        train_df, valid_df, test_df = self._split_dataframe(df)
        if self.downsample_upweight:
            train_df = self._downsample_and_upweight(train_df)
            valid_df = self._downsample_and_upweight(valid_df)
        # print("Dataset sizes:")
        # print(
        #     f"Training: {len(train_df)},\tValidation: {len(valid_df)},\tTesting:{len(test_df)}"
        # )
        train_set = self._df_to_dataset(train_df)
        valid_set = self._df_to_dataset(valid_df)
        test_set = self._df_to_dataset(test_df)
        return train_set, valid_set, test_set, test_df
