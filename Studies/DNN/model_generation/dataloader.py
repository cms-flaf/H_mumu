import numpy as np
import pandas as pd
import uproot
from math import floor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataLoader:

	def __init__(self, columns_to_use, signal_types, valid_size, test_size, additional_cut=None, **kwargs):
		self.columns_to_use = columns_to_use
		self.signal_types = signal_types
		self.valid_size = valid_size
		self.test_size = test_size
		self.additional_cut = additional_cut


	def _ensure_float(self, df):
		for col in self.columns_to_use + ["label"]:
			df[col] = df[col].astype("float")
		return df


	def _root_to_dataframe(self, filename):
		cols = ['sample_type'] + self.columns_to_use
		with uproot.open(filename) as f:
			tree = f['Events']
			df = tree.arrays(cols, cut=self.additional_cut, library='pd')
		df['source_file'] = filename
		df['label'] = df.sample_type.apply(lambda x: 1 if x in self.signal_types else 0)
		df = self._ensure_float(df)
		return df


	def _split_dataframe(self, data):
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
		y = df.label.values
		y = y.reshape([len(y), 1])
		# Get the input vectors
		x = df[self.columns_to_use].values
		# Return (x,y) tuple
		return (x, y)

	### Main runner function ###
	def gen_datasets(self, filename):
		print(f"Generating testing/training samples sets from {filename}")
		df = self._root_to_dataframe(filename)
		train_df, valid_df, test_df = self._split_dataframe(df)
		train_set = self._df_to_dataset(train_df)
		valid_set = self._df_to_dataset(valid_df)
		test_set = self._df_to_dataset(test_df)
		return train_set, valid_set, test_set, test_df
