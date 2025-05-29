import argparse
import os
from glob import glob
from multiprocessing import Pool
from pprint import pprint

import pandas as pd
import ROOT as root


class Parser:

    def __init__(self, selection_column, columns_to_use, signal_event_types, **kwargs):
        self.selection_column = selection_column
        self.columns_to_use = columns_to_use
        self.signal_event_types = signal_event_types

    def parse_event(self, filename, event):
        vec = [filename, event.sample_type]
        vec += [getattr(event, col) for col in self.columns_to_use]
        return vec

    def parse_file(self, filename):
        print(f"→ Starting {filename}")
        file_data = pd.DataFrame(
            columns=["source_file", "sample_type"] + self.columns_to_use
        )
        f = root.TFile(filename)
        tree = f.Get("Events")
        index = 0
        for event in tree:
            if getattr(event, self.selection_column) and (event.sample_type != 0):
                row = self.parse_event(filename, event)
                file_data.loc[index] = row
                index += 1
        print(f"← Finished file {filename}!")
        return file_data

    def label(self, sample_type):
        if sample_type in self.signal_event_types:
            return 1
        else:
            return 0

    def add_labels(self, df):
        df["label"] = df["sample_type"].apply(self.label)
        return df

    def parse(self, input_directory):
        os.chdir(input_directory)
        filelist = glob("*.root")
        with Pool(4) as pool:
            results = pool.map(self.parse_file, filelist)
        final_df = pd.concat(results)
        return self.add_labels(final_df)

    def save_result(self, df, output_dir, output_name):
        os.chdir(output_dir)
        df_name = f"{output_name}.pkl"
        df.to_pickle(df_name)
