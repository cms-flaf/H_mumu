import argparse
import os
from pprint import pprint

import numpy as np
import pandas as pd
import ROOT as root
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm


def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser(
        prog="Feature importance runner",
        description="Calculates Shapely values for the provided model and data",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        required=True,
        help="Path to the results directory containing evaluated_testing_df and model",
    )
    args = parser.parse_args()
    return args


def calc_transformed_hist(df, n_bins=20):
    """
    Counts the populations for the DNN' plots from AN2019_205, fig 14
    """
    # Calculate bin edges for percentiles
    signal = df[df.Label == 1]
    wq = DescrStatsW(data=signal.NN_Output, weights=signal.Class_Weight)
    p = np.linspace(0, 1, n_bins + 1)
    bin_edges = wq.quantile(p, return_pandas=False)
    # Calculate the bin populations for each process
    counts_lookup = {}
    for p in pd.unique(df.process):
        selected = df[df.process == p]
        counts, _ = np.histogram(
            selected.NN_Output, weights=selected.Class_Weight, bins=bin_edges
        )
        counts_lookup[p] = counts
    print("Bin edges:", bin_edges)
    pprint(counts_lookup)
    return bin_edges, counts_lookup


def make_thist(df, n_bins=20):
    """
    Saves a THist to be used with combine.
    Combine datacard expect two hists named:
    "signal" and "background"
    """
    bin_edges, _ = calc_transformed_hist(df, n_bins)
    # Init the histograms
    histo_dict = {}
    for p in pd.unique(df.process):
        hist = root.TH1F(p, p, n_bins, bin_edges)
        hist.Sumw2(True)
        histo_dict[p] = hist
        count_name = f"{p}_counts"
        histo_dict[count_name] = root.TH1F(count_name, count_name, n_bins, bin_edges)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        hist = histo_dict[row.process]
        hist.Fill(row.NN_Output, row.Class_Weight)
        count_hist = histo_dict[f"{row.process}_counts"]
        count_hist.Fill(row.NN_Output)
    # Save to a root file
    outname = f"hists_cut={cut}.root"
    with root.TFile.Open(outname, "RECREATE") as outFile:
        for k, v in histo_dict.items():
            outFile.WriteObject(v, k)


if __name__ == "__main__":
    args = get_arguments()
    print("Input args:")
    pprint(args)
    os.chdir(args.results_dir)
    df = pd.read_pickle("evaluated_testing_df.pkl")
    os.mkdir("thists")
    os.chdir("thists")
    basedir = os.getcwd()
    cuts = np.linspace(0, 0.95, 20)
    for cut in cuts:
        os.chdir(basedir)
        outname = f"cut_{cut}"
        os.mkdir(outname)
        os.chdir(outname)
        selected = df[df.VBFNet_Output >= cut]
        make_thist(selected)
