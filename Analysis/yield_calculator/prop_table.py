import ROOT
import sys
import os
import yaml
import importlib
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib import ticker
from cycler import cycler

# Importa le funzioni necessarie

import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup


def get_histograms_from_dir(directory, sample_type, hist_dict, key_to_select):
    """
    Funzione ricorsiva per attraversare le directory del file ROOT e
    raccogliere gli istogrammi desiderati.
    """
    key_to_select_split = key_to_select.split("/")
    keys = [k.GetName() for k in directory.GetListOfKeys()]

    if sample_type in keys:
        obj = directory.Get(sample_type)
        if obj.IsA().InheritsFrom(ROOT.TH1.Class()):
            obj.SetDirectory(0)
            path = directory.GetPath().split(':')[-1].strip('/')
            if path not in hist_dict:
                hist_dict[path] = {}

            if sample_type not in hist_dict[path]:
                hist_dict[path][sample_type] = obj
            # else:
            #     hist_dict[path][sample_type].Add(obj)
    for key in keys:
        if key not in key_to_select_split: continue
        sub_dir = directory.Get(key)
        if sub_dir.IsA().InheritsFrom(ROOT.TDirectory.Class()):
            get_histograms_from_dir(sub_dir, sample_type, hist_dict, key_to_select)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--inFile', required=True, type=str, help="Input ROOT file containing histograms")
    parser.add_argument('--var', required=False, type=str, default='m_mumu', help="Variable to plot (e.g., M_HH)")
    parser.add_argument('--channel', required=False, type=str, default='muMu', help="Analysis channel (common to all regions)")
    parser.add_argument('--period', required=False, type=str, default='Run3_2022', help="Run era")
    parser.add_argument('--mass_region', required=False, type=str, default='OS_Iso', help="muMu mass region")
    parser.add_argument('--subregion', required=False, type=str, default=None, help="muMu mass region")
    parser.add_argument('--category', required=False, type=str, default='inclusive', help="Analysis category (base for qcdregions, or compared if category type)")
    parser.add_argument('--wantData', action='store_true', help="Include data")
    parser.add_argument('--wantSignal', action='store_true', help="Include Signal")
    parser.add_argument('--wantScaledToRun2', action='store_true', help="Scale the histograms to the Run 2 luminosity")


    args = parser.parse_args()

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"], args.period, None
    )

    analysis_import = setup.global_params["analysis_import"]

    analysis = importlib.import_module(f"{analysis_import}")
    signals = setup.phys_model['signals']
    backgrounds = setup.phys_model['backgrounds']
    all_types = backgrounds
    if args.wantSignal:
        all_types+=signals
    if args.wantData:
        all_types+=['data']
    # all_types=['m_mumu']
    hists_to_plot = {}
    inFile_root = ROOT.TFile.Open(args.inFile, "READ")
    key_to_select = f"{args.channel}/{args.mass_region}/{args.category}"
    if args.subregion:
        key_to_select += f"/{args.subregion}"
    for sample_type in all_types:
        get_histograms_from_dir(inFile_root, sample_type, hists_to_plot,key_to_select)

    period_dict = {
        "Run3_2022": 7.9804,
        "Run3_2022EE": 26.6717,
        "Run3_2023": 16.8,
        "Run3_2023BPix": 19.5,
    }
    lumi = period_dict[args.period]
    run_2_factor_scale =  137.3 / lumi if args.wantScaledToRun2 else 1.0
    # run_2018_factor_scale =  59.83 / lumi if args.wantScaledToRun2 else 1.0

    print("Sample name\tYield\tNumber of events")

    for contrib in hists_to_plot[key_to_select]:
        yield_value = run_2_factor_scale * hists_to_plot[key_to_select][contrib].Integral()
        n_events = run_2_factor_scale * hists_to_plot[key_to_select][contrib].GetEntries()
        print(f"{contrib}\t{yield_value}\t{n_events}")

    '''
    for contrib in hists_to_plot[key_to_select]:
        # print(args.category, setup.global_params["category_definition"][args.category])
        print(f"scaling factor to Run2? {run_2_factor_scale}")
        print(f"contrib = {contrib}, yield = {run_2_factor_scale * hists_to_plot[key_to_select][contrib].Integral()}")
        # print(f"contrib = {contrib}, yield = {run_2_factor_scale * hists_to_plot[key_to_select][contrib].Integral(0, hists_to_plot[key_to_select][contrib].GetNbinsX()+1)}")
        print(f"contrib = {contrib}, NUMBER of events= {run_2_factor_scale * hists_to_plot[key_to_select][contrib].GetEntries()}")
        print()
        # print(f"contrib = {contrib}, yield (not scaled to run 2) = { hists_to_plot[key_to_select][contrib].Integral()}")

        # print(f"contrib = {contrib}, NUMBER of events (not scaled to run 2)= { hists_to_plot[key_to_select][contrib].GetEntries()}")
        # print()

    # print(f"""VBFHto2Mu/DY = {hists_to_plot[key_to_select]["VBFHto2Mu"].Integral()} / {hists_to_plot[key_to_select]["DY"].Integral()} = {hists_to_plot[key_to_select]["VBFHto2Mu"].Integral()/hists_to_plot[key_to_select]["DY"].Integral()}""")
    # print(f"""GluGluHto2Mu/DY = {hists_to_plot[key_to_select]["GluGluHto2Mu"].Integral()} / {hists_to_plot[key_to_select]["DY"].Integral()} = {hists_to_plot[key_to_select]["GluGluHto2Mu"].Integral()/hists_to_plot[key_to_select]["DY"].Integral()}""")
    inFile_root.Close()
    '''