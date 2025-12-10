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
from drawer_functions import *
from HelpersForHistograms import *

import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a specific contribution for one or different regions.")
    parser.add_argument('--outFile', required=True, help="Output file name for the plot (e.g., my_plot)")
    parser.add_argument('--inFile', required=True, type=str, help="Input ROOT file containing histograms")
    parser.add_argument('--var', required=False, type=str, default='m_mumu', help="Variable to plot (e.g., M_HH)")
    parser.add_argument('--channel', required=False, type=str, default='muMu', help="Analysis channel (common to all regions)")
    parser.add_argument('--period', required=False, type=str, default='Run3_2022', help="Run era")
    parser.add_argument('--mass_region', required=False, type=str, default='OS_Iso', help="muMu mass region")
    parser.add_argument('--category', required=False, type=str, default='inclusive', help="Analysis category (base for qcdregions, or compared if category type)")
    parser.add_argument('--wantLogY', action='store_true', help="Apply log scale to y-axis")
    parser.add_argument('--wantData', action='store_true', help="Include data")
    parser.add_argument('--wantRatio', action='store_true', help="Include data")
    parser.add_argument('--wantSignal', action='store_true', help="Include data")
    parser.add_argument('--contribution', required=False, type=str, help="Specific contribution to plot (e.g., 'DY,TT').", default="all")
    parser.add_argument('--rebin', action='store_true', help="Enable rebinning based on histogram configuration")
    parser.add_argument('--compare_list', required=False, type=str, help="comma-separated subcategory to compare")
    parser.add_argument('--unstacked', action='store_true', help="Plot contributions unstacked.")
    # parser.add_argument('--reverse_order', action='store_true', help="Reverse the order of contributions.")


    args = parser.parse_args()

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"], args.period, None
    )

    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")

    global_cfg_dict = setup.global_params
    sample_cfg_dict = setup.samples
    hist_cfg_dict = setup.hists
    bckg_cfg_dict = setup.bckg_config
    sig_cfg_dict = setup.signal_config
    unc_cfg_dict = setup.weights_config

    page_cfg = os.path.join(os.environ["ANALYSIS_PATH"], f'config', f'plot',f"cms_stacked.yaml")
    with open(page_cfg, 'r') as f:
        page_cfg_dict = yaml.safe_load(f)
    page_cfg_custom = os.path.join(os.environ["ANALYSIS_PATH"], f'config', f'plot',f"{args.period}.yaml")
    with open(page_cfg_custom, 'r') as f:
        page_cfg_custom_dict = yaml.safe_load(f)
    inputs_cfg = os.path.join(os.environ["ANALYSIS_PATH"], "config", "plot", "inputs.yaml")
    with open(inputs_cfg, 'r') as f:
        raw_inputs_cfg_dict = yaml.safe_load(f)
    all_samples_types = {}
    if args.wantData:
        all_samples_types = {
                'data':
                {
                    'type':'data',
                    'plot':'data'
                },
            }

    for sample_name in bckg_cfg_dict.keys():
        if 'sampleType' not in bckg_cfg_dict[sample_name].keys(): continue
        bckg_sample_type = bckg_cfg_dict[sample_name]['sampleType']
        bckg_sample_name = bckg_sample_type if bckg_sample_type in global_cfg_dict['sample_types_to_merge'] else sample_name
        if bckg_sample_name in all_samples_types.keys():
            continue
        all_samples_types[bckg_sample_name] = {}
        all_samples_types[bckg_sample_name]['type']= bckg_sample_type

        for sample_for_plot_dict in raw_inputs_cfg_dict:
            plot_types = sample_for_plot_dict['types']
            if bckg_sample_type in plot_types:
                all_samples_types[bckg_sample_name]['plot'] = sample_for_plot_dict['name']
        if 'plot' not in all_samples_types[bckg_sample_name].keys():
            all_samples_types[bckg_sample_name]['plot'] = 'Other'

    if args.wantSignal:
        for sig_sample_name in sig_cfg_dict.keys():
            if 'sampleType' not in sig_cfg_dict[sig_sample_name].keys(): continue
            sig_sample_type = sig_cfg_dict[sig_sample_name]['sampleType']
            # print(sig_sample_type)
            if sig_sample_type not in global_cfg_dict['signal_types']: continue
            for sample_for_plot_dict in raw_inputs_cfg_dict:
                if sample_for_plot_dict['name']== sig_sample_name:
                    all_samples_types[sig_sample_name] = {
                        'type' : sig_sample_type,
                        'plot' : sig_sample_name
                    }

    hists_to_plot = {}

    rebin_condition = args.rebin and "x_rebin" in hist_cfg_dict[args.var].keys()
    new_bins = None
    if rebin_condition:
        bins_to_compute = findNewBins(hist_cfg_dict, args.var, channel=args.channel, category=args.category, region=args.mass_region)


        new_bins = getNewBins(bins_to_compute)

    inFile_root = ROOT.TFile.Open(args.inFile, "READ")

    contributions_to_plot = []
    if args.contribution == "all":
        for sample_for_plot_dict in raw_inputs_cfg_dict:
            plot_name = sample_for_plot_dict.get('name')
            plot_types = sample_for_plot_dict.get('types', [])

            if 'data' in plot_types and args.wantData:
                if 'data' not in contributions_to_plot:
                    contributions_to_plot.append('data')
            for sample_name, sample_content in all_samples_types.items():
                if sample_content['plot'] == plot_name and sample_name not in contributions_to_plot:
                    contributions_to_plot.append(sample_name)

        # if args.reverse_order:
        contributions_to_plot.reverse()

    else:
        contributions_to_plot = args.contribution.split(",")

    if args.wantData and 'data' not in contributions_to_plot:
        contributions_to_plot.append('data')


    print(f"contributions = {contributions_to_plot}")

    for sample_name, sample_content in all_samples_types.items():
        if sample_name not in contributions_to_plot:
            continue

        sample_type = sample_content['type']
        sample_plot_name = sample_content['plot']

        get_histograms_from_dir(inFile_root, sample_type, sample_plot_name, hists_to_plot)
    # print(f"hists_to_plot = {hists_to_plot}")
    hists_to_plot_binned = {}
    for path, hist_dict in hists_to_plot.items():
        if path not in hists_to_plot_binned:
            hists_to_plot_binned[path] = {}

        for hist_key, hist in hist_dict.items():
            if rebin_condition:
                new_hist = RebinHisto(hist, new_bins, hist_key, wantOverflow=False, verbose=False)
                hists_to_plot_binned[path][hist_key] = new_hist
            else:
                hists_to_plot_binned[path][hist_key] = hist
    # print(f"hists_to_plot_binned = {hists_to_plot_binned.keys()}")
    # -------------------------------------------------------------------------
    # Esecuzione dei plot
    # -------------------------------------------------------------------------

    pre_path = f"{args.channel}/{args.mass_region}/{args.category}"
    # main_region_path = f"{pre_path}/eta_incl"
    main_region_path = f"{pre_path}"

    regions_to_compare = args.compare_list.split(",") if args.compare_list else []

    if regions_to_compare:
        comparison_paths = [f"{pre_path}/{reg}" for reg in regions_to_compare]
        if all(path in hists_to_plot_binned for path in comparison_paths):
            comparison_data = {path: hists_to_plot_binned[path] for path in comparison_paths}
            print(f"Eseguendo il plot in modalità confronto tra regioni per: {', '.join(contributions_to_plot)}")
            plot_histogram_from_config(
                variable=args.var,
                histograms_dict=comparison_data,
                inputs_cfg=raw_inputs_cfg_dict,
                axes_cfg_dict=hist_cfg_dict,
                page_cfg_dict=page_cfg_dict,
                page_cfg_custom_dict=page_cfg_custom_dict,
                filename_base=f"{args.outFile}_regions_comparison",
                period=args.period,
                stacked=False,
                compare_mode=True,
                wantLogX=False,
                wantLogY=args.wantLogY,
                wantData=args.wantData,
                wantSignal=args.wantSignal,
                wantRatio = args.wantRatio,
                category=args.category,
                channel=args.channel
            )
        else:
            print("Alcune delle regioni da confrontare non sono state trovate.")

    elif main_region_path in hists_to_plot_binned:
        main_region_data = {contrib: hists_to_plot_binned[main_region_path][contrib] for contrib in contributions_to_plot if contrib in hists_to_plot_binned[main_region_path]}

        if len(contributions_to_plot) >= 1:
            print(f"Eseguendo il plot per la regione: {main_region_path}")
            plot_histogram_from_config(
                variable=args.var,
                histograms_dict=main_region_data,
                inputs_cfg=raw_inputs_cfg_dict,
                axes_cfg_dict=hist_cfg_dict,
                page_cfg_dict=page_cfg_dict,
                page_cfg_custom_dict=page_cfg_custom_dict,
                filename_base=f"{args.outFile}_stacked",
                period=args.period,
                stacked=True,
                wantLogX=False,
                wantLogY=args.wantLogY,
                wantData=args.wantData,
                wantSignal=args.wantSignal,
                wantRatio=args.wantRatio,
                category=args.category,
                channel=args.channel
            )
        else:
            print("Nessun contributo valido da plottare.")
    else:
        print(f"La regione principale '{main_region_path}' non è stata trovata.")

    print("Script completato.")
    inFile_root.Close()
