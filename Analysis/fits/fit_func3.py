import ROOT
import os
import sys
import argparse

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from Analysis.plotting_tools.HelpersForHistograms import *
from Analysis.fits.fitting_functions import *


from FLAF.Common.Setup import Setup



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui un fit su istogrammi di dati o MC.")
    parser.add_argument('--inFile', required=True, help="Percorso del file ROOT di input.")
    parser.add_argument('--channel', required=False, default="muMu", help="")
    parser.add_argument('--region', required=False, default="Z_sideband", help="")
    parser.add_argument('--category', required=False, default="baseline", help="")
    parser.add_argument('--subregion', required=False, default=None, help="")
    parser.add_argument('--contribution', required=False, type=str, help="Specific contribution to plot (e.g., 'DY,TT').", default="all")
    parser.add_argument('--outFile', required=False, default="dcb_fit", help="outFileName")
    parser.add_argument('--fitRangeMin', required=False, type=float, default=85.0, help="Valore minimo del range di fit.")
    parser.add_argument('--fitRangeMax', required=False, type=float, default=95.0, help="Valore massimo del range di fit.")
    parser.add_argument('--isMC', action='store_true', help="Set this flag if the input is a Monte Carlo signal.")
    parser.add_argument('--wantLogY', action='store_true', help="plot log Y ")
    parser.add_argument('--var', required=False, type=str, default="m_mumu", help="Variable")
    parser.add_argument('--year', required=False, type=str, default="2022", help="year")

    parser.add_argument('--fitFunc', required=False, type=str, default="DoubleSidedCB",
                        choices=["DoubleSidedCB", "CrystalBall", "Voigtian", "BreitWigner", "BW_conv_DCS", "Gaussian"],
                        help="Fit function (DoubleSidedCB, CrystalBall, Voigtian, BreitWigner).")
    parser.add_argument('--bgFunc', required=False, type=str, default="Erf_conv_Exp",
                        choices=["Exponential", "Erf_conv_Exp"],
                        help="Funzione di fit per il background (Exponential, Erf_conv_Exp).")

    args = parser.parse_args()

    if not os.path.exists(args.inFile):
        print(f"Errore: File non trovato - {args.inFile}")
        exit()

    try:

        import yaml
        hist_cfg_path = "/afs/cern.ch/work/v/vdamante/H_mumu/config/plot/histograms.yaml"
        hist_cfg_dict = {}
        with open(hist_cfg_path, "r") as hist_cfg_fl:
            hist_cfg_dict=yaml.safe_load(hist_cfg_fl)
        bins_to_compute = findNewBins(hist_cfg_dict, args.var, channel=args.channel, category=args.category, region=args.region)

        inFile_root = ROOT.TFile.Open(args.inFile, "READ")
        setup = Setup.getGlobal(
            os.environ["ANALYSIS_PATH"], f"Run3_{args.year}", None
        )
        phys_model_cfg_dict = setup.phys_model
        backgrounds = phys_model_cfg_dict['backgrounds'] +  phys_model_cfg_dict['signals']
        signals=['DY', 'EWK', 'VV', 'VH_inclusive','VVV']
        backgrounds= ['TT', 'W_NJets','TW', 'TTX', 'H_mainBckg'] 
        hist_prePath = f"{args.channel}/{args.region}/{args.category}/"
        if args.subregion:
            hist_prePath+=f"{args.subregion}/"
        all_contributions = backgrounds if args.isMC else ['data']
        print(backgrounds, signals)

        if args.contribution != 'all':
            all_contributions = args.contribution.split(",")

        print(hist_prePath)

        objsToMerge = ROOT.TList()

        all_contributions_new = [contrib for contrib in all_contributions if contrib not in signals]
        hist_bckg = None
        hist_data = inFile_root.Get(f"{hist_prePath}/data")
        hist_for_fit = hist_data
        hist_signal = inFile_root.Get(f"{hist_prePath}/{signals[0]}")
        hist_signal.SetDirectory(0)

        for signal in signals[1:]:
            hist_signal_to_add = inFile_root.Get(f"{hist_prePath}/{signal}")
            print(signal, hist_signal_to_add)
            objsToMerge.Add(hist_signal_to_add)
        hist_signal.Merge(objsToMerge)

        if args.isMC :
            hist_for_fit = hist_signal

        hist_bckg = inFile_root.Get(f"{hist_prePath}/{backgrounds[0]}")
        hist_bckg.SetDirectory(0)

        for contrib in backgrounds[1:]:
            hist_to_add = inFile_root.Get(f"{hist_prePath}/{contrib}")
            print(contrib, hist_to_add)
            objsToMerge.Add(hist_to_add)
        hist_bckg.Merge(objsToMerge)

        var_x_title = hist_cfg_dict[args.var]["x_title"]

        do_fit_and_plot(
            hist=hist_for_fit,
            hist_signal=hist_signal,
            hist_background=hist_bckg,
            fit_range=(args.fitRangeMin, args.fitRangeMax),
            var_name=args.var,
            var_x_title = var_x_title,
            fit_func=args.fitFunc,
            bg_func=args.bgFunc,
            rebin_bins=bins_to_compute,
            isMC=args.isMC,
            year=args.year,
            out_file_name=args.outFile
        )
        inFile_root.Close()

    except Exception as e:
        print(f"Error = {e}")