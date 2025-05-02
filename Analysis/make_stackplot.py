import os
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    parser.add_argument('--var', required=True, type=str, help="vars, separated by commas")
    parser.add_argument('--era', required= False, type=str, default="Run3_2022EE", help="era")
    parser.add_argument('--histDir', required= False, type=str, default="/eos/user/v/vdamante/H_mumu/histograms/", help="era")
    parser.add_argument('--version', required= False, type=str, default="Run3_2022EE_Hmumu_v2", help="version for input files")
    parser.add_argument('--categories', required= False, type=str, help="categories")
    parser.add_argument('--wantNonLog', required= False, type=bool, default=False,help="use uncertainties")
    parser.add_argument('--wantLog', required= False, type=bool, default=False,help="use uncertainties")
    parser.add_argument('--use_unc', required= False, type=bool, default=False,help="use uncertainties")
    args = parser.parse_args()

    era = args.era #"Run3_2022EE"
    ver = args.version # "Run3_2022EE_Hmumu_v2"

    common_path = os.getcwd() # f"/afs/cern.ch/work/v/vdamante/H_mumu"
    using_uncertainties = args.use_unc #True #When we turn on Up/Down, the file storage changes due to renameHists.py

    varnames = args.var.split(",")
    categories = ["baseline", "VBF", "ggH", "baseline_Zmumu", "VBF_Zmumu", "ggH_Zmumu", "VBF_JetVeto", "VBF_Zmumu_JetVeto"]
    if args.categories:
        categories = args.categories.split(",")

    indir = os.path.join(args.histDir, ver, era, "merged")
    plotdir = os.path.join(args.histDir, ver, era, "plots")

    for var in varnames:
        for cat in categories:
            filename = os.path.join(indir, var, f"{var}.root")
            os.makedirs(os.path.join(plotdir,cat), exist_ok=True)

            if not using_uncertainties:
                if args.wantLog:
                    outname = os.path.join(plotdir,cat, f"{var}_yLog.pdf")
                    os.system(f"python3 {common_path}/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig {common_path}/config/background_samples.yaml --globalConfig {common_path}/config/global.yaml --outFile {outname} --var {var} --category {cat} --channel muMu --uncSource Central --wantData --year {era} --wantQCD False --wantLogScale y --rebin False --analysis H_mumu --qcdregion OS --sigConfig {common_path}/config/signal_samples.yaml --wantSignals")

                if args.wantNonLog:
                    outname = os.path.join(plotdir,cat, f"{var}.pdf")
                    os.system(f"python3 {common_path}/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig {common_path}/config/background_samples.yaml --globalConfig {common_path}/config/global.yaml --outFile {outname} --var {var} --category {cat} --channel muMu --uncSource Central --wantData --year {era} --wantQCD False --rebin False --analysis H_mumu --qcdregion OS --sigConfig {common_path}/config/{era}/samples.yaml")

            else:
                filename = os.path.join(indir, var, 'tmp', f"all_histograms_{var}_hadded.root")
                os.system(f"python3 /afs/cern.ch/work/v/vdamante/H_mumu/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig ../config/background_samples.yaml --globalConfig ../config/global.yaml --outFile {outname} --var {var} --category {cat} --channel muMu --uncSource Central --wantData --year {era} --wantQCD False --rebin False --analysis H_mumu --qcdregion OS_Iso --sigConfig ../config/{era}/samples.yaml")