import os

era = "Run3_2022EE"
ver = "Run3_2022EE_Hmumu_v2"
common_path = f"/afs/cern.ch/user/s/sabhosal/H_mumu" # CHANGE TO YOURS
using_uncertainties = False #True #When we turn on Up/Down, the file storage changes due to renameHists.py
# varnames = [ "mu1_eta" , "mu2_eta" ]
# varnames = ["m_mumu","pt_ll","mu1_pt","mu2_pt" ,"dR_mumu" ,  "mu1_eta" , "mu1_phi" ,  "mu2_eta" , "mu2_phi"  ]
# varnames = ["VBF_etaSeparation","VBFjet1_eta","VBFjet2_eta","mu1_pt","mu2_pt","VBFjet1_pt","VBFjet2_pt","m_mumu", "VBF_mInv"]
# varnames = ["m_mumu", "pt_ll", "VBF_etaSeparation", "VBFjet1_eta"]
# varnames = ["pt_mumu"]
# varnames = ["m_mumu", "mu1_pt", "mu2_pt", "pt_mumu"] # "m_jj",
varnames = ["pt_mumu"]

channellist = ["muMu"]

categories = ["baseline", "VBF", "ggH", "VBF_JetVeto"]

# Including QCD control regions
qcd_regions = ["Z_sideband", "H_sideband"]

indir = f"/eos/user/s/sabhosal/H_mumu/histograms/Run3_2022EE_Hmumu_v2/Run3_2022EE/merged" # CHANGE TO YOURS
plotdir = f"/eos/user/s/sabhosal/H_mumu/histograms/Run3_2022EE_Hmumu_v2/Run3_2022EE/plots" # CHANGE TO YOURS

for var in varnames:
    for channel in channellist:
        for cat in categories:

            # Looping over every QCD control region and making a plot for eachof them
            for qcd_region in qcd_regions:

                filename = os.path.join(indir, var, f"{var}.root")
                # print("Loading fname ", filename)
                os.makedirs(os.path.join(plotdir, cat), exist_ok=True)
                outname = os.path.join(plotdir, cat, f"{var}_{qcd_region}_yLog.pdf")

                if not using_uncertainties:
                    os.system(
                        f"python3 {common_path}/FLAF/Analysis/HistPlotter.py "
                        f"--inFile {filename} "
                        f"--bckgConfig {common_path}/config/background_samples.yaml "
                        f"--globalConfig {common_path}/config/global.yaml "
                        f"--outFile {outname} "
                        f"--var {var} "
                        f"--category {cat} "
                        f"--channel {channel} "
                        f"--uncSource Central "
                        f"--wantData "
                        f"--year {era} "
                        f"--wantQCD False "
                        f"--wantLogScale y "
                        f"--rebin False "
                        f"--analysis H_mumu "
                        f"--qcdregion {qcd_region} "  # <- QCDRegion name passed here
                        f"--sigConfig {common_path}/config/signal_samples.yaml "
                        f"--wantSignals"
                    )

                    # outname = os.path.join(plotdir, f"H_mumu_{var}_StackPlot.pdf")
                    # os.system(f"python3 {common_path}/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig {common_path}/config/background_samples.yaml --globalConfig {common_path}/config/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantData --year {era} --wantQCD False --rebin False --analysis H_mumu --qcdregion OS --sigConfig {common_path}/config/{era}/samples.yaml")
                else:
                    filename_tmp = os.path.join(indir, var, "tmp", f"all_histograms_{var}_hadded.root")
                    os.system(
                        f"python3 /afs/cern.ch/work/v/vdamante/H_mumu/FLAF/Analysis/HistPlotter.py "
                        f"--inFile {filename_tmp} "
                        f"--bckgConfig ../config/background_samples.yaml "
                        f"--globalConfig ../config/global.yaml "
                        f"--outFile {outname} "
                        f"--var {var} "
                        f"--category {cat} "
                        f"--channel {channel} "
                        f"--uncSource Central "
                        f"--wantData "
                        f"--year {era} "
                        f"--wantQCD False "
                        f"--rebin False "
                        f"--analysis H_mumu "
                        f"--qcdregion {qcd_region} "
                        f"--sigConfig ../config/{era}/samples.yaml"
                    )
