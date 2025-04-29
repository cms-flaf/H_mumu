import os

era = "Run3_2022EE"
ver = "Run3_2022EE_Hmumu_v2"
common_path = f"/afs/cern.ch/work/v/vdamante/H_mumu"
using_uncertainties = False #True #When we turn on Up/Down, the file storage changes due to renameHists.py
# varnames = [ "mu1_eta" , "mu2_eta" ]
# varnames = ["m_mumu","pt_ll","mu1_pt","mu2_pt" ,"dR_mumu" ,  "mu1_eta" , "mu1_phi" ,  "mu2_eta" , "mu2_phi"  ]
# varnames = ["VBF_etaSeparation","VBFjet1_eta","VBFjet2_eta","mu1_pt","mu2_pt","VBFjet1_pt","VBFjet2_pt","m_mumu", "VBF_mInv"]
# varnames = ["m_mumu", "pt_ll", "VBF_etaSeparation", "VBFjet1_eta"]
# varnames = ["pt_mumu"]
# varnames = [  "m_mumu", "mu1_pt", "mu2_pt", "pt_mumu"] # "m_jj",
varnames = [  "m_jj"]
channellist = ["muMu"]

region = "Inclusive"
categories = ["baseline", "VBF", "ggH", "baseline_Zmumu", "VBF_Zmumu", "ggH_Zmumu", "VBF_JetVeto", "ggH_VBFJetVeto", "VBF_Zmumu_JetVeto", "ggH_Zmumu_JetVeto"]
indir = f"/eos/user/v/vdamante/H_mumu/histograms/{ver}/{era}/merged/"
plotdir = f"/eos/user/v/vdamante/H_mumu/histograms/{ver}/{era}/plots/"

for var in varnames:
    for channel in channellist:
        for cat in categories:
            filename = os.path.join(indir, var, f"{var}.root")
            # print("Loading fname ", filename)
            os.makedirs(os.path.join(plotdir,cat), exist_ok=True)
            outname = os.path.join(plotdir,cat, f"{var}_yLog.pdf")

            if not using_uncertainties:
                os.system(f"python3 {common_path}/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig {common_path}/config/background_samples.yaml --globalConfig {common_path}/config/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantData --year {era} --wantQCD False --wantLogScale y --rebin False --analysis H_mumu --qcdregion OS --sigConfig {common_path}/config/signal_samples.yaml --wantSignals")


                # outname = os.path.join(plotdir, f"H_mumu_{var}_StackPlot.pdf")
                # os.system(f"python3 {common_path}/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig {common_path}/config/background_samples.yaml --globalConfig {common_path}/config/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantData --year {era} --wantQCD False --rebin False --analysis H_mumu --qcdregion OS --sigConfig {common_path}/config/{era}/samples.yaml")

            else:
                filename = os.path.join(indir, var, 'tmp', f"all_histograms_{var}_hadded.root")
                os.system(f"python3 /afs/cern.ch/work/v/vdamante/H_mumu/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig ../config/background_samples.yaml --globalConfig ../config/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantData --year {era} --wantQCD False --rebin False --analysis H_mumu --qcdregion OS_Iso --sigConfig ../config/{era}/samples.yaml")