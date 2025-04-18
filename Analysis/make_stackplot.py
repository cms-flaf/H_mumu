import os

era = "Run3_2022EE"
ver = "Run3_2022EE_Hmumu_v1"
indir = f"/eos/user/v/vdamante/H_mumu/histograms/{ver}/{era}/merged/"
plotdir = f"/eos/user/v/vdamante/H_mumu/histograms/{ver}/{era}/plots/"
common_path = f"/afs/cern.ch/work/v/vdamante/H_mumu"
varnames = [ "mu1_eta" , "mu2_eta" ]
# varnames = ["m_mumu","pt_ll","mu1_pt","mu2_pt" ,"dR_mumu" ,  "mu1_eta" , "mu1_phi" ,  "mu2_eta" , "mu2_phi"  ]

channellist = ["muMu"]

categories = ["inclusive", "Zmumu"]

using_uncertainties = False #True #When we turn on Up/Down, the file storage changes due to renameHists.py

for var in varnames:
    for channel in channellist:
        for cat in categories:
            filename = os.path.join(indir, var, f"{var}.root")
            print("Loading fname ", filename)
            os.makedirs(plotdir, exist_ok=True)
            outname = os.path.join(plotdir,cat, f"{var}_yLog.pdf")

            if not using_uncertainties:
                os.system(f"python3 {common_path}/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig {common_path}/config/background_samples.yaml --globalConfig {common_path}/config/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantData --year {era} --wantQCD False --wantLogScale y --rebin False --analysis H_mumu --qcdregion OS --sigConfig {common_path}/config/{era}/samples.yaml")


                # outname = os.path.join(plotdir, f"H_mumu_{var}_StackPlot.pdf")
                # os.system(f"python3 {common_path}/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig {common_path}/config/background_samples.yaml --globalConfig {common_path}/config/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantData --year {era} --wantQCD False --rebin False --analysis H_mumu --qcdregion OS --sigConfig {common_path}/config/{era}/samples.yaml")

            else:
                filename = os.path.join(indir, var, 'tmp', f"all_histograms_{var}_hadded.root")
                os.system(f"python3 /afs/cern.ch/work/v/vdamante/H_mumu/FLAF/Analysis/HistPlotter.py --inFile {filename} --bckgConfig ../config/background_samples.yaml --globalConfig ../config/global.yaml --outFile {outname} --var {var} --category {cat} --channel {channel} --uncSource Central --wantData --year {era} --wantQCD False --rebin False --analysis H_mumu --qcdregion OS_Iso --sigConfig ../config/{era}/samples.yaml")