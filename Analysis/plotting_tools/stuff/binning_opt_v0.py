import ROOT
import yaml
from HelpersForHistograms import *
input_file_path = "/eos/user/v/vdamante/H_mumu/merged_hists/v3_m_mumu_noIso/Run3_2022/m_mumu/m_mumu.root"

inFile_root = ROOT.TFile.Open(input_file_path, "READ")

phys_model_config_file = "config/phys_models.yaml"
with open(phys_model_config_file, 'r') as phys_mod_file:
    phys_model_cfg_dict = yaml.safe_load(phys_mod_file)
signals = phys_model_cfg_dict['BaseModel']['signals']
backgrounds = phys_model_cfg_dict['BaseModel']['backgrounds']
all_contributions = backgrounds

all_contributions+=signals

all_contributions+=['data']
folder_path = "muMu/Z_sideband/VBF_JetVeto"

hists_to_plot = {}
for sample_type in all_contributions:
    get_histograms_from_dir(inFile_root, sample_type, hists_to_plot)

# print(hists_to_plot[folder_path])
## getting histogram
contribution = "DY"
hist_object = hists_to_plot[folder_path][contribution]
print(hist_object.Integral())


## get bins

bins_file_path = "/afs/cern.ch/work/v/vdamante/H_mumu/config/plot/histograms.yaml"
with open(bins_file_path, 'r') as bins_file_path_file:
    hist_cfg_dict = yaml.safe_load(bins_file_path_file)


bins_to_compute = findNewBins(hist_cfg_dict, "m_mumu", channel='muMu', category="VBF_JetVeto", region="Z_sideband")
new_bins = getNewBins(bins_to_compute)

new_hist = RebinHisto(hist_object, new_bins, contribution, wantOverflow=False, verbose=False)
print(new_hist.Integral())

# 1 calculate integral in each bin for each contribution
# 2 look at the contributions and different regions
# 3 put these values in a table --> starting point for optimization
# 4 we will discuss togeter how to optimize the binning
# 5 we will see later