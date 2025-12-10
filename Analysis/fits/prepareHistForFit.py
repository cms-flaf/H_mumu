import ROOT
inFiles = []

cuts = []


hist_cfg_dict = yaml.safe_load(open('/afs/cern.ch/work/v/vdamante/H_mumu/Analysis/config/plot/histograms.yaml'))
def prepareHistograms(cut, rdf, var):
    model = GetModel(hist_cfg_dict, var)
    hist = rdf.Filter(filter_to_apply).Histo1D(
        model, f"{var}_bin", "weight_Central"
    )
    return hist

DY_samples = 


