#!/usr/bin/env python3
import ROOT, sys, os, glob
ROOT.EnableImplicitMT()

# append analysis path
if "ANALYSIS_PATH" not in os.environ:
    raise RuntimeError("Devi avere ANALYSIS_PATH settato nell'ambiente.")
sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.HistHelper import *
from Analysis.H_mumu import *

# user packages (usati nel tuo esempio)
import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
import importlib
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
args = parser.parse_args()
period = f"Run3_{args.year}"
# ---------------------------------------------------------------------
def expand_filelist(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    return files

# -------------------------
# File lists
# -------------------------
sig_patterns = [
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/GluGluHto2Mu/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/VBFHto2Mu/anaTuple*.root"
]

ggh_files = expand_filelist([
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/GluGluHto2Mu/anaTuple*.root"
])

vbf_files = expand_filelist([
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/VBFHto2Mu/anaTuple*.root"
])


# bkg_patterns = [
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2E_M_50_0J_amcatnloFXFX/anaTuple*.root",
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2E_M_50_1J_amcatnloFXFX/anaTuple*.root",
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2E_M_50_2J_amcatnloFXFX/anaTuple*.root",
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2Mu_M_50_0J_amcatnloFXFX/anaTuple*.root",
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2Mu_M_50_1J_amcatnloFXFX/anaTuple*.root",
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2Mu_M_50_2J_amcatnloFXFX/anaTuple*.root",
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2Tau_M_50_0J_amcatnloFXFX/anaTuple*.root",
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2Tau_M_50_1J_amcatnloFXFX/anaTuple*.root",
#     f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/DYto2Tau_M_50_2J_amcatnloFXFX/anaTuple*.root"
# ]

bkg_patterns = [
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/TTto2L2Nu/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/TTto4Q/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/{period}/TTtoLNu2Q/anaTuple*.root",
]

sig_files = [f for p in sig_patterns for f in glob.glob(p)]
bkg_files = [f for p in bkg_patterns for f in glob.glob(p)]
all_sig_files = vbf_files+ggh_files
# setup from your framework
setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], period, "")
analysis_import = setup.global_params["analysis_import"]
analysis = importlib.import_module(analysis_import)

# samples: set process_name for signal and background builders
gp_sig = dict(setup.global_params)
gp_sig["process_name"] = setup.samples["VBFHto2Mu"]["process_name"]
gp_sig["process_group"] = setup.samples["VBFHto2Mu"]["process_group"]

gp_bkg = dict(setup.global_params)
gp_bkg["process_name"] = "DY"
gp_bkg["process_group"] = "DY"

kwargset = {}
kwargset["isData"] = False
kwargset["wantTriggerSFErrors"] = False
kwargset["colToSave"] = []

# expand file lists
full_sig_list = expand_filelist(sig_patterns)
full_bkg_list = expand_filelist(bkg_patterns)

print(f"Found {len(full_sig_list)} signal files, {len(full_bkg_list)} background files.")

if len(full_sig_list) == 0 or len(full_bkg_list) == 0:
    raise RuntimeError("File lists vuote: controlla i pattern passati.")



rdf_sig = ROOT.RDataFrame("Events", Utilities.ListToVector(all_sig_files))

rdf_sig = rdf_sig.Define(
    "isVBF",
    'std::string(ROOT::RDF::RSampleInfo().AsString()).find("VBFHto2Mu") != std::string::npos'
)
rdf_sig = rdf_sig.Define(
    "isggH",
    'std::string(ROOT::RDF::RSampleInfo().AsString()).find("ggHto2Mu") != std::string::npos'
)
rdf_bkg = ROOT.RDataFrame("Events", Utilities.ListToVector(full_bkg_list))
rdf_bkg = rdf_bkg.Define("isVBF","0")
rdf_bkg = rdf_bkg.Define("isggH","0")

# Build DataFrame wrappers using the analysis-provided builder
dfw_sig = analysis.DataFrameBuilderForHistograms(rdf_sig, gp_sig, period, **kwargset)
dfw_bkg = analysis.DataFrameBuilderForHistograms(rdf_bkg, gp_bkg, period, **kwargset)

# common definitions and pre-caching (appliazione una sola volta)
for dfw in (dfw_sig, dfw_bkg):
    dfw.df = Utilities.defineP4(dfw.df, f"mu1")
    dfw.df = Utilities.defineP4(dfw.df, f"mu2")
    dfw.df = dfw.df.Define("m_mumu", f"(mu1_p4 + mu2_p4).M()")
    dfw.defineChannels()
    dfw.defineTriggers()

    # dfw.df = AddNewDYWeights(dfw.df, dfw.period, False)
    # dfw.df = AddMuTightIDWeights(dfw.df, dfw.period)
    # dfw.df = dfw.df.Define("Jet_vetoMap","Jet_pt > 15 && ( Jet_passJetIdTightLepVeto ) && (Jet_chEmEF + Jet_neEmEF < 0.9) && Jet_isInsideVetoRegion")
    dfw.df = JetCollectionDef(dfw.df)
    dfw.df = VBFJetSelection(dfw.df)
    dfw.SignRegionDef()
    dfw.defineRegions()
    dfw.defineCategories()
    dfw.df = dfw.df.Define(f"baseline_noID_Iso","OS && trigger_sel")
    dfw.df = dfw.df.Define(f"VBF_JetVeto_noID_Iso","baseline_noID_Iso && HasVBF && j1_pt >= 35 && j2_pt >= 25")
    dfw.df = dfw.df.Define(f"ggH_noID_Iso","baseline_noID_Iso && !(VBF_JetVeto_noID_Iso)")
    dfw.df = dfw.df.Define(f"mu1_isTrigger","(mu1_pt > 26 && mu1_HasMatching_singleMu)")
    dfw.df = dfw.df.Define(f"mu2_isTrigger","(mu2_pt > 26 && mu2_HasMatching_singleMu)")
    dfw.df = dfw.df.Define("Signal_Fit", "m_mumu > 115 && m_mumu < 135")
    # dfw.df = dfw.df.Define("looseID_mu1_looseID_mu2",  "mu1_looseId  && mu2_looseId")
    # dfw.df = dfw.df.Define("looseID_mu1_mediumID_mu2",  "mu1_looseId  && mu2_mediumId")
    # dfw.df = dfw.df.Define("mediumID_mu1_looseID_mu2",  "mu1_mediumId  && mu2_looseId")
    # dfw.df = dfw.df.Define("mediumID_mu1_mediumID_mu2",  "mu1_mediumId  && mu2_mediumId")
    # dfw.df = dfw.df.Define("tightID_mu1_tightID_mu2",  "mu1_tightId  && mu2_tightId")

# -------------------------
# Snapshot
# -------------------------
cols_to_save = [
    "Signal_Fit","baseline_noID_Iso", "VBF_JetVeto_noID_Iso","ggH_noID_Iso","weight_MC_Lumi_pu","isggH","isVBF"]

for muCol in ["looseId","mediumId","tightId","pfRelIso04_all","isTrigger"]:
    for muIdx in [1,2]:
        cols_to_save.append(f"mu{muIdx}_{muCol}")

# dfw_sig.df.Snapshot("Mini", f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/mini_signal_{args.year}.root", cols_to_save)
dfw_bkg.df.Snapshot("Mini", f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/mini_bkgTTbar_{args.year}.root", cols_to_save)

print("Mini-ntupla salvata. Pronta per calcolo efficienze.")
