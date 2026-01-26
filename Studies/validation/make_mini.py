
import os
import sys
import ROOT
import glob
import argparse

sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
# import FLAF.Common.BaselineSelection as Baseline
from Analysis.MuonRelatedFunctions import GetMuMuP4Observables,GetAllMuMuCorrectedPtRelatedObservables
from Analysis.JetRelatedFunctions import JetCollectionDef

# from Corrections.Corrections import Corrections
ROOT.EnableThreadSafety()

# ---------------------------------------------------------------------
def expand_filelist(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    return files

mu_pt_for_selection = "pt_ScaRe"
### definitions needed for DFs
trigger_sel = f"( HLT_singleMu && ( (mu1_{mu_pt_for_selection} > 26 && mu1_HasMatching_singleMu) || (mu2_{mu_pt_for_selection} > 26 && mu2_HasMatching_singleMu) ) ) "
trigger_sel_withDiMuon = f"{trigger_sel} || ( HLT_diMu && ( (mu1_{mu_pt_for_selection} > 19 && mu1_HasMatching_diMu) && (mu2_{mu_pt_for_selection} > 10 && mu2_HasMatching_diMu) ) )"
SignalRegion_def = "(m_mumu < 135 && m_mumu > 115)"

ID_WPs = {
    "looseId":"mu{idx}_looseId",
    "mediumId":"mu{idx}_mediumId",
    "tightId":"mu{idx}_tightId",
    "mediumMVAMuId":"mu{idx}_mvaMuID_WP == 1",
    "tightMVAMuId":"mu{idx}_mvaMuID_WP == 2",
    "looseLowPtMVA":"mu{idx}_mvaLowPt > -0.6",
    "mediumLowPtMVA":"mu{idx}_mvaLowPt > -0.2",
    "tightLowPtMVA":"mu{idx}_mvaLowPt > 0.15",
}

Iso_WPs = {
    "pfRelIso_loose": "mu{idx}_pfRelIso04_all < 0.25",
    "pfRelIso_medium": "mu{idx}_pfRelIso04_all < 0.2",
    "pfRelIso_tight": "mu{idx}_pfRelIso04_all < 0.15",
    "tkRelIso_loose":"mu{idx}_tkRelIso < 0.1",
    "tkRelIso_tight":"mu{idx}_tkRelIso < 0.05",
    "miniPFRelIso_loose":"mu{idx}_miniPFRelIso_all < 0.4",
    "miniPFRelIso_medium":"mu{idx}_miniPFRelIso_all < 0.2",
    "miniPFRelIso_tight":"mu{idx}_miniPFRelIso_all < 0.1",
    "multiIso_loose":"mu{idx}_multiIsoId == 1",
    "multiIso_medium":"mu{idx}_multiIsoId == 2",
}

all_cuts_mu1 = {}
all_cuts_mu2 = {}
for idCut_name,idCut_def in ID_WPs.items():
    for isoCut_name,isoCut_def in Iso_WPs.items():
        all_cuts_mu1[f"mu1_{idCut_name}_{isoCut_name}"]= idCut_def.format(idx=1) + " && " + isoCut_def.format(idx=1)
        all_cuts_mu2[f"mu2_{idCut_name}_{isoCut_name}"]= idCut_def.format(idx=2) + " && " + isoCut_def.format(idx=2)

def GetAllDfStuff(input_df, outFileName):
    input_df = input_df.Define(
        "isVBF",
        'std::string(ROOT::RDF::RSampleInfo().AsString()).find("VBFHto2Mu") != std::string::npos',
    )
    input_df = input_df.Define(
        "isggH",
        'std::string(ROOT::RDF::RSampleInfo().AsString()).find("ggHto2Mu") != std::string::npos',
    )
    input_df = GetMuMuP4Observables(input_df)
    input_df = GetAllMuMuCorrectedPtRelatedObservables(input_df)
    input_df = JetCollectionDef(input_df)
    # -------------------------
    # Define all WP combinations
    # -------------------------
    wp_columns = []
    columns = [
        "weight_MC_Lumi_pu",
        "isVBF",
        "isggH",
        # "Signal_Fit",
        # "JetTagSel",
        # "m_mumu",
    ]

    for mu1_colName,mu1_colExpr in all_cuts_mu1.items():
        if mu1_colName not in input_df.GetColumnNames():
            input_df = input_df.Define(mu1_colName, mu1_colExpr)
        for mu2_colName,mu2_colExpr in all_cuts_mu2.items():
            if mu2_colName not in input_df.GetColumnNames():
                input_df = input_df.Define(mu2_colName, mu2_colExpr)
            colname = f"{mu1_colName}_{mu2_colName}"
            # print(f"defining {colname}")
            input_df = input_df.Define(
                f"{colname}",
                f"{mu1_colName} && {mu2_colName}",
            )
            columns.append(colname)
    # -------------------------
    # Columns to save
    # -------------------------
    df_singleMuonOnly = input_df.Filter(f"({trigger_sel} && JetTagSel && {SignalRegion_def} )")
    df_singleAndDiMuonTrg = input_df.Filter(f"({trigger_sel_withDiMuon}) && JetTagSel && {SignalRegion_def}")
    return df_singleMuonOnly,df_singleAndDiMuonTrg,columns

### setup everything

parser = argparse.ArgumentParser()
parser.add_argument("--year", required=False, default="2022")
args = parser.parse_args()
period = f"Run3_{args.year}"

setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], period)
histTupleDef = Utilities.load_module(os.path.join(os.environ["ANALYSIS_PATH"],"Analysis/histTupleDef.py"))
histTupleDef.Initialize()
histTupleDef.analysis_setup(setup)

### input files
input_path = f"/eos/user/e/eusebi/Hmumu/anaTuples/MuID_Iso_Studies_diMuonTrg/{period}"
output_path = f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/new_miniTuples/{period}"
input_df = ROOT.RDataFrame("Events",f"/eos/user/v/vdamante/H_mumu/anaTuples/test_PR/Run3_2022/data/anaTuple_0.root")

sig_patterns = [
    f"{input_path}/GluGluHto2Mu/anaTuple*.root",
    f"{input_path}/VBFHto2Mu/anaTuple*.root",
]

DY_patterns = [
    f"{input_path}/DYto2E_M_50_0J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/DYto2E_M_50_1J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/DYto2E_M_50_2J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/DYto2Mu_M_50_0J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/DYto2Mu_M_50_1J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/DYto2Mu_M_50_2J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/DYto2Tau_M_50_0J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/DYto2Tau_M_50_1J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/DYto2Tau_M_50_2J_amcatnloFXFX/anaTuple*.root"
]

TT_patterns = [
    f"{input_path}/TTto2L2Nu/anaTuple*.root",
    f"{input_path}/TTto4Q/anaTuple*.root",
    f"{input_path}/TTtoLNu2Q/anaTuple*.root",
]

W_patterns = [
    f"{input_path}/WtoLNu_0J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/WtoLNu_1J_amcatnloFXFX/anaTuple*.root",
    f"{input_path}/WtoLNu_2J_amcatnloFXFX/anaTuple*.root",
]
# 1. select only single muon trigger, look at other ID/Iso vars:

full_sig_list = expand_filelist(sig_patterns)
full_DY_list = expand_filelist(DY_patterns)
full_TT_list = expand_filelist(TT_patterns)
full_W_list = expand_filelist(W_patterns)

samples_names_list = {"sig":full_sig_list, "DY": full_DY_list, "TT":full_TT_list, "W":full_W_list}

for sample_name,samples_list in samples_names_list.items():
    print(f"processing {sample_name}")
    inputDataFrame = ROOT.RDataFrame("Events", Utilities.ListToVector(samples_list))
    output_file = f"{output_path}/mini_{sample_name}.root"
    df_singleMuonOnly,df_singleAndDiMuonTrg,columns = GetAllDfStuff(inputDataFrame, output_file)
    df_singleMuonOnly.Snapshot("singleMu",output_file,columns)
    df_singleAndDiMuonTrg.Snapshot("singleMuOrDiMuon",output_file,columns)
    print(f"Mini-ntupla salvata in {output_file}")



