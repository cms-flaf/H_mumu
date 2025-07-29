import ROOT
import sys
if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

# from FLAF.Common.HistHelper import *
from FLAF.Analysis.HistHelper import *
from FLAF.Common.Utilities import *
from Analysis.GetTriggerWeights import *


JetObservables = ["PNetRegPtRawCorr","PNetRegPtRawCorrNeutrino","PNetRegPtRawRes","area","btagDeepFlavB","btagDeepFlavCvB","btagDeepFlavCvL","btagDeepFlavQG","btagPNetB","btagPNetCvB","btagPNetCvL","btagPNetCvNotB","btagPNetQvG","btagPNetTauVJet","chEmEF","chHEF","chMultiplicity","electronIdx1","electronIdx2","eta","hfEmEF","hfHEF","hfadjacentEtaStripsSize","hfcentralEtaStripSize","hfsigmaEtaEta","hfsigmaPhiPhi","jetId","mass","muEF","muonIdx1","muonIdx2","muonSubtrFactor","nConstituents","nElectrons","nMuons","nSVs","neEmEF","neHEF","neMultiplicity","partonFlavour","phi","pt","rawFactor","svIdx1","svIdx2"]
JetObservablesMC = ["hadronFlavour","partonFlavour", "genJetIdx"]

# def createKeyFilterDict(global_params):
#     filter_dict = {}
#     filter_str = ""
#     channels_to_consider = global_params['channels_to_consider']
#     sign_regions_to_consider = global_params['MuMuMassRegions']
#     categories_to_consider = global_params["categories"]
#     triggers_dict = global_params['hist_triggers']
#     period = global_params["era"]
#     for ch in channels_to_consider:
#         triggers = triggers_dict[ch]['default']
#         if period in triggers_dict[ch].keys():
#             triggers = triggers_dict[ch][period]
#         for reg in sign_regions_to_consider:
#             for cat in categories_to_consider:
#                 filter_base = f" ({ch} && {triggers}&& {reg} && {cat})"
#                 filter_str = f"(" + filter_base
#                 filter_str += ")"
#                 key = (ch, reg, cat)
#                 filter_dict[key] = filter_str
#     return filter_dict

def createKeyFilterDict(global_params, period):
    filter_dict = {}
    filter_str = ""
    channels_to_consider = global_params['channels_to_consider']
    sign_regions_to_consider = global_params['MuMuMassRegions']
    categories_to_consider = global_params["categories"]
    triggers_dict = global_params['hist_triggers']
    for ch in channels_to_consider:
        triggers = triggers_dict[ch]['default']
        if period in triggers_dict[ch].keys():
            triggers = triggers_dict[ch][period]
        for reg in sign_regions_to_consider:
            for cat in categories_to_consider:
                filter_base = f" ({ch} && {triggers}&& {reg} && {cat})"
                filter_str = f"(" + filter_base
                filter_str += ")"
                key = (ch, reg, cat)
                filter_dict[key] = filter_str
    return filter_dict

def GetBTagWeight(global_cfg_dict,cat,applyBtag=False):
    btag_weight = "1"
    btagshape_weight = "1"
    global_cfg_dict["ApplyBweight"]
    if global_cfg_dict["ApplyBweight"]:
        if applyBtag:
            if global_cfg_dict['btag_wps'][cat]!='' : btag_weight = f"weight_bTagSF_{btag_wps[cat]}_Central"
        else:
            if cat not in global_cfg_dict['boosted_categories'] and not cat.startswith("baseline"):
                btagshape_weight = "weight_bTagShape_Central"
    return f'{btag_weight}*{btagshape_weight}'

def VBFJetSelection(df):
    if "SelectedJet_idx" not in df.GetColumnNames():
        print("SelectedJet_idx not in df.GetColumnNames")
        df = df.Define(f"SelectedJet_idx", f"CreateIndexes(SelectedJet_pt.size())")
    df = df.Define(f"SelectedJet_p4", f"GetP4(SelectedJet_pt, SelectedJet_eta, SelectedJet_phi, SelectedJet_mass, SelectedJet_idx)")
    # df = df.Define(f"JetVeto", f"SelectedJet_btagDeepFlavB")
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/#ak4-b-tagging
    df = df.Define("Jet_Veto_loose", "SelectedJet_btagPNetB >= 0.0499")  # 0.0499 is the loose working point for PNet B-tagging in Run3
    df = df.Define("Jet_Veto_medium", "SelectedJet_btagPNetB >= 0.2605")  # 0.2605 is the medium working point for PNet B-tagging in Run3
    # df = df.Define("Jet_Veto_tight", "SelectedJet_btagPNetB >= 0.6484")  # 0.6484 is the tight working point for PNet B-tagging in Run3

    df = df.Define("SelectedJet_vetoMap_inverted", "!SelectedJet_vetoMap").Define("Jet_preselection","SelectedJet_p4[Jet_Veto_medium].size() < 1  && SelectedJet_p4[Jet_Veto_loose].size() < 2 && SelectedJet_p4[SelectedJet_vetoMap_inverted].size()>0 ") # "Remove events with at least one medium b-tagged jet and events with at least two loose b-tagged jets")

    df = df.Define("VBFJetCand","FindVBFJets(SelectedJet_p4)")
    df = df.Define("HasVBF", "return static_cast<bool>(VBFJetCand.isVBF)")
    df = df.Define("m_jj", "if (HasVBF) return static_cast<float>(VBFJetCand.m_inv); return -1000.f")
    df = df.Define("delta_eta_jj", "if (HasVBF) return static_cast<float>(VBFJetCand.eta_separation); return -1000.f")
    df = df.Define("j1_idx", "if (HasVBF) return static_cast<int>(VBFJetCand.leg_index[0]); return -1000; ")
    df = df.Define("j2_idx", "if (HasVBF) return static_cast<int>(VBFJetCand.leg_index[1]); return -1000; ")
    df = df.Define("j1_pt", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Pt()); return -1000.f; ")
    df = df.Define("j2_pt", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Pt()); return -1000.f; ")
    df = df.Define("j1_eta", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Eta()); return -1000.f; ")
    df = df.Define("j2_eta", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Eta()); return -1000.f; ")
    df = df.Define("j1_phi", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Phi()); return -1000.f; ")
    df = df.Define("j2_phi", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Phi()); return -1000.f; ")
    df = df.Define("j1_y", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Rapidity()); return -1000.f; ")
    df = df.Define("j2_y", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Rapidity()); return -1000.f; ")
    df = df.Define("delta_phi_jj", "if (HasVBF) return static_cast<float>(ROOT::Math::VectorUtil::DeltaPhi( VBFJetCand.leg_p4[0], VBFJetCand.leg_p4[1] ) ); return -1000.f;")
    df = df.Define(f"pt_jj", "(VBFJetCand.leg_p4[0]+VBFJetCand.leg_p4[1]).Phi()")
    df = df.Define("VBFjets_pt", f"RVecF void_pt {{}} ; if (HasVBF) return v_ops::pt(VBFJetCand.legs_p4); return void_pt;" )
    df = df.Define("VBFjets_eta", f"RVecF void_eta {{}} ; if (HasVBF) return v_ops::eta(VBFJetCand.legs_p4); return void_eta;" )
    df = df.Define("VBFjets_phi", f"RVecF void_phi {{}} ; if (HasVBF) return v_ops::phi(VBFJetCand.legs_p4); return void_phi;" )
    df = df.Define("VBFjets_y", f"RVecF void_y {{}} ; if (HasVBF) return v_ops::rapidity(VBFJetCand.legs_p4); return void_y;" )
    for var in JetObservables:
        if f"SelectedJet_{var}" not in df.GetColumnNames():
            continue
        if f"j1_{var}" not in df.GetColumnNames():
            df = df.Define("j1_"+var, f"if (HasVBF && j1_idx >= 0) return static_cast<float>(SelectedJet_{var}[j1_idx]); return -1000.f;")
        if f"j2_{var}" not in df.GetColumnNames():
            df = df.Define("j2_"+var, f"if (HasVBF && j2_idx >= 0) return static_cast<float>(SelectedJet_{var}[j2_idx]); return -1000.f;")
    return df

def GetMuMuObservables(df):
    for idx in [0,1]:
        df = Utilities.defineP4(df, f"mu{idx+1}")
    df = df.Define(f"pt_mumu", "(mu1_p4+mu2_p4).Pt()")
    df = df.Define(f"y_mumu", "(mu1_p4+mu2_p4).Rapidity()")
    df = df.Define(f"eta_mumu", "(mu1_p4+mu2_p4).Eta()")
    df = df.Define(f"phi_mumu", "(mu1_p4+mu2_p4).Phi()")
    df = df.Define("m_mumu", "static_cast<float>((mu1_p4+mu2_p4).M())")
    for idx in [0,1]:
        df = df.Define(f"mu{idx+1}_pt_rel", f"mu{idx+1}_pt/m_mumu")
    df = df.Define("dR_mumu", 'ROOT::Math::VectorUtil::DeltaR(mu1_p4, mu2_p4)')

    df = df.Define("Ebeam", "13600.0/2")
    df = df.Define("cosTheta_Phi_CS","ComputeCosThetaPhiCS(mu1_p4, mu2_p4,  Ebeam)")
    df = df.Define("cosTheta_CS", "static_cast<float>(std::get<0>(cosTheta_Phi_CS))")
    df = df.Define("phi_CS", "static_cast<float>(std::get<1>(cosTheta_Phi_CS))")
    return df

def GetMuMuMassResolution(df):
    delta_mu_expr = "sqrt( 0.5 * (pow( ({0}/{1}), 2) + pow( ({2}/{3}), 2) ) ) "
    df = df.Define("m_mumu_resolution", delta_mu_expr.format("mu1_pt", "(mu1_pt-mu1_pt_nano)/mu1_pt", "mu2_pt", "(mu2_pt-mu2_pt_nano)/mu2_pt"))
    df = df.Define("m_mumu_resolution_relerr", delta_mu_expr.format("mu1_pt", "(mu1_pt-mu1_pt_nano)", "mu2_pt", "(mu2_pt-mu2_pt_nano)"))
    df = df.Define("m_mumu_resolution_bsConstrained", delta_mu_expr.format("mu1_bsConstrainedPt", "mu1_bsConstrainedPtErr", "mu2_bsConstrainedPt", "mu2_bsConstrainedPtErr"))
    return df

def VBFJetMuonsObservables(df):
    df = df.Define("Zepperfield_Var", "if (HasVBF) return static_cast<float>((y_mumu - 0.5*(j1_y+j2_y))/std::abs(j1_y - j2_y)); return -10000.f;")
    df = df.Define("pT_all_sum", "if(HasVBF) return static_cast<float>(pT_sum ({mu1_p4, mu2_p4, VBFJetCand.leg_p4[0], VBFJetCand.leg_p4[1]})); return -10000.f;")
    df = df.Define("R_pt", "if(HasVBF) return static_cast<float>((pT_all_sum)/(pt_mumu + j1_pt + j2_pt)); return -10000.f;")
    df = df.Define("pT_jj_sum", "if(HasVBF) return static_cast<float>(pT_sum ({VBFJetCand.leg_p4[0], VBFJetCand.leg_p4[1]})); return -10000.f;")
    df = df.Define("pt_centrality", "if(HasVBF) return static_cast<float>(( (pt_mumu-0.5*(pT_jj_sum)) / pT_diff(VBFJetCand.leg_p4[0], VBFJetCand.leg_p4[1]) )); return -10000.f;")

    df = df.Define("minDeltaPhi", "if(HasVBF) return static_cast<float>(std::min(ROOT::Math::VectorUtil::DeltaPhi( (mu1_p4+mu2_p4), VBFJetCand.leg_p4[0]), ROOT::Math::VectorUtil::DeltaPhi((mu1_p4+mu2_p4), VBFJetCand.leg_p4[1]) ) )  ; return -10000.f;")
    df = df.Define("minDeltaEta", "if(HasVBF) return static_cast<float>(std::min(std::abs(eta_mumu - j1_eta),std::abs(eta_mumu - j2_eta))) ; return -10000.f;")
    df = df.Define("minDeltaEtaSigned", "if(HasVBF) return static_cast<float>(std::min((eta_mumu - j1_eta),(eta_mumu - j2_eta))) ; return -10000.f;")

    return df

def GetSoftJets(df):
    df = df.Define("SoftJet_def_vtx", "(SelectedJet_svIdx1 < 0 && SelectedJet_svIdx2< 0 ) ")  # no secondary vertex associated
    df = df.Define("SoftJet_def_pt", " (SelectedJet_pt>2) ") # pT > 2 GeV
    df = df.Define("SoftJet_def_muon", "(SelectedJet_idx != mu1_jetIdx && SelectedJet_idx != mu2_jetIdx)") # TMP PATCH. For next round it will be changed to the commented one in next line --> the muon index of the jets (because there can be muons associated to jets) has to be different than the signal muons (i.e. those coming from H decay)

    # df = df.Define("SoftJet_def_muon", "(Jet_muonIdx1 != mu1_idx && Jet_muonIdx1 != mu2_idx && Jet_muonIdx2 != mu1_idx && Jet_muonIdx2 != mu2_idx)") # mu1_idx and mu2_idx are not present in the current anaTuples, but need to be introduced for next round . The idx is the index in the original muon collection as well as Jet_muonIdx()

    df = df.Define("SoftJet_def_VBF", " (HasVBF && SelectedJet_idx != j1_idx && SelectedJet_idx != j2_idx) ") # if it is a VBF event, the soft jets are not the VBF jets
    df = df.Define("SoftJet_def_noVBF", " (!(HasVBF)) ")

    df = df.Define("SoftJet_def", "SoftJet_def_vtx && SoftJet_def_pt && SoftJet_def_muon && (SoftJet_def_VBF || SoftJet_def_noVBF )")

    df = df.Define("N_softJet", "SelectedJet_p4[SoftJet_def].size()")
    df = df.Define("SoftJet_energy", "v_ops::energy(SelectedJet_p4[SoftJet_def])")
    df = df.Define("SoftJet_Et", "v_ops::Et(SelectedJet_p4[SoftJet_def])")
    df = df.Define("SoftJet_HtCh_fraction", "SelectedJet_chHEF[SoftJet_def]")
    df = df.Define("SoftJet_HtNe_fraction", "SelectedJet_neHEF[SoftJet_def]")
    df = df.Define("SoftJet_HtHF_fraction", "SelectedJet_hfHEF[SoftJet_def]")
    for var in JetObservables:
        if f"SoftJet_{var}" not in df.GetColumnNames():
            if f"SoftJet_{var}" not in df.GetColumnNames() and f"SelectedJet_{var}" in df.GetColumnNames():
                df = df.Define(f"SoftJet_{var}", f"SelectedJet_{var}[SoftJet_def]")
    for var in JetObservablesMC:
        if f"SoftJet_{var}" not in df.GetColumnNames() and f"SelectedJet_{var}" in df.GetColumnNames():
            df = df.Define(f"SoftJet_{var}", f"SelectedJet_{var}[SoftJet_def]")
    return df


def defineP4AndInvMass(df):
    if "SelectedJet_idx" not in df.GetColumnNames():
        print("SelectedJet_idx not in df.GetColumnNames")
        df = df.Define(f"SelectedJet_idx", f"CreateIndexes(SelectedJet_pt.size())")
    df = df.Define(f"SelectedJet_p4", f"GetP4(SelectedJet_pt, SelectedJet_eta, SelectedJet_phi, SelectedJet_mass, SelectedJet_idx)")
    for idx in [0,1]:
        df = Utilities.defineP4(df, f"mu{idx+1}")
    # for met_var in ['met','metnomu']:
    #     df = df.Define(f"{met_var}_p4", f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>({met_var}_pt,0.,{met_var}_phi,0.)")
    #     for leg_idx in [0,1]:
    #         df = df.Define(f"deltaPhi_{met_var}_tau{leg_idx+1}",f"ROOT::Math::VectorUtil::DeltaPhi({met_var}_p4,tau{leg_idx+1}_p4)")
            # df = df.Define(f"deltaPhi_{met_var}_b{leg_idx+1}",f"ROOT::Math::VectorUtil::DeltaPhi({met_var}_p4,b{leg_idx+1}_p4)")
    # df = df.Define(f"met_nano_p4", f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(met_pt_nano,0.,met_phi_nano,0.)")
    df = df.Define(f"pt_ll", "(mu1_p4+mu2_p4).Pt()")

    df = df.Define("m_mumu", "static_cast<float>((mu1_p4+mu2_p4).M())")
    df = df.Define("dR_mumu", 'ROOT::Math::VectorUtil::DeltaR(mu1_p4, mu2_p4)')

    return df


def SaveVarsForNNInput(vars_to_save):
    mumu_vars = ["pt_mumu","y_mumu","eta_mumu","phi_mumu","m_mumu","dR_mumu","cosTheta_CS","mu1_pt_rel","mu2_pt_rel","mu1_eta","mu2_eta","phi_CS"]#, "Ebeam"
    jj_vars = ["j1_pt","j1_eta","j2_pt","j2_eta","HasVBF","m_jj","delta_eta_jj"] #,"j1_idx","j1_y","j1_phi","delta_phi_jj"
    mumu_jj_vars = ["Zepperfield_Var", "R_pt",  "pt_centrality", "minDeltaPhi", "minDeltaEta", "minDeltaEtaSigned"]#, "pT_all_sum","pT_jj_sum",
    softJets_vars = ["N_softJet", "SoftJet_energy","SoftJet_Et","SoftJet_HtCh_fraction","SoftJet_HtNe_fraction","SoftJet_HtHF_fraction" ]# ATTENTION: THESE ARE VECTORS, NOT FLAT OBSERVABLES
    global_vars = ["entryIndex","luminosityBlock", "run","event", "sample_type", "sample_name", "period", "isData", "nJet"] # ,"PV_npvs"
    # global_vars = ["FullEventId","luminosityBlock", "run","event", "sample_type", "sample_name", "period", "isData", "nJet"] # ,"PV_npvs"
    for var in global_vars + mumu_vars + jj_vars + mumu_jj_vars + softJets_vars:
        vars_to_save.append(var)
    return vars_to_save


def GetWeight(channel, A, B):
    weights_to_apply = ["weight_MC_Lumi_pu", "weight_EWKCorr_VptCentral", "weight_DYw_DYWeightCentral"]#,"weight_EWKCorr_ewcorrCentral"] #

    trg_weights_dict = {
        'muMu':["weight_trigSF_singleMu"],
    }
    ID_weights_dict = {
        "muMu": ["weight_mu1_HighPt_MuonID_SF_MediumIDCentral",  "weight_mu1_HighPt_MuonID_SF_RecoCentral",  "weight_mu1_LowPt_MuonID_SF_MediumIDCentral",  "weight_mu1_MuonID_SF_LoosePFIsoCentral",  "weight_mu1_MuonID_SF_MediumIDLoosePFIsoCentral",  "weight_mu1_MuonID_SF_MediumID_TrkCentral",  "weight_mu2_HighPt_MuonID_SF_MediumIDCentral",  "weight_mu2_HighPt_MuonID_SF_RecoCentral",  "weight_mu2_LowPt_MuonID_SF_MediumIDCentral",  "weight_mu2_MuonID_SF_LoosePFIsoCentral",  "weight_mu2_MuonID_SF_MediumIDLoosePFIsoCentral",  "weight_mu2_MuonID_SF_MediumID_TrkCentral"]
        }
    # should be moved to config
    # what about :
    #   weight_mu1_HighPt_MuonID_SF_MediumIDLooseRelIsoHLTCentral ?
    #   weight_mu2_HighPt_MuonID_SF_MediumIDLooseRelIsoHLTCentral
    #   weight_mu1_HighPt_MuonID_SF_MediumIdLooseRelTkIsoCentral ?
    #   weight_mu2_HighPt_MuonID_SF_MediumIdLooseRelTkIsoCentral ?
    weights_to_apply.extend(ID_weights_dict[channel])
    weights_to_apply.extend(trg_weights_dict[channel])

    total_weight = '*'.join(weights_to_apply)
    return total_weight

class DataFrameBuilderForHistograms(DataFrameBuilderBase):

    def defineTriggers(self):
        for ch in self.config['channelSelection']:
            for trg in self.config['triggers'][ch]:
                trg_name = 'HLT_'+trg
                self.colToSave.append(trg_name)
                if trg_name not in self.df.GetColumnNames():
                    print(f"{trg_name} not present in colNames")
                    self.df = self.df.Define(trg_name, "1")


    def defineRegions(self):
        region_defs = self.config['MuMuMassRegions']
        for reg_name, reg_cut in region_defs.items():
            self.df = self.df.Define(reg_name, reg_cut)
            self.colToSave.append(reg_name)

    def SignRegionDef(self):
        self.df = self.df.Define("OS", "mu1_charge*mu2_charge < 0")
        self.colToSave.append("OS")
        self.df = self.df.Define("SS", "!OS")
        self.colToSave.append("SS")

    def defineCategories(self): # needs lot of stuff --> at the end
        singleMuTh = self.config["singleMu_th"][self.period]
        for category_to_def in self.config['category_definition'].keys():
            category_name = category_to_def
            cat_str = self.config['category_definition'][category_to_def].format(MuPtTh=singleMuTh)
            self.df = self.df.Define(category_to_def, cat_str)
            self.colToSave.append(category_to_def)

    def defineChannels(self):
        self.df = self.df.Define(f"muMu", f"return true;")
        self.colToSave.append("muMu")

    def __init__(self, df, config, period, isData=False, isCentral=False, wantTriggerSFErrors=False, colToSave=[]):
        super(DataFrameBuilderForHistograms, self).__init__(df)
        self.config = config
        self.isData = isData
        self.period = period
        self.isCentral = isCentral
        self.colToSave = colToSave
        self.wantTriggerSFErrors = wantTriggerSFErrors


def PrepareDfForHistograms(dfForHistograms):
    # dfForHistograms.df = defineP4AndInvMass(dfForHistograms.df)
    dfForHistograms.defineChannels()
    dfForHistograms.defineTriggers()
    dfForHistograms.df = GetMuMuObservables(dfForHistograms.df)
    dfForHistograms.df = GetMuMuMassResolution(dfForHistograms.df)
    dfForHistograms.df = VBFJetSelection(dfForHistograms.df)
    dfForHistograms.df = VBFJetMuonsObservables(dfForHistograms.df)
    dfForHistograms.df = GetSoftJets(dfForHistograms.df)
    if not dfForHistograms.isData:
        defineTriggerWeights(dfForHistograms)
        if dfForHistograms.wantTriggerSFErrors and dfForHistograms.isCentral:
            defineTriggerWeightsErrors(dfForHistograms)
    dfForHistograms.SignRegionDef()
    dfForHistograms.defineRegions()
    dfForHistograms.defineCategories()
    return dfForHistograms


def PrepareDfForNNInputs(dfBuilder):
    dfBuilder.df = GetMuMuObservables(dfBuilder.df)
    dfBuilder.df = GetMuMuMassResolution(dfBuilder.df)
    dfBuilder.defineSignRegions()
    dfBuilder.df = VBFJetSelection(dfBuilder.df)
    dfBuilder.df = VBFJetMuonsObservables(dfBuilder.df)
    dfBuilder.df = GetSoftJets(dfBuilder.df)
    # dfBuilder.defineRegions()
    dfBuilder.defineCategories()
    dfBuilder.colToSave = SaveVarsForNNInput(dfBuilder.colToSave)
    return dfBuilder



# def addNewCols(self):
#     self.colNames = []
#     self.colTypes = []
#     colNames = [str(c) for c in self.df.GetColumnNames()] #if 'kinFit_result' not in str(c)]
#     cols_to_remove = []
#     for colName in colNames:
#         col_name_split = colName.split("_")
#         if "p4" in col_name_split or "vec" in col_name_split:
#             cols_to_remove.append(colName)
#     for col_to_remove in cols_to_remove:
#         colNames.remove(col_to_remove)
#     FullEventIdIdx = colNames.index("FullEventId")
#     runIdx = colNames.index("run")
#     eventIdx = colNames.index("event")
#     lumiIdx = colNames.index("luminosityBlock")
#     colNames[FullEventIdIdx], colNames[0] = colNames[0], colNames[FullEventIdIdx]
#     colNames[runIdx], colNames[1] = colNames[1], colNames[runIdx]
#     colNames[eventIdx], colNames[2] = colNames[2], colNames[eventIdx]
#     colNames[lumiIdx], colNames[3] = colNames[3], colNames[lumiIdx]
#     self.colNames = colNames
#     self.colTypes = [str(self.df.GetColumnType(c)) for c in self.colNames]
#     for colName,colType in zip(self.colNames,self.colTypes):
#         print(colName,colType)