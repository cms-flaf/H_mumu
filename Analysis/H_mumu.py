import ROOT
if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

from FLAF.Analysis.HistHelper import *
from Analysis.GetTriggerWeights import *
from FLAF.Common.Utilities import *


def createKeyFilterDict(global_cfg_dict, year):
    filter_dict = {}
    filter_str = ""
    channels_to_consider = global_cfg_dict['channels_to_consider']
    sign_regions_to_consider = global_cfg_dict['SignRegions']
    categories_to_consider = global_cfg_dict["categories"]
    triggers_dict = global_cfg_dict['hist_triggers']
    for ch in channels_to_consider:
        triggers = triggers_dict[ch]['default']
        if year in triggers_dict[ch].keys():
            triggers = triggers_dict[ch][year]
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



def GetWeight(channel, cat, boosted_categories):
    weights_to_apply = ["weight_MC_Lumi_pu"] #  "weight_L1PreFiring_Central"
    trg_weights_dict = {
        'muMu':["weight_trigSF_singleMu"],
    }
    ID_weights_dict = {
        "muMu": ["weight_mu1_MuonID_SF_LoosePFIsoCentral","weight_mu1_MuonID_SF_TightID_TrkCentral","weight_mu2_MuonID_SF_LoosePFIsoCentral","weight_mu2_MuonID_SF_TightID_TrkCentral","weight_mu1_TrgSF_singleMu_Central","weight_mu2_TrgSF_singleMu_Central"]
        # 'muMu': ["weight_mu1_HighPt_MuonID_SF_Reco_Central", "weight_mu1_HighPt_MuonID_SF_TightID_Central", "weight_mu1_MuonID_SF_Reco_Central", "weight_mu1_MuonID_SF_TightID_Trk_Central", "weight_mu1_MuonID_SF_TightRelIso_Central", "weight_mu2_HighPt_MuonID_SF_Reco_Central", "weight_mu2_HighPt_MuonID_SF_TightID_Central", "weight_mu2_MuonID_SF_Reco_Central", "weight_mu2_MuonID_SF_TightID_Trk_Central", "weight_mu2_MuonID_SF_TightRelIso_Central","weight_mu1_TrgSF_singleMu_Central","weight_mu2_TrgSF_singleMu_Central"],
        }

    weights_to_apply.extend(ID_weights_dict[channel])
    weights_to_apply.extend(trg_weights_dict[channel])
    # if cat not in boosted_categories:
    #      weights_to_apply.extend(["weight_Jet_PUJetID_Central_b1_2", "weight_Jet_PUJetID_Central_b2_2"])
    # else:
    #     weights_to_apply.extend(["weight_pNet_Central"])
    total_weight = '*'.join(weights_to_apply)
    return total_weight

class DataFrameBuilderForHistograms(DataFrameBuilderBase):

    def defineTriggers(self):
        for ch in self.config['channelSelection']:
            for trg in self.config['triggers'][ch]:
                trg_name = 'HLT_'+trg
                if trg_name not in self.df.GetColumnNames():
                    print(f"{trg_name} not present in colNames")
                    self.df = self.df.Define(trg_name, "1")

    def defineSignRegions(self):
        self.df = self.df.Define("OS", "mu1_charge*mu2_charge < 0")
        self.df = self.df.Define("SS", "!OS")


    # def defineCRs(self): # needs inv mass def
    #     SR_mass_limits_bb_boosted = self.config['mass_cut_limits']['bb_m_vis']['boosted']
    #     SR_mass_limits_bb = self.config['mass_cut_limits']['bb_m_vis']['other']
    #     SR_mass_limits_tt = self.config['mass_cut_limits']['tautau_m_vis']
    #     self.df = self.df.Define("SR_tt", f"return (tautau_m_vis > {SR_mass_limits_tt[0]} && tautau_m_vis  < {SR_mass_limits_tt[1]});")
    #     self.df = self.df.Define("SR_bb", f"(bb_m_vis > {SR_mass_limits_bb[0]} && bb_m_vis < {SR_mass_limits_bb[1]});")
    #     self.df = self.df.Define("SR_bb_boosted", f"(bb_m_vis_softdrop > {SR_mass_limits_bb_boosted[0]} && bb_m_vis_softdrop < {SR_mass_limits_bb_boosted[1]});")
    #     self.df = self.df.Define("SR", f" SR_tt &&  SR_bb")
    #     self.df = self.df.Define("SR_boosted", f" SR_tt &&  SR_bb_boosted")


    #     self.df = self.df.Define("DYCR", "if(muMu || eE) {return (tautau_m_vis < 100 && tautau_m_vis > 80);} return true;")
    #     self.df = self.df.Define("DYCR_boosted", "DYCR")


    #     TTCR_mass_limits_eTau = self.config['TTCR_mass_limits']['eTau']
    #     TTCR_mass_limits_muTau = self.config['TTCR_mass_limits']['muTau']
    #     TTCR_mass_limits_tauTau = self.config['TTCR_mass_limits']['tauTau']
    #     TTCR_mass_limits_muMu = self.config['TTCR_mass_limits']['muMu']
    #     TTCR_mass_limits_eE = self.config['TTCR_mass_limits']['eE']
    #     self.df = self.df.Define("TTCR", f"""
    #                             if(eTau) {{return (tautau_m_vis < {TTCR_mass_limits_eTau[0]} || tautau_m_vis > {TTCR_mass_limits_eTau[1]});
    #                             }};
    #                              if(muTau) {{return (tautau_m_vis < {TTCR_mass_limits_muTau[0]} || tautau_m_vis > {TTCR_mass_limits_muTau[1]});
    #                              }};
    #                              if(tauTau) {{return (tautau_m_vis < {TTCR_mass_limits_tauTau[0]} || tautau_m_vis > {TTCR_mass_limits_tauTau[1]});
    #                              }};
    #                              if(muMu) {{return (tautau_m_vis < {TTCR_mass_limits_muMu[0]} || tautau_m_vis > {TTCR_mass_limits_muMu[1]});
    #                              }};
    #                              if(eE) {{return (tautau_m_vis < {TTCR_mass_limits_eE[0]} || tautau_m_vis > {TTCR_mass_limits_eE[1]});
    #                              }};
    #                              return true;""")defineTriggerWeights
    #     self.df = self.df.Define("TTCR_boosted", "TTCR")

    def defineCategories(self): # needs lot of stuff --> at the end
        for category_to_def in self.config['category_definition'].keys():
            category_name = category_to_def
            self.df = self.df.Define(category_to_def, self.config['category_definition'][category_to_def])#.format(region=self.region))

    def defineChannels(self):
        self.df = self.df.Define(f"muMu", f"return true;")
        # for channel in self.config['channelSelection']:
        #     ch_value = self.config['channelDefinition'][channel]
        #     self.df = self.df.Define(f"{channel}", f"channelId=={ch_value}")


    # def defineLeptonPreselection(self): # needs channel def
    #     if self.period == 'Run2_2016' or self.period == 'Run2_2016_HIPM':
    #         self.df = self.df.Define("eleEta2016", "if(eE) {return (abs(mu1_eta) < 2 && abs(mu2_eta)<2); } if(eTau||eMu) {return (abs(mu1_eta) < 2); } return true;")
    #     else:
    #         self.df = self.df.Define("eleEta2016", "return true;")
    #     self.df = self.df.Define("muon1_tightId", "if(muTau || muMu) {return (mu1_Muon_tightId && mu1_Muon_pfRelIso04_all < 0.15); } return true;")
    #     self.df = self.df.Define("muon2_tightId", "if(muMu || eMu) {return (mu2_Muon_tightId && mu2_Muon_pfRelIso04_all < 0.3);} return true;")
    #     self.df = self.df.Define("firstele_mvaIso", "if(eMu || eE){return mu1_Electron_mvaIso_WP80==1 && mu1_Electron_pfRelIso03_all < 0.15 ; } return true; ")
    #     self.df = self.df.Define("mu1_iso_medium", f"if(tauTau) return (mu1_idDeepTau{self.deepTauYear()}v{self.deepTauVersion}VSjet >= {Utilities.WorkingPointsTauVSjet.Medium.value}); return true;")
    #     if f"mu1_gen_kind" not in self.df.GetColumnNames():
    #         self.df=self.df.Define("mu1_gen_kind", "if(isData) return 5; return 0;")
    #     if f"mu2_gen_kind" not in self.df.GetColumnNames():
    #         self.df=self.df.Define("mu2_gen_kind", "if(isData) return 5; return 0;")
    #     self.df = self.df.Define("tau_true", f"""(mu1_gen_kind==5 && mu2_gen_kind==5)""")
    #     self.df = self.df.Define(f"lepton_preselection", "eleEta2016 && mu1_iso_medium && muon1_tightId && muon2_tightId && firstele_mvaIso")

    def addNewCols(self):
        self.colNames = []
        self.colTypes = []
        colNames = [str(c) for c in self.df.GetColumnNames()] #if 'kinFit_result' not in str(c)]
        cols_to_remove = []
        for colName in colNames:
            col_name_split = colName.split("_")
            if "p4" in col_name_split or "vec" in col_name_split:
                cols_to_remove.append(colName)
        for col_to_remove in cols_to_remove:
            colNames.remove(col_to_remove)
        entryIndexIdx = colNames.index("entryIndex")
        runIdx = colNames.index("run")
        eventIdx = colNames.index("event")
        lumiIdx = colNames.index("luminosityBlock")
        colNames[entryIndexIdx], colNames[0] = colNames[0], colNames[entryIndexIdx]
        colNames[runIdx], colNames[1] = colNames[1], colNames[runIdx]
        colNames[eventIdx], colNames[2] = colNames[2], colNames[eventIdx]
        colNames[lumiIdx], colNames[3] = colNames[3], colNames[lumiIdx]
        self.colNames = colNames
        self.colTypes = [str(self.df.GetColumnType(c)) for c in self.colNames]
        for colName,colType in zip(self.colNames,self.colTypes):
            print(colName,colType)

    def __init__(self, df, config, period,isData=False, isCentral=False):
        super(DataFrameBuilderForHistograms, self).__init__(df)
        self.config = config
        self.isData = isData
        self.period = period
        self.isCentral = isCentral
        #  deepTauVersion='v2p1', bTagWPString = "Medium", pNetWPstring="Loose", region="SR", , wantTriggerSFErrors=False, whichType=3, wantScales=True
        # self.deepTauVersion = deepTauVersion
        # self.bTagWPString = bTagWPString
        # self.pNetWPstring = pNetWPstring
        # self.pNetWP = WorkingPointsParticleNet[period][pNetWPstring]
        # self.bTagWP = WorkingPointsDeepFlav[period][bTagWPString]
        # self.region = region
        # self.whichType = whichType
        # self.wantTriggerSFErrors = wantTriggerSFErrors
        # self.wantScales = isCentral and wantScales




def PrepareDfForHistograms(dfForHistograms):
    dfForHistograms.df = defineP4AndInvMass(dfForHistograms.df)
    dfForHistograms.defineChannels()
    dfForHistograms.defineTriggers()
    # dfForHistograms.df = createInvMass(dfForHistograms.df)
    if not dfForHistograms.isData:
        defineTriggerWeights(dfForHistograms)
        # if dfForHistograms.wantTriggerSFErrors and dfForHistograms.isCentral:
        #     defineTriggerWeightsErrors(dfForHistograms)
    dfForHistograms.defineCategories()
    dfForHistograms.defineSignRegions()
    return dfForHistograms



def defineP4AndInvMass(df):
    if "SelectedJet_idx" not in df.GetColumnNames():
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
