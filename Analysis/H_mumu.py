import ROOT
if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])
    # ROOT.gInterpreter.Declare(f'#include "include/HmumuCore.h"')

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
    weights_to_apply = ["weight_MC_Lumi_pu", "weight_EWKCorr_VptCentral_scaled1"]#,"weight_EWKCorr_ewcorrCentral"] #  "weight_L1PreFiring_Central"

    trg_weights_dict = {
        'muMu':["weight_trigSF_singleMu"],
    }
    ID_weights_dict = {
        "muMu": ["weight_mu1_MuonID_SF_MediumIDLooseIsoCentral","weight_mu2_MuonID_SF_MediumIDLooseIsoCentral","weight_mu1_MuonID_SF_MediumID_TrkCentral","weight_mu2_MuonID_SF_MediumID_TrkCentral"]# "weight_mu1_TrgSF_singleMu_Central","weight_mu2_TrgSF_singleMu_Central"]
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


    def defineRegions(self): # needs inv mass def
        self.df = self.df.Define("Inclusive", f" return true;")
        # print("inclusive")
        # print(self.df.Filter("Inclusive").Count().GetValue())
        self.df = self.df.Define("DYEnriched", f" return (m_mumu > 70 && m_mumu < 100);")
        # print("DYEnriched")
        # print(self.df.Filter("DYEnriched").Count().GetValue())



    def defineCategories(self): # needs lot of stuff --> at the end
        singleMuTh = self.config["singleMu_th"][self.period]
        # print(singleMuTh)
        for category_to_def in self.config['category_definition'].keys():
            category_name = category_to_def
            cat_str = self.config['category_definition'][category_to_def].format(MuPtTh=singleMuTh, region=self.region)
            # print(cat_str)
            self.df = self.df.Define(category_to_def, cat_str)
            # print(self.df.Filter(category_to_def).Count().GetValue())

    def defineChannels(self):
        self.df = self.df.Define(f"muMu", f"return true;")
        # for channel in self.config['channelSelection']:
        #     ch_value = self.config['channelDefinition'][channel]
        #     self.df = self.df.Define(f"{channel}", f"channelId=={ch_value}")


    def VBFJetSelection(self):
        ROOT.gROOT.ProcessLine(".include "+ os.environ['ANALYSIS_PATH'])
        ROOT.gInterpreter.Declare(f'#include "include/Helper.h"')
        self.df = self.df.Define("VBFJetCand","FindVBFJets(SelectedJet_p4)")
        self.df = self.df.Define("HasVBF", "return static_cast<bool>(VBFJetCand.isVBF)")
        self.df = self.df.Define("VBF_mInv", "if (HasVBF) return static_cast<float>(VBFJetCand.m_inv); return -1000.f")
        self.df = self.df.Define("VBF_etaSeparation", "if (HasVBF) return static_cast<float>(VBFJetCand.eta_separation); return -1000.f")
        self.df = self.df.Define("VBFjet1_idx", "if (HasVBF) return static_cast<int>(VBFJetCand.leg_index[0]); return -1000; ")
        self.df = self.df.Define("VBFjet2_idx", "if (HasVBF) return static_cast<int>(VBFJetCand.leg_index[1]); return -1000; ")
        self.df = self.df.Define("VBFjet1_pt", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Pt()); return -1000.f; ")
        self.df = self.df.Define("VBFjet2_pt", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Pt()); return -1000.f; ")
        self.df = self.df.Define("VBFjet1_eta", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Eta()); return -1000.f; ")

        self.df = self.df.Define("VBFjet2_eta", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Eta()); return -1000.f; ")
        self.df = self.df.Define("VBFjet1_phi", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Phi()); return -1000.f; ")
        self.df = self.df.Define("VBFjet2_phi", "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Phi()); return -1000.f; ")

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

    def __init__(self, df, config, period,region,isData=False, isCentral=False):
        super(DataFrameBuilderForHistograms, self).__init__(df)
        self.config = config
        self.isData = isData
        self.period = period
        self.isCentral = isCentral
        self.region = region
        #  deepTauVersion='v2p1', bTagWPString = "Medium", pNetWPstring="Loose", region="SR", , wantTriggerSFErrors=False, whichType=3, wantScales=True
        # self.deepTauVersion = deepTauVersion
        # self.bTagWPString = bTagWPString
        # self.pNetWPstring = pNetWPstring
        # self.pNetWP = WorkingPointsParticleNet[period][pNetWPstring]
        # self.bTagWP = WorkingPointsDeepFlav[period][bTagWPString]
        # self.whichType = whichType
        # self.wantTriggerSFErrors = wantTriggerSFErrors
        # self.wantScales = isCentral and wantScales




def PrepareDfForHistograms(dfForHistograms):
    dfForHistograms.df = defineP4AndInvMass(dfForHistograms.df)
    dfForHistograms.defineChannels()
    dfForHistograms.defineTriggers()
    dfForHistograms.VBFJetSelection()
    # dfForHistograms.df = createInvMass(dfForHistograms.df)
    if not dfForHistograms.isData:
        defineTriggerWeights(dfForHistograms)
        # if dfForHistograms.wantTriggerSFErrors and dfForHistograms.isCentral:
        #     defineTriggerWeightsErrors(dfForHistograms)
    dfForHistograms.defineRegions()
    dfForHistograms.defineCategories()
    dfForHistograms.defineSignRegions()
    return dfForHistograms



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
    ### currently putting it here ####
    if "weight_EWKCorr_VptCentral" in df.GetColumnNames():
        df = df.Define("weight_EWKCorr_VptCentral_scaled1", "1+weight_EWKCorr_VptCentral")
    return df
