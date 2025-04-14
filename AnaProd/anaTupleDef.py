import AnaProd.baseline as AnaBaseline
import FLAF.Common.BaselineSelection as CommonBaseline
from Corrections.Corrections import Corrections

lepton_legs = [ "mu1", "mu2" ]

Muon_observables = ["IP_cov00","IP_cov10","IP_cov11","IP_cov20","IP_cov21","IP_cov22","IPx","IPy","IPz","bField_z","bsConstrainedChi2","bsConstrainedPt","bsConstrainedPtErr","charge","dxy","dxyErr","dxybs","dz","dzErr","eta","fsrPhotonIdx","genPartFlav","genPartIdx","highPtId","highPurity","inTimeMuon","ip3d","ipLengthSig","isGlobal","isPFcand","isStandalone","isTracker","jetIdx","jetNDauCharged","jetPtRelv2","jetRelIso","looseId","mass","mediumId","mediumPromptId","miniIsoId","miniPFRelIso_all","miniPFRelIso_chg","multiIsoId","mvaLowPt","mvaMuID","mvaMuID_WP","nStations","nTrackerLayers","pdgId","pfIsoId","pfRelIso03_all","pfRelIso03_chg","pfRelIso04_all","phi","promptMVA","pt","ptErr","puppiIsoId","segmentComp","sip3d","softId","softMva","softMvaId","softMvaRun3","svIdx","tightCharge","tightId","tkIsoId","tkRelIso","track_cov00","track_cov10","track_cov11","track_cov20","track_cov21","track_cov22","track_cov30","track_cov31","track_cov32","track_cov33","track_cov40","track_cov41","track_cov42","track_cov43","track_cov44","track_dsz","track_dxy","track_lambda","track_phi","track_qoverp","triggerIdLoose","tunepRelPt", "nMuon"]
# Electron_observables = ["Electron_mvaNoIso_WP80", "Electron_mvaIso_WP80","Electron_pfRelIso03_all"]
JetObservables = ["PNetRegPtRawCorr","PNetRegPtRawCorrNeutrino","PNetRegPtRawRes","area","btagDeepFlavB","btagDeepFlavCvB","btagDeepFlavCvL","btagDeepFlavQG","btagPNetB","btagPNetCvB","btagPNetCvL","btagPNetCvNotB","btagPNetQvG","btagPNetTauVJet","chEmEF","chHEF","chMultiplicity","electronIdx1","electronIdx2","eta","genJetIdx","hadronFlavour","hfEmEF","hfHEF","hfadjacentEtaStripsSize","hfcentralEtaStripSize","hfsigmaEtaEta","hfsigmaPhiPhi","jetId","mass","muEF","muonIdx1","muonIdx2","muonSubtrFactor","nConstituents","nElectrons","nMuons","nSVs","neEmEF","neHEF","neMultiplicity","partonFlavour","phi","pt","rawFactor","svIdx1","svIdx2"]

JetObservablesMC = ["hadronFlavour","partonFlavour"]

FatJetObservables = ["area", "btagCSVV2", "btagDDBvLV2", "btagDeepB", "btagHbb", "deepTagMD_HbbvsQCD",
                     "deepTagMD_ZHbbvsQCD", "deepTagMD_ZbbvsQCD", "deepTagMD_bbvsLight", "deepTag_H",
                     "jetId", "msoftdrop", "nBHadrons", "nCHadrons", "nConstituents","rawFactor",
                      "particleNetMD_QCD", "particleNetMD_Xbb", "particleNet_HbbvsQCD", "particleNet_mass", # 2018
                     "particleNet_QCD","particleNet_XbbVsQCD", # 2016
                     "particleNetLegacy_QCD", "particleNetLegacy_Xbb", "particleNetLegacy_mass", # 2016
                     "particleNetWithMass_QCD", "particleNetWithMass_HbbvsQCD", "particleNet_massCorr", # 2016
                     "ptRes", "idbtagPNetB"]

FatJetObservablesMC = ["hadronFlavour","partonFlavour"]

SubJetObservables = ["btagDeepB", "eta", "mass", "phi", "pt", "rawFactor"]
SubJetObservablesMC = ["hadronFlavour","partonFlavour"]

defaultColToSave = ["entryIndex","luminosityBlock", "run","event", "sample_type", "sample_name", "period", "isData","PuppiMET_pt", "PuppiMET_phi", "nJet","DeepMETResolutionTune_pt", "DeepMETResolutionTune_phi","DeepMETResponseTune_pt", "DeepMETResponseTune_phi","PV_npvs"]

def getDefaultColumnsToSave(isData):
    colToSave = defaultColToSave.copy()
    if not isData:
        colToSave.extend(['Pileup_nTrueInt'])
    return colToSave

def addAllVariables(dfw, syst_name, isData, trigger_class, lepton_legs, isSignal, global_params, channels):
    dfw.Apply(AnaBaseline.ObjReconstruction) # here go the reconstruction and vetos --> only two muons + veto
    dfw.Apply(AnaBaseline.ThirdLeptonVeto)
    dfw.Apply(Corrections.getGlobal().btag.getWPid)
    dfw.Apply(AnaBaseline.JetSelection, global_params["era"])
    dfw.Apply(Corrections.getGlobal().jet.getEnergyResolution)

    n_legs = 2

    for leg_idx in range(n_legs):
        def LegVar(var_name, var_expr, var_type=None, var_cond=None, check_leg_type=True, default=0):
            dfw.DefineAndAppend( f"mu{leg_idx+1}_{var_name}", define_expr)

        dfw.Define( f"mu{leg_idx+1}_idx", f"Hmumu_idx[{leg_idx}]")
        for var in Muon_observables:
            dfw.DefineAndAppend( f"mu{leg_idx+1}_{var}", f"Muon_{var}.at(mu{leg_idx+1}_idx)")

    for jetobs in JetObservables + ["idx"]:
        jet_obs_name = f"Jet_{jetobs}"
        if jet_obs_name in dfw.df.GetColumnNames():
            dfw.DefineAndAppend(f"SelectedJet_{jet_obs_name}", "Jet_{jet_obs_name}[Jet_B1]")

    # pf_str = global_params["met_type"]
    # dfw.DefineAndAppend(f"met_pt_nano", f"static_cast<float>({pf_str}_p4_nano.pt())")
    # dfw.DefineAndAppend(f"met_phi_nano", f"static_cast<float>({pf_str}_p4_nano.phi())")
    # dfw.DefineAndAppend("met_pt", f"static_cast<float>({pf_str}_p4.pt())")
    # dfw.DefineAndAppend("met_phi", f"static_cast<float>({pf_str}_p4.phi())")
    # dfw.DefineAndAppend("metnomu_pt_nano", f"static_cast<float>(GetMetNoMu(HttCandidate, {pf_str}_p4_nano).pt())")
    # dfw.DefineAndAppend("metnomu_phi_nano", f"static_cast<float>(GetMetNoMu(HttCandidate, {pf_str}_p4_nano).phi())")
    # dfw.DefineAndAppend("metnomu_pt", f"static_cast<float>(GetMetNoMu(HttCandidate, {pf_str}_p4).pt())")
    # dfw.DefineAndAppend("metnomu_phi", f"static_cast<float>(GetMetNoMu(HttCandidate, {pf_str}_p4).phi())")
    # for var in ["covXX", "covXY", "covYY"]:
    #     dfw.DefineAndAppend(f"met_{var}", f"static_cast<float>({pf_str}_{var})")

    if trigger_class is not None:
        hltBranches = dfw.Apply(trigger_class.ApplyMuonTriggers, lepton_legs, 'Htt',isData)
        dfw.colToSave.extend(hltBranches)
