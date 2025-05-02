import AnaProd.baseline as AnaBaseline
import FLAF.Common.BaselineSelection as CommonBaseline
from Corrections.Corrections import Corrections
from FLAF.Common.Utilities import *

lepton_legs = [ "mu1", "mu2" ]
loadTF = False
loadHHBtag = False


Muon_observables = ["IP_cov00","IP_cov10","IP_cov11","IP_cov20","IP_cov21","IP_cov22","IPx","IPy","IPz","bField_z","bsConstrainedChi2","bsConstrainedPt","bsConstrainedPtErr","charge","dxy","dxyErr","dxybs","dz","dzErr","eta","fsrPhotonIdx","genPartFlav","genPartIdx","highPtId","highPurity","inTimeMuon","ip3d","ipLengthSig","isGlobal","isPFcand","isStandalone","isTracker","jetIdx","jetNDauCharged","jetPtRelv2","jetRelIso","looseId","mass","mediumId","mediumPromptId","miniIsoId","miniPFRelIso_all","miniPFRelIso_chg","multiIsoId","mvaLowPt","mvaMuID","mvaMuID_WP","nStations","nTrackerLayers","pdgId","pfIsoId","pfRelIso03_all","pfRelIso03_chg","pfRelIso04_all","phi","promptMVA","pt","ptErr","puppiIsoId","segmentComp","sip3d","softId","softMva","softMvaId","softMvaRun3","svIdx","tightCharge","tightId","tkIsoId","tkRelIso","track_cov00","track_cov10","track_cov11","track_cov20","track_cov21","track_cov22","track_cov30","track_cov31","track_cov32","track_cov33","track_cov40","track_cov41","track_cov42","track_cov43","track_cov44","track_dsz","track_dxy","track_lambda","track_phi","track_qoverp","triggerIdLoose","tunepRelPt"]
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

defaultColToSave = ["FullEventId","luminosityBlock", "run","event", "sample_type", "period", "isData","PuppiMET_pt", "PuppiMET_phi", "nJet","DeepMETResolutionTune_pt", "DeepMETResolutionTune_phi","DeepMETResponseTune_pt", "DeepMETResponseTune_phi","PV_npvs"]
# defaultColToSave = ["entryIndex","luminosityBlock", "run","event", "sample_type", "sample_name", "period", "isData","PuppiMET_pt", "PuppiMET_phi", "nJet","DeepMETResolutionTune_pt", "DeepMETResolutionTune_phi","DeepMETResponseTune_pt", "DeepMETResponseTune_phi","PV_npvs"]

def getDefaultColumnsToSave(isData):
    colToSave = defaultColToSave.copy()
    if not isData:
        colToSave.extend(['Pileup_nTrueInt'])
    return colToSave

def addAllVariables(dfw, syst_name, isData, trigger_class, lepton_legs, isSignal, global_params, channels):
    dfw.Apply(AnaBaseline.RecoHttCandidateSelection, global_params)
    # dfw.Apply(AnaBaseline.ObjReconstruction) # here go the reconstruction and vetos --> only two muons + veto
    dfw.Apply(AnaBaseline.LeptonVeto)
    dfw.Apply(Corrections.getGlobal().btag.getWPid)
    dfw.Apply(AnaBaseline.JetSelection, global_params["era"])
    dfw.Apply(Corrections.getGlobal().jet.getEnergyResolution)
    dfw.Apply(AnaBaseline.GetMuMuCandidate)

    n_legs = 2

    for leg_idx in range(n_legs):
        def LegVar(var_name, var_expr, var_type=None, var_cond=None, check_leg_type=True, default=0):
            cond = var_cond
            if check_leg_type:
                type_cond = f"HttCandidate.leg_type[{leg_idx}] != Leg::none"
                cond = f"{type_cond} && ({cond})" if cond else type_cond
            define_expr = f'static_cast<{var_type}>({var_expr})' if var_type else var_expr
            if cond:
                define_expr = f'{cond} ? ({define_expr}) : {default}'
            dfw.DefineAndAppend( f"mu{leg_idx+1}_{var_name}", define_expr)

        LegVar('legType', f"HttCandidate.leg_type[{leg_idx}]", var_type='int', check_leg_type=False)
        for var in [ 'pt', 'eta', 'phi', 'mass' ]:
            LegVar(var, f'HttCandidate.leg_p4[{leg_idx}].{var}()', var_type='float', default='-1.f')
        LegVar('charge', f'HttCandidate.leg_charge[{leg_idx}]', var_type='int')

        dfw.Define(f"mu{leg_idx+1}_recoJetMatchIdx", f"""HttCandidate.leg_type[{leg_idx}] != Leg::none
                                                          ? FindMatching(HttCandidate.leg_p4[{leg_idx}], Jet_p4, 0.3)
                                                          : -1""")
        LegVar('iso', f"HttCandidate.leg_rawIso.at({leg_idx})")

        for muon_obs in Muon_observables:
            if f"mu{leg_idx+1}_{muon_obs}" in dfw.df.GetColumnNames(): continue
            if f"Muon_{muon_obs}" not in dfw.df.GetColumnNames(): continue

            LegVar(muon_obs, f"Muon_{muon_obs}.at(HttCandidate.leg_index[{leg_idx}])",
                   var_cond=f"HttCandidate.leg_type[{leg_idx}] == Leg::mu", default='-1.f')
        if not isData:
            dfw.Define(f"mu{leg_idx+1}_genMatchIdx",
                       f"HttCandidate.leg_type[{leg_idx}] != Leg::none ? HttCandidate.leg_genMatchIdx[{leg_idx}] : -1")
            LegVar('gen_kind', f'genLeptons.at(mu{leg_idx+1}_genMatchIdx).kind()',
                   var_type='int', var_cond=f"mu{leg_idx+1}_genMatchIdx>=0",
                   default='static_cast<int>(GenLeptonMatch::NoMatch)')
            LegVar('gen_charge', f'genLeptons.at(mu{leg_idx+1}_genMatchIdx).charge()',
                   var_type='int', var_cond=f"mu{leg_idx+1}_genMatchIdx>=0", default='-10')
        # if not isData:
        #     for var in [ 'pt', 'eta', 'phi', 'mass' ]:
        #         LegVar(f'gen_vis_{var}', f'genLeptons.at(tau{leg_idx+1}_genMatchIdx).visibleP4().{var}()',
        #                var_type='float', var_cond=f"tau{leg_idx+1}_genMatchIdx>=0", default='-1.f')
        #     LegVar('gen_nChHad', f'genLeptons.at(tau{leg_idx+1}_genMatchIdx).nChargedHadrons()',
        #            var_type='int', var_cond=f"tau{leg_idx+1}_genMatchIdx>=0", default='-1')
        #     LegVar('gen_nNeutHad', f'genLeptons.at(tau{leg_idx+1}_genMatchIdx).nNeutralHadrons()',
        #            var_type='int', var_cond=f"tau{leg_idx+1}_genMatchIdx>=0", default='-1')
        #     LegVar('seedingJet_partonFlavour', f'Jet_partonFlavour.at(tau{leg_idx+1}_recoJetMatchIdx)',
        #            var_type='int', var_cond=f"tau{leg_idx+1}_recoJetMatchIdx>=0", default='-10')
        #     LegVar('seedingJet_hadronFlavour', f'Jet_hadronFlavour.at(tau{leg_idx+1}_recoJetMatchIdx)',
        #            var_type='int', var_cond=f"tau{leg_idx+1}_recoJetMatchIdx>=0", default='-10')

        # for var in [ 'pt', 'eta', 'phi', 'mass' ]:
        #     LegVar(f'seedingJet_{var}', f"Jet_p4.at(tau{leg_idx+1}_recoJetMatchIdx).{var}()",
        #            var_type='float', var_cond=f"tau{leg_idx+1}_recoJetMatchIdx>=0", default='-1.f')

        #Save the lep* p4 and index directly to avoid using HwwCandidate in SF LUTs
        dfw.Define( f"mu{leg_idx+1}_p4", f"HttCandidate.leg_type.size() > {leg_idx} ? HttCandidate.leg_p4.at({leg_idx}) : LorentzVectorM()")
        dfw.Define( f"mu{leg_idx+1}_index", f"HttCandidate.leg_type.size() > {leg_idx} ? HttCandidate.leg_index.at({leg_idx}) : -1")
        dfw.Define( f"mu{leg_idx+1}_type", f"HttCandidate.leg_type.size() > {leg_idx} ? static_cast<int>(HttCandidate.leg_type.at({leg_idx})) : -1")

    # n_legs = 2

    # for leg_idx in range(n_legs):
    #     def LegVar(var_name, var_expr):
    #         dfw.DefineAndAppend( f"mu{leg_idx+1}_{var_name}", define_expr)

    #     dfw.Define( f"mu{leg_idx+1}_idx", f"Hmumu_idx[{leg_idx}]")
    #     for var in Muon_observables:
    #         dfw.DefineAndAppend( f"mu{leg_idx+1}_{var}", f"Muon_{var}.at(mu{leg_idx+1}_idx)")
    #     dfw.df = defineP4(dfw.df, f"mu{leg_idx+1}")
    #     print(f"mu{leg_idx+1}_p4" in dfw.df.GetColumnNames())
    #     # dfw.df.Display(f"mu{leg_idx+1}_p4.Pt()").Print()


    for jetobs in JetObservables + ["idx"]:
        jet_obs_name = f"Jet_{jetobs}"
        if jet_obs_name in dfw.df.GetColumnNames():
            dfw.DefineAndAppend(f"SelectedJet_{jetobs}", f"Jet_{jetobs}[Jet_B1]")

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
        print(f"mu{leg_idx+1}_p4" in dfw.df.GetColumnNames())

        hltBranches = dfw.Apply(trigger_class.ApplyTriggers, lepton_legs, 'Htt',isData, isSignal)
        # hltBranches = dfw.Apply(trigger_class.ApplyTriggers, isData)
        dfw.colToSave.extend(hltBranches)
