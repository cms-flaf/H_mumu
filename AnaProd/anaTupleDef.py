import AnaProd.baseline as AnaBaseline
import FLAF.Common.BaselineSelection as CommonBaseline
from Corrections.Corrections import Corrections
from FLAF.Common.Utilities import *

lepton_legs = ["mu1", "mu2"]
offline_legs = ["mu1", "mu2"]
loadTF = False
loadHHBtag = False

Muon_observables_v15 = [
    "dxybs",
    "dxybsErr",
    "IPx",
    "IPy",
    "IPz",
    "VXBS_Cov00",
    "VXBS_Cov03",
    "VXBS_Cov33",
    "bestTrackType",
    "ipLengthSig",
    "jetDF",
    "pnScore_heavy",
    "pnScore_light",
    "pnScore_prompt",
    "pnScore_tau",
    "promptMVA",
    "softMvaRun3",
    "tuneP_charge",
    "tuneP_pterr",
]
Muon_observables_base = [
    "bsConstrainedChi2",
    "bsConstrainedPt",
    "bsConstrainedPtErr",
    "charge",
    "dxy",
    "dxyErr",
    "dz",
    "dzErr",
    "fsrPhotonIdx",
    "genMatch",
    "genMatchIdx",
    "highPtId",
    "highPurity",
    "inTimeMuon",
    "ip3d",
    "isGlobal",
    "isPFcand",
    "isStandalone",
    "isTracker",
    "jetIdx",
    "jetNDauCharged",
    "jetPtRelv2",
    "jetRelIso",
    "looseId",
    "mediumId",
    "mediumPromptId",
    "miniIsoId",
    "miniPFRelIso_all",
    "miniPFRelIso_chg",
    "multiIsoId",
    "mvaLowPt",
    "mvaMuID",
    "mvaMuID_WP",
    "nStations",
    "nTrackerLayers",
    "pdgId",
    "pfIsoId",
    "pfRelIso03_all",
    "pfRelIso03_chg",
    "pfRelIso04_all",
    "ptErr",
    "puppiIsoId",
    "segmentComp",
    "sip3d",
    "softId",
    "softMva",
    "softMvaId",
    "svIdx",
    "tightCharge",
    "tightId",
    "tkIsoId",
    "tkRelIso",
    "triggerIdLoose",
    "tunepRelPt",
]
MuonObservables_MC = ["genPartFlav", "genPartIdx"]

PrimaryVertexObservables = [
    "PVBS_chi2",
    "PVBS_cov00",
    "PVBS_cov10",
    "PVBS_cov11",
    "PVBS_cov20",
    "PVBS_cov21",
    "PVBS_cov22",
    "PVBS_x",
    "PVBS_y",
    "PVBS_z",
    "PV_score",
    "PV_z",
    "PV_chi2",
    "PV_ndof",
    "PV_npvs",
    "PV_npvsGood",
    "PV_score",
    "PV_sumpt2",
    "PV_sumpx",
    "PV_sumpy",
    "PV_x",
    "PV_y",
    "PV_z",
]

SecondaryVertexObservables = [
    "SV_jetIdx",
    "SV_sVIdx",
    "SV_charge",
    "SV_chi2",
    "SV_dlen",
    "SV_dlenSig",
    "SV_dxy",
    "SV_dxySig",
    "SV_eta",
    "SV_mass",
    "SV_ndof",
    "SV_ntracks",
    "SV_pAngle",
    "SV_phi",
    "SV_pt",
    "SV_x",
    "SV_y",
    "SV_z",
]

PUObservables = [
    "Pileup_gpudensity",
    "Pileup_nPU",
    "Pileup_pthatmax",
    "Pileup_pudensity",
    "Pileup_sumEOOT",
    "Pileup_sumLOOT",
]

JetObservables_base = [
    "PNetRegPtRawCorr",
    "PNetRegPtRawCorrNeutrino",
    "PNetRegPtRawRes",
    "area",
    "btagDeepFlavB",
    "btagDeepFlavCvB",
    "btagDeepFlavCvL",
    "btagDeepFlavQG",
    "btagPNetB",
    "btagPNetCvB",
    "btagPNetCvL",
    "btagPNetQvG",
    "btagPNetTauVJet",
    "btagRobustParTAK4B",
    "btagRobustParTAK4CvB",
    "btagRobustParTAK4CvL",
    "btagRobustParTAK4QG",
    "chEmEF",
    "chHEF",
    "electronIdx1",
    "electronIdx2",
    "eta",
    "genJetIdx",
    "hfadjacentEtaStripsSize",
    "hfcentralEtaStripSize",
    "hfsigmaEtaEta",
    "hfsigmaPhiPhi",
    "jetId",
    "mass",
    "muEF",
    "muonIdx1",
    "muonIdx2",
    "muonSubtrFactor",
    "nConstituents",
    "nElectrons",
    "nMuons",
    "nSVs",
    "neEmEF",
    "neHEF",
    "phi",
    "pt",
    "rawFactor",
    "svIdx1",
    "svIdx2",
]
JetObservables_v15 = [
    "UParTAK4RegPtRawCorr",
    "UParTAK4RegPtRawCorrNeutrino",
    "UParTAK4RegPtRawRes",
    "UParTAK4V1RegPtRawCorr",
    "UParTAK4V1RegPtRawCorrNeutrino",
    "UParTAK4V1RegPtRawRes",
    "btagPNetCvNotB",
    "btagUParTAK4B",
    "btagUParTAK4CvB",
    "btagUParTAK4CvL",
    "btagUParTAK4CvNotB",
    "btagUParTAK4Ele",
    "btagUParTAK4Mu",
    "btagUParTAK4QvG",
    "btagUParTAK4SvCB",
    "btagUParTAK4SvUDG",
    "btagUParTAK4TauVJet",
    "btagUParTAK4UDG",
    "btagUParTAK4probb",
    "btagUParTAK4probbb",
    "chMultiplicity",
    "hfEmEF",
    "hfHEF",
    "muonSubtrDeltaEta",
    "muonSubtrDeltaPhi",
    "neMultiplicity",
    "puIdDisc",
]
JetObservablesMC = ["hadronFlavour", "partonFlavour"]

FSRPhotonObservables = [
    "FsrPhoton_dROverEt2",
    "FsrPhoton_electronIdx",
    "FsrPhoton_eta",
    "FsrPhoton_muonIdx",
    "FsrPhoton_phi",
    "FsrPhoton_pt",
    "FsrPhoton_relIso03",
]

SoftActivityJetObservables = [
    "SoftActivityJet_eta",
    "SoftActivityJet_phi",
    "SoftActivityJet_pt",
    "SoftActivityJetHT",
    "SoftActivityJetHT10",
    "SoftActivityJetHT2",
    "SoftActivityJetHT5",
    "SoftActivityJetNjets10",
    "SoftActivityJetNjets2",
    "SoftActivityJetNjets5",
]


defaultColToSave = [
    "FullEventId",
    "luminosityBlock",
    "run",
    "event",
    "period",
    "isData",
    "PV_npvs",
    "BeamSpot_sigmaZ",
    "BeamSpot_sigmaZError",
    "BeamSpot_type",
    "BeamSpot_z",
    "BeamSpot_zError",
]

LHE_vars = [
    "LHE_AlphaS",
    "LHE_HT",
    "LHE_HTIncoming",
    "LHE_Nb",
    "LHE_Nc",
    "LHE_Nglu",
    "LHE_Njets",
    "LHE_NpLO",
    "LHE_NpNLO",
    "LHE_Nuds",
    "LHE_Vpt",
    # the comment observables should be present in nanov15 (https://cms-xpog.docs.cern.ch/autoDoc/NanoAODv15/2024/doc_TTH-Hto2G_Par-M-125_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2.html#LHEScaleSumw) but actually if looking in the files it is not present (e.g. root -l davs://eoscms.cern.ch:443/eos/cms/store/mc/RunIII2024Summer24NanoAODv15/DYto2E-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/90000/0cab6044-c840-461b-a46c-c8e5861775aa.root   and then Events->Print("*Sumw*") returns no branches) ..
    # "LHEPdfSumw",
    # "LHEScaleSumw",
    # "PSSumw",
    "LHEPdfWeight",
    "LHEReweightingWeight",
    "LHEScaleWeight",
    "LHEWeight_originalXWGTUP",
    "LHEPart_eta",
    "LHEPart_incomingpz",
    "LHEPart_mass",
    "LHEPart_pdgId",
    "LHEPart_phi",
    "LHEPart_pt",
    "LHEPart_spin",
    "LHEPart_status",
    "PSWeight",
]
LHE_vars_v15 = ["LHEPart_firstMotherIdx", "LHEPart_lastMotherIdx"]
additional_VBFStudies_vars = [
    "GenJet_eta",
    "GenJet_hadronFlavour",
    "GenJet_mass",
    "GenJet_nBHadrons",
    "GenJet_nCHadrons",
    "GenJet_partonFlavour",
    "GenJet_phi",
    "GenJet_pt",
    "LHEPart_eta",
    "LHEPart_incomingpz",
    "LHEPart_mass",
    "LHEPart_pdgId",
    "LHEPart_phi",
    "LHEPart_pt",
    "LHEPart_spin",
    "LHEPart_status",
    "GenPart_eta",
    "GenPart_genPartIdxMother",
    "GenPart_iso",
    "GenPart_mass",
    "GenPart_pdgId",
    "GenPart_phi",
    "GenPart_pt",
    "GenPart_status",
    "GenPart_statusFlags",
    "GenPart_vx",
    "GenPart_vy",
    "GenPart_vz",
    "GenProton_isPU",
    "GenProton_px",
    "GenProton_py",
    "GenProton_pz",
    "GenProton_vz",
]


def getDefaultColumnsToSave(isData):
    colToSave = defaultColToSave.copy()
    if not isData:
        colToSave.extend(["Pileup_nTrueInt"])
        colToSave.extend(LHE_vars)
    return list(set(colToSave))


def addAllVariables(
    dfw,
    syst_name,
    isData,
    trigger_class,
    lepton_legs,
    isSignal,
    applyTriggerFilter,
    global_params,
    channels,
    dataset_cfg,
):

    dfw.Apply(
        AnaBaseline.LeptonVeto,
        muon_pt_to_use=global_params.get("muon_pt_for_presel", "pt_nano"),
    )
    dfw.Apply(Corrections.getGlobal().jet.getEnergyResolution)
    dfw.Apply(Corrections.getGlobal().btag.getWPid, "Jet")
    dfw.Apply(Corrections.getGlobal().JetVetoMap.GetJetVetoMap)

    isV12 = global_params["nano_version"] == "v12"
    dfw.Apply(
        CommonBaseline.ApplyJetVetoMap,
        apply_filter=False,
        defineElectronCleaning=True,
        isV12=isV12,
    )

    n_legs = 2

    for leg_idx in range(n_legs):

        def LegVar(
            var_name,
            var_expr,
            var_type=None,
            var_cond=None,
            check_leg_type=True,
            default=0,
        ):
            cond = var_cond
            # if check_leg_type:
            #     type_cond = f"HttCandidate.leg_type[{leg_idx}] != Leg::none"
            #     cond = f"{type_cond} && ({cond})" if cond else type_cond
            define_expr = (
                f"static_cast<{var_type}>({var_expr})" if var_type else var_expr
            )
            if cond:
                define_expr = f"{cond} ? ({define_expr}) : {default}"
            dfw.DefineAndAppend(f"mu{leg_idx+1}_{var_name}", define_expr)

        LegVar("legType", f"Leg::mu", check_leg_type=False)

        # LegVar(
        #     f"index",
        #     f"Muon_idx[mu{leg_idx+1}_idx]",
        #     var_type="int",
        #     default="-1",
        # )

        # LegVar("charge", f"Muon_charge[mu{leg_idx+1}_idx]", var_type="int")
        # LegVar(
        #     f"pt_nano",
        #     f"static_cast<float>(Muon_p4_nano.at(mu{leg_idx+1}_idx).pt())",
        # )
        # dfw.Define(
        #     f"mu{leg_idx+1}_p4",
        #     f"Muon_p4.at(mu{leg_idx+1}_idx)",
        # )
        # dfw.Define(
        #     f"mu{leg_idx+1}_p4_pt_nano",
        #     f"Muon_p4_nano.at(mu{leg_idx+1}_idx)",
        # )
        # dfw.Define(
        #     f"mu{leg_idx+1}_p4_bsConstrainedPt",
        #     f"Muon_p4_bsConstrainedPt.at(mu{leg_idx+1}_idx)",
        # )
        Muon_observables = Muon_observables_base
        if not isData:
            Muon_observables.extend(MuonObservables_MC)
        if global_params["nano_version"] == "v15":
            Muon_observables.extend(Muon_observables_v15)
        for muon_obs in list(set(Muon_observables)):
            LegVar(
                muon_obs,
                f"Muon_{muon_obs}.at(mu{leg_idx+1}_idx)",
                var_cond=f"mu{leg_idx+1}_idx>=0",
                default="-100000.f",
            )
        for var in ["pt", "eta", "phi", "mass"]:
            LegVar(
                var,
                f"Muon_p4[mu{leg_idx+1}_idx].{var}()",
                var_cond=f"mu{leg_idx+1}_idx>=0",
                var_type="float",
                default="-1000.f",
            )

        LegVar(
            "pt_nano",
            f"Muon_p4_nano.at(mu{leg_idx+1}_idx).Pt()",
            var_cond=f"mu{leg_idx+1}_idx>=0",
            default="-100000.f",
        )
        if not isData:
            LegVar(
                "gen_kind",
                f"genLeptons.at(mu{leg_idx+1}_genMatchIdx).kind()",
                var_type="int",
                var_cond=f"mu{leg_idx+1}_idx>=0 && mu{leg_idx+1}_genMatchIdx>=0",
                default="static_cast<int>(GenLeptonMatch::NoMatch)",
            )
            LegVar(
                "gen_charge",
                f"genLeptons.at(mu{leg_idx+1}_genMatchIdx).charge()",
                var_type="int",
                var_cond=f"mu{leg_idx+1}_idx>=0 && mu{leg_idx+1}_genMatchIdx>=0",
                default="-10",
            )
        else:
            LegVar(
                "gen_kind",
                f"static_cast<int>(GenLeptonMatch::NoMatch)",
                var_cond=f"mu{leg_idx+1}_idx>=0",
                var_type="int",
                default="static_cast<int>(GenLeptonMatch::NoMatch)",
            )
            LegVar(
                "gen_charge",
                f"-10",
                var_cond=f"mu{leg_idx+1}_idx>=0",
                var_type="int",
                default="-10",
            )

        # defining each leg p4 for FindMatching from Muon_p4

        for suffix in ["p4_bsConstrainedPt", "p4_nano", "p4"]:
            if f"mu{leg_idx+1}_{suffix}" not in dfw.df.GetColumnNames():
                dfw.df = dfw.df.Define(
                    f"mu{leg_idx+1}_{suffix}",
                    f"mu{leg_idx+1}_idx >= 0 ? Muon_{suffix}[mu{leg_idx+1}_idx] : LorentzVectorM(0.,0.,0.,0.)",
                )

    dfw.Apply(
        AnaBaseline.LowerMassCut,
        suffixes=["p4", "p4_nano", "p4_bsConstrainedPt"],
    )

    jet_obs_names = []
    for jvar in ["pt"]:
        jet_obs_name = f"v_ops::{var}(Jet_p4_nano)"
        if f"{jet_obs_name}" in dfw.df.GetColumnNames():
            dfw.DefineAndAppend(f"Jet_{jvar}_nano", jet_obs_name)

    JetObservables = JetObservables_base
    if global_params["nano_version"] == "v15":
        JetObservables.extend(JetObservables_v15)
    if not isData:
        JetObservables.extend(JetObservablesMC)

    for jetobs in list(set(JetObservables)) + ["idx"]:
        jet_obs_name = f"Jet_{jetobs}"
        if jet_obs_name in dfw.df.GetColumnNames():
            jet_obs_names.append(jet_obs_name)
    dfw.colToSave.extend(list(set(jet_obs_names)))
    # nemmeno i cazzo di jet sono il problems

    if not isData and global_params["nano_version"] == "v15":
        dfw.colToSave.extend(LHE_vars_v15)

    for recoObsNew in list(
        set(
            PUObservables
            + FSRPhotonObservables
            + SoftActivityJetObservables
            # + additional_VBFStudies_vars
        )
    ):
        if recoObsNew in dfw.df.GetColumnNames():
            dfw.colToSave.extend([recoObsNew])

    ### Keep commented in case we decide to store MET too

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
        hltBranches = dfw.Apply(
            trigger_class.ApplyTriggers,
            lepton_legs,
            isData,
            applyTriggerFilter,  # False # --> for sync purposes
            global_params.get("extraFormat_for_triggerMatchingAndSF", {}),
        )
        dfw.colToSave.extend(hltBranches)
