from FLAF.Common.Utilities import *


def ObjReconstruction(df):
    df = df.Define("Muon_iso", "Muon_pfRelIso04_all")
    df = df.Define("Muon_B0", f"""
        v_ops::pt(Muon_p4) > 10 && abs(v_ops::eta(Muon_p4)) < 2.4 && (Muon_mediumId && Muon_iso < 0.25)""") # && abs(Muon_dz) < 0.2 && abs(Muon_dxy) < 0.024 # in future: Muon_bsConstrainedPt, Muon_bsConstrainedChi2, and Muon_bsConstrainedPtErr
    df = df.Define("Electron_B0_veto", f"""
        v_ops::pt(Electron_p4) > 20 && abs(v_ops::eta(Electron_p4)) < 2.5  && ( Electron_mvaIso_WP90 == true )""") # && abs(Electron_dz) < 0.2 && abs(Electron_dxy) < 0.024
    return df

def LeptonVeto(df):
    df = df.Filter('MuonVeto', "Muon_idx[Muon_B0].size()==2")
    df = df.Filter("Electron_idx[Electron_B0_veto].size() == 0", "No extra electrons")
    return df

def JetSelection(df, era):
    jet_puID_cut = ""
    # jet_puID_cut = "&& (Jet_puId>0 || v_ops::pt(Jet_p4)>50)" if era.startswith("Run2") else ""
    df = df.Define("Jet_B0", f"""v_ops::pt(Jet_p4) > 25 && abs(v_ops::eta(Jet_p4))< 4.7 && ( Jet_jetId & 2 ) {jet_puID_cut} """)
    df = df.Define("Jet_B1", "RemoveOverlaps(Jet_p4, Jet_B0,{{Muon_p4},}, 2, 0.4)")
    df = df.Define("Jet_Veto_loose", "Jet_B1 && abs(v_ops::eta(Jet_p4))< 2.5 && Jet_idbtagPNetB >= 1")
    df = df.Define("Jet_Veto_medium", "Jet_Veto_loose && Jet_idbtagPNetB >= 2")
    df = df.Define("Jet_Veto", "Jet_idx[Jet_Veto_medium].size()==0 || Jet_idx[Jet_Veto_loose].size()==y2")
    df = df.Filter(Jet_Veto,"excl. events with two loose or one medium b-jet")
    return df

def GetMuMuCandidate(df):
    df = df.Define("Hmumu_idx", "Muon_idx[Muon_B0]")


# def GenRecoJetMatching(df):
#     df = df.Define("Jet_genJetIdx_matched", "GenRecoJetMatching(event,Jet_idx, GenJet_idx, Jet_bCand, GenJet_B2, GenJet_p4, Jet_p4 , 0.3)")
#     df = df.Define("Jet_genMatched", "Jet_genJetIdx_matched>=0")
#     return df.Filter("Jet_genJetIdx_matched[Jet_genMatched].size()>=2", "Two different gen-reco jet matches at least")
'''

def getChannelLegs(channel):
    ch_str = channel.lower()
    legs = []
    while len(ch_str) > 0:
        name_idx = None
        obj_name = None
        for idx, obj in enumerate(['e', 'mu', 'tau']):
            if ch_str.startswith(obj):
                name_idx = idx
                obj_name = obj
                break
        if name_idx is None:
            raise RuntimeError(f"Invalid channel name {channel}")
        legs.append(leg_names[name_idx])
        ch_str = ch_str[len(obj_name):]
    return legs

def PassGenAcceptance(df):
    df = df.Filter("genHttCandidate.get() != nullptr", "genHttCandidate present")
    return df.Filter("PassGenAcceptance(*genHttCandidate)", "genHttCandidate Acceptance")

def GenJetSelection(df):
    df = df.Define("GenJet_B1","GenJet_pt > 20 && abs(GenJet_eta) < 2.5 && GenJet_Hbb")
    df = df.Define("GenJetAK8_B1","GenJetAK8_pt > 170 && abs(GenJetAK8_eta) < 2.5 && GenJetAK8_Hbb")
    return df.Filter("GenJet_idx[GenJet_B1].size()==2 || (GenJetAK8_idx[GenJetAK8_B1].size()==1 && genHbb_isBoosted)", "(One)Two b-parton (Fat)jets at least")

def GenJetHttOverlapRemoval(df):
    for var in ["GenJet", "GenJetAK8"]:
        df = df.Define(f"{var}_B2", f"RemoveOverlaps({var}_p4, {var}_B1,{{{{genHttCandidate->leg_p4[0], genHttCandidate->leg_p4[1]}},}}, 2, 0.5)" )
    return df.Filter("GenJet_idx[GenJet_B2].size()==2 || (GenJetAK8_idx[GenJetAK8_B2].size()==1 && genHbb_isBoosted)", "No overlap between genJets and genHttCandidates")

def RequestOnlyResolvedGenJets(df):
    return df.Filter("GenJet_idx[GenJet_B2].size()==2", "Resolved topology")
'''