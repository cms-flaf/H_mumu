from FLAF.Common.Utilities import *

channels = ["muMu"]  
leg_names = ["Electron", "Muon", "Tau"]


def LowerMassCut(df, suffixes=None):
    if suffixes is None:
        suffixes = []
        for dfCol in df.GetColumnNames():
            if f"mu1_p4" in dfCol and "delta" not in dfCol:
                suffixes.append("_".join(dfCol.split("_")[1:]))
    masses_suffixes = []
    for suffix in suffixes:
        split_suffix = suffix.split("_")
        suffix_for_m_mumu = ""
        if len(split_suffix) > 1:
            suffix_for_m_mumu = "_" + ("_".join(split_suffix[1:]))
        if f"m_mumu{suffix_for_m_mumu}" not in df.GetColumnNames():
            df = df.Define(
                f"m_mumu{suffix_for_m_mumu}", f"(mu1_{suffix}+mu2_{suffix}).M()"
            )
        masses_suffixes.append(suffix_for_m_mumu)
    masses_cut = " || ".join([f"m_mumu{s} > 50" for s in masses_suffixes])
    df = df.Filter(masses_cut, "m(mumu) > 50 ")
    return df


def LeptonsSelection(df):
    ### muon selection ###
    df = df.Define(
        "Muon_B0", f"""(v_ops::pt(Muon_p4) > 15 && abs(v_ops::eta(Muon_p4)) < 2.4)"""
    )
    df = df.Define(
        "Muon_IsoIDOfficial", f"""(Muon_mediumId && Muon_miniPFRelIso_all < 0.25)"""
    )
    df = df.Filter(
        "Muon_idx[Muon_B0 && Muon_IsoIDOfficial].size()==2",
        "Consider events with exactly 2 muons",
    )
    df = df.Define("mu1_idx", "Muon_idx[Muon_B0 && Muon_IsoIDOfficial][0]")
    df = df.Define("mu2_idx", "Muon_idx[Muon_B0 && Muon_IsoIDOfficial][1]")
    df = df.Filter("Muon_charge[mu1_idx]*Muon_charge[mu2_idx]<0", "OS muons")

    ### electron veto ###
    df = df.Define(
        "Electron_B0_veto",
        f"""v_ops::pt(Electron_p4) > 20 && abs(v_ops::eta(Electron_p4)) < 2.5  && ( Electron_mvaIso_WP90 == true )""",
    )  # && abs(Electron_dz) < 0.2 && abs(Electron_dxy) < 0.024 --> to add?
    df = df.Filter("Electron_idx[Electron_B0_veto].size() == 0", "No extra electrons")
    return df


def LeptonsSelection_dev(df):
    df = df.Define(
        "Muon_B0", f"""(v_ops::pt(Muon_p4) > 15 && abs(v_ops::eta(Muon_p4)) < 2.4)"""
    )
    big_ID_OR = "(Muon_looseId || Muon_mvaLowPt > -0.6)"
    big_Iso_OR = "(Muon_pfIsoId >= 2 || Muon_miniIsoId >= 1 ||  Muon_miniPFRelIso_all <= 0.4 || Muon_pfRelIso04_all <= 0.25)"
    big_ID_Iso_OR = f"({big_ID_OR} || {big_Iso_OR})"
    df = df.Define("Big_ID_Iso_OR", big_ID_Iso_OR)
    df = df.Filter(
        "Muon_idx[Muon_B0 && Big_ID_Iso_OR].size()==2",
        "Consider events with exactly 2 muons",
    )
    df = df.Define("mu1_idx", "Muon_idx[Muon_B0 && Big_ID_Iso_OR][0]")
    df = df.Define("mu2_idx", "Muon_idx[Muon_B0 && Big_ID_Iso_OR][1]")

    ### electron veto ###
    df = df.Define(
        "Electron_B0_veto",
        f"""v_ops::pt(Electron_p4) > 20 && abs(v_ops::eta(Electron_p4)) < 2.5  && ( Electron_mvaIso_WP90 == true )""",
    )  # && abs(Electron_dz) < 0.2 && abs(Electron_dxy) < 0.024 --> to add?
    df = df.Filter("Electron_idx[Electron_B0_veto].size() == 0", "No extra electrons")
