from FLAF.Common.Utilities import *

channels = ["muMu"]
leg_names = ["Electron", "Muon", "Tau"]


def LowerMassCut(df, p4_cols=["p4"], cut_value=50):
    masses_suffixes = []
    for p4_col in p4_cols:
        for mu_idx in [1, 2]:
            if f"mu{mu_idx}_{p4_col}" not in df.GetColumnNames():
                raise RuntimeError(f"mu{mu_idx}_{p4_col} not in df col names!!")
        mass_suffix_split = p4_col.split("_")
        mass_suffix = ""
        if len(mass_suffix_split) > 1:
            mass_suffix = "_" + ("_".join(mass_suffix_split[1:]))
        if f"m_mumu{mass_suffix}" not in df.GetColumnNames():
            df = df.Define(f"m_mumu{mass_suffix}", f"(mu1_{p4_col}+mu2_{p4_col}).M()")
        masses_suffixes.append(mass_suffix)
    masses_cut = " || ".join([f"m_mumu{s} > {cut_value}" for s in masses_suffixes])
    df = df.Filter(masses_cut, masses_cut)
    return df


def LeptonsSelection(df):
    ### muon selection ###
    df = df.Define(
        "Muon_B0", f"""(v_ops::pt(Muon_p4) > 15 && abs(v_ops::eta(Muon_p4)) < 2.4)"""
    )
    df = df.Define(
        "Muon_IsoIDOfficial",
        f"""(Muon_mediumId && (Muon_pfRelIso04_all < 0.25 || Muon_pfIsoId >= 2) ) """,  # medium ID, loose Iso
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
