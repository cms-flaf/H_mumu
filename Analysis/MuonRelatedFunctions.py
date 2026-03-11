import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from FLAF.Common.Utilities import *


def GetMuMuMassResolution(df, pt_to_use):
    sigma_pt = {
        "scare": "mu{0}_ptErr/mu{0}_pt",
        "nano": "mu{0}_ptErr/mu{0}_pt",
        "scare_reapplied": "mu{0}_ptErr/mu{0}_pt",
        "BS": "mu{0}_bsConstrainedPtErr/mu{0}_bsConstrainedPt",
        "BS_scare": "mu{0}_bsConstrainedPtErr/mu{0}_bsConstrainedPt",
        "RoccoR": "mu{0}_ptErr/mu{0}_pt",
        "BS_RoccoR": "mu{0}_bsConstrainedPtErr/mu{0}_bsConstrainedPt",
    }
    sigma_scaleandresol = {
        "scare": "0.5*(((mu{0}_pt_1_corr_up-mu{0}_pt_1_scale_corr)*(mu{0}_pt_1_corr_up-mu{0}_pt_1_scale_corr))+((mu{0}_pt_1_corr_dn-mu{0}_pt_1_scale_corr)*(mu{0}_pt_1_corr_dn-mu{0}_pt_1_scale_corr)))",
        "nano": "0.",
        "scare_reapplied": "0.5*(((mu{0}_pt_1_corr_up-mu{0}_pt_1_scale_corr)*(mu{0}_pt_1_corr_up-mu{0}_pt_1_scale_corr))+((mu{0}_pt_1_corr_dn-mu{0}_pt_1_scale_corr)*(mu{0}_pt_1_corr_dn-mu{0}_pt_1_scale_corr)))",
        "BS": "0.",
        "BS_scare": "0.5*(((mu{0}_BS_pt_1_corr_up-mu{0}_BS_pt_1_corr)*(mu{0}_BS_pt_1_corr_up-mu{0}_BS_pt_1_corr))+((mu{0}_BS_pt_1_corr_dn-mu{0}_BS_pt_1_corr)*(mu{0}_BS_pt_1_corr_dn-mu{0}_BS_pt_1_corr)))",
        "RoccoR": "0.",
        "BS_RoccoR": "0.",
    }
    for mu_idx in [1, 2]:
        print(sigma_pt[pt_to_use].format(mu_idx))
        print(sigma_scaleandresol[pt_to_use].format(mu_idx))
        df = df.Define(f"sigma_mu{mu_idx}_pt_rel", sigma_pt[pt_to_use].format(mu_idx))
        # df.Display({f"sigma_mu{mu_idx}_pt_rel"}).Print()
        # df=df.Define(f"sigma_mu{mu_idx}_pt_rel", f"sigma_mu{mu_idx}_pt_rel/mu{mu_idx}_pt") # is it alreadt relative??
        df = df.Define(
            f"sigma_mu{mu_idx}_scaleresolution",
            sigma_scaleandresol[pt_to_use].format(mu_idx),
        )
        df = df.Define(
            f"sigma_mu{mu_idx}_scaleresolution_rel",
            f"sigma_mu{mu_idx}_scaleresolution/mu{mu_idx}_pt",
        )
        # df.Display({f"sigma_mu{mu_idx}_scaleresolution"}).Print()
        df = df.Define(
            f"sigma_mu{mu_idx}_total_pt_rel",
            f"sqrt( pow(sigma_mu{mu_idx}_pt_rel,2) + sigma_mu{mu_idx}_scaleresolution_rel )",
        )
        # df.Display({f"sigma_mu{mu_idx}_total_pt_rel"}).Print()
    delta_mu_expr = "0.5*sqrt({0}*{0}+{1}*{1}) "
    # delta_mu_expr = "sqrt( 0.5 * (pow( ({0}/{1}), 2) + pow( ({2}/{3}), 2) ) ) "
    df = df.Define(
        "m_mumu_resolution",
        delta_mu_expr.format("sigma_mu1_total_pt_rel", "sigma_mu2_total_pt_rel"),
    )
    # df.Display({"m_mumu_resolution"}).Print()
    # df = df.Define(
    #     "m_mumu_resolution_nano",
    #     delta_mu_expr.format(
    #         "mu1_ptErr",
    #         "mu1_pt_nano",
    #         "mu2_ptErr",
    #         "mu2_pt_nano",
    #     ),
    # )
    # df = df.Define(
    #     "m_mumu_resolution",
    #     delta_mu_expr.format(
    #         "(mu1_pt-mu1_pt_nano)/mu1_pt",
    #         "mu1_pt",
    #         "(mu2_pt-mu2_pt_nano)/mu2_pt",
    #         "mu2_pt",
    #     ),
    # )

    # df = df.Define(
    #     "m_mumu_resolution_BS",
    #     delta_mu_expr.format(
    #         "mu1_bsConstrainedPtErr",
    #         "mu1_bsConstrainedPt",
    #         "mu2_bsConstrainedPtErr",
    #         "mu2_bsConstrainedPt",
    #     ),
    # )
    # df = df.Define(
    #     "m_mumu_resolution_BS_ScaRe",
    #     delta_mu_expr.format(
    #         "(mu1_BS_pt_1_corr-mu1_bsConstrainedPt)",
    #         "mu1_BS_pt_1_corr",
    #         "(mu2_BS_pt_1_corr-mu2_bsConstrainedPt)",
    #         "mu2_BS_pt_1_corr",
    #     ),
    # )

    return df


def GetMuMuP4Observables(df):
    for mu_idx in [1, 2]:  # tmp patch for back compatibility with old AnaTuples
        if f"mu{mu_idx}_bsConstrainedPt" in df.GetColumnNames():
            if f"mu{mu_idx}_pt_bsConstrainedPt" not in df.GetColumnNames():
                # print(f"defining mu{mu_idx}_pt_bsConstrainedPt" )
                df = df.Define(
                    f"mu{mu_idx}_pt_bsConstrainedPt", f"mu{mu_idx}_bsConstrainedPt"
                )
    pt_def = [col for col in df.GetColumnNames() if f"mu1_pt_" in col]
    # print(f"pt defined are: {pt_def}")
    muon_p4_to_define = list(set(["_".join(pt.split("_")[2:]) for pt in pt_def]))
    # print(f"suffixes are : {muon_p4_to_define}")
    for pt_suffix in muon_p4_to_define + [""]:
        pt_suffix_plus_underscore = f"_{pt_suffix}" if pt_suffix != "" else ""
        for idx in [0, 1]:
            df = df.Define(
                f"mu{idx+1}_p4{pt_suffix_plus_underscore}",
                f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_pt{pt_suffix_plus_underscore},mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
            )

        df = df.Define(
            f"pt_mumu{pt_suffix_plus_underscore}",
            f"(mu1_p4{pt_suffix_plus_underscore}+mu2_p4{pt_suffix_plus_underscore}).Pt()",
        )
        df = df.Define(
            f"m_mumu{pt_suffix_plus_underscore}",
            f"(mu1_p4{pt_suffix_plus_underscore}+mu2_p4{pt_suffix_plus_underscore}).M()",
        )
    return df


def GetAllMuMuCorrectedPtRelatedObservables(df, suff=""):
    suffix = f"_{suff}" if suff else ""
    cols = set(df.GetColumnNames())

    def define_or_redefine(df, name, expr):
        if name in cols:
            return df.Redefine(name, expr)
        else:
            cols.add(name)
            return df.Define(name, expr)

    df = df.Define("Ebeam", "13600.0/2")

    dimu = f"(mu1_p4{suffix}+mu2_p4{suffix})"

    # dimuon observables
    dimu_obs = {
        "pt_mumu": f"{dimu}.Pt()",
        "m_mumu": f"{dimu}.M()",
        "y_mumu": f"{dimu}.Rapidity()",
        "eta_mumu": f"{dimu}.Eta()",
        "phi_mumu": f"{dimu}.Phi()",
    }

    for name, expr in dimu_obs.items():
        df = define_or_redefine(df, name, expr)

    df = df.Define(
        "dR_mumu", f"ROOT::Math::VectorUtil::DeltaR(mu1_p4{suffix}, mu2_p4{suffix})"
    )

    df = (
        df.Define(
            "cosTheta_Phi_CS",
            f"ComputeCosThetaPhiCS(mu1_p4{suffix}, mu2_p4{suffix}, Ebeam)",
        )
        .Define("cosTheta_CS", "static_cast<float>(std::get<0>(cosTheta_Phi_CS))")
        .Define("phi_CS", "static_cast<float>(std::get<1>(cosTheta_Phi_CS))")
    )

    # ensure m_mumu_suffix exists
    m_mumu_s = f"m_mumu{suffix}"
    if m_mumu_s not in cols:
        df = df.Define(m_mumu_s, f"{dimu}.M()")
        cols.add(m_mumu_s)

    for i in (1, 2):

        p4 = f"mu{i}_p4{suffix}"
        pt_s = f"mu{i}_pt{suffix}"
        pt = f"mu{i}_pt"

        if f"mu{i}_p4" not in cols:
            df = df.Define(f"mu{i}_p4", p4)
            cols.add(f"mu{i}_p4")

        df = df.Redefine(f"mu{i}_p4", p4)

        if pt_s not in cols:
            df = df.Define(pt_s, f"{p4}.Pt()")
            cols.add(pt_s)

        df = define_or_redefine(df, pt, f"{p4}.Pt()")

        df = df.Define(f"mu{i}_pt_rel{suffix}", f"{pt_s}/{m_mumu_s}")

        if suffix:
            df = df.Define(f"mu{i}_pt_rel", f"{pt_s}/{m_mumu_s}")

    return df
