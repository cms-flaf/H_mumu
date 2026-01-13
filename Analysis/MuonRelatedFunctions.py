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
    for idx in [0, 1]:
        df = df.Define(
            f"mu{idx+1}_p4_bsConstrainedPt",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_bsConstrainedPt,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
        if f"mu{idx+1}_p4_nano" not in df.GetColumnNames():
            df = df.Define(
                f"mu{idx+1}_p4_pt_nano",
                f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_pt_nano,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
            )
        if f"mu{idx+1}_p4_nano" not in df.GetColumnNames():
            df = df.Define(
                f"mu{idx+1}_p4_pt_nano",
                f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_pt_nano,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
            )
    df = df.Define(
        f"pt_mumu_BS", "(mu1_p4_bsConstrainedPt+mu2_p4_bsConstrainedPt).Pt()"
    )
    df = df.Define(f"pt_mumu_nano", "(mu1_p4_pt_nano+mu2_p4_pt_nano).Pt()")
    df = df.Define(f"m_mumu_BS", "(mu1_p4_bsConstrainedPt+mu2_p4_bsConstrainedPt).M()")
    df = df.Define(f"m_mumu_nano", "(mu1_p4_pt_nano+mu2_p4_pt_nano).M()")
    return df


def GetAllMuMuCorrectedPtRelatedObservables(
    df, suffix="corr"
):  # suffix can be "corr", "nano", "bsConstrainedPt", "roccor" (last not yet implemented)
    df = df.Define("Ebeam", "13600.0/2")
    df = df.Define(f"pt_mumu_corr", "(mu1_p4_corr+mu2_p4_corr).Pt()")
    df = df.Define(f"m_mumu_corr", "(mu1_p4_corr+mu2_p4_corr).M()")
    df = df.Define(f"y_mumu_corr", "(mu1_p4_corr+mu2_p4_corr).Rapidity()")
    df = df.Define(f"eta_mumu_corr", "(mu1_p4_corr+mu2_p4_corr).Eta()")
    df = df.Define(f"phi_mumu_corr", "(mu1_p4_corr+mu2_p4_corr).Phi()")
    df = df.Define(
        "dR_mumu_corr", "ROOT::Math::VectorUtil::DeltaR(mu1_p4_corr, mu2_p4_corr)"
    )

    df = df.Define(f"pt_mumu", f"(mu1_p4_{suffix}+mu2_p4_{suffix}).Pt()")
    df = df.Define(f"m_mumu", f"(mu1_p4_{suffix}+mu2_p4_{suffix}).M()")
    df = df.Define(f"y_mumu", f"(mu1_p4_{suffix}+mu2_p4_{suffix}).Rapidity()")
    df = df.Define(f"eta_mumu", f"(mu1_p4_{suffix}+mu2_p4_{suffix}).Eta()")
    df = df.Define(f"phi_mumu", f"(mu1_p4_{suffix}+mu2_p4_{suffix}).Phi()")
    df = df.Define(
        "dR_mumu", f"ROOT::Math::VectorUtil::DeltaR(mu1_p4_{suffix}, mu2_p4_{suffix})"
    )

    df = df.Define(
        "cosTheta_Phi_CS_corr", "ComputeCosThetaPhiCS(mu1_p4_corr, mu2_p4_corr,  Ebeam)"
    )
    df = df.Define(
        "cosTheta_CS_corr", "static_cast<float>(std::get<0>(cosTheta_Phi_CS_corr))"
    )
    df = df.Define(
        "phi_CS_corr", "static_cast<float>(std::get<1>(cosTheta_Phi_CS_corr))"
    )

    df = df.Define(
        f"cosTheta_Phi_CS",
        f"ComputeCosThetaPhiCS(mu1_p4_{suffix}, mu2_p4_{suffix},  Ebeam)",
    )
    df = df.Define("cosTheta_CS", "static_cast<float>(std::get<0>(cosTheta_Phi_CS))")
    df = df.Define("phi_CS", "static_cast<float>(std::get<1>(cosTheta_Phi_CS))")

    for mu_idx in [1, 2]:
        df = df.Define(f"mu{mu_idx}_p4", f"mu{mu_idx}_p4_{suffix}")
        df = df.Define(f"mu{mu_idx}_pt_corr", f"mu{mu_idx}_p4_corr.Pt()")
        if f"mu{mu_idx}_pt" not in df.GetColumnNames():
            df = df.Define(f"mu{mu_idx}_pt", f"mu{mu_idx}_p4_{suffix}.Pt()")
        df = df.Redefine(f"mu{mu_idx}_pt", f"mu{mu_idx}_p4_{suffix}.Pt()")
        df = df.Define(f"mu{mu_idx}_pt_rel_corr", f"mu{mu_idx}_pt_corr/m_mumu_corr")
        df = df.Define(f"mu{mu_idx}_pt_rel", f"mu{mu_idx}_pt_{suffix}/m_mumu_{suffix}")

    return df
