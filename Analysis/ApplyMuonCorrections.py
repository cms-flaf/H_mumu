import ROOT
import correctionlib
import os


def AddRoccoR(df, period, isData):
    year = period.split("_")[1]
    analysis_path = os.environ["ANALYSIS_PATH"]
    ROOT.gROOT.ProcessLine(
        f'auto cset = correction::CorrectionSet::from_file("{analysis_path}/Corrections/data/MUO/RoccoR/RoccoR{year}.json.gz");'
    )
    ROOT.gROOT.ProcessLine(f'#include "{analysis_path}/include/RoccoR.cc"')
    for mu_idx in [1, 2]:
        if isData:
            # Data apply scale correction
            df = df.Define(
                f"mu{mu_idx}_RoccoR_scale",
                f"""cset->at("kScaleDT")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}})"""
                # f"""if(mu{mu_idx}_pt_nano >= 20) return cset->at("kScaleDT")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}}); return -1."""
            ).Define(f"mu{mu_idx}_RoccoR_pt",f"mu{mu_idx}_pt_nano * mu{mu_idx}_RoccoR_scale")
            df = df.Define(
                f"mu{mu_idx}_BS_RoccoR_scale",
                # f"""if(mu{mu_idx}_bsConstrainedPt >= 20) return cset->at("kScaleDT")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}}); return -1."""

                f"""cset->at("kScaleDT")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}})"""
            ).Define(f"mu{mu_idx}_BS_RoccoR_pt",f"mu{mu_idx}_bsConstrainedPt * mu{mu_idx}_BS_RoccoR_scale")
        else:
            df = df.Define(f"genmu{mu_idx}_pT", f"mu{mu_idx}_genPartIdx >= 0 ? GenPart_pt.at(mu{mu_idx}_genPartIdx) : -1.")
            df.Display({f"mu{mu_idx}_charge"}).Print()
            df = df.Define(
                f"mu{mu_idx}_RoccoR_scale",
                f""" mu{mu_idx}_genPartIdx >= 0? cset->compound().at("kSpreadMC")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi,  genmu{mu_idx}_pT, 0, 0}}) : cset->at("kScaleMC")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}}) """
            ).Define(f"mu{mu_idx}_RoccoR_pt",f"mu{mu_idx}_pt_nano * mu{mu_idx}_RoccoR_scale")

            df = df.Define(
                f"mu{mu_idx}_BS_RoccoR_scale",
                f""" mu{mu_idx}_genPartIdx >= 0? cset->compound().at("kSpreadMC")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi,  genmu{mu_idx}_pT, 0, 0}}) : cset->at("kScaleMC")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}}) """
            ).Define(f"mu{mu_idx}_BS_RoccoR_pt",f"mu{mu_idx}_bsConstrainedPt * mu{mu_idx}_BS_RoccoR_scale")
    return df


def AddScaReOnBS(df, period, isData):
    import correctionlib

    period_files = {
        "Run3_2022": "2022_Summer22",
        "Run3_2022EE": "2022_Summer22EE",
        "Run3_2023": "2023_Summer23",
        "Run3_2023BPix": "2023_Summer23BPix",
    }
    correctionlib.register_pyroot_binding()
    file_name = period_files.get(period, "")
    analysis_path = os.environ["ANALYSIS_PATH"]
    ROOT.gROOT.ProcessLine(
        # f'auto cset = correction::CorrectionSet::from_file("{analysis_path}/Analysis/schemaV2_VXBS.json");' # only for 2024
        f'auto cset = correction::CorrectionSet::from_file("{analysis_path}/Corrections/data/MUO/MuonScaRe/{file_name}.json");'
    )
    ROOT.gROOT.ProcessLine(f'#include "{analysis_path}/include/MuonScaRe.cc"')
    for mu_idx in [1, 2]:
        if isData:
            # Data apply scale correction
            df = df.Define(
                f"mu{mu_idx}_BS_pt_1_corr",
                f"pt_scale(1, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, mu{mu_idx}_charge)",
            )
        else:
            df = df.Define(
                f"mu{mu_idx}_BS_pt_1_scale_corr",
                f"pt_scale(0, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, mu{mu_idx}_charge)",
            )

            df = df.Define(
                f"mu{mu_idx}_BS_pt_1_corr",
                f"pt_resol(mu{mu_idx}_BS_pt_1_scale_corr, mu{mu_idx}_eta, float(mu{mu_idx}_nTrackerLayers))",
            )
            # # MC evaluate scale uncertainty
            # df_mc = df_mc.Define(
            #     'pt_1_scale_corr_up',
            #     'pt_scale_var(pt_1_corr, eta_1, phi_1, charge_1, "up")'
            # )
            # df_mc = df_mc.Define(
            #     'pt_1_scale_corr_dn',
            #     'pt_scale_var(pt_1_corr, eta_1, phi_1, charge_1, "dn")'
            # )

            # # MC evaluate resolution uncertainty
            # df_mc = df_mc.Define(
            #     "pt_1_corr_resolup",
            #     'pt_resol_var(pt_1_scale_corr, pt_1_corr, eta_1, "up")'
            # )
            # df_mc = df_mc.Define(
            #     "pt_1_corr_resoldn",
            #     'pt_resol_var(pt_1_scale_corr, pt_1_corr, eta_1, "dn")'
            # )
    return df