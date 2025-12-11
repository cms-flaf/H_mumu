import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
# def parse_value_error(s):
#     """Converte stringhe del tipo '1.1411 ± 0.0022' → (1.1411, 0.0022)."""
#     val, err = s.replace(" ", "").split("±")
#     return float(val), float(err)
def parse_value_error(s):
    """parse robusto: se vuoto → None."""
    if pd.isna(s) or str(s).strip() == "":
        return None
    try:
        val, err = s.replace(" ", "").split("±")
        return float(val), float(err)
    except Exception:
        return None

def safe_parse(s):
    """Versione robusta: se la stringa è vuota o NaN, restituisce (NaN, NaN)."""
    if pd.isna(s) or str(s).strip() == "":
        return pd.Series([np.nan, np.nan])
    try:
        return pd.Series(parse_value_error(s))
    except Exception:
        return pd.Series([np.nan, np.nan])


parser = argparse.ArgumentParser(description="Esegui un fit su istogrammi di dati o MC.")
parser.add_argument('--year', required=True, help="year")
parser.add_argument('--pt_type', required=False, default="ScaRe_reapplied_subregions_1Dec", help="")


args = parser.parse_args()
if not os.path.exists(f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_data_BW_conv_DCS.csv"):
    print(f"file non trovato: /afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_data_BW_conv_DCS.csv")
else:
    print(f"considering {args.year} and {args.pt_type}")
    # df_data = pd.read_csv(f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_data_BW_conv_DCS.csv", sep="\t")
    # df_mc = pd.read_csv(f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_MC_BW_conv_DCS.csv", sep="\t")


    # -----------------------
    # 2) Parsing colonne mean e sigma
    # -----------------------

    df_data = pd.read_csv(
        f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_data_BW_conv_DCS.csv",
        sep="\t",
        skip_blank_lines=True
    )
    df_mc = pd.read_csv(
        f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_MC_BW_conv_DCS.csv",
        sep="\t",
        skip_blank_lines=True
    )

    # Parsing robusto e pulizia righe
    for df in (df_data, df_mc):
        tmp_sigma = df["sigma (res)"].apply(parse_value_error)
        tmp_mean  = df["mean (mZ)"].apply(parse_value_error)

        df["sigma"] = tmp_sigma.apply(lambda x: x[0] if x is not None else None)
        df["sigma_err"] = tmp_sigma.apply(lambda x: x[1] if x is not None else None)

        df["mean"] = tmp_mean.apply(lambda x: x[0] if x is not None else None)
        df["mean_err"] = tmp_mean.apply(lambda x: x[1] if x is not None else None)


    # for df in (df_data, df_mc):
    #     df[["sigma", "sigma_err"]] = df["sigma (res)"].apply(
    #         lambda s: pd.Series(parse_value_error(s))
    #     )
    #     df[["mean", "mean_err"]] = df["mean (mZ)"].apply(
    #         lambda s: pd.Series(parse_value_error(s))
    #     )
    #     # print(df.head())

    # -----------------------
    # 3) Merge dei due file su region + mu1 pT
    # -----------------------

    df = df_data.merge(df_mc, on=["region", "mu1 pT"], suffixes=("_data", "_mc"))

    # Tieni solo punti validi
    df = df.dropna(subset=["sigma_data", "sigma_mc", "mean_data", "mean_mc"])


    # -----------------------
    # 4) Ratio e propagazione dell’errore
    #     R = sD / sM
    #     dR = R * sqrt( (dD/sD)^2 + (dM/sM)^2 )
    # -----------------------
    df["ratio"] = df["sigma_data"] / df["sigma_mc"]
    df["ratio_err"] = df["ratio"] * np.sqrt(
        (df["sigma_err_data"] / df["sigma_data"])**2 +
        (df["sigma_err_mc"]   / df["sigma_mc"])**2
    )
    df["ratio_mean"] = df["mean_data"] / df["mean_mc"]
    df["ratio_err_mean"] = df["ratio_mean"] * np.sqrt(
        (df["mean_err_data"] / df["mean_data"])**2 +
        (df["mean_err_mc"]   / df["mean_mc"])**2
    )

    # -----------------------
    # 5) Etichette X ordinate
    # -----------------------
    df["label"] = df["region"] + "  " + df["mu1 pT"]

    # -----------------------
    # 6) Plot
    # -----------------------
    plt.figure(figsize=(16, 6))

    plt.errorbar(
        df.index,
        df["ratio"],
        yerr=df["ratio_err"],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3
    )

    plt.axhline(1.0, linestyle="--", color="red", linewidth=1)

    plt.xticks(df.index, df["label"], rotation=75, ha="right")
    plt.ylabel("Data / MC σ")
    plt.title("Mass resolution ratio - Data / MC")

    plt.ylim(0.8,1.2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    # plt.show()
    prepath_to_save = "/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions" # /eos/user/v/vdamante/www/H_mumu/fits/
    plt.savefig(f"{prepath_to_save}/{args.pt_type}/Run3_{args.year}/ratio_sigma_dataMC.png", bbox_inches="tight")
    print(f"file salvato in {prepath_to_save}/{args.pt_type}/Run3_{args.year}/ratio_sigma_dataMC.png")

    plt.figure(figsize=(16, 6))

    plt.errorbar(
        df.index,
        df["ratio_mean"],
        yerr=df["ratio_err_mean"],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3
    )

    plt.axhline(1.0, linestyle="--", color="red", linewidth=1)

    plt.xticks(df.index, df["label"], rotation=75, ha="right")
    plt.ylabel("Data / MC mZ")
    plt.title("Mass mean ratio - Data / MC")

    plt.ylim(0.99, 1.01)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(
        f"{prepath_to_save}/{args.pt_type}/Run3_{args.year}/ratio_mean_dataMC.png",
        bbox_inches="tight"
    )
    print(
        f"file salvato in {prepath_to_save}/{args.pt_type}/Run3_{args.year}/ratio_mean_dataMC.png"
    )

    plt.close()
