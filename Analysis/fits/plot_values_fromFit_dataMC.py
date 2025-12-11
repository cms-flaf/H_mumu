import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
# def parse_value_error(s):
#     """Converte stringhe del tipo '1.1411 ± 0.0022' → (1.1411, 0.0022)."""
#     val, err = s.replace(" ", "").split("±")
#     return float(val), float(err)

# parser = argparse.ArgumentParser(description="Esegui un fit su istogrammi di dati o MC.")
# parser.add_argument('--year', required=True, help="year")
# parser.add_argument('--pt_type', required=False, default="ScaRe_reapplied_subregions_1Dec", help="")
# parser.add_argument('--var_to_plot', required=False, default="sigma", help="sigma or mean")
# parser.add_argument('--isMC', action='store_true', help="Set this flag if the input is a Monte Carlo signal.")

def parse_value_error(s):
    """parse robusto: se vuoto → None."""
    if pd.isna(s) or str(s).strip() == "":
        return None
    try:
        val, err = s.replace(" ", "").split("±")
        return float(val), float(err)
    except Exception:
        return None


args = parser.parse_args()
if not os.path.exists(f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_data_BW_conv_DCS.csv"):
    print(f"file non trovato: /afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_data_BW_conv_DCS.csv")
else:
    print(f"considering {args.year} and {args.pt_type}")
    df_data = pd.read_csv(f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_data_BW_conv_DCS.csv", sep="\t")
    df_mc = pd.read_csv(f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/summary_results_MC_BW_conv_DCS.csv", sep="\t")


    df = df_mc if args.isMC else df_data
    dataorMC = "MC" if args.isMC else  "data"

    df[["sigma", "sigma_err"]] = df["sigma (res)"].apply(
        lambda s: pd.Series(parse_value_error(s))
    )
    df[["mean", "mean_err"]] = df["mean (mZ)"].apply(
        lambda s: pd.Series(parse_value_error(s))
    )

    df["label"] = df["region"] + "  " + df["mu1 pT"]

    plt.figure(figsize=(16, 6))

    plt.errorbar(
        df.index,
        df[args.var_to_plot],
        yerr=df[f"{args.var_to_plot}_err"],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3
    )
    ax_val = 91.1880 if args.var_to_plot=="mean" else df[args.var_to_plot].mean()
    plt.axhline(ax_val, linestyle="--", color="red", linewidth=1)

    plt.xticks(df.index, df["label"], rotation=75, ha="right")
    plt.ylabel(args.var_to_plot)
    plt.title(args.var_to_plot + f" in {dataorMC}")

    # plt.ylim(0.8,1.2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    # plt.show()
    prepath_to_save = "/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions" # /eos/user/v/vdamante/www/H_mumu/fits/
    plt.savefig(f"{prepath_to_save}/{args.pt_type}/Run3_{args.year}/{args.var_to_plot}_{dataorMC}.png", bbox_inches="tight")
    print(f"file salvato in {prepath_to_save}/{args.pt_type}/Run3_{args.year}/{args.var_to_plot}_{dataorMC}.png")


    plt.close()
