import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from collections import defaultdict
import argparse

# Stile CMS
hep.style.use("CMS")

# ------------------------
# Argomenti
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
args = parser.parse_args()

# ------------------------
# Carica dati
# ------------------------
with open(f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/efficiencies_{args.year}.json") as f:
    data = json.load(f)

period_dict = {
    "Run3_2022": "7.9804",
    "Run3_2022EE": "26.6717",
    "Run3_2023": "18.063",
    "Run3_2023BPix": "9.693",
}
lumi = period_dict[f"Run3_{args.year}"]

regions = ["baseline_noID_Iso", "VBF_JetVeto_noID_Iso", "ggH_noID_Iso"]

# ------------------------
# Raggruppa per regione
# ------------------------
region_dict = defaultdict(list)
for entry in data:
    region_dict[entry["region"]].append(entry)

# ------------------------
# Label compatti (puoi modificarli!)
# ------------------------
def compact_label(entry):
    """Label nella forma: ID  |  WP1=... WP2=..."""
    mu1_id= entry['ID'].split('_')[0].replace('ID','')
    mu2_id= entry['ID'].split('_')[2].replace('ID','')
    id_dict = {"loose":"L", "medium":"M", "tight":"T"}
    iso_dict = {0.25:"L", 0.2:"M", 0.15:"T"}
    return f"mu1{id_dict[mu1_id]},{iso_dict[entry['WP1']]}\n mu2{id_dict[mu2_id]},{iso_dict[entry['WP2']]}"


# ------------------------
# Funzione plotting SENZA TABELLA
# ------------------------
def plot_region(region_entries, region_name, lumi):

    # Ordine coerente
    region_entries = sorted(
        region_entries,
        key=lambda e: (e["ID"], float(e["WP1"]), float(e["WP2"]))
    )

    x = np.arange(len(region_entries))

    labels = [compact_label(e) for e in region_entries]

    # Estrai quantità
    s_sqrtB   = np.array([e["signal"]["s_sqrtB"] for e in region_entries])
    eff_sig   = np.array([e["signal"]["eff"]       for e in region_entries])
    eff_bck   = np.array([e["background"]["eff"]   for e in region_entries])

    s_sqrtB_e = np.array([e["signal"].get("s_sqrtB_err", 0.0) for e in region_entries])
    eff_sig_e = np.array([e["signal"].get("err",        0.0) for e in region_entries])
    eff_bck_e = np.array([e["background"].get("err",    0.0) for e in region_entries])

    # -----------------------------------
    # FUNZIONE DI PLOT UNIFICATA
    # -----------------------------------
    def make_plot(y, yerr, ylabel, outname):
        fig, ax = plt.subplots(figsize=(50, 10))

        ax.errorbar(
            x, y, yerr=yerr,
            fmt="o", markersize=6, capsize=3
        )

        # Imposta label sull'asse x
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=15)


        ax.set_ylabel(ylabel)
        y_min = min(y)
        y_max = max(y)
        ax.set_ylim(y_min*0.95,y_max*1.01)
        ax.grid()
        hep.cms.label("Preliminary", data=True, lumi=lumi)

        # margine per evitare tagli delle x labels
        plt.subplots_adjust(
            left=0.12,
            right=0.95,
            top=0.92,
            bottom=0.25
        )

        plt.savefig(outname, dpi=500)
        plt.close()

    # --------------------------
    # Plot 1 — S/sqrt(B)
    # --------------------------
    make_plot(
        s_sqrtB, s_sqrtB_e,
        ylabel="S/√B",
        outname=f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/s_sqrtB_{region_name}_{args.year}.png"
    )

    # --------------------------
    # Plot 2 — Efficienza segnale
    # --------------------------
    make_plot(
        eff_sig, eff_sig_e,
        ylabel="Signal efficiency",
        outname=f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/eff_signal_{region_name}_{args.year}.png"
    )

    # --------------------------
    # Plot 3 — Efficienza background
    # --------------------------
    make_plot(
        eff_bck, eff_bck_e,
        ylabel="Background efficiency",
        outname=f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/eff_background_{region_name}_{args.year}.png"
    )


# ------------------------
# Loop sulle regioni
# ------------------------
for reg in regions:
    if reg in region_dict:
        plot_region(region_dict[reg], reg, lumi)
