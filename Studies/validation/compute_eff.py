#!/usr/bin/env python3
import uproot
import numpy as np
import json
from itertools import product


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
args = parser.parse_args()

# -------------------------
# Load mini-ntupla
# -------------------------
def load(fname):
    f = uproot.open(fname)["Mini"]
    return {k: f[k].array(library="np") for k in f.keys()}

sig = load(f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/stuffmini_signal_{args.year}.root")
bkg = load(f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/stuffmini_bkg_{args.year}.root")

# -------------------------
# Definizioni
# -------------------------
isoWPs = [0.25, 0.20, 0.15]

# Colonne ID disponibili

# Regioni disponibili
regions = [ "baseline_noID_Iso", "VBF_JetVeto_noID_Iso","ggH_noID_Iso"]
ID_cols = [k for k in sig.keys() if "ID" in k and k not in regions]
print(ID_cols)

# -------------------------
# Funzione calcolo efficienze per una categoria e ID combo
# -------------------------
def compute_ssqrtb(data_sig, data_bckg, wp1, wp2, ID_col, region):
    print(ID_col, wp1, wp2)
    signal_fit_mask_sig = data_sig["Signal_Fit"]
    signal_fit_mask_bckg = data_bckg["Signal_Fit"]


    iso_cut_sig = (data_sig["mu1_pfRelIso04_all"] < wp1) & (data_sig["mu2_pfRelIso04_all"] < wp2)
    iso_cut_bckg = (data_bckg["mu1_pfRelIso04_all"] < wp1) & (data_bckg["mu2_pfRelIso04_all"] < wp2)

    sel_sig = signal_fit_mask_sig & data_sig[region] & data_sig[ID_col] & iso_cut_sig
    sel_bckg = signal_fit_mask_bckg & data_bckg[region] & data_bckg[ID_col] & iso_cut_bckg

    selected_sig = sel_sig.sum()
    selected_bckg = sel_bckg.sum()

    s_sqrtB = selected_sig / np.sqrt(max(selected_bckg,1))

    return {"s_sqrtB": float(s_sqrtB)}


def compute_eff(data, wp1, wp2, ID_col, region):
    print(ID_col, wp1, wp2)
    # tutti gli eventi devono essere in Signal_Fit
    signal_fit_mask = data["Signal_Fit"]

    # applica il WP di isolamento
    iso_cut = (data["mu1_pfRelIso04_all"] < wp1) & (data["mu2_pfRelIso04_all"] < wp2)

    # combinazione: Signal_Fit & region & ID & iso
    sel = signal_fit_mask & data[region] & data[ID_col] & iso_cut

    total_base = (signal_fit_mask & data[region]).sum()  # totale nella regione + Signal_Fit
    selected = sel.sum()

    eff = selected / max(total_base,1)

    return {"count": int(selected), "total": int(total_base), "eff": float(eff)}

# -------------------------
# Loop su WP, ID e regioni
# -------------------------
results = []

for wp1 in isoWPs:
    for wp2 in isoWPs:
        for ID_col in ID_cols:
            for region in regions:
                res_sig = compute_eff(sig, wp1, wp2, ID_col, region)
                res_bkg = compute_eff(bkg, wp1, wp2, ID_col, region)
                res_ssqrtb = compute_ssqrtb(sig, bkg, wp1, wp2, ID_col, region)
                results.append({
                    "WP1": wp1,
                    "WP2": wp2,
                    "ID": ID_col,
                    "region": region,
                    "signal": res_sig,
                    "background": res_bkg,
                    "s_sqrtB": res_ssqrtb
                })

# -------------------------
# Salvataggio
# -------------------------
json.dump(results, open(f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/efficiencies_{args.year}.json","w"), indent=2)
print(f"Efficienze salvate in /afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/efficiencies_{args.year}.json")
