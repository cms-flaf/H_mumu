#!/usr/bin/env python3
import json
import pandas as pd
import os
import argparse

# -------------------------------
# Argomenti
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
parser.add_argument("--region", default=None, help="Filtra per regione specifica (opzionale)")
args = parser.parse_args()

# -------------------------------
# Percorsi
# -------------------------------
# input_path = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/json/efficienciesWithFake_{}.json"
# output_path_template = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/tables/efficienciesWithFake_{}_{}.tsv"
input_path = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/json/efficienciesTTbar_{}.json"
output_path_template = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/tables/efficienciesTTbar_{}_{}.tsv"

# -------------------------------
# Mappa ISO
# -------------------------------
ISO_MAP = {
    0.15: "tight Iso",
    0.20: "medium Iso",
    0.25: "loose Iso",
}

# -------------------------------
# Funzioni di parsing
# -------------------------------
def parse_entry(entry):
    """Estrae ISO e ID da un entry JSON e costruisce la chiave unica"""
    wp1 = entry.get("WP1", 0.0)
    wp2 = entry.get("WP2", 0.0)
    id1 = entry.get("Id1", "N/A")
    id2 = entry.get("Id2", "N/A")

    iso1 = ISO_MAP.get(wp1, f"WP1={wp1}")
    iso2 = ISO_MAP.get(wp2, f"WP2={wp2}")

    key = f"{iso1}|{iso2}|{id1}|{id2}"
    return key, iso1, iso2, id1, id2

# -------------------------------
# Funzione pivot e scrittura TSV
# -------------------------------
def create_pivot_and_format(df, value_col, year, region_filter=None):
    """
    Pivotta il DataFrame su Key x Region e salva TSV.
    Filtra opzionalmente per regione.
    """
    if region_filter is not None:
        df = df[df["Region"] == region_filter]

    if df.empty:
        print(f"[WARNING] Nessun dato valido per {year}, regione={region_filter}")
        return None

    df_pivot = df.pivot(index="Key", columns="Region", values=value_col)

    # Split dell'indice Key in colonne separate
    new_cols = df_pivot.index.str.split("|", expand=True)
    new_cols.names = ["Iso_mu1", "Iso_mu2", "ID_mu1", "ID_mu2"]
    df_pivot.index = new_cols
    df_pivot = df_pivot.reset_index()
    df_pivot.columns.name = None

    # Formattazione numerica
    if value_col == "S_sqrtB":
        fmt = "{:.4f}"
    else:
        fmt = "{:.5f}"

    for col in df_pivot.columns:
        if col not in ["Iso_mu1", "Iso_mu2", "ID_mu1", "ID_mu2"]:
            df_pivot[col] = df_pivot[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else "")

    # Nome file
    metric_name = value_col.replace(" (%)", "").replace("_", "")
    if region_filter is not None:
        filename = output_path_template.format(year, f"{metric_name}_{region_filter}")
    else:
        filename = output_path_template.format(year, metric_name)

    df_pivot.to_csv(filename, sep="\t", index=False)
    return filename

# -------------------------------
# MAIN
# -------------------------------
file_input = input_path.format(args.year)
print(f"\n--- Elaborazione anno {args.year} ---")

try:
    with open(file_input, "r") as f:
        data = json.load(f)
except Exception as e:
    print(f"Errore nel file {file_input}: {e}")
    exit()

rows = []
for entry in data:
    if "signal" not in entry or "background" not in entry:
        continue

    signal_eff = entry["signal"].get("eff", 0.0) * 100
    background_eff = entry["background"].get("eff", 0.0) * 100

    # s_sqrtB come float
    s_sqrtB = entry.get("s_sqrtB", {}).get("s_sqrtB", 0.0)

    key, iso1, iso2, id1, id2 = parse_entry(entry)

    rows.append({
        "Key": key,
        "Iso_mu1": iso1,
        "Iso_mu2": iso2,
        "ID_mu1": id1,
        "ID_mu2": id2,
        "Region": entry.get("region", "N/A"),
        "Eff_Signal (%)": signal_eff,
        "Eff_Background (%)": background_eff,
        "S_sqrtB": s_sqrtB
    })

if not rows:
    print("Nessun dato valido trovato. Uscita.")
    exit()

df = pd.DataFrame(rows)

# Creazione TSV
f1 = create_pivot_and_format(df.copy(), "S_sqrtB", args.year, region_filter=args.region)
f2 = create_pivot_and_format(df.copy(), "Eff_Signal (%)", args.year, region_filter=args.region)
f3 = create_pivot_and_format(df.copy(), "Eff_Background (%)", args.year, region_filter=args.region)

print(f"→ S/sqrt(B) salvato in: {f1}")
print(f"→ Efficienza Signal salvata in: {f2}")
print(f"→ Efficienza Background salvata in: {f3}")

print("\nElaborazione completata.")

# import json
# import pandas as pd
# import os

# # -------------------------------
# # Percorsi
# # -------------------------------

# input_path = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/json/efficiencies_{}.json"
# output_path_template = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/tables/efficiencies_{}_{}.tsv"

# # -------------------------------
# # Mappa ISO
# # -------------------------------

# ISO_MAP = {
#     0.15: "tight Iso",
#     0.20: "medium Iso",
#     0.25: "loose Iso",
# }

# years = ["2022", "2022EE", "2023", "2023BPix"]

# # -------------------------------
# # Parsing nuovo formato JSON
# # -------------------------------

# def parse_entry(entry):
#     """
#     Parsing corretto per nuovo JSON:
#     - WP1, WP2
#     - Id1, Id2
#     """
#     wp1 = entry.get("WP1", 0.0)
#     wp2 = entry.get("WP2", 0.0)

#     id1 = entry.get("Id1", "N/A")
#     id2 = entry.get("Id2", "N/A")

#     iso1 = ISO_MAP.get(wp1, f"WP1={wp1}")
#     iso2 = ISO_MAP.get(wp2, f"WP2={wp2}")

#     key = f"{iso1}|{iso2}|{id1}|{id2}"

#     return key, iso1, iso2, id1, id2


# # -------------------------------
# # Pivot e scrittura TSV
# # -------------------------------

# def create_pivot_and_format(df, value_col, year):

#     df_pivot = df.pivot(index='Key', columns='Region', values=value_col)

#     # split indice Key
#     new_cols = df_pivot.index.str.split('|', expand=True)
#     new_cols.names = ['Iso_mu1', 'Iso_mu2', 'ID_mu1', 'ID_mu2']
#     df_pivot.index = new_cols
#     df_pivot = df_pivot.reset_index()
#     df_pivot.columns.name = None

#     # naming file
#     metric_name = value_col.replace(" (%)", "").replace("_", "")
#     filename = output_path_template.format(year, metric_name)

#     # formatting valori numerici
#     if value_col == "S_sqrtB":
#         fmt = "{:.4f}"
#     else:
#         fmt = "{:.5f}"

#     for col in df_pivot.columns:
#         if col not in ['Iso_mu1', 'Iso_mu2', 'ID_mu1', 'ID_mu2']:
#             df_pivot[col] = df_pivot[col].apply(lambda x: fmt.format(x))

#     df_pivot.to_csv(filename, sep="\t", index=False)
#     return filename


# # -------------------------------
# # MAIN LOOP
# # -------------------------------

# for year in years:

#     file_input = input_path.format(year)
#     print(f"\n--- Elaborazione anno {year} ---")

#     # Caricamento JSON
#     try:
#         with open(file_input, 'r') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"Errore nel file {file_input}: {e}")
#         continue

#     rows = []

#     for entry in data:

#         if "signal" not in entry or "background" not in entry:
#             continue

#         signal_eff = entry["signal"].get("eff", 0.0) * 100
#         background_eff = entry["background"].get("eff", 0.0) * 100

#         # nuovo formato: s_sqrtB è un valore diretto
#         s_sqrtB = entry.get("s_sqrtB", 0.0)

#         key, iso1, iso2, id1, id2 = parse_entry(entry)

#         rows.append({
#             "Key": key,
#             "Iso_mu1": iso1,
#             "Iso_mu2": iso2,
#             "ID_mu1": id1,
#             "ID_mu2": id2,
#             "Region": entry.get("region", "N/A"),
#             "Eff_Signal (%)": signal_eff,
#             "Eff_Background (%)": background_eff,
#             "S_sqrtB": s_sqrtB
#         })

#     if not rows:
#         print(f"Nessun dato valido per {year}.")
#         continue

#     df = pd.DataFrame(rows)

#     # TSV finali
#     f1 = create_pivot_and_format(df.copy(), "S_sqrtB", year)
#     print(f"   → S/sqrt(B) salvato in: {f1}")

#     f2 = create_pivot_and_format(df.copy(), "Eff_Signal (%)", year)
#     print(f"   → Efficienza Signal salvata in: {f2}")

#     f3 = create_pivot_and_format(df.copy(), "Eff_Background (%)", year)
#     print(f"   → Efficienza Background salvata in: {f3}")

# print("\nTutti gli anni processati.\n")
