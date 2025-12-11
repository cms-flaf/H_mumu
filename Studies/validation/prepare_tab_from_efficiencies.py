import json
import pandas as pd
import os

# Definizioni dei percorsi
input_path = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/efficiencies_{}.json"
# Modifico output_path per includere il nome della metrica (Sig, Bckg, S_sqrtB)
output_path_template = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/efficiencies_{}_{}.tsv"

# Mappa per la conversione dei valori WP in nomi di Isolamento
ISO_MAP = {
    0.15: "tight Iso",
    0.20: "medium Iso",
    0.25: "loose Iso",
    # Aggiungere altri valori se necessario
}

# Anni/versioni da processare
years = ["2022", "2022EE", "2023", "2023BPix"]

# Funzione per parsare la riga e separare i campi
def parse_and_map_entry(entry):
    """
    Esegue il parsing dei campi ID e WP e li converte in stringhe leggibili.
    """
    wp1 = entry.get('WP1', 0.0)
    wp2 = entry.get('WP2', 0.0)
    id_full = entry.get('ID', 'N/A_mu1_N/A_mu2')

    # 1. Separazione ID1 e ID2
    parts = id_full.split('_mu1_')

    id1 = parts[0]
    id2_full = parts[1] if len(parts) > 1 else 'N/A_mu2'
    id2 = id2_full.replace('_mu2', '')

    # 2. Mappatura WP (0.15 -> tight Iso, ecc.)
    iso1 = ISO_MAP.get(wp1, f"WP1={wp1}")
    iso2 = ISO_MAP.get(wp2, f"WP2={wp2}")

    # 3. Creazione della nuova Key (per l'indice della tabella)
    new_key = f"{iso1}|{iso2}|{id1}|{id2}"

    return new_key, iso1, iso2, id1, id2


# Funzione per pivotare e formattare i dati
def create_pivot_and_format(df, value_col, year, output_path_template):
    """Pivotta il DataFrame per creare una tabella con le regioni come colonne e la salva come TSV."""

    # Pivot: Raggruppa per 'Key' (riga) e usa 'Region' (colonna) come valori
    df_pivot = df.pivot(index='Key', columns='Region', values=value_col)

    # Split dell'indice 'Key' nelle colonne richieste
    new_cols = df_pivot.index.str.split('|', expand=True)
    new_cols.names = ['Iso_mu1', 'Iso_mu2', 'ID_mu1', 'ID_mu2']

    # Unione del nuovo indice separato con il DataFrame
    df_pivot.index = new_cols
    df_pivot = df_pivot.reset_index()

    # Rimuove il nome della colonna 'Region' dall'indice
    df_pivot.columns.name = None

    # Formattazione per la scrittura del file (precisione)
    if value_col in ['Eff_Signal (%)', 'Eff_Background (%)']:
        format_func = lambda x: '{:.5f}'.format(x)
        metric_name = value_col.split(' ')[0]
    else: # S_sqrtB
        format_func = lambda x: '{:.4f}'.format(x)
        metric_name = "S_sqrtB"

    # Applicare la formattazione solo alle colonne numeriche (Regioni)
    region_cols = [col for col in df_pivot.columns if col not in ['Iso_mu1', 'Iso_mu2', 'ID_mu1', 'ID_mu2']]

    for col in region_cols:
        df_pivot[col] = df_pivot[col].apply(format_func)

    # Determina il nome del file di output
    filename = output_path_template.format(year, metric_name)

    # Scrivi sul file TSV
    df_pivot.to_csv(filename, sep='\t', index=False)

    return filename

# --- Ciclo principale sugli anni ---

for year in years:
    file_input = input_path.format(year)
    print(f"\n--- Elaborazione Anno: {year} ---")

    # 1. Carica i dati
    try:
        with open(file_input, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ATTENZIONE: Il file '{file_input}' non è stato trovato. Salto.")
        continue
    except json.JSONDecodeError:
        print(f"Errore: Impossibile decodificare il file '{file_input}'. Salto.")
        continue

    # 2. Prepara e Calcola le metriche
    prepared_data = []
    for entry in data:
        if 'signal' not in entry or 'background' not in entry:
            continue

        # Calcola le metriche
        eff_signal = entry.get('signal', {}).get('eff', 0.0) * 100
        eff_background = entry.get('background', {}).get('eff', 0.0) * 100

        # *** MODIFICA PER IL NUOVO FORMATO s_sqrtB ***
        # s_sqrtB ora si recupera da entry.get('s_sqrtB', {}).get('s_sqrtB', 0.0)
        s_sqrtB = entry.get('s_sqrtB', {}).get('s_sqrtB', 0.0)

        # Parsa e mappa i campi
        new_key, iso1, iso2, id1, id2 = parse_and_map_entry(entry)

        prepared_data.append({
            'Key': new_key,
            'Iso_mu1': iso1,
            'Iso_mu2': iso2,
            'ID_mu1': id1,
            'ID_mu2': id2,
            'Region': entry.get('region', 'N/A'),
            'Eff_Signal (%)': eff_signal,
            'Eff_Background (%)': eff_background,
            'S_sqrtB': s_sqrtB # Valore aggiornato
        })

    if not prepared_data:
        print(f"Nessun dato valido trovato nel file per l'anno {year}. Salto.")
        continue

    df = pd.DataFrame(prepared_data)

    # 3. Generazione delle Tre Tabelle TSV

    file_s_sqrtB = create_pivot_and_format(df.copy(), 'S_sqrtB', year, output_path_template)
    print(f"-> Significatività (S/sqrt(B)) salvata in: '{file_s_sqrtB}'")

    file_sig = create_pivot_and_format(df.copy(), 'Eff_Signal (%)', year, output_path_template)
    print(f"-> Efficienza Signal salvata in: '{file_sig}'")

    file_bckg = create_pivot_and_format(df.copy(), 'Eff_Background (%)', year, output_path_template)
    print(f"-> Efficienza Background salvata in: '{file_bckg}'")

print("\nElaborazione di tutti gli anni completata.")
# import json
# import pandas as pd
# import os

# # Definizioni dei percorsi
# input_path = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/efficiencies_{}.json"
# # Modifico output_path per includere il nome della metrica (Sig, Bckg, S_sqrtB)
# output_path_template = "/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/efficiencies_{}_{}.tsv"

# # Mappa per la conversione dei valori WP in nomi di Isolamento
# ISO_MAP = {
#     0.15: "tight Iso",
#     0.20: "medium Iso",
#     0.25: "loose Iso",
#     # Aggiungere altri valori se necessario
# }

# # Anni/versioni da processare
# years = ["2022", "2022EE", "2023", "2023BPix"]

# # Funzione per parsare la riga e separare i campi
# def parse_and_map_entry(entry):
#     """
#     Esegue il parsing dei campi ID e WP e li converte in stringhe leggibili.
#     """
#     wp1 = entry.get('WP1', 0.0)
#     wp2 = entry.get('WP2', 0.0)
#     id_full = entry.get('ID', 'N/A_mu1_N/A_mu2')

#     # 1. Separazione ID1 e ID2
#     # L'ID è nel formato: "ID_mu1_ID_mu2". Dividiamo per '_mu1_' e '_mu2'
#     parts = id_full.split('_mu1_')

#     # Assumiamo che il formato sia corretto per il parsing
#     id1 = parts[0]
#     id2_full = parts[1] if len(parts) > 1 else 'N/A_mu2'
#     id2 = id2_full.replace('_mu2', '')

#     # 2. Mappatura WP (0.15 -> tight Iso, ecc.)
#     iso1 = ISO_MAP.get(wp1, f"WP1={wp1}")
#     iso2 = ISO_MAP.get(wp2, f"WP2={wp2}")

#     # 3. Creazione della nuova Key (per l'indice della tabella)
#     new_key = f"{iso1}|{iso2}|{id1}|{id2}"

#     return new_key, iso1, iso2, id1, id2


# # Funzione per pivotare e formattare i dati
# def create_pivot_and_format(df, value_col, year, output_path_template):
#     """Pivotta il DataFrame per creare una tabella con le regioni come colonne e la salva come TSV."""

#     # Pivot: Raggruppa per 'Key' (riga) e usa 'Region' (colonna) come valori
#     df_pivot = df.pivot(index='Key', columns='Region', values=value_col)

#     # Split dell'indice 'Key' nelle colonne richieste
#     new_cols = df_pivot.index.str.split('|', expand=True)
#     new_cols.names = ['Iso_mu1', 'Iso_mu2', 'ID_mu1', 'ID_mu2']

#     # Unione del nuovo indice separato con il DataFrame
#     df_pivot.index = new_cols
#     df_pivot = df_pivot.reset_index()

#     # Rimuove il nome della colonna 'Region' dall'indice
#     df_pivot.columns.name = None

#     # Formattazione per la scrittura del file (precisione)
#     if value_col in ['Eff_Signal (%)', 'Eff_Background (%)']:
#         format_func = lambda x: '{:.5f}'.format(x)
#         metric_name = value_col.split(' ')[0]
#     else: # S_sqrtB
#         format_func = lambda x: '{:.4f}'.format(x)
#         metric_name = "S_sqrtB"

#     # Applicare la formattazione solo alle colonne numeriche (Regioni)
#     region_cols = [col for col in df_pivot.columns if col not in ['Iso_mu1', 'Iso_mu2', 'ID_mu1', 'ID_mu2']]

#     for col in region_cols:
#         df_pivot[col] = df_pivot[col].apply(format_func)

#     # Determina il nome del file di output
#     filename = output_path_template.format(year, metric_name)

#     # Scrivi sul file TSV
#     df_pivot.to_csv(filename, sep='\t', index=False)

#     return filename

# # --- Ciclo principale sugli anni ---

# for year in years:
#     file_input = input_path.format(year)
#     print(f"\n--- Elaborazione Anno: {year} ---")

#     # 1. Carica i dati
#     try:
#         with open(file_input, 'r') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"ATTENZIONE: Il file '{file_input}' non è stato trovato. Salto.")
#         continue
#     except json.JSONDecodeError:
#         print(f"Errore: Impossibile decodificare il file '{file_input}'. Salto.")
#         continue

#     # 2. Prepara e Calcola le metriche
#     prepared_data = []
#     for entry in data:
#         if 'signal' not in entry or 'background' not in entry:
#             continue

#         # Calcola le metriche
#         eff_signal = entry.get('signal', {}).get('eff', 0.0) * 100
#         eff_background = entry.get('background', {}).get('eff', 0.0) * 100
#         s_sqrtB = entry.get('signal', {}).get('s_sqrtB', 0.0)

#         # Parsa e mappa i campi
#         new_key, iso1, iso2, id1, id2 = parse_and_map_entry(entry)

#         prepared_data.append({
#             'Key': new_key, # Manteniamo 'Key' per il pivot
#             'Iso_mu1': iso1, # Nuove colonne mappate
#             'Iso_mu2': iso2,
#             'ID_mu1': id1,
#             'ID_mu2': id2,
#             'Region': entry.get('region', 'N/A'),
#             'Eff_Signal (%)': eff_signal,
#             'Eff_Background (%)': eff_background,
#             'S_sqrtB': s_sqrtB
#         })

#     if not prepared_data:
#         print(f"Nessun dato valido trovato nel file per l'anno {year}. Salto.")
#         continue

#     df = pd.DataFrame(prepared_data)

#     # 3. Generazione delle Tre Tabelle TSV

#     # Le tabelle devono contenere come prime colonne le nuove righe separate.
#     # Per farlo, uso la pivotazione sull'indice 'Key' e poi lo scompongo
#     # in 'Iso_mu1', 'Iso_mu2', 'ID_mu1', 'ID_mu2' subito dopo.

#     # Per S/sqrt(B)
#     file_s_sqrtB = create_pivot_and_format(df.copy(), 'S_sqrtB', year, output_path_template)
#     print(f"-> Significatività (S/sqrt(B)) salvata in: '{file_s_sqrtB}'")

#     # Per Efficienza Signal
#     file_sig = create_pivot_and_format(df.copy(), 'Eff_Signal (%)', year, output_path_template)
#     print(f"-> Efficienza Signal salvata in: '{file_sig}'")

#     # Per Efficienza Background
#     file_bckg = create_pivot_and_format(df.copy(), 'Eff_Background (%)', year, output_path_template)
#     print(f"-> Efficienza Background salvata in: '{file_bckg}'")

# print("\nElaborazione di tutti gli anni completata.")

