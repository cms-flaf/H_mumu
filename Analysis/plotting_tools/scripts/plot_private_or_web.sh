#!/bin/bash

# ==========================================================
# GESTIONE ARGOMENTI E VARIABILI
# Esecuzione: ./script.sh <dir> <year> <final_folder> <regions> <categories> <subregion_opt> [web_flag]
# ==========================================================
dir=$1                # $1: Directory specifica dei merged_hists (es. 2023_v1)
year=$2               # $2: Anno (es. 2023)
final_folder=$3       # $3: Sotto-cartella finale (es. plots_final)
regions=$4            # $4: Lista di regioni (es. "Z_sideband H_sideband")
categories=$5         # $5: Lista di categorie (es. "ggH VBF")
# $6: Flag opzionale per l'output web (es. "web" o "1")
web_flag=$6
# $7: Opzioni aggiuntive per subregion (es. --isInclusive)
subregion_opt=$7

# Base path per gli input (merged histograms)
base_path="/eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/"

# ==========================================================
# LOGICA PER LA CARTELLA BASE DI OUTPUT
# ==========================================================

# Controlla se il $7 (web_flag) è stato fornito e non è vuoto, e lo confronta con "web" o "1"
if [[ -n "$web_flag" ]] && [[ "$web_flag" == "web" || "$web_flag" == "1" ]]; then
    # Se il flag è attivo, usa la directory specificata per il web
    OUTPUT_BASE_DIR="/eos/user/v/vdamante/www/H_mumu/plots/"
    echo "Output directory set to WEB path: ${OUTPUT_BASE_DIR}"
else
    # Altrimenti, usa la directory di default
    OUTPUT_BASE_DIR="stuff/plots/"
    echo "Output directory set to local path: ${OUTPUT_BASE_DIR}"
fi
echo "---"

# ==========================================================
# CICLO DI LAVORO
# ==========================================================
# Ciclo sulle variabili
# Ciclo sulle variabili
for var in $(find "${base_path}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'); do
    input_file="${base_path}/${var}/${var}.root"

    # Verifica se il file di input esiste
    if [ ! -f "${input_file}" ]; then
        echo "File not found: ${input_file}"
        continue
    fi

    echo "Found variable: ${var}"
    read -t 2 -p "Do you wish to process this variable? [y/n/q] (default: yes in 2s) " yn
    yn=${yn:-y}  # se timeout o input vuoto, consideriamo "n"

    case $yn in
        [Yy]* )
            # Ciclo su region e category solo se l'utente conferma
            for region in ${regions}; do
                for cat in ${categories}; do
                    # Crea la cartella di output
                    mkdir -p "${OUTPUT_BASE_DIR}/Run3_${year}/${final_folder}/${region}/${cat}"

                    output_file="${OUTPUT_BASE_DIR}/Run3_${year}/${final_folder}/${region}/${cat}/${var}_sig"

                    echo "Processing variable: ${var} in ${region}/${cat}"
                    echo python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py \
                        --inFile "${input_file}" \
                        --outFile "${output_file}" \
                        --var "${var}" \
                        --period "Run3_${year}" \
                        --region "${region}" \
                        --category "${cat}" \
                        --contribution DY,TT,VV,data,ST,EWK,W_NJets,TW,VVV,TTX,VBFHto2Mu,GluGluHto2Mu \
                        --wantRatio \
                        --wantData \
                        --wantSignal \
                        ${subregion_opt} # --wantLogY  --rebin
                    python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py \
                        --inFile "${input_file}" \
                        --outFile "${output_file}" \
                        --var "${var}" \
                        --period "Run3_${year}" \
                        --region "${region}" \
                        --category "${cat}" \
                        --contribution DY,TT,VV,data,ST,EWK,W_NJets,TW,VVV,TTX,VBFHto2Mu,GluGluHto2Mu \
                        --wantRatio \
                        --wantData \
                        --wantSignal \
                        ${subregion_opt} # --wantLogY  --rebin
                done
            done
            ;;
        [Nn]* )
            continue
            ;;
        [Qq]* )
            echo "Quitting."
            break
            ;;
        * )
            echo "Please answer yes, no, or q to quit."
            ;;
    esac
done

