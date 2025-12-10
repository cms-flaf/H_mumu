#!/bin/bash

# ========================
# Script: make_all_plots.sh
# Usage: ./make_all_plots.sh 21Oct v3_muons_observables
# ========================

# Controllo argomenti
if [ $# -ne 2 ]; then
    echo "❌ Uso corretto: $0 <data> <dir>"
    echo "Esempio: $0 21Oct v3_muons_observables"
    exit 1
fi

# Input
date_tag=$1
dir=$2

# Loop sui vari anni, categorie e regioni
for year in 2022 ; #2022EE 2023 2023BPix; do
  for cat in baseline ggH VBF_JetVeto; do
    for region in mass_inclusive Z_sideband; do

      # Crea la directory di output
      outdir="stuff/plots_${date_tag}/Run3_${year}/${dir}/${region}/${cat}"
      mkdir -p "$outdir"

      # Trova tutti i file ROOT nella directory di input
      input_base="/eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}"
      find "$input_base" -mindepth 2 -maxdepth 2 -name "*.root" | while read infile; do

          # Estraggo il nome della variabile dal file
          var=$(basename "$infile" .root)

          echo "➡️  Plotting ${var} (${year}, ${cat}, ${region})"

          # Esegui il plotter
          python3 Analysis/plotting_tools/main_plotter.py \
            --inFile "$infile" \
            --outFile "${outdir}/${var}" \
            --var "$var" \
            --period "Run3_${year}" \
            --region "$region" \
            --category "$cat" \
            --contribution DY,TT,VV,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu \
            --rebin --wantLogY --wantSignal --wantRatio --wantData

      done
    done
  done
done

echo "✅ Tutti i plot sono stati generati in stuff/plots_${date_tag}/"


# ./afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/scripts/new_command_allplots.sh.sh 21Oct v3_muonObservables