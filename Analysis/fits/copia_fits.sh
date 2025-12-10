#!/bin/bash

SRC_BASE="/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions"
DEST_BASE="/eos/user/v/vdamante/www/H_mumu/fits"
INDEX_SRC="/eos/user/v/vdamante/www/H_mumu/index.php"

echo "Copio la struttura in:"
echo "  sorgente: $SRC_BASE"
echo "  destinazione: $DEST_BASE"
echo ""

# Copia la struttura di directory (senza file)
echo ">> Creazione struttura directory..."
cd "$SRC_BASE" || exit 1
find . -type d -exec mkdir -p "$DEST_BASE"/{} \;

# Copia solo i file desiderati
echo ">> Copia dei file fit_data* e fit_MC*..."
find . -type f \( -name "ratio_sigma.*" -o -name "ratio_mean.*" \) | while read -r FILE; do
    DEST_DIR=$(dirname "$DEST_BASE/$FILE")
    cp "$FILE" "$DEST_DIR"/
done

# Copia ricorsivamente l'index.php in ogni cartella
echo ">> Copia dell'index.php in tutte le directory..."
find "$DEST_BASE" -type d -exec cp "$INDEX_SRC" {} \;

echo "Done."
