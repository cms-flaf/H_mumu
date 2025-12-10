import argparse
import glob
import os
import csv
import re
import math
from collections import defaultdict

def smart_round(value, error, sig_digits=2):
    value = float(value)
    error = float(error)
    err_order = math.floor(math.log10(abs(error))) if error != 0. else 0
    rounded_error = round(error, sig_digits - 1 - err_order)
    rounded_value = round(value, sig_digits - 1 - err_order)

    if abs(rounded_value) > 1e4 or abs(rounded_value) < 1e-3:
        return f"{rounded_value:.6e} ± {rounded_error:.6e}"

    return f"{rounded_value} ± {rounded_error}"


folder_names = {
 "inclusive_etainclusive" : "$p_T$ ",
 "inclusive_BB" : "$p_T$ incl",
 "inclusive_BO" : "$p_T$ incl",
 "inclusive_BE" : "$p_T$ incl",
 "inclusive_OB" : "$p_T$ incl",
 "inclusive_OO" : "$p_T$ incl",
 "inclusive_OE" : "$p_T$ incl",
 "inclusive_EB" : "$p_T$ incl",
 "inclusive_EO" : "$p_T$ incl",
 "inclusive_EE" : "$p_T$ incl",
 "leading_mu_pt_upto26_etainclusive" : " $p_T$ < 26 GeV",
 "leading_mu_pt_upto26_BB" : "$p_T$ < 26 GeV",
 "leading_mu_pt_upto26_BO" : "$p_T$ < 26 GeV",
 "leading_mu_pt_upto26_BE" : "$p_T$ < 26 GeV",
 "leading_mu_pt_upto26_OB" : "$p_T$ < 26 GeV",
 "leading_mu_pt_upto26_OO" : "$p_T$ < 26 GeV",
 "leading_mu_pt_upto26_OE" : "$p_T$ < 26 GeV",
 "leading_mu_pt_upto26_EB" : "$p_T$ < 26 GeV",
 "leading_mu_pt_upto26_EO" : "$p_T$ < 26 GeV",
 "leading_mu_pt_upto26_EE" : "$p_T$ < 26 GeV",
 "leading_mu_pt_26to45_etainclusive" : "26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_BB" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_BO" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_BE" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_OB" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_OO" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_OE" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_EB" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_EO" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_26to45_EE" : " 26 < $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_etainclusive" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_BB" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_BO" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_BE" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_OB" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_OO" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_OE" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_EB" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_EO" : " $p_T$ < 45 GeV",
 "leading_mu_pt_upto45_EE" : " $p_T$ < 45 GeV",
 "leading_mu_pt_45to52_etainclusive" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_BB" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_BO" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_BE" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_OB" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_OO" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_OE" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_EB" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_EO" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_45to52_EE" : " 45 < $p_T$ < 52 GeV",
 "leading_mu_pt_52to62_etainclusive" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_BB" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_BO" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_BE" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_OB" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_OO" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_OE" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_EB" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_EO" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_52to62_EE" : " 52 < $p_T$ < 62 GeV",
 "leading_mu_pt_above62_etainclusive" : "$p_T$ > 62 GeV",
 "leading_mu_pt_above62_BB" : " $p_T$ > 62 GeV",
 "leading_mu_pt_above62_BO" : " $p_T$ > 62 GeV",
 "leading_mu_pt_above62_BE" : " $p_T$ > 62 GeV",
 "leading_mu_pt_above62_OB" : " $p_T$ > 62 GeV",
 "leading_mu_pt_above62_OO" : " $p_T$ > 62 GeV",
 "leading_mu_pt_above62_OE" : " $p_T$ > 62 GeV",
 "leading_mu_pt_above62_EB" : " $p_T$ > 62 GeV",
 "leading_mu_pt_above62_EO" : " $p_T$ > 62 GeV",
 "leading_mu_pt_above62_EE" : " $p_T$ > 62 GeV",
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui un fit su istogrammi di dati o MC.")
    parser.add_argument('--year', required=True, help="year")
    parser.add_argument('--function', required=False, default="BW_conv_DCS", help="")
    parser.add_argument('--isMC', action='store_true', help="Set this flag if the input is a Monte Carlo signal.")
    parser.add_argument('--pt_type', required=False, default="ScaRe_reapplied_subregions_1Dec", help="")


    args = parser.parse_args()

    data = "MC" if args.isMC else "data"
    base_path_all = f"/afs/cern.ch/work/v/vdamante/H_mumu/stuff/fits/fits_02Dec_subregions/{args.pt_type}/Run3_{args.year}/"
    base_pattern = f"{base_path_all}/Z_sideband/*/baseline_muonJet/{args.function}/m_mumu/log_{data}.txt"

    files = glob.glob(base_pattern)

    # regex per estrarre num +/- num
    value_regex = re.compile(r"=\s*([0-9\.\-eE]+)\s*\+/-\s*([0-9\.\-eE]+)")

    # dizionario: categoria → regione → righe
    tables = defaultdict(lambda: defaultdict(list))

    for f in files:
        folder = f.split("/Z_sideband/")[1].split("/")[0]
        parts = folder.split("_")
        region = parts[-1]              # BB, BO, ...
        category = "_".join(parts[:-1]) # leading_mu_pt_45
        physical_mean = ""
        res_sigma = ""

        with open(f) as infile:
            for line in infile:
                m = value_regex.search(line)
                if not m:
                    continue
                val, err = m.groups()

                if "physical_mean" in line:
                    physical_mean = smart_round(val, err)
                if "res_sigma" in line:
                    res_sigma = smart_round(val, err)
        tables[category][region].append([folder_names[folder], physical_mean, res_sigma])

    # produci un TSV per ogni categoria
    outname = f"{base_path_all}/summary_results_{data}_{args.function}.csv"
    with open(outname, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        writer.writerow(["region", "mu1 pT", "mean (mZ)", "sigma (res)"])
        for category, regions in tables.items():

            for region in sorted(regions.keys()):
                for row in regions[region]:
                    writer.writerow([region] + row)

        print(f"Creato: {outname}")

