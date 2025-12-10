year=$1;
var=$2; # m_mumu
dir=$3;
for region in Z_sideband ;
    do for cat in baseline_AtLeastTwoJetOutOfVetoRegions ;
        do mkdir -p stuff/proportions_07Nov/${year}/${region}/${cat}/ ;
        echo python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/yield_calculator/prop_table.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/${var}/${var}.root --var ${var} --period Run3_${year} --mass_region ${region} --category ${cat}  --wantSignal --wantData # --subregion inclusive
        python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/yield_calculator/prop_table.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/${var}/${var}.root --var ${var} --period Run3_${year}  --mass_region ${region} --category ${cat} --wantSignal --wantData # > stuff/proportions_07Nov/${year}/${region}/${cat}/proportions_${var}.txt # --subregion inclusive
    done
done


# /eos/user/v/vdamante/H_mumu/merged_hists/ScaRe_reapplied_25Nov/