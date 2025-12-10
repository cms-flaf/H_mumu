year=$1;
var=$2; # m_mumu
dir=$3;
for region in mass_inclusive Z_sideband Signal_Fit H_sideband
    do for cat in baseline JetTagSel_def_sel ggH VBF VBF_JetVeto ;
        do mkdir -p stuff/proportions_27Oct/${year}/${region}/${cat}/ ;
        echo python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/yield_calculator/prop_table.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/${var}/${var}.root --var ${var} --period Run3_${year} --mass_region ${region} --category ${cat} --subregion inclusive --wantSignal --wantData --wantScaledToRun2 > stuff/proportions_27Oct/${year}/${region}/${cat}/proportions_${var}_Run2Scaled.txt
        python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/yield_calculator/prop_table.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/${var}/${var}.root --var ${var} --period Run3_${year} --subregion inclusive --mass_region ${region} --category ${cat} --wantSignal --wantData --wantScaledToRun2 > stuff/proportions_27Oct/${year}/${region}/${cat}/proportions_${var}_Run2Scaled.txt
    done
done
