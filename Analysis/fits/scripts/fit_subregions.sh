indir=$1
subregions=$2
year=$3
for region in Z_sideband;
    do for cat in baseline_muonJet ; #VBF ggH VBF_JetVeto;
        do for year in ${year} ; #2022EE 2023 2023BPix; # 2022EE
            do for func in BW_conv_DCS  ; # Voigtian DoubleSidedCB BreitWigner;
                do for subregion in ${subregions} ;
                do for var in m_mumu;
                    do
                    mkdir -p stuff/fits/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var};
                    # mkdir -p /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var};
                    # cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/;
                    # cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/ ;
                    # cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/ ;
                    # cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion} ;
                    # cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/ ;
                    # cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/ ;
                    # cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/ ;

                    echo python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/fit_func3.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${indir}/Run3_${year}/${var}/${var}.root --outFile  stuff/fits/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/fit_data --fitRangeMin 75 --fitRangeMax 105 --var ${var} --year ${year} --region ${region} --subregion ${subregion} --category ${cat} --fitFunc ${func}
                    python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/fit_func3.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${indir}/Run3_${year}/${var}/${var}.root --outFile  stuff/fits/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/fit_data --fitRangeMin 75 --fitRangeMax 105 --var ${var} --year ${year} --region ${region} --subregion ${subregion} --category ${cat} --fitFunc ${func} >  stuff/fits/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/log_data.txt;
                    echo python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/fit_func3.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${indir}/Run3_${year}/${var}/${var}.root --outFile  stuff/fits/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/fit_MC --fitRangeMin 75 --fitRangeMax 105 --var ${var} --year ${year} --region ${region} --subregion ${subregion} --isMC --category ${cat} --fitFunc ${func}
                    python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/fit_func3.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${indir}/Run3_${year}/${var}/${var}.root --outFile  stuff/fits/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/fit_MC --fitRangeMin 75 --fitRangeMax 105 --var ${var} --year ${year} --region ${region} --subregion ${subregion} --isMC --category ${cat} --fitFunc ${func} >  stuff/fits/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/log_MC.txt;

                    # ### log Y
                    # python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/fit_func3.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${indir}/Run3_${year}/${var}/${var}.root --outFile  /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/fit_data_logY --fitRangeMin 75 --fitRangeMax 105 --var ${var} --year ${year} --region ${region} --subregion ${subregion} --category ${cat} --fitFunc ${func} --wantLogY
                    # python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/fit_func3.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${indir}/Run3_${year}/${var}/${var}.root --outFile  /eos/user/v/vdamante/www/H_mumu/fits_02Dec_subregions/${indir}/Run3_${year}/${region}/${subregion}/${cat}/${func}/${var}/fit_MC_logY --fitRangeMin 75 --fitRangeMax 105 --var ${var} --year ${year} --region ${region} --subregion ${subregion} --isMC --category ${cat} --fitFunc ${func} --wantLogY
                done
                done
            done
        done
    done
done
