dir=$1
subregion_opt=$2
year=$3
for region in  Z_sideband  mass_inclusive ;
    do for cat in JetTagSel_def_sel VBF_JetVeto ggH ; #baseline ;
        do mkdir -p /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat} ;
        cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/ ;
        cp /eos/user/v/vdamante/www/H_mumu/Parsedown.php /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/ ;
        for var in m_mumu m_mumu_BS_ScaRe mu1_bsConstrainedPt mu1_pt mu2_bsConstrainedPt mu2_pt_nano pt_mumu_BS pt_mumu_nano m_mumu_BS m_mumu_nano mu1_BS_pt_1_corr mu1_pt_nano mu2_BS_pt_1_corr mu2_RoccoR_pt pt_mumu_BS_RoccoR pt_mumu_RoccoR m_mumu_BS_RoccoR m_mumu_RoccoR mu1_BS_RoccoR_pt mu1_RoccoR_pt mu2_BS_RoccoR_pt pt_mumu pt_mumu_BS_ScaRe
        do

            echo  python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/${var}_LogY_sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution DY,TT,ST,VV,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu --rebin --wantLogY --wantSignal --wantRatio --wantData ${subregion_opt}
            python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/${var}_LogY_sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution DY,TT,ST,VV,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu --rebin --wantLogY --wantSignal --wantRatio --wantData ${subregion_opt}

            # echo python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/${var}_Sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantSignal --wantData --wantRatio ;
            # python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/${var}_Sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantSignal --wantData --wantRatio ;
            # echo python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/${var}_LogY_noSig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantData --wantRatio --wantLogY ;
            # python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/${var}_LogY_noSig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantData --wantRatio --wantLogY ;
            # echo python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/${var}_noSig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantData --wantRatio ;
            # python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/27Oct/${region}/${cat}/${var}_noSig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantData --wantRatio ;
        done
    done
done
