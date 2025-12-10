for region in mass_inclusive Z_sideband Signal_Fit H_sideband
 do for year in 2023BPix; # 2022 2022EE 2023 
    do for cat in VBF_JetVeto ;
        do mkdir -p /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat} ;
        cp /eos/user/v/vdamante/www/H_mumu/index.php /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/ ;
        cp /eos/user/v/vdamante/www/H_mumu/Parsedown.php /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/ ;
        for var in  delta_eta_jj  m_jj  VBFjets_eta  VBFjets_phi  VBFjets_pt ;
        do

            echo  python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_jetObservables/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/${var}_LogY_sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution DY,TT,VV,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu --rebin --wantLogY --wantSignal --wantRatio --wantData
            python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_jetObservables/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/${var}_LogY_sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution DY,TT,VV,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu --rebin --wantLogY --wantSignal --wantRatio --wantData

            # echo python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/${var}_Sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantSignal --wantData --wantRatio ;
            # python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/${var}_Sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantSignal --wantData --wantRatio ;
            # echo python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/${var}_LogY_noSig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantData --wantRatio --wantLogY ;
            # python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/${var}_LogY_noSig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantData --wantRatio --wantLogY ;
            # echo python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/${var}_noSig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantData --wantRatio ;
            # python3 Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/v3_dimuon_system/Run3_${year}/${var}/${var}.root --outFile /eos/user/v/vdamante/www/H_mumu/plots/Run3_${year}/22Oct/${region}/${cat}/${var}_noSig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution all --rebin --wantData --wantRatio ;
        done
    done
done
done