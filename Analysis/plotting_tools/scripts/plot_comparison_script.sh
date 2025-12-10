# var1=m_mumu;var2=m_mumu_BS;var3=m_mumu_BS_ScaRe;var4=m_mumu_nano;period=Run3_2022;
# for region in Signal_Fit mass_inclusive
#     do for cat in baseline ggH VBF_JetVeto VBF ;
#         do mkdir -p /eos/user/v/vdamante/www/H_mumu/comparison_plots/Run3_2022/25Aug/${region}/${cat} ;
#         for subreg in etamm_0p9 etamm_1p4 etamm_2p4 etamm_above2p4;
#             do python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/plot_more_files.py  --inFiles /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var1}/${var1}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var2}/${var2}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var3}/${var3}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var4}/${var4}_Central.root  --vars ${var1},${var2},${var3},${var4} --outFile /eos/user/v/vdamante/www/H_mumu/comparison_plots/Run3_2022/25Aug/${region}/${cat}/m_mumu_comparisons_signals_${subreg} --sub_region ${subreg} --contribution ggH,VBF --rebin --category ${cat} --wantLogY --compare_vars --mass_region ${region} --wantSignal ;
#         done
#     done
# done

var1=m_mumu;var2=m_mumu_BS;var3=m_mumu_BS_ScaRe;var4=m_mumu_nano;period=Run3_2022;
for region in Z_sideband H_sideband
    do for cat in baseline ggH VBF_JetVeto VBF ;
        do mkdir -p /eos/user/v/vdamante/www/H_mumu/comparison_plots/Run3_2022/25Aug/${region}/${cat} ;
        for subreg in etamm_0p9 etamm_1p4 etamm_2p4 etamm_above2p4;
            do python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/plot_more_files.py  --inFiles /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var1}/${var1}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var2}/${var2}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var3}/${var3}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var4}/${var4}_Central.root  --vars ${var1},${var2},${var3},${var4} --outFile /eos/user/v/vdamante/www/H_mumu/comparison_plots/Run3_2022/25Aug/${region}/${cat}/m_mumu_comparisons_DY_${subreg} --sub_region ${subreg} --contribution DY --rebin --category ${cat} --wantLogY --compare_vars --mass_region ${region};
            python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/plot_more_files.py  --inFiles /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var1}/${var1}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var2}/${var2}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var3}/${var3}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var4}/${var4}_Central.root  --vars ${var1},${var2},${var3},${var4} --outFile /eos/user/v/vdamante/www/H_mumu/comparison_plots/Run3_2022/25Aug/${region}/${cat}/m_mumu_comparisons_TT_${subreg} --sub_region ${subreg} --contribution TT --rebin --category ${cat} --wantLogY --compare_vars --mass_region ${region};
            python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/plot_more_files.py  --inFiles /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var1}/${var1}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var2}/${var2}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var3}/${var3}_Central.root /eos/user/v/vdamante/H_mumu/merged_hists/v2_subcats_new/${period}/${var4}/${var4}_Central.root  --vars ${var1},${var2},${var3},${var4} --outFile /eos/user/v/vdamante/www/H_mumu/comparison_plots/Run3_2022/25Aug/${region}/${cat}/m_mumu_comparisons_TT_${subreg} --sub_region ${subreg} --contribution all --rebin --category ${cat} --wantLogY --compare_vars --mass_region ${region} ;
        done
    done
done


