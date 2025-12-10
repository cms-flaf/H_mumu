dir=$1
year=$2
final_folder=$3
regions=$4
categories=$5
subregion_opt=$6
base_path="/eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/"

for region in  ${regions} ; # H_sideband mass_inclusive;
    do for cat in ${categories} ; #baseline_MuonPOG; # VBF_JetVeto baseline_muonJet VBF ; # ggH baseline_muon baseline_muonJet VBF_def VBF VBF_JetVeto; # ggH
        do mkdir -p stuff/plots/Run3_${year}/${final_folder}/${region}/${cat} ;

        # ----------------------------------------------------
        #  NUOVO CICLO PER LE VARIABILI (var)
        # ----------------------------------------------------
        # 1. Utilizza find per elencare solo le directory (type d) immediatamente sotto base_path
        # 2. Usa -mindepth 1 -maxdepth 1 per assicurarsi di prendere solo i nomi delle sottocartelle
        # 3. Utilizza basename per ottenere solo il nome della cartella (la variabile) senza il percorso completo
        # 4. Assegna l'output a una variabile (VAR_LIST) o usa direttamente il comando nel ciclo (come qui)

        for var in $(find "${base_path}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n');
        do
            # Verifica se la sottocartella trovata è effettivamente un file .root
            # Non è strettamente necessario, ma è una buona pratica
            if [ -f "${base_path}/${var}/${var}.root" ]; then
                echo "Processing variable: ${var}"

                # Esecuzione del primo comando (per la traccia)
                echo  python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py --inFile "${base_path}/${var}/${var}.root" --outFile stuff/plots/Run3_${year}/${final_folder}/${region}/${cat}/${var}_sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution DY,TT,ST,VV,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu --rebin --wantSignal --wantRatio --wantData ${subregion_opt} --wantLogY

                # Esecuzione del comando effettivo
                python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py --inFile "${base_path}/${var}/${var}.root" --outFile stuff/plots/Run3_${year}/${final_folder}/${region}/${cat}/${var}_sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution DY,TT,VV,ST,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu  --rebin --wantSignal --wantRatio --wantData ${subregion_opt} --wantLogY
            else
                echo "Warning: Expected file ${base_path}/${var}/${var}.root not found. Skipping."
            fi
        done
    done
done


# for region in  Z_sideband ; # H_sideband mass_inclusive;
#     do for cat in baseline_muonJet ggH ; #baseline_MuonPOG; # VBF_JetVeto baseline_muonJet VBF ; # ggH baseline_muon baseline_muonJet VBF_def VBF VBF_JetVeto; # ggH
#         do mkdir -p stuff/plots/Run3_${year}/${final_folder}/${region}/${cat} ;
#         for var in pt_mumu_nano pt_mumu_reapplied pt_mumu_BS pt_mumu_BS_ScaRe ; #m_mumu_reapplied m_mumu_nano m_mumu_BS m_mumu_BS_ScaRe m_mumu_RoccoR m_mumu_BS_RoccoR  pt_mumu_reapplied pt_mumu_nano pt_mumu_BS pt_mumu_BS_ScaRe pt_mumu_RoccoR pt_mumu_BS_RoccoR;
#         do
#             echo  python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/${var}/${var}.root --outFile stuff/plots/Run3_${year}/${final_folder}/${region}/${cat}/${var}_sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution DY,TT,ST,VV,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu --rebin --wantSignal --wantRatio --wantData ${subregion_opt} --wantLogY
#             python3 /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/plotting_tools/main_plotter.py --inFile /eos/user/v/vdamante/H_mumu/merged_hists/${dir}/Run3_${year}/${var}/${var}.root --outFile stuff/plots/Run3_${year}/${final_folder}/${region}/${cat}/${var}_sig --var ${var} --period Run3_${year} --region ${region} --category ${cat} --contribution DY,TT,VV,ST,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu  --rebin --wantSignal --wantRatio --wantData ${subregion_opt} --wantLogY
#         done
#     done
# done

# #  DY,TT,ST,VV,EWK,W_NJets,TW,VVV,TTX,data,VBFHto2Mu,GluGluHto2Mu



# # delta_eta_jj j2_phi m_mumu mu1_bsConstrainedPt mu1_pt_nano mu2_phi pt_mumu R_pt eta_mumu j2_pt m_mumu_BS mu1_BS_pt_1_corr mu1_RoccoR_pt mu2_pt pt_mumu_BS VBFjets_eta j1_eta minDeltaEta m_mumu_BS_RoccoR mu1_BS_RoccoR_pt mu2_bsConstrainedPt mu2_pt_nano pt_mumu_BS_RoccoR VBFjets_phi j1_phi minDeltaEtaSigned m_mumu_BS_ScaRe mu1_eta mu2_BS_pt_1_corr mu2_RoccoR_pt pt_mumu_BS_ScaRe VBFjets_pt j1_pt minDeltaPhi m_mumu_nano mu1_phi mu2_BS_RoccoR_pt phi_mumu pt_mumu_nano y_mumu j2_eta m_jj m_mumu_RoccoR mu1_pt mu2_eta pt_centrality pt_mumu_RoccoR Zepperfield_Var