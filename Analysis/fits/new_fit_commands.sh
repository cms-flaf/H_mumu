year=$1
dir=$2

for reg in inclusive_etainclusive inclusive_BB inclusive_BO inclusive_BE inclusive_OB inclusive_OO inclusive_OE inclusive_EB inclusive_EO inclusive_EE ; do sh /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/scripts/fit_subregions.sh ${dir} "${reg}" ${year} ;  done

for reg in leading_mu_pt_upto26_etainclusive leading_mu_pt_upto26_BB leading_mu_pt_upto26_BO leading_mu_pt_upto26_BE leading_mu_pt_upto26_OB leading_mu_pt_upto26_OO leading_mu_pt_upto26_OE leading_mu_pt_upto26_EB leading_mu_pt_upto26_EO leading_mu_pt_upto26_EE; do sh /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/scripts/fit_subregions.sh ${dir}  "${reg}" ${year} ;  done

for reg in leading_mu_pt_26to45_etainclusive leading_mu_pt_26to45_BB leading_mu_pt_26to45_BO leading_mu_pt_26to45_BE leading_mu_pt_26to45_OB leading_mu_pt_26to45_OO leading_mu_pt_26to45_OE leading_mu_pt_26to45_EB leading_mu_pt_26to45_EO leading_mu_pt_26to45_EE; do sh /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/scripts/fit_subregions.sh ${dir}  "${reg}" ${year} ;  done

for reg in leading_mu_pt_upto45_etainclusive leading_mu_pt_upto45_BB leading_mu_pt_upto45_BO leading_mu_pt_upto45_BE leading_mu_pt_upto45_OB leading_mu_pt_upto45_OO leading_mu_pt_upto45_OE leading_mu_pt_upto45_EB leading_mu_pt_upto45_EO leading_mu_pt_upto45_EE ; do sh /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/scripts/fit_subregions.sh ${dir}  "${reg}" ${year} ;  done

for reg in leading_mu_pt_upto45_etainclusive leading_mu_pt_upto45_BB leading_mu_pt_upto45_BO leading_mu_pt_upto45_BE leading_mu_pt_upto45_OB leading_mu_pt_upto45_OO leading_mu_pt_upto45_OE leading_mu_pt_upto45_EB leading_mu_pt_upto45_EO leading_mu_pt_upto45_EE ; do sh /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/scripts/fit_subregions.sh ${dir}  "${reg}" ${year} ;  done
for reg in leading_mu_pt_45to52_etainclusive leading_mu_pt_45to52_BB leading_mu_pt_45to52_BO leading_mu_pt_45to52_BE leading_mu_pt_45to52_OB leading_mu_pt_45to52_OO leading_mu_pt_45to52_OE leading_mu_pt_45to52_EB leading_mu_pt_45to52_EO leading_mu_pt_45to52_EE ; do sh /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/scripts/fit_subregions.sh ${dir}  "${reg}" ${year} ;  done

for reg in leading_mu_pt_52to62_etainclusive leading_mu_pt_52to62_BB leading_mu_pt_52to62_BO leading_mu_pt_52to62_BE leading_mu_pt_52to62_OB leading_mu_pt_52to62_OO leading_mu_pt_52to62_OE leading_mu_pt_52to62_EB leading_mu_pt_52to62_EO leading_mu_pt_52to62_EE ; do sh /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/scripts/fit_subregions.sh ${dir}  "${reg}" ${year} ;  done

for reg in leading_mu_pt_above62_etainclusive leading_mu_pt_above62_BB leading_mu_pt_above62_BO leading_mu_pt_above62_BE leading_mu_pt_above62_OB leading_mu_pt_above62_OO leading_mu_pt_above62_OE leading_mu_pt_above62_EB leading_mu_pt_above62_EO leading_mu_pt_above62_EE ; do sh /afs/cern.ch/work/v/vdamante/H_mumu/Analysis/fits/scripts/fit_subregions.sh ${dir}  "${reg}" ${year} ;  done
