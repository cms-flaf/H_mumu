#pragma once
#include "FLAF/include/AnalysisTools.h"
#include "HmumuCore.h"
#include <optional>


VBFJets FindVBFJets(const RVecLV& Jet_p4)
{

  VBFJets VBF_jets_collection ;

  VBF_jets_collection.isVBF=false;

  for(int j = 0; j < VBF_jets_collection.n_legs; j++){
    VBF_jets_collection.leg_index[j]=-1;
  }
  if (Jet_p4.size() < VBF_jets_collection.n_legs)
    return VBF_jets_collection;


  float inv_mass_th = VBF_jets_collection.m_inv_th;
  float eta_th = VBF_jets_collection.eta_th;

  for (size_t i = 0; i < Jet_p4.size(); ++i) {

    for (size_t j = i+1; j < Jet_p4.size(); ++j) {  // j > i per evitare doppioni
        float inv_mass = (Jet_p4.at(i) + Jet_p4.at(j)).M();
        float eta = std::abs(Jet_p4.at(i).Eta() - Jet_p4.at(j).Eta());
        if (inv_mass >= inv_mass_th && eta >= eta_th){
          inv_mass_th = inv_mass;
          eta_th = eta;
          VBF_jets_collection.leg_index[0] = i;
          VBF_jets_collection.leg_index[1] = j;
          VBF_jets_collection.leg_p4[0] = Jet_p4.at(i);
          VBF_jets_collection.leg_p4[1] = Jet_p4.at(j);
          VBF_jets_collection.m_inv = inv_mass;
          VBF_jets_collection.eta_separation = eta;
          VBF_jets_collection.isVBF = true;
        }
    }
  }
  return VBF_jets_collection;

}
