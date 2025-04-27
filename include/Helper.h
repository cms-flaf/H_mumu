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
        float eta = Jet_p4.at(i).Eta() - Jet_p4.at(j).Eta();
        if (inv_mass >= inv_mass_th && std::abs(eta) >= eta_th){
          inv_mass_th = inv_mass;
          // eta_th = eta;
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



std::pair<double, double> ComputeCosThetaPhiCS(const LorentzVectorM& mu1_p4, const LorentzVectorM& mu2_p4, double Ebeam) {
    // Calcolare il 4-vettore della coppia dilepton


    // LVector ROOT::Math::VectorUtil::boost
    // Calcolare il vettore di boost verso il rest frame della coppia dilepton




    // Applicare il boost ai muoni
    LorentzVectorXYZ mu1_p4_XYZ = LorentzVectorXYZ{mu1_p4.Px(),mu1_p4.Py(),mu1_p4.Pz(),mu1_p4.E()};
    LorentzVectorXYZ mu2_p4_XYZ = LorentzVectorXYZ{mu2_p4.Px(),mu2_p4.Py(),mu2_p4.Pz(),mu2_p4.E()};
    LorentzVectorXYZ dilepton = mu1_p4_XYZ + mu2_p4_XYZ;
    ROOT::Math::XYZVector boost = -dilepton.BoostToCM();

    LorentzVectorXYZ mu1_boosted_XYZ =  ROOT::Math::VectorUtil::boost (mu1_p4_XYZ, boost);
    LorentzVectorXYZ mu2_boosted_XYZ =  ROOT::Math::VectorUtil::boost (mu2_p4_XYZ, boost);

    // Boost dei protoni (pA e pB)
    LorentzVectorXYZ pA(0, 0, Ebeam, Ebeam);
    LorentzVectorXYZ pB(0, 0, -Ebeam, Ebeam);
    LorentzVectorXYZ pA_boosted =  ROOT::Math::VectorUtil::boost (pA, boost);
    LorentzVectorXYZ pB_boosted =  ROOT::Math::VectorUtil::boost (pB, boost);


    // Calcolare gli assi di Collins-Soper
    const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>> z_cs = (pA_boosted.Vect().Unit() - pB_boosted.Vect().Unit()).Unit();
    const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>> y_cs = (pA_boosted.Vect().Unit().Cross(pB_boosted.Vect().Unit())).Unit();
    const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>> x_cs = (y_cs.Cross(z_cs)).Unit();

    // Calcolare cos(theta_CS)
    double cos_theta_cs = mu1_boosted_XYZ.Vect().Unit().Dot(z_cs);

    // Calcolare phi_CS
    double phi_cs = atan2(mu1_boosted_XYZ.Vect().Dot(y_cs), mu1_boosted_XYZ.Vect().Dot(x_cs));

    return std::make_pair(cos_theta_cs, phi_cs);
}
