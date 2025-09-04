#pragma once
#include "FLAF/include/AnalysisTools.h"
#include "HmumuCore.h"
#include <optional>

VBFJets FindVBFJets(const RVecLV& Jet_p4) {
    VBFJets VBF_jets_collection;

    VBF_jets_collection.isVBF = false;

    for (int j = 0; j < VBF_jets_collection.n_legs; j++) {
        VBF_jets_collection.leg_index[j] = -1;
    }
    if (Jet_p4.size() < VBF_jets_collection.n_legs)
        return VBF_jets_collection;

    float inv_mass_th = VBF_jets_collection.m_inv_th;
    float eta_th = VBF_jets_collection.eta_th;

    for (size_t i = 0; i < Jet_p4.size(); i++) {
        for (size_t j = i + 1; j < Jet_p4.size(); j++) {  // j > i per evitare doppioni
            float inv_mass = (Jet_p4.at(i) + Jet_p4.at(j)).M();
            float eta = Jet_p4.at(i).Eta() - Jet_p4.at(j).Eta();
            if (inv_mass >= inv_mass_th && std::abs(eta) >= eta_th) {
                inv_mass_th = inv_mass;
                // eta_th = eta;
                VBF_jets_collection.leg_index[0] = i;
                VBF_jets_collection.leg_index[1] = j;
                VBF_jets_collection.leg_p4[0] = Jet_p4.at(i);
                VBF_jets_collection.leg_p4[1] = Jet_p4.at(j);
                VBF_jets_collection.m_inv = inv_mass;
                VBF_jets_collection.eta_separation = eta;
                VBF_jets_collection.isVBF = true;
                VBF_jets_collection.legs_p4 = {Jet_p4.at(i), Jet_p4.at(j)};
            }
        }
    }
    return VBF_jets_collection;
}

std::pair<double, double> ComputeCosThetaPhiCS(const LorentzVectorM& mu1_p4,
                                               const LorentzVectorM& mu2_p4,
                                               double Ebeam) {
    // muons p4 in XYZ coordinates
    LorentzVectorXYZ mu1_p4_XYZ = LorentzVectorXYZ{mu1_p4.Px(), mu1_p4.Py(), mu1_p4.Pz(), mu1_p4.E()};
    LorentzVectorXYZ mu2_p4_XYZ = LorentzVectorXYZ{mu2_p4.Px(), mu2_p4.Py(), mu2_p4.Pz(), mu2_p4.E()};
    // dilepton boosted p4 in XYZ coordinates
    LorentzVectorXYZ dilepton = mu1_p4_XYZ + mu2_p4_XYZ;
    // boost vector in XYZ coordinates
    ROOT::Math::XYZVector boost = -dilepton.BoostToCM();
    // boost muons in XYZ coordinates
    LorentzVectorXYZ mu1_boosted_XYZ = ROOT::Math::VectorUtil::boost(mu1_p4_XYZ, boost);
    LorentzVectorXYZ mu2_boosted_XYZ = ROOT::Math::VectorUtil::boost(mu2_p4_XYZ, boost);
    // proton p4 in XYZ coordinates
    LorentzVectorXYZ pA(0, 0, Ebeam, Ebeam);
    LorentzVectorXYZ pB(0, 0, -Ebeam, Ebeam);
    // boost proton p4 in XYZ coordinates
    LorentzVectorXYZ pA_boosted = ROOT::Math::VectorUtil::boost(pA, boost);
    LorentzVectorXYZ pB_boosted = ROOT::Math::VectorUtil::boost(pB, boost);
    // Collins-Soper axes
    // z axis is the bisector of the angle between the two boosted muons
    // y axis is the normal to the plane defined by the two boosted muons and the two boosted protons
    const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>> z_cs =
        (pA_boosted.Vect().Unit() - pB_boosted.Vect().Unit()).Unit();
    const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>> y_cs =
        (pA_boosted.Vect().Unit().Cross(pB_boosted.Vect().Unit())).Unit();
    const ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>> x_cs = (y_cs.Cross(z_cs)).Unit();
    // cos(theta_CS)
    double cos_theta_cs = mu1_boosted_XYZ.Vect().Unit().Dot(z_cs);
    // phi_CS
    double phi_cs = atan2(mu1_boosted_XYZ.Vect().Dot(y_cs), mu1_boosted_XYZ.Vect().Dot(x_cs));

    return std::make_pair(cos_theta_cs, phi_cs);
}
