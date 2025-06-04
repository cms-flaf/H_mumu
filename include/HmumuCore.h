#pragma once

#include "FLAF/include/AnalysisTools.h"

struct VBFJets
{
  static constexpr size_t n_legs = 2;
  static constexpr float m_inv_th = 400.;
  static constexpr float eta_th = 2.5;
  std::array<int, n_legs> leg_index;
  std::array<LorentzVectorM, n_legs> leg_p4; // p4 of quark from H->bb
  float m_inv;
  float eta_separation;
  bool isVBF;
  RVecLV legs_p4;
};


float pT_sum (const RVecLV & all_p4s ) {
  float pT_x_sum = 0.;
  float pT_y_sum = 0.;
  for (int idx = 0; idx < all_p4s.size(); idx++){
    pT_x_sum+=all_p4s[idx].Px();
    pT_y_sum+=all_p4s[idx].Py();
  }
  return sqrt(pT_x_sum*pT_x_sum + pT_y_sum*pT_y_sum);
}

float pT_diff (const LorentzVectorM & p4_1,const LorentzVectorM & p4_2) {
  float delta_px = p4_1.Px() - p4_2.Px();
  float delta_py = p4_1.Py() - p4_2.Py();
  return sqrt(delta_px * delta_px + delta_py * delta_py);
}


