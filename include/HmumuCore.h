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
};