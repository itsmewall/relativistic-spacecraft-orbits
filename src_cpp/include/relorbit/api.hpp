#pragma once
#include <array>
#include <vector>

#include "relorbit/types.hpp"

namespace relorbit {

// Core: Newton 2-corpos plano com RK4 (passo fixo) usando array (forte, determinístico)
TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::array<double,4>& state0, // [x,y,vx,vy]
    double t0,
    double tf,
    const SolverCfg& cfg
);

// Wrapper: versão amigável ao Python (lista -> vector)
TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::vector<double>& state0, // espera size==4
    double t0,
    double tf,
    const SolverCfg& cfg
);

} // namespace relorbit
