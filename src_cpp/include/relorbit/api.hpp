#pragma once
#include <array>
#include "relorbit/types.hpp"

namespace relorbit {

// Simula Newton 2-corpos com RK4 de passo fixo
TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::array<double,4>& state0,
    double t0,
    double tf,
    const SolverCfg& cfg
);

} // namespace relorbit
