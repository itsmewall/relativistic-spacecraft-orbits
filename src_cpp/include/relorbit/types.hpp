// src_cpp/include/relorbit/types.hpp
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace relorbit {

enum class OrbitStatus : std::uint8_t {
    BOUND = 0,
    UNBOUND = 1,
    CAPTURE = 2,
    ERROR = 3
};

// Estrutura para manobras de missão
struct Maneuver {
    double tau = 0.0;     // Tempo próprio do disparo
    double dv_r = 0.0;    // Delta-v radial (muda pr)
    double dv_phi = 0.0;  // Delta-v tangencial (muda L)
};

struct SolverCfg {
    double dt = 1.0e-3;
    int n_steps = 0;
    int record_every = 1;
    std::vector<Maneuver> maneuvers; 
};

struct TrajectoryNewton {
    std::vector<double> t;
    std::vector<std::array<double, 4>> y;
    std::vector<double> energy;
    std::vector<double> h;
    std::vector<double> mass; // NOVO: Acompanhamento de massa
    OrbitStatus status = OrbitStatus::BOUND;
    std::string message;
};

} // namespace relorbit