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

struct SolverCfg {
    double dt = 1.0e-3;   // passo fixo
    int n_steps = 0;      // se 0, calculamos via (tf-t0)/dt
};

struct TrajectoryNewton {
    std::vector<double> t;                 // tempos
    std::vector<std::array<double, 4>> y;  // estado [x,y,vx,vy]
    std::vector<double> energy;            // energia específica
    std::vector<double> h;                 // momento angular específico (z)
    OrbitStatus status = OrbitStatus::BOUND;
    std::string message;
};

} // namespace relorbit
