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
    double dt = 1.0e-3;      // passo fixo (por enquanto)
    std::int32_t n_steps = 10000;
};

struct TrajectoryNewton {
    // y = [x, y, vx, vy] em cada step
    std::vector<double> t;                 // size N
    std::vector<std::array<double, 4>> y;  // size N

    // invariantes por step
    std::vector<double> energy;            // energia específica
    std::vector<double> h;                 // momento angular específico (z)

    OrbitStatus status = OrbitStatus::BOUND;
    std::string message;
};

} // namespace relorbit
