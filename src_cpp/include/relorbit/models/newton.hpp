#pragma once
#include <array>
#include <cmath>

namespace relorbit {

// RHS do problema de 2 corpos planar: state = [x, y, vx, vy]
inline std::array<double, 4> rhs_two_body(double mu, const std::array<double, 4>& s) {
    const double x = s[0];
    const double y = s[1];
    const double vx = s[2];
    const double vy = s[3];

    const double r2 = x*x + y*y;
    const double r = std::sqrt(r2);
    const double r3 = r2 * r;

    const double ax = -mu * x / r3;
    const double ay = -mu * y / r3;

    return {vx, vy, ax, ay};
}

inline double specific_energy(double mu, const std::array<double,4>& s) {
    const double x = s[0];
    const double y = s[1];
    const double vx = s[2];
    const double vy = s[3];

    const double r = std::sqrt(x*x + y*y);
    const double v2 = vx*vx + vy*vy;
    return 5.0e-1 * v2 - mu / r;
}

inline double specific_angular_momentum_z(const std::array<double,4>& s) {
    const double x = s[0];
    const double y = s[1];
    const double vx = s[2];
    const double vy = s[3];
    return x*vy - y*vx;
}

} // namespace relorbit
