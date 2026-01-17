#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "relorbit/types.hpp"  // <<< fonte única: OrbitStatus, SolverCfg, TrajectoryNewton

namespace relorbit {

// Trajetória Schwarzschild (equatorial)
struct TrajectorySchwarzschildEq {
    // evolução em tempo próprio tau
    std::vector<double> tau;

    // coordenadas (equatorial): r(tau), phi(tau)
    std::vector<double> r;
    std::vector<double> phi;

    // tempo coordenado t(tau)
    std::vector<double> tcoord;

    // momento radial pr = dr/dtau
    std::vector<double> pr;

    // diagnóstico "por construção" (não usar como validação principal)
    // epsilon(tau) = pr^2 + V_eff(r) - E^2  (deve ser ~0)
    std::vector<double> epsilon;

    // séries de E e L (parâmetros constantes; guardado para auditoria)
    std::vector<double> E_series;
    std::vector<double> L_series;

    // =========================================
    // NOVO: checagem independente via FD
    // =========================================
    // u^t, u^r, u^phi obtidos por derivada numérica de (tcoord,r,phi) em função de tau
    std::vector<double> ut_fd;
    std::vector<double> ur_fd;
    std::vector<double> uphi_fd;

    // norm_u = g_{μν} u^μ u^ν + 1 (timelike). Deve tender a 0 quando dt -> 0.
    std::vector<double> norm_u;

    OrbitStatus status = OrbitStatus::ERROR;
    std::string message;

    // parâmetros do caso
    double M = 0.0;
    double E = 0.0;
    double L = 0.0;
    double r0 = 0.0;
    double phi0 = 0.0;
};

// Potencial efetivo Schwarzschild (partícula massiva, equatorial)
inline double Veff_schw(double M, double r, double L) {
    const double A = 1.0 - 2.0 * M / r;
    const double B = 1.0 + (L * L) / (r * r);
    return A * B;
}

// dV/dr
inline double dVeff_dr_schw(double M, double r, double L) {
    const double A = 1.0 - 2.0 * M / r;
    const double B = 1.0 + (L * L) / (r * r);
    const double dA = 2.0 * M / (r * r);
    const double dB = -2.0 * (L * L) / (r * r * r);
    return dA * B + A * dB;
}

// Helpers teóricos (circular, equatorial)
inline double E_circular(double M, double r) {
    // E = (1-2M/r)/sqrt(1-3M/r)
    return (1.0 - 2.0 * M / r) / std::sqrt(1.0 - 3.0 * M / r);
}
inline double L_circular(double M, double r) {
    // L = sqrt(M r)/sqrt(1-3M/r)
    return std::sqrt(M * r) / std::sqrt(1.0 - 3.0 * M / r);
}

// API Schwarzschild
TrajectorySchwarzschildEq simulate_schwarzschild_equatorial_rk4(
    double M,
    double E,
    double L,
    double r0,
    double phi0,
    double tau0,
    double tauf,
    const SolverCfg& cfg,
    double capture_r = 2.0,     // r <= capture_r*M
    double capture_eps = 1e-12
);

} // namespace relorbit
