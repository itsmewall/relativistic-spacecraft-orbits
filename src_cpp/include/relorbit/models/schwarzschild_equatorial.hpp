#pragma once
#include <cmath>
#include <string>
#include <vector>

namespace relorbit {

enum class OrbitStatus { BOUND, UNBOUND, CAPTURE, ERROR };

struct SolverCfg {
    double dt = 1e-3;
    int n_steps = 0; // se 0, calcula por (tf-t0)/dt
};

// Trajetória Newton (mantém compatibilidade com o que você já tinha)
struct TrajectoryNewton {
    std::vector<double> t;
    std::vector<std::vector<double>> y; // [x,y,vx,vy]
    std::vector<double> energy;
    std::vector<double> h;
    OrbitStatus status = OrbitStatus::ERROR;
    std::string message;
};

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

    // diagnóstico do constraint:
    // epsilon(tau) = pr^2 + V_eff(r) - E^2  (deve ser ~0)
    std::vector<double> epsilon;

    // séries de E e L "reconstruídas" do estado (devem ser constantes)
    // Aqui, por construção, E e L são parâmetros; ainda assim guardamos para auditoria.
    std::vector<double> E_series;
    std::vector<double> L_series;

    OrbitStatus status = OrbitStatus::ERROR;
    std::string message;

    // parâmetros do caso (úteis para relatório)
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

// API (declarações)
TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::vector<double>& state0, // [x,y,vx,vy]
    double t0,
    double tf,
    const SolverCfg& cfg
);

TrajectorySchwarzschildEq simulate_schwarzschild_equatorial_rk4(
    double M,
    double E,
    double L,
    double r0,
    double phi0,
    double tau0,
    double tauf,
    const SolverCfg& cfg,
    double capture_r = 2.0,     // limiar de captura em unidades de M: r <= capture_r*M
    double capture_eps = 1e-12  // tolerância numérica (evita falso “cruzou” por ruído)
);

} // namespace relorbit
