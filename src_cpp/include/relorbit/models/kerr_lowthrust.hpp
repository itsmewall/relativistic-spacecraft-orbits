// src_cpp/include/relorbit/models/kerr_lowthrust.hpp
// Item 5 — Low-Thrust em Kerr Equatorial (ZAMO frame)
//
// TETRADA ZAMO (Zero Angular Momentum Observer):
//   Delta = r^2 - 2Mr + a^2
//   Sigma = r^2  (equatorial: theta=pi/2)
//   rho2  = r^2 + a^2 + 2Ma^2/r   (= g_phiphi efetivo)
//
//   e_(r^)   = sqrt(Delta)/r * d_r
//   e_(phi^) = 1/sqrt(rho2) * (d_phi - Omega_ZAMO * d_t)
//   Omega_ZAMO = 2Ma / (r * rho2)
//
// Componentes de coordenada do empuxo:
//   f^r   = F_r   * sqrt(Delta)/r
//   f^phi = F_phi / sqrt(rho2)
//   f^t   determinado por ortogonalidade f^mu * u_mu = 0
//
// SISTEMA DIFERENCIAL — Estado 6D: y = (r, phi, pr, L, E, m_kg)
//
//  [1] dr/dtau   = pr
//  [2] dphi/dtau = (2MaE/r + (1-2M/r)*L) / Delta
//  [3] dpr/dtau  = [termo geodesico Kerr]  +  F_r * sqrt(Delta)/r
//  [4] dL/dtau   = sqrt(rho2) * F_phi
//  [5] dE/dtau   = via ortogonalidade f^mu u_mu = 0 (ver cpp)
//  [6] dm/dtau   = -m * |F| * c / (Isp * g0)
//
// [3] geodesico Kerr:
//   dpr/dtau_geo = -M/r^2 + (L^2 + a^2(1-E^2))/r^3 - 3M*K^2/r^4
//   com K = L - a*E  (constante de Carter projetada, equatorial)
//   Nota: com empuxo, L e E variam => K = L(tau) - a*E(tau)
//
// [5] calculo de dE/dtau:
//   dE/dtau = A_eff * f^t  onde A_eff = -g_tt - Omega_ZAMO * g_tphi
//   Na pratica: dE/dtau = -(g_tt f^t + g_tphi f^phi)
//   f^t de g_mu_nu * f^mu * u^nu = 0 => resolvido analiticamente no cpp.

#pragma once
#include <cmath>
#include <string>
#include <vector>
#include "relorbit/types.hpp"
#include "relorbit/models/schwarzschild_lowthrust.hpp"  // reutiliza ThrustCfg e ThrustMode

namespace relorbit {

struct TrajectoryKerrLT {
    std::vector<double> tau, r, phi, pr;
    std::vector<double> L, E, mass, epsilon, thrust_mag;
    std::vector<std::string> event_kind;
    std::vector<double> event_tau, event_r, event_phi;
    std::vector<double> event_pr, event_L, event_E, event_mass;
    OrbitStatus status = OrbitStatus::ERROR;
    std::string message;
    double M=1.0, a=0.0, r0=0.0, phi0=0.0, E0=0.0, L0=0.0;
};

// ── Helpers inline para Kerr ──────────────────────────────────

inline double lt_kerr_Delta(double M, double a, double r) {
    return r*r - 2.0*M*r + a*a;
}
inline double lt_kerr_rho2(double M, double a, double r) {
    return r*r + a*a + 2.0*M*a*a/r;   // g_phiphi no equatorial
}
// Veff de Carter (equatorial), com L e E variaveis
inline double lt_kerr_Veff(double M, double a, double r, double L, double E) {
    const double K = L - a*E;
    return (E*E-1.0) + 2.0*M/r - (L*L+a*a*(1.0-E*E))/(r*r) + 2.0*M*K*K/(r*r*r);
    // epsilon = pr^2 - Veff  (= 0 na geodesica)
}
// dVeff/dr com L e E variaveis
inline double lt_kerr_dVeff_dr(double M, double a, double r, double L, double E) {
    const double r2=r*r, r3=r2*r, r4=r3*r;
    const double K = L - a*E;
    return -2.0*M/r2 + 2.0*(L*L+a*a*(1.0-E*E))/r3 - 6.0*M*K*K/r4;
}

TrajectoryKerrLT simulate_kerr_lowthrust_rk4(
    double M, double a,
    double E0, double L0, double r0, double phi0, double pr0,
    double tau0, double tauf,
    const ThrustCfg& thrust, const SolverCfg& cfg,
    double capture_r=2.0, double capture_eps=1e-12);

} // namespace relorbit