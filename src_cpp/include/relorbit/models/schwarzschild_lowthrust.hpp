// src_cpp/include/relorbit/models/schwarzschild_lowthrust.hpp
// Item 5 — Low-Thrust em Schwarzschild Equatorial
//
// SISTEMA DIFERENCIAL — Estado 6D: y = (r, phi, pr, L, E, m_kg)
//
//  [1] dr/dtau   = pr
//  [2] dphi/dtau = L / r^2
//  [3] dpr/dtau  = -0.5 * dVeff/dr(r,L)  +  F_r * sqrt(A)     A = 1-2M/r
//  [4] dL/dtau   = r * F_phi
//  [5] dE/dtau   = (F_r * pr * sqrt(A)  +  F_phi * L * A / r) / E
//  [6] dm/dtau   = -m * |F| * c / (Isp * g0)
//
// Nota: Veff = A*(1 + L^2/r^2) usa o L *corrente* do estado — nao constante.
// Nota: epsilon = pr^2 + Veff - E^2 sera != 0 durante o empuxo (diagnostico).

#pragma once
#include <cmath>
#include <string>
#include <vector>
#include "relorbit/types.hpp"

namespace relorbit {

enum class ThrustMode {
    CONSTANT = 0, TANGENTIAL_ONLY = 1, RADIAL_ONLY = 2, COAST = 3
};

struct ThrustCfg {
    double F_r = 0.0, F_phi = 0.0;
    double isp_s = 3000.0;
    double mass0_kg = 1000.0, dry_mass_kg = 300.0;
    ThrustMode mode = ThrustMode::CONSTANT;
    double tau_on = 0.0, tau_off = 1e18;

    double magnitude() const { return std::sqrt(F_r*F_r + F_phi*F_phi); }
    bool active(double tau) const {
        return tau>=tau_on && tau<=tau_off && mode!=ThrustMode::COAST && magnitude()>0.0;
    }
    double eff_F_r  (double tau) const { return (active(tau)&&mode!=ThrustMode::TANGENTIAL_ONLY)?F_r  :0.0; }
    double eff_F_phi(double tau) const { return (active(tau)&&mode!=ThrustMode::RADIAL_ONLY   )?F_phi:0.0; }
};

struct TrajectorySchwarzschildLT {
    std::vector<double> tau, r, phi, pr;
    std::vector<double> L, E, mass, epsilon, thrust_mag;
    std::vector<std::string> event_kind;
    std::vector<double> event_tau, event_r, event_phi;
    std::vector<double> event_pr, event_L, event_E, event_mass;
    OrbitStatus status = OrbitStatus::ERROR;
    std::string message;
    double M=1.0, r0=0.0, phi0=0.0, E0=0.0, L0=0.0;
};

static constexpr double LT_C_MS = 2.99792458e8;
static constexpr double LT_G0   = 9.80665;

inline double lt_schw_A      (double M, double r)           { return 1.0-2.0*M/r; }
inline double lt_schw_Veff   (double M, double r, double L) { return lt_schw_A(M,r)*(1.0+L*L/(r*r)); }
inline double lt_schw_dVeff_dr(double M, double r, double L){
    const double r2=r*r, A=lt_schw_A(M,r);
    return (2.0*M/r2)*(1.0+L*L/r2) + A*(-2.0*L*L/(r2*r));
}

TrajectorySchwarzschildLT simulate_schwarzschild_lowthrust_rk4(
    double M, double E0, double L0, double r0, double phi0, double pr0,
    double tau0, double tauf, const ThrustCfg& thrust, const SolverCfg& cfg,
    double capture_r=2.0, double capture_eps=1e-12);

} // namespace relorbit