// src_cpp/lib/schwarzschild_lowthrust.cpp
#include "relorbit/models/schwarzschild_lowthrust.hpp"
#include <algorithm>
#include <cmath>

namespace relorbit {

// ───────────────────────────────────────────────────────────────
// RHS: as 6 equacoes diferenciais do empuxo continuo
// ───────────────────────────────────────────────────────────────
static void rhs_schw_lt(
    double M, double tau,
    double r, double /*phi*/, double pr, double L, double E, double m_kg,
    const ThrustCfg& thr,
    double& dr, double& dphi, double& dpr, double& dL, double& dE, double& dm
) {
    const double A    = lt_schw_A(M, r);
    const double Asq  = std::sqrt(std::max(A, 1e-300));
    const double Fr   = thr.eff_F_r  (tau);
    const double Fphi = thr.eff_F_phi(tau);
    const double Fmag = thr.active(tau) ? thr.magnitude() : 0.0;

    // [1]
    dr   = pr;
    // [2]
    dphi = L / (r*r);
    // [3]  geodesica + empuxo radial
    dpr  = -0.5 * lt_schw_dVeff_dr(M, r, L)  +  Fr * Asq;
    // [4]  torque
    dL   = r * Fphi;
    // [5]  trabalho da 4-forca
    const double Esafe = (std::abs(E) > 1e-14) ? E : 1e-14;
    dE   = (Fr * pr * Asq  +  Fphi * L * A / r) / Esafe;
    // [6]  exaustao de massa em tempo proprio geometrico
    dm   = -m_kg * Fmag * LT_C_MS / (thr.isp_s * LT_G0);
}

// ───────────────────────────────────────────────────────────────
// Integrador RK4 com estado 6D
// ───────────────────────────────────────────────────────────────
TrajectorySchwarzschildLT simulate_schwarzschild_lowthrust_rk4(
    double M, double E0, double L0, double r0, double phi0, double pr0,
    double tau0, double tauf, const ThrustCfg& thrust, const SolverCfg& cfg,
    double capture_r, double capture_eps
) {
    TrajectorySchwarzschildLT traj;
    traj.M=M; traj.r0=r0; traj.phi0=phi0; traj.E0=E0; traj.L0=L0;
    traj.status = OrbitStatus::BOUND;

    const double h_step = cfg.dt;
    if (!(h_step>0.0) || !std::isfinite(h_step) || !(tauf>=tau0)) {
        traj.status=OrbitStatus::ERROR; traj.message="invalid parameters"; return traj;
    }
    const double r_hor = 2.0*M*(1.0+capture_eps);
    const double r_cap = capture_r*M;
    int n_steps = cfg.n_steps>0 ? cfg.n_steps
                : std::max(1,(int)std::ceil((tauf-tau0)/h_step));
    int rec = cfg.record_every>0 ? cfg.record_every : 1;

    // Estado corrente
    double tau=tau0, r=r0, phi=phi0, pr=pr0, L=L0, E=E0;
    double m = thrust.mass0_kg;

    auto push = [&]() {
        traj.tau.push_back(tau);  traj.r.push_back(r);    traj.phi.push_back(phi);
        traj.pr.push_back(pr);    traj.L.push_back(L);    traj.E.push_back(E);
        traj.mass.push_back(m);
        traj.epsilon.push_back(pr*pr + lt_schw_Veff(M,r,L) - E*E);
        traj.thrust_mag.push_back(thrust.active(tau) ? thrust.magnitude() : 0.0);
    };

    if (!(r > r_hor)) { traj.status=OrbitStatus::ERROR; traj.message="r0 inside horizon"; return traj; }
    push();
    bool rcap_logged = false;

    for (int step=0; step<n_steps; ++step) {
        double h = h_step;
        if (tau+h > tauf) h = tauf-tau;
        if (!(h > 0.0)) break;

        // k1
        double k1r,k1p,k1ph,k1L,k1E,k1m;
        rhs_schw_lt(M,tau,  r,phi,pr,L,E,m, thrust, k1r,k1p,k1ph,k1L,k1E,k1m);
        // k2
        double k2r,k2p,k2ph,k2L,k2E,k2m;
        rhs_schw_lt(M,tau+0.5*h,
            r+0.5*h*k1r, phi+0.5*h*k1p, pr+0.5*h*k1ph,
            L+0.5*h*k1L, E+0.5*h*k1E,  m+0.5*h*k1m,
            thrust, k2r,k2p,k2ph,k2L,k2E,k2m);
        // k3
        double k3r,k3p,k3ph,k3L,k3E,k3m;
        rhs_schw_lt(M,tau+0.5*h,
            r+0.5*h*k2r, phi+0.5*h*k2p, pr+0.5*h*k2ph,
            L+0.5*h*k2L, E+0.5*h*k2E,  m+0.5*h*k2m,
            thrust, k3r,k3p,k3ph,k3L,k3E,k3m);
        // k4
        double k4r,k4p,k4ph,k4L,k4E,k4m;
        rhs_schw_lt(M,tau+h,
            r+h*k3r, phi+h*k3p, pr+h*k3ph,
            L+h*k3L, E+h*k3E,  m+h*k3m,
            thrust, k4r,k4p,k4ph,k4L,k4E,k4m);

        const double r_n   = r   + (h/6.0)*(k1r  +2*k2r  +2*k3r  +k4r);
        const double phi_n = phi + (h/6.0)*(k1p  +2*k2p  +2*k3p  +k4p);
        const double pr_n  = pr  + (h/6.0)*(k1ph +2*k2ph +2*k3ph +k4ph);
        const double L_n   = L   + (h/6.0)*(k1L  +2*k2L  +2*k3L  +k4L);
        const double E_n   = E   + (h/6.0)*(k1E  +2*k2E  +2*k3E  +k4E);
        const double m_n   = std::max(m + (h/6.0)*(k1m+2*k2m+2*k3m+k4m), thrust.dry_mass_kg);
        const double tau_n = tau+h;

        if (!std::isfinite(r_n)||!std::isfinite(phi_n)||!std::isfinite(pr_n)||
            !std::isfinite(L_n)||!std::isfinite(E_n)) {
            traj.status=OrbitStatus::ERROR; traj.message="non-finite state"; break;
        }

        // Detecao de eventos
        bool cap_hor  = (r>r_hor  && r_n<=r_hor);
        bool cap_rcap = !rcap_logged && (r>r_cap && r_n<=r_cap);
        if (cap_rcap) {
            traj.event_kind.push_back("r_cap");
            traj.event_tau.push_back(tau_n); traj.event_r.push_back(r_cap);
            traj.event_phi.push_back(phi_n); traj.event_pr.push_back(pr_n);
            traj.event_L.push_back(L_n);     traj.event_E.push_back(E_n);
            traj.event_mass.push_back(m_n);  rcap_logged=true;
        }
        if (cap_hor) {
            traj.event_kind.push_back("horizon");
            traj.event_tau.push_back(tau_n); traj.event_r.push_back(r_hor);
            traj.event_phi.push_back(phi_n); traj.event_pr.push_back(pr_n);
            traj.event_L.push_back(L_n);     traj.event_E.push_back(E_n);
            traj.event_mass.push_back(m_n);
            tau=tau_n; r=r_hor; phi=phi_n; pr=pr_n; L=L_n; E=E_n; m=m_n; push();
            traj.status=OrbitStatus::CAPTURE; traj.message="horizon crossed"; break;
        }

        tau=tau_n; r=r_n; phi=phi_n; pr=pr_n; L=L_n; E=E_n; m=m_n;
        if ((step+1)%rec==0 || step==n_steps-1 || tau>=tauf) push();
        if (tau>=tauf) break;
    }
    return traj;
}

} // namespace relorbit