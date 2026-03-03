// src_cpp/lib/kerr_lowthrust.cpp
#include "relorbit/models/kerr_lowthrust.hpp"
#include <algorithm>
#include <cmath>

namespace relorbit {

// ───────────────────────────────────────────────────────────────
// RHS Kerr Low-Thrust
// ───────────────────────────────────────────────────────────────
static void rhs_kerr_lt(
    double M, double a, double tau,
    double r, double /*phi*/, double pr, double L, double E, double m_kg,
    const ThrustCfg& thr,
    double& dr, double& dphi, double& dpr, double& dL, double& dE, double& dm
) {
    const double r2    = r*r, r3=r2*r, r4=r3*r;
    const double a2    = a*a;
    const double Delta = lt_kerr_Delta(M, a, r);
    const double Dsafe = (Delta > 1e-300) ? Delta : 1e-300;
    const double rho2  = lt_kerr_rho2(M, a, r);
    const double K     = L - a*E;  // constante de Carter (varia com L,E)

    const double Fr   = thr.eff_F_r  (tau);
    const double Fphi = thr.eff_F_phi(tau);
    const double Fmag = thr.active(tau) ? thr.magnitude() : 0.0;

    // [1] dr/dtau = pr
    dr = pr;

    // [2] dphi/dtau = (2MaE/r + (1-2M/r)*L) / Delta
    dphi = (2.0*M*a*E/r + (1.0-2.0*M/r)*L) / Dsafe;

    // [3] dpr/dtau = geodesico + empuxo radial na tetrada ZAMO
    //   geodesico: -M/r^2 + (L^2+a^2(1-E^2))/r^3 - 3M*K^2/r^4
    //   empuxo: F_r * sqrt(Delta)/r
    const double dpr_geo = -M/r2 + (L*L + a2*(1.0-E*E))/r3 - 3.0*M*K*K/r4;
    const double Dsq     = std::sqrt(std::max(Delta, 0.0));
    dpr = dpr_geo + Fr * Dsq / r;

    // [4] dL/dtau = sqrt(rho2) * F_phi   (torque no frame ZAMO)
    dL = std::sqrt(std::max(rho2, 1e-300)) * Fphi;

    // [5] dE/dtau via ortogonalidade f^mu u_mu = 0
    //   u^t = ( (r^2+a^2+2Ma^2/r)*E - 2Ma*L/r ) / Delta
    //   u^phi = (2MaE/r + (1-2M/r)*L) / Delta
    //   g_tt*f^t + g_tphi*f^phi = -(E contribution)
    //
    //   Resultado compacto (derivado expandindo g_mu_nu f^mu u^nu = 0):
    //   dE/dtau = (F_r * pr * Dsq/r  +  F_phi * L_eff * (Delta/rho2) / r) / E_eff
    //   onde L_eff = rho2 * dphi  (componente de L no frame ZAMO)
    //
    //   Forma equivalente segura numericamente:
    //   dE/dtau = [Fr*(Dsq/r)*pr + Fphi*sqrt(rho2)*dphi*(rho2-2Ma^2/r)/(rho2)] / E
    //
    //   Usamos a forma direta derivada de f^t pela constrainte:
    //   g_tt*u^t*f^t + g_tphi*(u^t*f^phi + u^phi*f^t) + g_phiphi*u^phi*f^phi + g_rr*u^r*f^r = 0
    {
        const double g_tt    = -(1.0 - 2.0*M/r);
        const double g_tphi  = -2.0*M*a/r;
        const double g_rr    = r2 / Dsafe;
        const double ut      = ((r2+a2+2.0*M*a2/r)*E - 2.0*M*a*L/r) / Dsafe;
        const double uphi    = dphi;  // ja calculado
        const double fr_coord  = Fr * Dsq / r;
        const double fphi_coord= Fphi / std::sqrt(std::max(rho2, 1e-300));
        // g_tt*ut*ft + g_tphi*(ut*fphi + uphi*ft) + g_phiphi*uphi*fphi + g_rr*ur*fr = 0
        // ft*(g_tt*ut + g_tphi*uphi) = -(g_tphi*ut*fphi + rho2*uphi*fphi + g_rr*pr*fr)
        const double denom_ft = g_tt*ut + g_tphi*uphi;
        double ft = 0.0;
        if (std::abs(denom_ft) > 1e-14) {
            ft = -(g_tphi*ut*fphi_coord + rho2*uphi*fphi_coord + g_rr*pr*fr_coord) / denom_ft;
        }
        // dE/dtau = -g_tt * ut * ... simplificado: dE/dtau = A_eff * ft
        // Na metrica de Kerr: E = -u_t = -(g_tt*ut + g_tphi*uphi)
        // => dE/dtau = -d(u_t)/dtau = -(g_tt*ft + g_tphi*fphi)
        dE = -(g_tt*ft + g_tphi*fphi_coord);
    }

    // [6] dm/dtau
    dm = -m_kg * Fmag * LT_C_MS / (thr.isp_s * LT_G0);
}

// ───────────────────────────────────────────────────────────────
// Integrador RK4
// ───────────────────────────────────────────────────────────────
TrajectoryKerrLT simulate_kerr_lowthrust_rk4(
    double M, double a,
    double E0, double L0, double r0, double phi0, double pr0,
    double tau0, double tauf,
    const ThrustCfg& thrust, const SolverCfg& cfg,
    double capture_r, double capture_eps
) {
    TrajectoryKerrLT traj;
    traj.M=M; traj.a=a; traj.r0=r0; traj.phi0=phi0; traj.E0=E0; traj.L0=L0;
    traj.status = OrbitStatus::BOUND;

    const double h_step = cfg.dt;
    if (!(h_step>0.0) || !std::isfinite(h_step) || !(tauf>=tau0)) {
        traj.status=OrbitStatus::ERROR; traj.message="invalid parameters"; return traj;
    }
    const double r_plus = M + std::sqrt(std::max(0.0, M*M-a*a));
    const double r_hor  = r_plus*(1.0+capture_eps);
    const double r_cap  = capture_r*M;
    int n_steps = cfg.n_steps>0 ? cfg.n_steps : std::max(1,(int)std::ceil((tauf-tau0)/h_step));
    int rec = cfg.record_every>0 ? cfg.record_every : 1;

    double tau=tau0, r=r0, phi=phi0, pr=pr0, L=L0, E=E0, m=thrust.mass0_kg;

    auto push = [&]() {
        traj.tau.push_back(tau); traj.r.push_back(r); traj.phi.push_back(phi);
        traj.pr.push_back(pr);   traj.L.push_back(L); traj.E.push_back(E);
        traj.mass.push_back(m);
        // epsilon = pr^2 - Veff_Carter (0 na geodesica)
        const double K=L-a*E, r2=r*r;
        const double veff = (E*E-1.0)+2.0*M/r-(L*L+a*a*(1.0-E*E))/r2+2.0*M*K*K/(r2*r);
        traj.epsilon.push_back(pr*pr - veff);
        traj.thrust_mag.push_back(thrust.active(tau)?thrust.magnitude():0.0);
    };

    if (!(r>r_hor)) { traj.status=OrbitStatus::ERROR; traj.message="r0 inside horizon"; return traj; }
    push();
    bool rcap_logged=false;

    for (int step=0; step<n_steps; ++step) {
        double h=h_step;
        if (tau+h>tauf) h=tauf-tau;
        if (!(h>0.0)) break;

        double k1r,k1p,k1ph,k1L,k1E,k1m;
        rhs_kerr_lt(M,a,tau,   r,phi,pr,L,E,m,thrust,k1r,k1p,k1ph,k1L,k1E,k1m);
        double k2r,k2p,k2ph,k2L,k2E,k2m;
        rhs_kerr_lt(M,a,tau+0.5*h, r+0.5*h*k1r,phi+0.5*h*k1p,pr+0.5*h*k1ph,
                    L+0.5*h*k1L,E+0.5*h*k1E,m+0.5*h*k1m, thrust,k2r,k2p,k2ph,k2L,k2E,k2m);
        double k3r,k3p,k3ph,k3L,k3E,k3m;
        rhs_kerr_lt(M,a,tau+0.5*h, r+0.5*h*k2r,phi+0.5*h*k2p,pr+0.5*h*k2ph,
                    L+0.5*h*k2L,E+0.5*h*k2E,m+0.5*h*k2m, thrust,k3r,k3p,k3ph,k3L,k3E,k3m);
        double k4r,k4p,k4ph,k4L,k4E,k4m;
        rhs_kerr_lt(M,a,tau+h,     r+h*k3r,phi+h*k3p,pr+h*k3ph,
                    L+h*k3L,E+h*k3E,m+h*k3m, thrust,k4r,k4p,k4ph,k4L,k4E,k4m);

        const double r_n  =r  +(h/6.0)*(k1r +2*k2r +2*k3r +k4r);
        const double p_n  =phi+(h/6.0)*(k1p +2*k2p +2*k3p +k4p);
        const double pr_n =pr +(h/6.0)*(k1ph+2*k2ph+2*k3ph+k4ph);
        const double L_n  =L  +(h/6.0)*(k1L +2*k2L +2*k3L +k4L);
        const double E_n  =E  +(h/6.0)*(k1E +2*k2E +2*k3E +k4E);
        const double m_n  =std::max(m+(h/6.0)*(k1m+2*k2m+2*k3m+k4m), thrust.dry_mass_kg);
        const double tn   =tau+h;

        if (!std::isfinite(r_n)||!std::isfinite(p_n)||!std::isfinite(pr_n)||
            !std::isfinite(L_n)||!std::isfinite(E_n)) {
            traj.status=OrbitStatus::ERROR; traj.message="non-finite state"; break;
        }
        bool cap_rcap = !rcap_logged && (r>r_cap && r_n<=r_cap);
        bool cap_hor  = (r>r_hor && r_n<=r_hor);
        if (cap_rcap) {
            traj.event_kind.push_back("r_cap");
            traj.event_tau.push_back(tn); traj.event_r.push_back(r_cap);
            traj.event_phi.push_back(p_n); traj.event_pr.push_back(pr_n);
            traj.event_L.push_back(L_n); traj.event_E.push_back(E_n);
            traj.event_mass.push_back(m_n); rcap_logged=true;
        }
        if (cap_hor) {
            traj.event_kind.push_back("horizon");
            traj.event_tau.push_back(tn); traj.event_r.push_back(r_hor);
            traj.event_phi.push_back(p_n); traj.event_pr.push_back(pr_n);
            traj.event_L.push_back(L_n); traj.event_E.push_back(E_n);
            traj.event_mass.push_back(m_n);
            tau=tn; r=r_hor; phi=p_n; pr=pr_n; L=L_n; E=E_n; m=m_n; push();
            traj.status=OrbitStatus::CAPTURE; traj.message="horizon crossed"; break;
        }
        tau=tn; r=r_n; phi=p_n; pr=pr_n; L=L_n; E=E_n; m=m_n;
        if ((step+1)%rec==0||step==n_steps-1||tau>=tauf) push();
        if (tau>=tauf) break;
    }
    return traj;
}

} // namespace relorbit