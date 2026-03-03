// src_cpp/lib/kerr_equatorial.cpp
#include "relorbit/models/kerr_equatorial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace relorbit {

static inline bool is_finite(double x) { return std::isfinite(x); }

static inline void push_event(
    TrajectoryKerrEq& traj,
    const std::string& kind,
    double tau, double tcoord, double vcoord, double r, double phi, double pr
) {
    traj.event_kind.push_back(kind);
    traj.event_tau.push_back(tau);
    traj.event_tcoord.push_back(tcoord);
    traj.event_vcoord.push_back(vcoord);
    traj.event_r.push_back(r);
    traj.event_phi.push_back(phi);
    traj.event_pr.push_back(pr);
}

static inline double lerp(double x0, double x1, double alpha) {
    return x0 + alpha * (x1 - x0);
}

static inline bool crossing_r(
    double r_prev, double r_next, double r_thr,
    double& alpha_out
) {
    if (!(r_prev > r_thr && r_next <= r_thr)) return false;
    const double denom = (r_prev - r_next);
    double alpha = 0.0;
    if (std::abs(denom) > 0.0) alpha = (r_prev - r_thr) / denom;
    alpha_out = std::clamp(alpha, 0.0, 1.0);
    return true;
}

// Raiz externa do horizonte de Kerr
static inline double r_plus_kerr(double M, double a) {
    return M + std::sqrt(std::max(0.0, M * M - a * a));
}

// ODE RHS para Kerr Equatorial em tempo próprio tau
static inline void rhs_kerr_eq(
    double M, double a, double E, double L,
    double r, double /*phi*/, double /*tcoord*/, double /*vcoord*/, double pr,
    double& dr, double& dphi, double& dt, double& dv, double& dpr
) {
    dr = pr;
    
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double r4 = r3 * r;
    const double a2 = a * a;
    const double E2 = E * E;
    
    const double Delta = r2 - 2.0 * M * r + a2;
    constexpr double Delta_floor = 1e-300;
    const double D_safe = (Delta >= Delta_floor) ? Delta : Delta_floor;

    // Equações de movimento acopladas (Frame Dragging)
    dphi = (2.0 * M * a * E / r + (1.0 - 2.0 * M / r) * L) / D_safe;
    dt   = ((r2 + a2 + 2.0 * M * a2 / r) * E - 2.0 * M * a * L / r) / D_safe;
    
    // Tempo regular (análogo a ingoing EF)
    dv   = dt + ((r2 + a2) / D_safe) * pr;

    const double K = L - a * E;
    // Aceleração radial
    dpr = -M / r2 + (L * L + a2 * (1.0 - E2)) / r3 - 3.0 * M * K * K / r4;
}

TrajectoryKerrEq simulate_kerr_equatorial_rk4(
    double M, double a, double E, double L,
    double r0, double phi0, double pr0,
    double tau0, double tauf,
    const SolverCfg& cfg,
    double capture_r, double capture_eps
) {
    TrajectoryKerrEq traj;
    traj.M = M; traj.a = a; traj.E = E; traj.L = L; traj.r0 = r0; traj.phi0 = phi0;

    traj.status = OrbitStatus::BOUND;
    traj.message.clear();

    const double dt0 = cfg.dt;
    if (!(dt0 > 0.0) || !is_finite(dt0) || !(tauf >= tau0)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid dt or span";
        return traj;
    }

    const double r_plus = r_plus_kerr(M, a);
    const double r_cap = capture_r * M;
    const double r_hor = r_plus * (1.0 + capture_eps);

    int n_steps = cfg.n_steps;
    if (n_steps <= 0) {
        n_steps = std::max(1, static_cast<int>(std::ceil((tauf - tau0) / dt0)));
    }

    // Reservas de memória
    size_t res_size = static_cast<size_t>(n_steps) + 1;
    traj.tau.reserve(res_size); traj.r.reserve(res_size); traj.phi.reserve(res_size);
    traj.tcoord.reserve(res_size); traj.vcoord.reserve(res_size); traj.pr.reserve(res_size);
    traj.epsilon.reserve(res_size); traj.E_series.reserve(res_size); traj.L_series.reserve(res_size);

    double tau = tau0, r = r0, phi = phi0, tcoord = 0.0, vcoord = 0.0, pr = pr0;

    auto append_sample = [&](double tau_s, double r_s, double phi_s, double t_s, double v_s, double pr_s) {
        traj.tau.push_back(tau_s); traj.r.push_back(r_s); traj.phi.push_back(phi_s);
        traj.tcoord.push_back(t_s); traj.vcoord.push_back(v_s); traj.pr.push_back(pr_s);
        
        // Conservação de energia pseudo-hamiltoniana
        const double K = L - a * E;
        const double theory_pr2 = (E*E - 1.0) + 2.0*M/r_s - (L*L + a*a*(1.0 - E*E))/(r_s*r_s) + 2.0*M*K*K/(r_s*r_s*r_s);
        traj.epsilon.push_back(pr_s * pr_s - theory_pr2);
        
        traj.E_series.push_back(E); traj.L_series.push_back(L);
    };

    if (!(r > r_hor)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid initial state (r must be > r_plus*(1+eps))";
        return traj;
    }

    append_sample(tau, r, phi, tcoord, vcoord, pr);
    bool rcap_logged = false;

    // Loop Principal RK4
    for (int step = 0; step < n_steps; ++step) {
        double h = dt0;
        if (tau + h > tauf) h = (tauf - tau);
        if (!(h > 0.0)) break;

        const double tau_prev = tau, r_prev = r, phi_prev = phi, t_prev = tcoord, v_prev = vcoord, pr_prev = pr;

        double k1_r, k1_phi, k1_t, k1_v, k1_pr;
        rhs_kerr_eq(M, a, E, L, r, phi, tcoord, vcoord, pr, k1_r, k1_phi, k1_t, k1_v, k1_pr);

        double k2_r, k2_phi, k2_t, k2_v, k2_pr;
        rhs_kerr_eq(M, a, E, L, r + 0.5*h*k1_r, phi + 0.5*h*k1_phi, tcoord + 0.5*h*k1_t, vcoord + 0.5*h*k1_v, pr + 0.5*h*k1_pr, k2_r, k2_phi, k2_t, k2_v, k2_pr);

        double k3_r, k3_phi, k3_t, k3_v, k3_pr;
        rhs_kerr_eq(M, a, E, L, r + 0.5*h*k2_r, phi + 0.5*h*k2_phi, tcoord + 0.5*h*k2_t, vcoord + 0.5*h*k2_v, pr + 0.5*h*k2_pr, k3_r, k3_phi, k3_t, k3_v, k3_pr);

        double k4_r, k4_phi, k4_t, k4_v, k4_pr;
        rhs_kerr_eq(M, a, E, L, r + h*k3_r, phi + h*k3_phi, tcoord + h*k3_t, vcoord + h*k3_v, pr + h*k3_pr, k4_r, k4_phi, k4_t, k4_v, k4_pr);

        const double r_next  = r  + (h / 6.0) * (k1_r  + 2.0*k2_r  + 2.0*k3_r  + k4_r);
        const double phi_next= phi+ (h / 6.0) * (k1_phi+ 2.0*k2_phi+ 2.0*k3_phi+ k4_phi);
        const double t_next  = tcoord+(h / 6.0)* (k1_t  + 2.0*k2_t  + 2.0*k3_t  + k4_t);
        const double v_next  = vcoord+(h / 6.0)* (k1_v  + 2.0*k2_v  + 2.0*k3_v  + k4_v);
        const double pr_next = pr + (h / 6.0) * (k1_pr + 2.0*k2_pr + 2.0*k3_pr + k4_pr);
        const double tau_next= tau + h;

        if (!is_finite(r_next) || !is_finite(phi_next) || !is_finite(t_next) || !is_finite(pr_next)) {
            traj.status = OrbitStatus::ERROR;
            traj.message = "non-finite state encountered";
            break;
        }

        // Turning events (periapse/apoapse)
        if (pr_prev != 0.0) {
            if ((pr_prev < 0.0 && pr_next >= 0.0) || (pr_prev > 0.0 && pr_next <= 0.0)) {
                const double denom = (pr_prev - pr_next);
                double alpha = (std::abs(denom) > 0.0) ? (pr_prev / denom) : 0.0;
                alpha = std::clamp(alpha, 0.0, 1.0);

                push_event(traj, (pr_prev < 0.0) ? "periapse" : "apoapse",
                           tau_prev + alpha*h, lerp(t_prev, t_next, alpha), lerp(v_prev, v_next, alpha),
                           lerp(r_prev, r_next, alpha), lerp(phi_prev, phi_next, alpha), 0.0);
            }
        }

        // r_cap marker
        if (!rcap_logged) {
            double alpha = 0.0;
            if (crossing_r(r_prev, r_next, r_cap, alpha)) {
                push_event(traj, "r_cap", tau_prev + alpha*h, lerp(t_prev, t_next, alpha), lerp(v_prev, v_next, alpha),
                           r_cap, lerp(phi_prev, phi_next, alpha), lerp(pr_prev, pr_next, alpha));
                rcap_logged = true;
            }
        }

        // Horizon crossing
        {
            double alpha = 0.0;
            if (crossing_r(r_prev, r_next, r_hor, alpha)) {
                push_event(traj, "horizon", tau_prev + alpha*h, lerp(t_prev, t_next, alpha), lerp(v_prev, v_next, alpha),
                           r_hor, lerp(phi_prev, phi_next, alpha), lerp(pr_prev, pr_next, alpha));
                
                append_sample(tau_prev + alpha*h, r_hor, lerp(phi_prev, phi_next, alpha), lerp(t_prev, t_next, alpha), lerp(v_prev, v_next, alpha), lerp(pr_prev, pr_next, alpha));
                traj.status = OrbitStatus::CAPTURE;
                traj.message = "horizon crossed (r <= r_plus*(1+eps))";
                break;
            }
        }

        tau = tau_next; r = r_next; phi = phi_next; tcoord = t_next; vcoord = v_next; pr = pr_next;
        append_sample(tau, r, phi, tcoord, vcoord, pr);
    }

    // Pós-processamento Teórico / FD para métrica
    const size_t N = traj.tau.size();
    traj.ut_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.vt_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.ur_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.uphi_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.norm_u.assign(N, std::numeric_limits<double>::quiet_NaN());

    traj.ut_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.vt_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.ur_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.uphi_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.norm_u_theory.assign(N, std::numeric_limits<double>::quiet_NaN());

    constexpr double Delta_min_theory = 1e-12;

    for (size_t i = 0; i < N; ++i) {
        const double rr = traj.r[i];
        const double Delta = rr * rr - 2.0 * M * rr + a * a;
        
        if (!is_finite(rr) || !(rr > 0.0) || !is_finite(Delta) || Delta < Delta_min_theory) continue;

        double tdot, phidot, dummy_r, dummy_v, dummy_pr;
        rhs_kerr_eq(M, a, E, L, rr, 0.0, 0.0, 0.0, traj.pr[i], dummy_r, phidot, tdot, dummy_v, dummy_pr);

        traj.ut_theory[i] = tdot;
        traj.vt_theory[i] = dummy_v;
        traj.ur_theory[i] = traj.pr[i];
        traj.uphi_theory[i] = phidot;

        const double g_tt = -(1.0 - 2.0*M/rr);
        const double g_tphi = -2.0*M*a/rr;
        const double g_phiphi = rr*rr + a*a + 2.0*M*a*a/rr;
        const double g_rr = rr*rr / Delta;

        const double g_uu = g_tt*(tdot*tdot) + 2.0*g_tphi*(tdot*phidot) + g_phiphi*(phidot*phidot) + g_rr*(traj.pr[i]*traj.pr[i]);
        traj.norm_u_theory[i] = g_uu + 1.0;
    }

    return traj;
}

} // namespace relorbit