// src_cpp/lib/kerr_equatorial.cpp
#include "relorbit/models/kerr_equatorial.hpp"
#include "relorbit/solvers/hermite.hpp"

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

static inline double r_plus_kerr(double M, double a) {
    return M + std::sqrt(std::max(0.0, M * M - a * a));
}

static inline void rhs_kerr_eq(
    double M, double a, double E, double L,
    double r, double /*phi*/, double /*tcoord*/, double /*vcoord*/, double pr,
    double& dr, double& dphi, double& dt, double& dv, double& dpr
) {
    dr = pr;
    const double r2 = r * r, r3 = r2 * r, r4 = r3 * r, a2 = a * a, E2 = E * E;
    const double Delta = r2 - 2.0 * M * r + a2;
    constexpr double Delta_floor = 1e-300;
    const double D_safe = (Delta >= Delta_floor) ? Delta : Delta_floor;

    dphi = (2.0 * M * a * E / r + (1.0 - 2.0 * M / r) * L) / D_safe;
    dt   = ((r2 + a2 + 2.0 * M * a2 / r) * E - 2.0 * M * a * L / r) / D_safe;
    dv   = dt + ((r2 + a2) / D_safe) * pr;

    const double K = L - a * E;
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

    const double h_step = cfg.dt;
    if (!(h_step > 0.0) || !is_finite(h_step) || !(tauf >= tau0)) {
        traj.status = OrbitStatus::ERROR; traj.message = "invalid parameters"; return traj;
    }

    const double r_plus = r_plus_kerr(M, a);
    const double r_cap = capture_r * M;
    const double r_hor = r_plus * (1.0 + capture_eps);

    int n_steps = cfg.n_steps;
    if (n_steps <= 0) n_steps = std::max(1, static_cast<int>(std::ceil((tauf - tau0) / h_step)));

    int record_every = cfg.record_every > 0 ? cfg.record_every : 1;
    size_t res_size = static_cast<size_t>(n_steps / record_every) + 2;

    traj.tau.reserve(res_size); traj.r.reserve(res_size); traj.phi.reserve(res_size);
    traj.tcoord.reserve(res_size); traj.vcoord.reserve(res_size); traj.pr.reserve(res_size);
    traj.epsilon.reserve(res_size); traj.E_series.reserve(res_size); traj.L_series.reserve(res_size);

    double tau = tau0, r = r0, phi = phi0, tcoord = 0.0, vcoord = 0.0, pr = pr0;

    auto append_sample = [&](double tau_s, double r_s, double phi_s, double t_s, double v_s, double pr_s) {
        traj.tau.push_back(tau_s); traj.r.push_back(r_s); traj.phi.push_back(phi_s);
        traj.tcoord.push_back(t_s); traj.vcoord.push_back(v_s); traj.pr.push_back(pr_s);
        const double K = L - a * E;
        const double theory_pr2 = (E*E - 1.0) + 2.0*M/r_s - (L*L + a*a*(1.0 - E*E))/(r_s*r_s) + 2.0*M*K*K/(r_s*r_s*r_s);
        traj.epsilon.push_back(pr_s * pr_s - theory_pr2);
        traj.E_series.push_back(E); traj.L_series.push_back(L);
    };

    if (!(r > r_hor)) { traj.status = OrbitStatus::ERROR; traj.message = "invalid initial state"; return traj; }
    append_sample(tau, r, phi, tcoord, vcoord, pr);
    bool rcap_logged = false;

    for (int step = 0; step < n_steps; ++step) {
        double h = h_step;
        if (tau + h > tauf) h = (tauf - tau);
        if (!(h > 0.0)) break;

        const double tau_prev = tau, r_prev = r, phi_prev = phi, t_prev = tcoord, v_prev = vcoord, pr_prev = pr;

        double k1_r, k1_phi, k1_t, k1_v, k1_pr;
        rhs_kerr_eq(M, a, E, L, r, phi, tcoord, vcoord, pr, k1_r, k1_phi, k1_t, k1_v, k1_pr);

        double k2_r, k2_phi, k2_t, k2_v, k2_pr; rhs_kerr_eq(M, a, E, L, r + 0.5*h*k1_r, phi + 0.5*h*k1_phi, tcoord + 0.5*h*k1_t, vcoord + 0.5*h*k1_v, pr + 0.5*h*k1_pr, k2_r, k2_phi, k2_t, k2_v, k2_pr);
        double k3_r, k3_phi, k3_t, k3_v, k3_pr; rhs_kerr_eq(M, a, E, L, r + 0.5*h*k2_r, phi + 0.5*h*k2_phi, tcoord + 0.5*h*k2_t, vcoord + 0.5*h*k2_v, pr + 0.5*h*k2_pr, k3_r, k3_phi, k3_t, k3_v, k3_pr);
        double k4_r, k4_phi, k4_t, k4_v, k4_pr; rhs_kerr_eq(M, a, E, L, r + h*k3_r, phi + h*k3_phi, tcoord + h*k3_t, vcoord + h*k3_v, pr + h*k3_pr, k4_r, k4_phi, k4_t, k4_v, k4_pr);

        const double r_next  = r  + (h/6.0)*(k1_r  + 2.0*k2_r  + 2.0*k3_r  + k4_r);
        const double phi_next= phi+ (h/6.0)*(k1_phi+ 2.0*k2_phi+ 2.0*k3_phi+ k4_phi);
        const double t_next  = tcoord+(h/6.0)*(k1_t  + 2.0*k2_t  + 2.0*k3_t  + k4_t);
        const double v_next  = vcoord+(h/6.0)*(k1_v  + 2.0*k2_v  + 2.0*k3_v  + k4_v);
        const double pr_next = pr + (h/6.0)*(k1_pr + 2.0*k2_pr + 2.0*k3_pr + k4_pr);
        const double tau_next= tau + h;

        if (!is_finite(r_next) || !is_finite(phi_next) || !is_finite(t_next) || !is_finite(pr_next)) {
            traj.status = OrbitStatus::ERROR; traj.message = "non-finite state encountered"; break;
        }

        // --- Detecção de Eventos com Hermite ---
        bool crossed_pr = (pr_prev != 0.0) && ((pr_prev < 0.0 && pr_next >= 0.0) || (pr_prev > 0.0 && pr_next <= 0.0));
        bool crossed_rcap = !rcap_logged && (r_prev > r_cap && r_next <= r_cap);
        bool crossed_hor = (r_prev > r_hor && r_next <= r_hor);

        if (crossed_pr || crossed_rcap || crossed_hor) {
            double dr_nx, dphi_nx, dt_nx, dv_nx, dpr_nx;
            rhs_kerr_eq(M, a, E, L, r_next, phi_next, t_next, v_next, pr_next, dr_nx, dphi_nx, dt_nx, dv_nx, dpr_nx);

            if (crossed_pr) {
                double alpha = hermite_root(pr_prev, pr_next, k1_pr, dpr_nx, h, 0.0);
                push_event(traj, (pr_prev < 0.0) ? "periapse" : "apoapse", tau_prev + alpha*h, 
                           hermite_eval(t_prev, t_next, k1_t, dt_nx, h, alpha), hermite_eval(v_prev, v_next, k1_v, dv_nx, h, alpha),
                           hermite_eval(r_prev, r_next, k1_r, dr_nx, h, alpha), hermite_eval(phi_prev, phi_next, k1_phi, dphi_nx, h, alpha), 0.0);
            }
            if (crossed_rcap) {
                double alpha = hermite_root(r_prev, r_next, k1_r, dr_nx, h, r_cap);
                push_event(traj, "r_cap", tau_prev + alpha*h, hermite_eval(t_prev, t_next, k1_t, dt_nx, h, alpha), hermite_eval(v_prev, v_next, k1_v, dv_nx, h, alpha),
                           r_cap, hermite_eval(phi_prev, phi_next, k1_phi, dphi_nx, h, alpha), hermite_eval(pr_prev, pr_next, k1_pr, dpr_nx, h, alpha));
                rcap_logged = true;
            }
            if (crossed_hor) {
                double alpha = hermite_root(r_prev, r_next, k1_r, dr_nx, h, r_hor);
                double t_ev = hermite_eval(t_prev, t_next, k1_t, dt_nx, h, alpha), v_ev = hermite_eval(v_prev, v_next, k1_v, dv_nx, h, alpha),
                       phi_ev = hermite_eval(phi_prev, phi_next, k1_phi, dphi_nx, h, alpha), pr_ev = hermite_eval(pr_prev, pr_next, k1_pr, dpr_nx, h, alpha);
                push_event(traj, "horizon", tau_prev + alpha*h, t_ev, v_ev, r_hor, phi_ev, pr_ev);
                append_sample(tau_prev + alpha*h, r_hor, phi_ev, t_ev, v_ev, pr_ev);
                traj.status = OrbitStatus::CAPTURE; traj.message = "horizon crossed"; break;
            }
        }

        tau = tau_next; r = r_next; phi = phi_next; tcoord = t_next; vcoord = v_next; pr = pr_next;
        if ((step + 1) % record_every == 0 || (step == n_steps - 1) || (tau >= tauf)) {
            append_sample(tau, r, phi, tcoord, vcoord, pr);
        }
        if (tau >= tauf) break;
    }

    // Pós-processamento Teórico
    const size_t N = traj.tau.size();
    traj.ut_fd.assign(N, std::numeric_limits<double>::quiet_NaN()); traj.vt_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.ur_fd.assign(N, std::numeric_limits<double>::quiet_NaN()); traj.uphi_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.norm_u.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.ut_theory.assign(N, std::numeric_limits<double>::quiet_NaN()); traj.vt_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.ur_theory.assign(N, std::numeric_limits<double>::quiet_NaN()); traj.uphi_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.norm_u_theory.assign(N, std::numeric_limits<double>::quiet_NaN());

    for (size_t i = 0; i < N; ++i) {
        const double rr = traj.r[i];
        const double Delta = rr * rr - 2.0 * M * rr + a * a;
        if (!is_finite(rr) || !(rr > 0.0) || Delta < 1e-12) continue;
        double tdot, phidot, dr, dv, dpr;
        rhs_kerr_eq(M, a, E, L, rr, 0.0, 0.0, 0.0, traj.pr[i], dr, phidot, tdot, dv, dpr);
        traj.ut_theory[i] = tdot; traj.vt_theory[i] = dv; traj.ur_theory[i] = traj.pr[i]; traj.uphi_theory[i] = phidot;
        const double g_tt = -(1.0 - 2.0*M/rr), g_tphi = -2.0*M*a/rr, g_phiphi = rr*rr + a*a + 2.0*M*a*a/rr, g_rr = rr*rr / Delta;
        traj.norm_u_theory[i] = g_tt*(tdot*tdot) + 2.0*g_tphi*(tdot*phidot) + g_phiphi*(phidot*phidot) + g_rr*(traj.pr[i]*traj.pr[i]) + 1.0;
    }
    return traj;
}

} // namespace relorbit