// src_cpp/lib/schwarzschild_equatorial.cpp
#include "relorbit/models/schwarzschild_equatorial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace relorbit {

static inline bool is_finite(double x) { return std::isfinite(x); }

static inline void push_event(
    TrajectorySchwarzschildEq& traj,
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

static inline double safe_A(double M, double r) {
    return 1.0 - 2.0 * M / r;
}

// Interp linear no passo: x(t)=x0 + a*(x1-x0)
static inline double lerp(double x0, double x1, double a) {
    return x0 + a * (x1 - x0);
}

// crossing de r: r_prev > r_thr && r_next <= r_thr (com interp linear)
static inline bool crossing_r(
    double r_prev, double r_next, double r_thr,
    double& alpha_out
) {
    if (!(r_prev > r_thr && r_next <= r_thr)) return false;
    const double denom = (r_prev - r_next);
    double a = 0.0;
    if (std::abs(denom) > 0.0) a = (r_prev - r_thr) / denom;
    a = std::clamp(a, 0.0, 1.0);
    alpha_out = a;
    return true;
}

// Forma estável para dv/dtau (ingoing EF):
//   dv/dtau = (E + pr)/A  -> sofre cancelamento quando pr ~ -E e A -> 0.
// Usamos identidade (exata em geodésica ideal):
//   (E + pr) = (A*B)/(E - pr), com B = (1 + L^2/r^2)
// então:
//   dv/dtau = (E + pr)/A = B/(E - pr), estável no horizonte.
static inline double stable_dv_dtau(double M, double E, double L, double r, double pr) {
    (void)M;
    const double rr = std::max(r, 1e-300);
    const double B  = 1.0 + (L * L) / (rr * rr);

    double denom = (E - pr);
    constexpr double denom_floor = 1e-14;
    if (!is_finite(denom) || std::abs(denom) < denom_floor) {
        denom = (denom >= 0.0 ? denom_floor : -denom_floor);
    }
    return B / denom;
}

// ODE RHS em tau: state = [r, phi, tcoord, vcoord, pr]
//
// Importante: (r,phi,pr) carregam a dinâmica. (tcoord,vcoord) são "outputs".
// Near-horizon A->0 não pode matar a simulação por overflow em dt/dtau.
// Porém: NÃO vamos "floor" agressivo em A aqui, porque isso distorce t.
// O que a gente faz:
//  - dt/dtau = E/A (com um floor microscópico só pra evitar /0 acidental)
//  - dv/dtau: usa forma estável B/(E-pr) quando A pequeno ou quando há cancelamento.
static inline void rhs_schw_eq(
    double M, double E, double L,
    double r, double /*phi*/, double /*tcoord*/, double /*vcoord*/, double pr,
    double& dr, double& dphi, double& dt, double& dv, double& dpr
) {
    dr = pr;
    dphi = L / (r * r);

    const double A = safe_A(M, r);

    // Segurança mínima contra divisão por zero (não muda nada nos seus casos, pois r > r_hor => A>0)
    constexpr double A_floor = 1e-300;
    const double A_safe = (A >= A_floor) ? A : A_floor;

    dt = E / A_safe;

    // dv/dtau: forma padrão longe do horizonte; forma estável perto do horizonte/cancelamento
    constexpr double A_switch = 1e-8;        // abaixo disso, preferimos estável
    constexpr double cancel_switch = 1e-10;  // |E+pr| pequeno => cancelamento
    if (A > A_switch && std::abs(E + pr) > cancel_switch) {
        dv = (E + pr) / A_safe;
    } else {
        dv = stable_dv_dtau(M, E, L, r, pr);
    }

    dpr = -0.5 * dVeff_dr_schw(M, r, L);
}

TrajectorySchwarzschildEq simulate_schwarzschild_equatorial_rk4(
    double M,
    double E,
    double L,
    double r0,
    double phi0,
    double pr0,
    double tau0,
    double tauf,
    const SolverCfg& cfg,
    double capture_r,
    double capture_eps
) {
    TrajectorySchwarzschildEq traj;
    traj.M = M;
    traj.E = E;
    traj.L = L;
    traj.r0 = r0;
    traj.phi0 = phi0;

    // default saudável
    traj.status = OrbitStatus::BOUND;
    traj.message.clear();

    const double dt0 = cfg.dt;
    if (!(dt0 > 0.0) || !is_finite(dt0)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid dt (must be > 0)";
        return traj;
    }
    if (!(tauf >= tau0) || !is_finite(tau0) || !is_finite(tauf)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid tau span";
        return traj;
    }
    if (!(M > 0.0) || !is_finite(M)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid M (must be > 0)";
        return traj;
    }
    if (!is_finite(E) || !is_finite(L)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid (E,L) (non-finite)";
        return traj;
    }
    if (!(capture_r > 0.0) || !is_finite(capture_r) || !(capture_eps >= 0.0) || !is_finite(capture_eps)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid capture params";
        return traj;
    }

    // r_cap: marcador operacional (não encerra)
    const double r_cap = capture_r * M;

    // r_hor: evento físico, com margem eps
    const double r_hor = (2.0 * M) * (1.0 + capture_eps);

    // passos
    int n_steps = cfg.n_steps;
    if (n_steps <= 0) {
        const double span = (tauf - tau0);
        n_steps = static_cast<int>(std::ceil(span / dt0));
        if (n_steps < 1) n_steps = 1;
    }

    // reserva
    traj.tau.reserve(static_cast<size_t>(n_steps) + 1);
    traj.r.reserve(static_cast<size_t>(n_steps) + 1);
    traj.phi.reserve(static_cast<size_t>(n_steps) + 1);
    traj.tcoord.reserve(static_cast<size_t>(n_steps) + 1);
    traj.vcoord.reserve(static_cast<size_t>(n_steps) + 1);
    traj.pr.reserve(static_cast<size_t>(n_steps) + 1);
    traj.epsilon.reserve(static_cast<size_t>(n_steps) + 1);
    traj.E_series.reserve(static_cast<size_t>(n_steps) + 1);
    traj.L_series.reserve(static_cast<size_t>(n_steps) + 1);

    // estado inicial
    double tau = tau0;
    double r = r0;
    double phi = phi0;
    double tcoord = 0.0;
    double vcoord = 0.0;
    double pr = pr0;

    auto append_sample = [&](double tau_s, double r_s, double phi_s, double t_s, double v_s, double pr_s) {
        traj.tau.push_back(tau_s);
        traj.r.push_back(r_s);
        traj.phi.push_back(phi_s);
        traj.tcoord.push_back(t_s);
        traj.vcoord.push_back(v_s);
        traj.pr.push_back(pr_s);

        const double eps = pr_s * pr_s + Veff_schw(M, r_s, L) - E * E;
        traj.epsilon.push_back(eps);

        traj.E_series.push_back(E);
        traj.L_series.push_back(L);
    };

    // validações iniciais
    if (!is_finite(r) || !is_finite(phi) || !is_finite(pr)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid initial state (non-finite)";
        return traj;
    }
    if (!(r > r_hor)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid initial state (r must be > 2M*(1+eps))";
        return traj;
    }

    append_sample(tau, r, phi, tcoord, vcoord, pr);

    bool rcap_logged = false;

    // integração
    for (int step = 0; step < n_steps; ++step) {
        double h = dt0;
        if (tau + h > tauf) h = (tauf - tau);
        if (!(h > 0.0)) break;

        // prev
        const double tau_prev = tau;
        const double r_prev = r;
        const double phi_prev = phi;
        const double t_prev = tcoord;
        const double v_prev = vcoord;
        const double pr_prev = pr;

        // RK4
        double k1_r, k1_phi, k1_t, k1_v, k1_pr;
        rhs_schw_eq(M, E, L, r, phi, tcoord, vcoord, pr, k1_r, k1_phi, k1_t, k1_v, k1_pr);

        const double r2  = r      + 0.5 * h * k1_r;
        const double p2  = phi    + 0.5 * h * k1_phi;
        const double t2  = tcoord + 0.5 * h * k1_t;
        const double v2  = vcoord + 0.5 * h * k1_v;
        const double pr2 = pr     + 0.5 * h * k1_pr;

        double k2_r, k2_phi, k2_t, k2_v, k2_pr;
        rhs_schw_eq(M, E, L, r2, p2, t2, v2, pr2, k2_r, k2_phi, k2_t, k2_v, k2_pr);

        const double r3  = r      + 0.5 * h * k2_r;
        const double p3  = phi    + 0.5 * h * k2_phi;
        const double t3  = tcoord + 0.5 * h * k2_t;
        const double v3  = vcoord + 0.5 * h * k2_v;
        const double pr3 = pr     + 0.5 * h * k2_pr;

        double k3_r, k3_phi, k3_t, k3_v, k3_pr;
        rhs_schw_eq(M, E, L, r3, p3, t3, v3, pr3, k3_r, k3_phi, k3_t, k3_v, k3_pr);

        const double r4  = r      + h * k3_r;
        const double p4  = phi    + h * k3_phi;
        const double t4  = tcoord + h * k3_t;
        const double v4  = vcoord + h * k3_v;
        const double pr4 = pr     + h * k3_pr;

        double k4_r, k4_phi, k4_t, k4_v, k4_pr;
        rhs_schw_eq(M, E, L, r4, p4, t4, v4, pr4, k4_r, k4_phi, k4_t, k4_v, k4_pr);

        const double r_next   = r      + (h / 6.0) * (k1_r   + 2.0 * k2_r   + 2.0 * k3_r   + k4_r);
        const double phi_next = phi    + (h / 6.0) * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi);
        const double t_next   = tcoord + (h / 6.0) * (k1_t   + 2.0 * k2_t   + 2.0 * k3_t   + k4_t);
        const double v_next   = vcoord + (h / 6.0) * (k1_v   + 2.0 * k2_v   + 2.0 * k3_v   + k4_v);
        const double pr_next  = pr     + (h / 6.0) * (k1_pr  + 2.0 * k2_pr  + 2.0 * k3_pr  + k4_pr);
        const double tau_next = tau + h;

        if (!is_finite(r_next) || !is_finite(phi_next) || !is_finite(t_next) || !is_finite(v_next) || !is_finite(pr_next)) {
            traj.status = OrbitStatus::ERROR;
            traj.message = "non-finite state encountered";
            break;
        }

        // EVENTO: turning (periapse/apoapse) via pr=0
        if (pr_prev != 0.0) {
            const bool crossed = (pr_prev < 0.0 && pr_next >= 0.0) || (pr_prev > 0.0 && pr_next <= 0.0);
            if (crossed) {
                const double denom = (pr_prev - pr_next);
                double alpha = 0.0;
                if (std::abs(denom) > 0.0) alpha = pr_prev / denom;
                alpha = std::clamp(alpha, 0.0, 1.0);

                const double tau_ev = tau_prev + alpha * h;
                const double r_ev   = lerp(r_prev,   r_next,   alpha);
                const double phi_ev = lerp(phi_prev, phi_next, alpha);
                const double t_ev   = lerp(t_prev,   t_next,   alpha);
                const double v_ev   = lerp(v_prev,   v_next,   alpha);

                const std::string kind = (pr_prev < 0.0) ? "periapse" : "apoapse";
                push_event(traj, kind, tau_ev, t_ev, v_ev, r_ev, phi_ev, 0.0);
            }
        }

        // EVENTO: crossing de r_cap (marcador operacional; NÃO encerra)
        if (!rcap_logged) {
            double alpha = 0.0;
            if (crossing_r(r_prev, r_next, r_cap, alpha)) {
                const double tau_ev = tau_prev + alpha * h;
                const double phi_ev = lerp(phi_prev, phi_next, alpha);
                const double t_ev   = lerp(t_prev,   t_next,   alpha);
                const double v_ev   = lerp(v_prev,   v_next,   alpha);
                const double pr_ev  = lerp(pr_prev,  pr_next,  alpha);
                push_event(traj, "r_cap", tau_ev, t_ev, v_ev, r_cap, phi_ev, pr_ev);
                rcap_logged = true;
            }
        }

        // EVENTO: horizon crossing (CAPTURE físico)
        {
            double alpha = 0.0;
            if (crossing_r(r_prev, r_next, r_hor, alpha)) {
                const double tau_ev = tau_prev + alpha * h;
                const double phi_ev = lerp(phi_prev, phi_next, alpha);
                const double t_ev   = lerp(t_prev,   t_next,   alpha);
                const double v_ev   = lerp(v_prev,   v_next,   alpha);
                const double pr_ev  = lerp(pr_prev,  pr_next,  alpha);

                push_event(traj, "horizon", tau_ev, t_ev, v_ev, r_hor, phi_ev, pr_ev);

                tau = tau_ev;
                r = r_hor;
                phi = phi_ev;
                tcoord = t_ev;
                vcoord = v_ev;
                pr = pr_ev;

                append_sample(tau, r, phi, tcoord, vcoord, pr);

                traj.status = OrbitStatus::CAPTURE;
                traj.message = "horizon crossed (r <= 2M*(1+eps))";
                break;
            }
        }

        // aceita passo normal
        tau = tau_next;
        r = r_next;
        phi = phi_next;
        tcoord = t_next;
        vcoord = v_next;
        pr = pr_next;

        append_sample(tau, r, phi, tcoord, vcoord, pr);

        if (tau >= tauf) break;
    }

    // =========================================
    // Pós-processamento: FD + norm_u (diagnóstico)
    // =========================================
    const size_t N = traj.tau.size();
    traj.ut_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.vt_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.ur_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.uphi_fd.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.norm_u.assign(N, std::numeric_limits<double>::quiet_NaN());

    constexpr double dtau_floor = 1e-300;

    // Aqui é o pulo do gato: FD perto do horizonte vira ruído (principalmente no g_rr = 1/A).
    // Como você encerra em r_hor = 2M(1+eps), A(r_hor) ~ eps/(1+eps).
    // Então use um corte bem acima do "A_min_theory" para não poluir com números absurdos.
    constexpr double A_min_norm = 1e-5; // <<--- ajustado (evita norm_u_fd insano em plunge)

    if (N >= 2) {
        for (size_t i = 0; i < N; ++i) {
            double dtau;
            double dt, dv, dr, dphi;

            if (i == 0) {
                dtau = traj.tau[1] - traj.tau[0];
                dt   = traj.tcoord[1] - traj.tcoord[0];
                dv   = traj.vcoord[1] - traj.vcoord[0];
                dr   = traj.r[1] - traj.r[0];
                dphi = traj.phi[1] - traj.phi[0];
            } else if (i == N - 1) {
                dtau = traj.tau[N - 1] - traj.tau[N - 2];
                dt   = traj.tcoord[N - 1] - traj.tcoord[N - 2];
                dv   = traj.vcoord[N - 1] - traj.vcoord[N - 2];
                dr   = traj.r[N - 1] - traj.r[N - 2];
                dphi = traj.phi[N - 1] - traj.phi[N - 2];
            } else {
                dtau = traj.tau[i + 1] - traj.tau[i - 1];
                dt   = traj.tcoord[i + 1] - traj.tcoord[i - 1];
                dv   = traj.vcoord[i + 1] - traj.vcoord[i - 1];
                dr   = traj.r[i + 1] - traj.r[i - 1];
                dphi = traj.phi[i + 1] - traj.phi[i - 1];
            }

            if (!is_finite(dtau) || std::abs(dtau) < dtau_floor) dtau = dtau_floor;

            traj.ut_fd[i]   = dt   / dtau;
            traj.vt_fd[i]   = dv   / dtau;
            traj.ur_fd[i]   = dr   / dtau;
            traj.uphi_fd[i] = dphi / dtau;

            const double rr = traj.r[i];
            const double A  = safe_A(M, rr);

            if (!is_finite(rr) || !(rr > 0.0) || !is_finite(A) || A < A_min_norm) {
                traj.norm_u[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            const double ut = traj.ut_fd[i];
            const double ur = traj.ur_fd[i];
            const double up = traj.uphi_fd[i];

            // g_tt=-A, g_rr=1/A, g_phiphi=r^2 (assinatura - + + +)
            const double g_uu = (-A) * (ut * ut) + (1.0 / A) * (ur * ur) + (rr * rr) * (up * up);
            traj.norm_u[i] = g_uu + 1.0;
        }
    }

    // =========================================
    // Séries por construção (teóricas) — MÉTRICA LIMPA
    // =========================================
    traj.ut_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.vt_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.ur_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.uphi_theory.assign(N, std::numeric_limits<double>::quiet_NaN());
    traj.norm_u_theory.assign(N, std::numeric_limits<double>::quiet_NaN());

    constexpr double A_min_theory = 1e-12;
    constexpr double A_switch_th  = 1e-8;
    constexpr double cancel_switch_th = 1e-10;

    for (size_t i = 0; i < N; ++i) {
        const double rr = traj.r[i];
        const double A  = safe_A(M, rr);

        if (!is_finite(rr) || !(rr > 0.0) || !is_finite(A) || A < A_min_theory) {
            continue; // mantém NaN
        }

        const double ut = E / A;
        const double ur = traj.pr[i];           // por definição: dr/dtau = pr
        const double up = L / (rr * rr);        // por definição: dphi/dtau = L/r^2

        // vt teórico: mesma lógica estável do RHS
        double vt;
        if (A > A_switch_th && std::abs(E + ur) > cancel_switch_th) {
            vt = (E + ur) / A;
        } else {
            vt = stable_dv_dtau(M, E, L, rr, ur);
        }

        traj.ut_theory[i] = ut;
        traj.vt_theory[i] = vt;
        traj.ur_theory[i] = ur;
        traj.uphi_theory[i] = up;

        const double g_uu = (-A) * (ut * ut) + (1.0 / A) * (ur * ur) + (rr * rr) * (up * up);
        traj.norm_u_theory[i] = g_uu + 1.0;
    }

    return traj;
}

} // namespace relorbit
