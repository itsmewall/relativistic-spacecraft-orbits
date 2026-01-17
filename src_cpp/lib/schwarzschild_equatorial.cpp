// src_cpp/lib/schwarzschild_equatorial.cpp
#include "relorbit/models/schwarzschild_equatorial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace relorbit {

static inline bool is_finite(double x) {
    return std::isfinite(x);
}

static inline void push_event(
    TrajectorySchwarzschildEq& traj,
    const std::string& kind,
    double tau, double tcoord, double r, double phi, double pr
) {
    traj.event_kind.push_back(kind);
    traj.event_tau.push_back(tau);
    traj.event_tcoord.push_back(tcoord);
    traj.event_r.push_back(r);
    traj.event_phi.push_back(phi);
    traj.event_pr.push_back(pr);
}

static inline double safe_A(double M, double r) {
    return 1.0 - 2.0 * M / r;
}

// ODE RHS em tau: state = [r, phi, tcoord, pr]
static inline void rhs_schw_eq(double M, double E, double L,
                               double r, double phi, double tcoord, double pr,
                               double& dr, double& dphi, double& dt, double& dpr) {
    (void)phi;
    (void)tcoord;

    // dr/dtau = pr
    dr = pr;

    // dphi/dtau = L/r^2
    dphi = L / (r * r);

    // dt/dtau = E / (1 - 2M/r)
    const double A = safe_A(M, r);
    dt = E / A;

    // dpr/dtau = -(1/2) dVeff/dr
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

    traj.status = OrbitStatus::BOUND;  // <<< CRÍTICO: default de sucesso (se nada der errado)
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

    const double r_cap = capture_r * M; // regra: r <= capture_r * M

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
    traj.pr.reserve(static_cast<size_t>(n_steps) + 1);
    traj.epsilon.reserve(static_cast<size_t>(n_steps) + 1);
    traj.E_series.reserve(static_cast<size_t>(n_steps) + 1);
    traj.L_series.reserve(static_cast<size_t>(n_steps) + 1);

    // estado inicial
    double tau = tau0;
    double r = r0;
    double phi = phi0;
    double tcoord = 0.0; // pode começar em 0, pois só diferenças importam
    double pr = pr0;

    auto append_sample = [&](double tau_s, double r_s, double phi_s, double t_s, double pr_s) {
        traj.tau.push_back(tau_s);
        traj.r.push_back(r_s);
        traj.phi.push_back(phi_s);
        traj.tcoord.push_back(t_s);
        traj.pr.push_back(pr_s);

        const double eps = pr_s * pr_s + Veff_schw(M, r_s, L) - E * E;
        traj.epsilon.push_back(eps);

        traj.E_series.push_back(E);
        traj.L_series.push_back(L);
    };

    // validações básicas iniciais
    if (!is_finite(r) || !(r > 2.0 * M) || !is_finite(phi) || !is_finite(pr)) {
        traj.status = OrbitStatus::ERROR;
        traj.message = "invalid initial state (r must be > 2M and finite)";
        return traj;
    }

    append_sample(tau, r, phi, tcoord, pr);

    // integração
    for (int step = 0; step < n_steps; ++step) {
        // ajusta último passo para bater em tauf
        double h = dt0;
        if (tau + h > tauf) h = (tauf - tau);
        if (!(h > 0.0)) break;

        // guarda prev
        const double tau_prev = tau;
        const double r_prev = r;
        const double phi_prev = phi;
        const double t_prev = tcoord;
        const double pr_prev = pr;

        // RK4
        double k1_r, k1_phi, k1_t, k1_pr;
        rhs_schw_eq(M, E, L, r, phi, tcoord, pr, k1_r, k1_phi, k1_t, k1_pr);

        double r2 = r + 0.5 * h * k1_r;
        double p2 = phi + 0.5 * h * k1_phi;
        double t2 = tcoord + 0.5 * h * k1_t;
        double pr2 = pr + 0.5 * h * k1_pr;

        double k2_r, k2_phi, k2_t, k2_pr;
        rhs_schw_eq(M, E, L, r2, p2, t2, pr2, k2_r, k2_phi, k2_t, k2_pr);

        double r3 = r + 0.5 * h * k2_r;
        double p3 = phi + 0.5 * h * k2_phi;
        double t3 = tcoord + 0.5 * h * k2_t;
        double pr3 = pr + 0.5 * h * k2_pr;

        double k3_r, k3_phi, k3_t, k3_pr;
        rhs_schw_eq(M, E, L, r3, p3, t3, pr3, k3_r, k3_phi, k3_t, k3_pr);

        double r4 = r + h * k3_r;
        double p4 = phi + h * k3_phi;
        double t4 = tcoord + h * k3_t;
        double pr4 = pr + h * k3_pr;

        double k4_r, k4_phi, k4_t, k4_pr;
        rhs_schw_eq(M, E, L, r4, p4, t4, pr4, k4_r, k4_phi, k4_t, k4_pr);

        double r_next = r + (h / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r);
        double phi_next = phi + (h / 6.0) * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi);
        double t_next = tcoord + (h / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t);
        double pr_next = pr + (h / 6.0) * (k1_pr + 2.0 * k2_pr + 2.0 * k3_pr + k4_pr);
        double tau_next = tau + h;

        // sanity numérica
        if (!is_finite(r_next) || !is_finite(phi_next) || !is_finite(t_next) || !is_finite(pr_next)) {
            traj.status = OrbitStatus::ERROR;
            traj.message = "non-finite state encountered";
            break;
        }

        // -----------------------------------------
        // EVENTO: turning point (periapse/apoapse)
        // -----------------------------------------
        // Detecta crossing de pr = 0 no passo (linear em pr)
        if (pr_prev != 0.0) {
            const bool crossed = (pr_prev < 0.0 && pr_next >= 0.0) || (pr_prev > 0.0 && pr_next <= 0.0);
            if (crossed) {
                const double denom = (pr_prev - pr_next);
                double alpha = 0.0;
                if (std::abs(denom) > 0.0) alpha = pr_prev / denom; // alpha in [0,1] típico
                alpha = std::clamp(alpha, 0.0, 1.0);

                const double tau_ev = tau_prev + alpha * h;
                const double r_ev = r_prev + alpha * (r_next - r_prev);
                const double phi_ev = phi_prev + alpha * (phi_next - phi_prev);
                const double t_ev = t_prev + alpha * (t_next - t_prev);
                const double pr_ev = 0.0;

                const std::string kind = (pr_prev < 0.0) ? "periapse" : "apoapse";
                push_event(traj, kind, tau_ev, t_ev, r_ev, phi_ev, pr_ev);
            }
        }

        // -----------------------------------------
        // EVENTO: capture crossing (r cruza r_cap)
        // -----------------------------------------
        // regra: se veio de fora e cruzou pra dentro, registra e encerra
        if ((r_prev > r_cap) && (r_next <= r_cap)) {
            const double denom = (r_prev - r_next);
            double alpha = 0.0;
            if (std::abs(denom) > 0.0) alpha = (r_prev - r_cap) / denom;
            alpha = std::clamp(alpha, 0.0, 1.0);

            const double tau_ev = tau_prev + alpha * h;
            const double phi_ev = phi_prev + alpha * (phi_next - phi_prev);
            const double t_ev = t_prev + alpha * (t_next - t_prev);
            const double pr_ev = pr_prev + alpha * (pr_next - pr_prev);

            // anexa amostra exatamente no crossing
            tau = tau_ev;
            r = r_cap;
            phi = phi_ev;
            tcoord = t_ev;
            pr = pr_ev;

            append_sample(tau, r, phi, tcoord, pr);
            push_event(traj, "capture", tau, tcoord, r, phi, pr);

            traj.status = OrbitStatus::CAPTURE;

            std::ostringstream oss;
            oss.setf(std::ios::scientific);
            oss << "capture: r crossed capture radius "
                << "(r_prev=" << r_prev
                << ", r_now=" << r_next
                << ", r_cap=" << r_cap
                << ", capture_r=" << capture_r
                << ", M=" << M << ")";
            traj.message = oss.str();

            break;
        }

        // Se chegou aqui, aceita passo normal
        tau = tau_next;
        r = r_next;
        phi = phi_next;
        tcoord = t_next;
        pr = pr_next;

        // protege contra entrar no horizonte por acidente
        if (!(r > 2.0 * M + capture_eps)) {
            traj.status = OrbitStatus::ERROR;
            traj.message = "r approached/entered horizon region (r <= 2M + eps)";
            break;
        }

        append_sample(tau, r, phi, tcoord, pr);

        // fim exato
        if (tau >= tauf) break;
    }

    // =========================================
    // Pós-processamento FD + norm_u
    // =========================================
    const size_t N = traj.tau.size();
    traj.ut_fd.assign(N, 0.0);
    traj.ur_fd.assign(N, 0.0);
    traj.uphi_fd.assign(N, 0.0);
    traj.norm_u.assign(N, 0.0);

    if (N >= 2) {
        // diferenças centradas internas, forward/backward nas bordas
        for (size_t i = 0; i < N; ++i) {
            double dtau;
            double dt, dr, dphi;

            if (i == 0) {
                dtau = traj.tau[1] - traj.tau[0];
                dt = traj.tcoord[1] - traj.tcoord[0];
                dr = traj.r[1] - traj.r[0];
                dphi = traj.phi[1] - traj.phi[0];
            } else if (i == N - 1) {
                dtau = traj.tau[N - 1] - traj.tau[N - 2];
                dt = traj.tcoord[N - 1] - traj.tcoord[N - 2];
                dr = traj.r[N - 1] - traj.r[N - 2];
                dphi = traj.phi[N - 1] - traj.phi[N - 2];
            } else {
                dtau = traj.tau[i + 1] - traj.tau[i - 1];
                dt = traj.tcoord[i + 1] - traj.tcoord[i - 1];
                dr = traj.r[i + 1] - traj.r[i - 1];
                dphi = traj.phi[i + 1] - traj.phi[i - 1];
            }

            if (std::abs(dtau) < 1e-300) dtau = 1e-300;

            traj.ut_fd[i] = dt / dtau;
            traj.ur_fd[i] = dr / dtau;
            traj.uphi_fd[i] = dphi / dtau;

            const double rr = traj.r[i];
            const double A = safe_A(M, rr);
            // g_tt = -A, g_rr = 1/A, g_phiphi = r^2
            const double ut = traj.ut_fd[i];
            const double ur = traj.ur_fd[i];
            const double up = traj.uphi_fd[i];

            const double g_uu = (-A) * (ut * ut) + (1.0 / A) * (ur * ur) + (rr * rr) * (up * up);
            traj.norm_u[i] = g_uu + 1.0; // deve ser ~0
        }
    }

    // Se terminou naturalmente e ninguém setou ERROR/CAPTURE, mantém BOUND
    // (isso impede o teu bug atual de status ERROR em caso saudável)
    if (traj.status == OrbitStatus::BOUND) {
        // ok
    }

    return traj;
}

} // namespace relorbit
