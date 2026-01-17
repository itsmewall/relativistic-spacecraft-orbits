#include "relorbit/models/schwarzschild_equatorial.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace relorbit {

struct SchwState {
    double tcoord; // t
    double r;      // r
    double phi;    // phi
    double pr;     // dr/dtau
};

static inline SchwState f_schw(double M, double E, double L, const SchwState& s) {
    const double A = 1.0 - 2.0 * M / s.r;   // 1 - 2M/r

    SchwState ds;
    ds.tcoord = E / A;                      // dt/dtau
    ds.r      = s.pr;                       // dr/dtau
    ds.phi    = L / (s.r * s.r);            // dphi/dtau
    ds.pr     = -0.5 * dVeff_dr_schw(M, s.r, L); // dpr/dtau
    return ds;
}

static inline double fd_first(const std::vector<double>& x, const std::vector<double>& t, size_t i) {
    // derivada dx/dt com malha uniforme ou não-uniforme (usa diferenças)
    const size_t n = x.size();
    if (n < 2) return 0.0;

    if (i == 0) {
        const double dt = t[1] - t[0];
        return (dt != 0.0) ? (x[1] - x[0]) / dt : 0.0;
    }
    if (i == n - 1) {
        const double dt = t[n-1] - t[n-2];
        return (dt != 0.0) ? (x[n-1] - x[n-2]) / dt : 0.0;
    }

    const double dt = t[i+1] - t[i-1];
    return (dt != 0.0) ? (x[i+1] - x[i-1]) / dt : 0.0;
}

TrajectorySchwarzschildEq simulate_schwarzschild_equatorial_rk4(
    double M,
    double E,
    double L,
    double r0,
    double phi0,
    double tau0,
    double tauf,
    const SolverCfg& cfg,
    double capture_r,
    double capture_eps
) {
    if (M <= 0) throw std::runtime_error("M must be > 0");
    if (r0 <= 2.0 * M) throw std::runtime_error("r0 must be > 2M");
    if (tauf <= tau0) throw std::runtime_error("tauf must be > tau0");

    int n = cfg.n_steps;
    if (n <= 0) {
        if (cfg.dt <= 0) throw std::runtime_error("cfg.dt must be > 0 if n_steps==0");
        n = static_cast<int>(std::ceil((tauf - tau0) / cfg.dt));
        if (n < 1) n = 1;
    }
    const double dt = (tauf - tau0) / static_cast<double>(n);

    TrajectorySchwarzschildEq out;
    out.M = M; out.E = E; out.L = L; out.r0 = r0; out.phi0 = phi0;

    out.tau.reserve(static_cast<size_t>(n) + 1);
    out.r.reserve(static_cast<size_t>(n) + 1);
    out.phi.reserve(static_cast<size_t>(n) + 1);
    out.tcoord.reserve(static_cast<size_t>(n) + 1);
    out.pr.reserve(static_cast<size_t>(n) + 1);
    out.epsilon.reserve(static_cast<size_t>(n) + 1);
    out.E_series.reserve(static_cast<size_t>(n) + 1);
    out.L_series.reserve(static_cast<size_t>(n) + 1);

    SchwState s;
    s.tcoord = 0.0;
    s.r = r0;
    s.phi = phi0;

    // pr0: se você quiser inbound/outbound, isso vira parâmetro (vamos fazer depois).
    // Por ora, default = turning point
    s.pr = 0.0;

    double tau = tau0;

    out.status = OrbitStatus::BOUND;

    auto push = [&]() {
        out.tau.push_back(tau);
        out.r.push_back(s.r);
        out.phi.push_back(s.phi);
        out.tcoord.push_back(s.tcoord);
        out.pr.push_back(s.pr);

        const double eps = (s.pr * s.pr) + Veff_schw(M, s.r, L) - (E * E);
        out.epsilon.push_back(eps);

        out.E_series.push_back(E);
        out.L_series.push_back(L);
    };

    push();

    const double r_capture = capture_r * M;

    for (int i = 0; i < n; ++i) {
        const auto k1 = f_schw(M, E, L, s);

        SchwState s2 { s.tcoord + 0.5*dt*k1.tcoord, s.r + 0.5*dt*k1.r, s.phi + 0.5*dt*k1.phi, s.pr + 0.5*dt*k1.pr };
        const auto k2 = f_schw(M, E, L, s2);

        SchwState s3 { s.tcoord + 0.5*dt*k2.tcoord, s.r + 0.5*dt*k2.r, s.phi + 0.5*dt*k2.phi, s.pr + 0.5*dt*k2.pr };
        const auto k3 = f_schw(M, E, L, s3);

        SchwState s4 { s.tcoord + dt*k3.tcoord, s.r + dt*k3.r, s.phi + dt*k3.phi, s.pr + dt*k3.pr };
        const auto k4 = f_schw(M, E, L, s4);

        s.tcoord += (dt/6.0) * (k1.tcoord + 2.0*k2.tcoord + 2.0*k3.tcoord + k4.tcoord);
        s.r      += (dt/6.0) * (k1.r      + 2.0*k2.r      + 2.0*k3.r      + k4.r);
        s.phi    += (dt/6.0) * (k1.phi    + 2.0*k2.phi    + 2.0*k3.phi    + k4.phi);
        s.pr     += (dt/6.0) * (k1.pr     + 2.0*k2.pr     + 2.0*k3.pr     + k4.pr);

        tau += dt;
        push();

        if (!std::isfinite(s.r) || !std::isfinite(s.phi) || !std::isfinite(s.pr) || !std::isfinite(s.tcoord)) {
            out.status = OrbitStatus::ERROR;
            out.message = "non-finite state encountered";
            break;
        }

        if (s.r <= r_capture + capture_eps) {
            out.status = OrbitStatus::CAPTURE;
            out.message = "capture: r crossed capture radius";
            break;
        }

        if (s.r > 1e6 * M) {
            out.status = OrbitStatus::UNBOUND;
            out.message = "unbound: r grew too large";
            break;
        }
    }

    // ============================
    // Pós-processamento FD: u^μ e norm_u
    // ============================
    const size_t N = out.tau.size();
    out.ut_fd.resize(N);
    out.ur_fd.resize(N);
    out.uphi_fd.resize(N);
    out.norm_u.resize(N);

    for (size_t i = 0; i < N; ++i) {
        const double ut = fd_first(out.tcoord, out.tau, i);
        const double ur = fd_first(out.r, out.tau, i);
        const double up = fd_first(out.phi, out.tau, i);

        out.ut_fd[i] = ut;
        out.ur_fd[i] = ur;
        out.uphi_fd[i] = up;

        const double r = out.r[i];
        const double A = 1.0 - 2.0 * M / r;

        // métrica Schwarzschild (equatorial), assinatura (-,+,+,+):
        // ds^2 = -A dt^2 + A^{-1} dr^2 + r^2 dphi^2
        const double g_tt = -A;
        const double g_rr = 1.0 / A;
        const double g_pp = r * r;

        const double norm = g_tt*ut*ut + g_rr*ur*ur + g_pp*up*up + 1.0;
        out.norm_u[i] = norm;
    }

    return out;
}

} // namespace relorbit
