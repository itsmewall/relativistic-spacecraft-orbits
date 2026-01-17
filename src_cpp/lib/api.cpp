#include "relorbit/models/schwarzschild_equatorial.hpp"
#include <stdexcept>
#include <array>

namespace relorbit {

// ---------- NEWTON RK4 (o que você já tinha, mantendo) ----------
static inline std::array<double,4> f_newton(double mu, const std::array<double,4>& s) {
    const double x = s[0], y = s[1], vx = s[2], vy = s[3];
    const double r2 = x*x + y*y;
    const double r = std::sqrt(r2);
    const double ax = -mu * x / (r*r*r);
    const double ay = -mu * y / (r*r*r);
    return {vx, vy, ax, ay};
}

static inline double energy_newton(double mu, const std::array<double,4>& s) {
    const double x=s[0], y=s[1], vx=s[2], vy=s[3];
    const double r = std::sqrt(x*x+y*y);
    const double v2 = vx*vx+vy*vy;
    return 0.5*v2 - mu/r;
}
static inline double h_newton(const std::array<double,4>& s) {
    const double x=s[0], y=s[1], vx=s[2], vy=s[3];
    return x*vy - y*vx;
}

TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::vector<double>& state0,
    double t0,
    double tf,
    const SolverCfg& cfg
) {
    if (state0.size() != 4) throw std::runtime_error("state0 must be [x,y,vx,vy]");
    if (tf <= t0) throw std::runtime_error("tf must be > t0");

    int n = cfg.n_steps;
    if (n <= 0) {
        if (cfg.dt <= 0) throw std::runtime_error("cfg.dt must be > 0 if n_steps==0");
        n = static_cast<int>(std::ceil((tf - t0) / cfg.dt));
        if (n < 1) n = 1;
    }
    const double dt = (tf - t0) / static_cast<double>(n);

    TrajectoryNewton out;
    out.t.reserve(static_cast<size_t>(n)+1);
    out.y.reserve(static_cast<size_t>(n)+1);
    out.energy.reserve(static_cast<size_t>(n)+1);
    out.h.reserve(static_cast<size_t>(n)+1);
    out.status = OrbitStatus::BOUND;

    std::array<double,4> s { state0[0], state0[1], state0[2], state0[3] };
    double t = t0;

    auto push = [&]() {
        out.t.push_back(t);
        out.y.push_back({s[0],s[1],s[2],s[3]});
        out.energy.push_back(energy_newton(mu,s));
        out.h.push_back(h_newton(s));
    };

    push();
    for (int i=0;i<n;i++) {
        const auto k1 = f_newton(mu, s);
        std::array<double,4> s2 { s[0] + 0.5*dt*k1[0], s[1] + 0.5*dt*k1[1], s[2] + 0.5*dt*k1[2], s[3] + 0.5*dt*k1[3] };
        const auto k2 = f_newton(mu, s2);
        std::array<double,4> s3 { s[0] + 0.5*dt*k2[0], s[1] + 0.5*dt*k2[1], s[2] + 0.5*dt*k2[2], s[3] + 0.5*dt*k2[3] };
        const auto k3 = f_newton(mu, s3);
        std::array<double,4> s4 { s[0] + dt*k3[0], s[1] + dt*k3[1], s[2] + dt*k3[2], s[3] + dt*k3[3] };
        const auto k4 = f_newton(mu, s4);

        for (int j=0;j<4;j++) {
            s[j] = s[j] + (dt/6.0)*(k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
        }

        t += dt;
        push();
        if (!std::isfinite(s[0]) || !std::isfinite(s[1]) || !std::isfinite(s[2]) || !std::isfinite(s[3])) {
            out.status = OrbitStatus::ERROR;
            out.message = "non-finite state encountered";
            break;
        }
    }

    return out;
}

// ---------- SCHWARZSCHILD EQUATORIAL RK4 ----------
struct SchwState {
    double tcoord; // t
    double r;      // r
    double phi;    // phi
    double pr;     // dr/dtau
};

static inline SchwState f_schw(double M, double E, double L, const SchwState& s) {
    // dt/dtau = E / (1 - 2M/r)
    const double A = 1.0 - 2.0*M/s.r;

    // dphi/dtau = L / r^2
    const double dphi = L / (s.r*s.r);

    // dr/dtau = pr
    const double dr = s.pr;

    // dpr/dtau = - 1/2 dV/dr
    const double dV = dVeff_dr_schw(M, s.r, L);
    const double dpr = -0.5 * dV;

    SchwState ds;
    ds.tcoord = E / A;
    ds.r = dr;
    ds.phi = dphi;
    ds.pr = dpr;
    return ds;
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
    if (r0 <= 2.0*M) throw std::runtime_error("r0 must be > 2M");
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
    out.tau.reserve(static_cast<size_t>(n)+1);
    out.r.reserve(static_cast<size_t>(n)+1);
    out.phi.reserve(static_cast<size_t>(n)+1);
    out.tcoord.reserve(static_cast<size_t>(n)+1);
    out.pr.reserve(static_cast<size_t>(n)+1);
    out.epsilon.reserve(static_cast<size_t>(n)+1);
    out.E_series.reserve(static_cast<size_t>(n)+1);
    out.L_series.reserve(static_cast<size_t>(n)+1);

    SchwState s;
    s.tcoord = 0.0;
    s.r = r0;
    s.phi = phi0;

    // pr inicial vindo do constraint: pr^2 = E^2 - Veff(r)
    const double V0 = Veff_schw(M, r0, L);
    double pr2 = E*E - V0;
    if (pr2 < 0) pr2 = 0; // permite caso em turning point (pr=0)
    s.pr = 0.0; // por padrão: inicia em turning point; se quiser inbound/outbound, ajuste no YAML depois

    double tau = tau0;
    out.status = OrbitStatus::BOUND;

    auto push = [&]() {
        out.tau.push_back(tau);
        out.r.push_back(s.r);
        out.phi.push_back(s.phi);
        out.tcoord.push_back(s.tcoord);
        out.pr.push_back(s.pr);

        const double eps = (s.pr*s.pr) + Veff_schw(M, s.r, L) - (E*E);
        out.epsilon.push_back(eps);

        out.E_series.push_back(E);
        out.L_series.push_back(L);
    };

    push();

    const double r_capture = capture_r * M;

    for (int i=0;i<n;i++) {
        const auto k1 = f_schw(M,E,L,s);

        SchwState s2 { s.tcoord + 0.5*dt*k1.tcoord, s.r + 0.5*dt*k1.r, s.phi + 0.5*dt*k1.phi, s.pr + 0.5*dt*k1.pr };
        const auto k2 = f_schw(M,E,L,s2);

        SchwState s3 { s.tcoord + 0.5*dt*k2.tcoord, s.r + 0.5*dt*k2.r, s.phi + 0.5*dt*k2.phi, s.pr + 0.5*dt*k2.pr };
        const auto k3 = f_schw(M,E,L,s3);

        SchwState s4 { s.tcoord + dt*k3.tcoord, s.r + dt*k3.r, s.phi + dt*k3.phi, s.pr + dt*k3.pr };
        const auto k4 = f_schw(M,E,L,s4);

        s.tcoord += (dt/6.0)*(k1.tcoord + 2*k2.tcoord + 2*k3.tcoord + k4.tcoord);
        s.r      += (dt/6.0)*(k1.r      + 2*k2.r      + 2*k3.r      + k4.r);
        s.phi    += (dt/6.0)*(k1.phi    + 2*k2.phi    + 2*k3.phi    + k4.phi);
        s.pr     += (dt/6.0)*(k1.pr     + 2*k2.pr     + 2*k3.pr     + k4.pr);

        tau += dt;
        push();

        if (!std::isfinite(s.r) || !std::isfinite(s.phi) || !std::isfinite(s.pr) || !std::isfinite(s.tcoord)) {
            out.status = OrbitStatus::ERROR;
            out.message = "non-finite state encountered";
            break;
        }

        // captura física: cruzou r <= 2M (com folga numérica)
        if (s.r <= r_capture + capture_eps) {
            out.status = OrbitStatus::CAPTURE;
            out.message = "capture: r crossed capture radius";
            break;
        }

        // escape simples (não é completo, mas útil): se r cresce demais, trate como UNBOUND
        if (s.r > 1e6 * M) {
            out.status = OrbitStatus::UNBOUND;
            out.message = "unbound: r grew too large";
            break;
        }
    }

    return out;
}

} // namespace relorbit
