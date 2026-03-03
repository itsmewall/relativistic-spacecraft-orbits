// src_cpp/lib/api.cpp
#include "relorbit/api.hpp"
#include <cmath>
#include <stdexcept>

namespace relorbit {

static inline std::array<double,4> f_newton(double mu, const std::array<double,4>& s) {
    const double x = s[0], y = s[1], vx = s[2], vy = s[3];
    const double r2 = x*x + y*y;
    const double r = std::sqrt(r2);
    const double invr3 = 1.0 / (r*r*r);
    const double ax = -mu * x * invr3;
    const double ay = -mu * y * invr3;
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

static inline OrbitStatus status_from_energy(double E) {
    return (E < 0.0) ? OrbitStatus::BOUND : OrbitStatus::UNBOUND;
}

TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::array<double,4>& state0,
    double t0,
    double tf,
    const SolverCfg& cfg
) {
    TrajectoryNewton out;
    out.status = OrbitStatus::BOUND;

    const double dt = cfg.dt;
    if (!(dt > 0.0) || !std::isfinite(dt)) {
        out.status = OrbitStatus::ERROR;
        out.message = "invalid dt";
        return out;
    }
    if (!(tf >= t0) || !std::isfinite(t0) || !std::isfinite(tf)) {
        out.status = OrbitStatus::ERROR;
        out.message = "invalid span";
        return out;
    }
    if (!(mu > 0.0) || !std::isfinite(mu)) {
        out.status = OrbitStatus::ERROR;
        out.message = "invalid mu";
        return out;
    }

    int n_steps = cfg.n_steps;
    if (n_steps <= 0) {
        n_steps = static_cast<int>(std::ceil((tf - t0) / dt));
        if (n_steps < 1) n_steps = 1;
    }

    int record_every = cfg.record_every > 0 ? cfg.record_every : 1;
    size_t res_size = static_cast<size_t>(n_steps / record_every) + 2;

    out.t.reserve(res_size);
    out.y.reserve(res_size);
    out.energy.reserve(res_size);
    out.h.reserve(res_size);

    double t = t0;
    std::array<double,4> s = state0;

    auto push = [&]() {
        out.t.push_back(t);
        out.y.push_back(s);
        out.energy.push_back(energy_newton(mu, s));
        out.h.push_back(h_newton(s));
    };

    if (!std::isfinite(s[0]) || !std::isfinite(s[1]) || !std::isfinite(s[2]) || !std::isfinite(s[3])) {
        out.status = OrbitStatus::ERROR;
        out.message = "invalid initial state (non-finite)";
        return out;
    }

    push();

    for (int i = 0; i < n_steps; ++i) {
        const auto k1 = f_newton(mu, s);
        
        std::array<double,4> s2 {
            s[0] + 0.5*dt*k1[0],
            s[1] + 0.5*dt*k1[1],
            s[2] + 0.5*dt*k1[2],
            s[3] + 0.5*dt*k1[3]
        };
        const auto k2 = f_newton(mu, s2);

        std::array<double,4> s3 {
            s[0] + 0.5*dt*k2[0],
            s[1] + 0.5*dt*k2[1],
            s[2] + 0.5*dt*k2[2],
            s[3] + 0.5*dt*k2[3]
        };
        const auto k3 = f_newton(mu, s3);

        std::array<double,4> s4 {
            s[0] + dt*k3[0],
            s[1] + dt*k3[1],
            s[2] + dt*k3[2],
            s[3] + dt*k3[3]
        };
        const auto k4 = f_newton(mu, s4);

        for (int j = 0; j < 4; ++j) {
            s[j] = s[j] + (dt/6.0)*(k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
        }

        t += dt;

        bool is_last = (i == n_steps - 1) || (t >= tf);
        bool broken = !std::isfinite(s[0]) || !std::isfinite(s[1]) || !std::isfinite(s[2]) || !std::isfinite(s[3]);

        if ((i + 1) % record_every == 0 || is_last || broken) {
            push();
        }

        if (broken) {
            out.status = OrbitStatus::ERROR;
            out.message = "non-finite state encountered";
            break;
        }

        if (is_last) break;
    }

    if (out.status != OrbitStatus::ERROR && !out.energy.empty()) {
        const double E_init = out.energy.front();
        out.status = status_from_energy(E_init);
    }

    return out;
}

TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::vector<double>& state0,
    double t0,
    double tf,
    const SolverCfg& cfg
) {
    if (state0.size() != 4) {
        TrajectoryNewton out;
        out.status = OrbitStatus::ERROR;
        out.message = "state0 must have size 4 [x,y,vx,vy]";
        return out;
    }
    std::array<double,4> arr { state0[0], state0[1], state0[2], state0[3] };
    return simulate_newton_rk4(mu, arr, t0, tf, cfg);
}

} // namespace relorbit