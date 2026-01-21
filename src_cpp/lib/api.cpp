// src_cpp/lib/api.cpp
#include "relorbit/api.hpp"
#include <cmath>
#include <stdexcept>

namespace relorbit {

static inline std::array<double,4> f_newton(double mu, const std::array<double,4>& s) {
    const double x = s[0], y = s[1], vx = s[2], vy = s[3];
    const double r2 = x*x + y*y;
    const double r = std::sqrt(r2);

    // evita divisão por zero / NaN
    if (!(r > 0.0) || !std::isfinite(r)) {
        return { std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN() };
    }

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

TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::array<double,4>& state0,
    double t0,
    double tf,
    const SolverCfg& cfg
) {
    if (tf <= t0) throw std::runtime_error("tf must be > t0");
    if (!(cfg.dt > 0.0) && cfg.n_steps <= 0) throw std::runtime_error("cfg.dt must be > 0 if n_steps==0");
    if (!(mu > 0.0) || !std::isfinite(mu)) throw std::runtime_error("mu must be > 0 and finite");

    // energia inicial define bound/unbound (teoria)
    const double e0 = energy_newton(mu, state0);
    if (!std::isfinite(e0)) throw std::runtime_error("invalid initial state: energy is non-finite");

    int n = cfg.n_steps;
    if (n <= 0) {
        n = static_cast<int>(std::ceil((tf - t0) / cfg.dt));
        if (n < 1) n = 1;
    }
    const double dt = (tf - t0) / static_cast<double>(n);

    TrajectoryNewton out;
    out.t.reserve(static_cast<size_t>(n)+1);
    out.y.reserve(static_cast<size_t>(n)+1);
    out.energy.reserve(static_cast<size_t>(n)+1);
    out.h.reserve(static_cast<size_t>(n)+1);

    // status físico default (só cai pra ERROR se quebrar)
    out.status = (e0 < 0.0) ? OrbitStatus::BOUND : OrbitStatus::UNBOUND;
    out.message.clear();

    std::array<double,4> s = state0;
    double t = t0;

    auto push = [&]() {
        out.t.push_back(t);
        out.y.push_back(s);
        out.energy.push_back(energy_newton(mu, s));
        out.h.push_back(h_newton(s));
    };

    push();

    for (int i = 0; i < n; ++i) {
        const auto k1 = f_newton(mu, s);

        std::array<double,4> s2 {
            s[0] + 0.5*dt*k1[0], s[1] + 0.5*dt*k1[1],
            s[2] + 0.5*dt*k1[2], s[3] + 0.5*dt*k1[3]
        };
        const auto k2 = f_newton(mu, s2);

        std::array<double,4> s3 {
            s[0] + 0.5*dt*k2[0], s[1] + 0.5*dt*k2[1],
            s[2] + 0.5*dt*k2[2], s[3] + 0.5*dt*k2[3]
        };
        const auto k3 = f_newton(mu, s3);

        std::array<double,4> s4 {
            s[0] + dt*k3[0], s[1] + dt*k3[1],
            s[2] + dt*k3[2], s[3] + dt*k3[3]
        };
        const auto k4 = f_newton(mu, s4);

        for (int j = 0; j < 4; ++j) {
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

    // Se quiser, pode deixar uma msg curta quando for UNBOUND (opcional)
    // if (out.status == OrbitStatus::UNBOUND && out.message.empty()) out.message = "specific energy >= 0 => unbound";

    return out;
}

// Wrapper vector -> array (resolve teu LNK2001 com pybind)
TrajectoryNewton simulate_newton_rk4(
    double mu,
    const std::vector<double>& state0,
    double t0,
    double tf,
    const SolverCfg& cfg
) {
    if (state0.size() != 4) throw std::runtime_error("state0 must be [x,y,vx,vy] (size==4)");
    std::array<double,4> a { state0[0], state0[1], state0[2], state0[3] };
    return simulate_newton_rk4(mu, a, t0, tf, cfg);
}

} // namespace relorbit
