// src_cpp/lib/attitude.cpp
//
// Módulo C — Item 7: Dinâmica de Atitude 6-DOF com Quaternions
//
// Integrador RK4 sobre y ∈ R^7 = [q0 q1 q2 q3 wx wy wz].
//
//   Cinemática:  q' = 1/2 * Omega(w) * q
//   Dinâmica:    w' = I^{-1} [ tau - w x (I w) ]
//
// Toda a álgebra linear usa Eigen.

#include "relorbit/models/attitude.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace relorbit {

// ───────────────────────────────────────────────────────────────
// omega_matrix — 4×4 tal que q' = 1/2 Omega(w) q
//
//  Omega(w) =  |  0  -wx -wy -wz |
//              | wx   0   wz -wy |
//              | wy  -wz   0   wx|
//              | wz   wy -wx   0 |
// ───────────────────────────────────────────────────────────────
static Eigen::Matrix4d omega_matrix(const Vec3& w)
{
    const double wx = w[0], wy = w[1], wz = w[2];
    Eigen::Matrix4d Om;
    Om <<  0.0, -wx,  -wy,  -wz,
           wx,   0.0,  wz,  -wy,
           wy,  -wz,   0.0,  wx,
           wz,   wy,  -wx,   0.0;
    return Om;
}


// ───────────────────────────────────────────────────────────────
// rhs_attitude — RHS do ODE para y = [q ; w] ∈ R^7
//
//   y' = [ 1/2 Omega(w) q  ;  I^{-1} (tau - w x Iw) ]
// ───────────────────────────────────────────────────────────────
static State7 rhs_attitude(
    double               t,
    const State7&        y,
    const Mat3&          Iinv,
    const InertiaTensor& inertia,
    const TorqueCfg&     torque
) {
    const Vec4 q = y.head<4>();
    const Vec3 w = y.tail<3>();

    // Cinemática: q' = 1/2 Omega(w) q
    const Vec4 dq = 0.5 * (omega_matrix(w) * q);

    // Dinâmica: w' = I^{-1} (tau - w x Iw)
    const Vec3 Iw    = inertia.mul(w);
    const Vec3 cross = w.cross(Iw);
    const Vec3 tau   = torque.get(t);
    const Vec3 dw    = Iinv * (tau - cross);

    State7 dy;
    dy.head<4>() = dq;
    dy.tail<3>() = dw;
    return dy;
}


// ───────────────────────────────────────────────────────────────
// simulate_attitude_rk4
// ───────────────────────────────────────────────────────────────
TrajectoryAttitude simulate_attitude_rk4(
    const AttitudeState&  state0,
    const InertiaTensor&  inertia,
    const TorqueCfg&      torque,
    double t0, double tf,
    const AttitudeCfg&    cfg
) {
    TrajectoryAttitude traj;

    // Parâmetros de inércia no resultado
    traj.Ixx = inertia.I(0,0); traj.Iyy = inertia.I(1,1); traj.Izz = inertia.I(2,2);
    traj.Ixy = inertia.I(0,1); traj.Ixz = inertia.I(0,2); traj.Iyz = inertia.I(1,2);

    // Validação básica
    if (!(cfg.dt > 0.0) || !std::isfinite(cfg.dt) || !(tf >= t0)) {
        traj.status  = "ERROR";
        traj.message = "invalid parameters: dt or time interval";
        return traj;
    }

    // Inversa do tensor de inércia
    Mat3 Iinv;
    if (!inertia.invert(Iinv)) {
        traj.status  = "ERROR";
        traj.message = "inertia tensor is singular";
        return traj;
    }

    const double h_step = cfg.dt;
    int n_steps = cfg.n_steps;
    if (n_steps <= 0)
        n_steps = std::max(1, static_cast<int>(std::ceil((tf - t0) / h_step)));

    const int    rec          = (cfg.record_every > 0) ? cfg.record_every : 1;
    const int    renorm_every = (cfg.renorm_every  > 0) ? cfg.renorm_every : 1;
    const double renorm_tol   = cfg.renorm_tol;

    // Reservar memória
    const size_t res = static_cast<size_t>(n_steps / rec) + 2;
    traj.t .reserve(res); traj.q0.reserve(res); traj.q1.reserve(res);
    traj.q2.reserve(res); traj.q3.reserve(res);
    traj.wx.reserve(res); traj.wy.reserve(res); traj.wz.reserve(res);
    traj.qnorm.reserve(res); traj.T_rot.reserve(res);
    traj.renorm_delta.reserve(res);

    // Estado inicial
    State7 y;
    y << state0.q[0], state0.q[1], state0.q[2], state0.q[3],
         state0.w[0], state0.w[1], state0.w[2];
    double t = t0;

    // Normalizar quaternion inicial
    const double n0 = y.head<4>().norm();
    if (n0 > 0.0) y.head<4>() /= n0;

    // Lambda: grava ponto actual
    auto push_sample = [&]() {
        const Vec4 q = y.head<4>();
        const Vec3 w = y.tail<3>();
        traj.t .push_back(t);
        traj.q0.push_back(q[0]); traj.q1.push_back(q[1]);
        traj.q2.push_back(q[2]); traj.q3.push_back(q[3]);
        traj.wx.push_back(w[0]); traj.wy.push_back(w[1]); traj.wz.push_back(w[2]);
        const double nq = q.norm();
        traj.qnorm       .push_back(nq);
        traj.T_rot       .push_back(inertia.T_rot(w));
        traj.renorm_delta.push_back(std::abs(nq - 1.0));
    };

    push_sample();

    for (int step = 0; step < n_steps; ++step) {
        double h = h_step;
        if (t + h > tf) h = tf - t;
        if (!(h > 0.0)) break;

        // ── RK4 (operações directas sobre Eigen::Matrix) ────────
        const State7 k1 = rhs_attitude(t,          y,               Iinv, inertia, torque);
        const State7 k2 = rhs_attitude(t + 0.5*h,  y + 0.5*h * k1, Iinv, inertia, torque);
        const State7 k3 = rhs_attitude(t + 0.5*h,  y + 0.5*h * k2, Iinv, inertia, torque);
        const State7 k4 = rhs_attitude(t + h,       y + h      * k3, Iinv, inertia, torque);

        const State7 y_new = y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
        t += h;

        // Verificação de finitude (Eigen)
        if (!y_new.allFinite()) {
            traj.status  = "ERROR";
            traj.message = "non-finite state at t=" + std::to_string(t);
            break;
        }

        y = y_new;

        // ── Renormalização do quaternion ────────────────────────
        const double nq      = y.head<4>().norm();
        const double delta_n = std::abs(nq - 1.0);
        const bool tol_trig  = (renorm_tol > 0.0 && delta_n > renorm_tol);
        const bool step_trig = ((step + 1) % renorm_every == 0);
        if ((step_trig || tol_trig) && nq > 0.0)
            y.head<4>() /= nq;

        // ── Gravação ────────────────────────────────────────────
        if ((step + 1) % rec == 0 || step == n_steps - 1 || t >= tf)
            push_sample();

        if (t >= tf) break;
    }

    if (traj.status.empty()) {
        traj.status  = "OK";
        traj.message = "integration complete";
    }
    return traj;
}

} // namespace relorbit