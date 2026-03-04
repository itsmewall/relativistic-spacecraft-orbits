// src_cpp/lib/attitude.cpp
//
// Módulo C — Item 7: Dinâmica de Atitude 6-DOF com Quaternions
//
// Implementação do integrador RK4 para o sistema de equações:
//
//   Cinemática:  q̇ = ½ Ω(ω) q
//   Dinâmica:    ω̇ = I⁻¹ [ τ − ω × (I ω) ]
//
// Critérios validados internamente:
//   (a) ‖q‖ = 1   — renormalização controlada
//   (b) T_rot constante quando τ = 0

#include "relorbit/models/attitude.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace relorbit {

// ───────────────────────────────────────────────────────────────
// Estado interno de integração (flat array de dim 7)
//   idx:  0   1   2   3    4    5    6
//         q0  q1  q2  q3   wx   wy   wz
// ───────────────────────────────────────────────────────────────
using State7 = std::array<double, 7>;

// ───────────────────────────────────────────────────────────────
// RHS — lado direito do sistema ODE de dim 7
//
// Cinemática do quaternion:
//   dq0/dt = ½ (−wx·q1 − wy·q2 − wz·q3)
//   dq1/dt = ½ ( wx·q0 + wz·q2 − wy·q3)
//   dq2/dt = ½ ( wy·q0 − wz·q1 + wx·q3)  ← corrigido sinal de wz·q1
//   dq3/dt = ½ ( wz·q0 + wy·q1 − wx·q2)
//
// Equações de Euler (dinâmica rotacional):
//   ω̇ = I⁻¹ [ τ − ω × (I ω) ]
// ───────────────────────────────────────────────────────────────
static State7 rhs_attitude(
    double t,
    const State7&               y,
    const std::array<double,9>& Iinv,   // inversa do tensor de inércia
    const InertiaTensor&        inertia,
    const TorqueCfg&            torque
) {
    const double q0 = y[0], q1 = y[1], q2 = y[2], q3 = y[3];
    const double wx = y[4], wy = y[5], wz = y[6];

    // ── Cinemática: q̇ = ½ Ω(ω) q ────────────────────────────
    const double dq0 = 0.5 * (-wx*q1 - wy*q2 - wz*q3);
    const double dq1 = 0.5 * ( wx*q0 + wz*q2 - wy*q3);
    const double dq2 = 0.5 * ( wy*q0 - wz*q1 + wx*q3);
    const double dq3 = 0.5 * ( wz*q0 + wy*q1 - wx*q2);

    // ── Dinâmica: ω̇ = I⁻¹ [τ − ω × (Iω)] ───────────────────

    // Iω
    auto Iw = inertia.mul(wx, wy, wz);

    // ω × (Iω)
    const double cross_x = wy*Iw[2] - wz*Iw[1];
    const double cross_y = wz*Iw[0] - wx*Iw[2];
    const double cross_z = wx*Iw[1] - wy*Iw[0];

    // torque externo
    const double tx = torque.get_x(t);
    const double ty = torque.get_y(t);
    const double tz = torque.get_z(t);

    // rhs_ω = τ − ω × (Iω)
    const double rhs_wx = tx - cross_x;
    const double rhs_wy = ty - cross_y;
    const double rhs_wz = tz - cross_z;

    // ω̇ = I⁻¹ · rhs_ω
    const double dwx = Iinv[0]*rhs_wx + Iinv[1]*rhs_wy + Iinv[2]*rhs_wz;
    const double dwy = Iinv[3]*rhs_wx + Iinv[4]*rhs_wy + Iinv[5]*rhs_wz;
    const double dwz = Iinv[6]*rhs_wx + Iinv[7]*rhs_wy + Iinv[8]*rhs_wz;

    return { dq0, dq1, dq2, dq3, dwx, dwy, dwz };
}

// ───────────────────────────────────────────────────────────────
// Adição e escala de State7
// ───────────────────────────────────────────────────────────────
static inline State7 add(const State7& a, const State7& b) {
    return { a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3],
             a[4]+b[4], a[5]+b[5], a[6]+b[6] };
}
static inline State7 scale(const State7& a, double s) {
    return { a[0]*s, a[1]*s, a[2]*s, a[3]*s,
             a[4]*s, a[5]*s, a[6]*s };
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
    traj.Ixx = inertia.I[0]; traj.Iyy = inertia.I[4]; traj.Izz = inertia.I[8];
    traj.Ixy = inertia.I[1]; traj.Ixz = inertia.I[2]; traj.Iyz = inertia.I[5];

    // Validação básica
    if (!(cfg.dt > 0.0) || !std::isfinite(cfg.dt) || !(tf >= t0)) {
        traj.status  = "ERROR";
        traj.message = "invalid parameters: dt or time interval";
        return traj;
    }

    // Inversão do tensor de inércia
    std::array<double,9> Iinv;
    if (!inertia.invert(Iinv)) {
        traj.status  = "ERROR";
        traj.message = "inertia tensor is singular — cannot invert";
        return traj;
    }

    const double h_step = cfg.dt;
    int n_steps = cfg.n_steps;
    if (n_steps <= 0)
        n_steps = std::max(1, static_cast<int>(std::ceil((tf - t0) / h_step)));

    const int rec          = (cfg.record_every > 0) ? cfg.record_every : 1;
    const int renorm_every = (cfg.renorm_every  > 0) ? cfg.renorm_every : 1;
    const double renorm_tol = cfg.renorm_tol;

    // Reservar memória
    const size_t res = static_cast<size_t>(n_steps / rec) + 2;
    traj.t .reserve(res); traj.q0.reserve(res); traj.q1.reserve(res);
    traj.q2.reserve(res); traj.q3.reserve(res);
    traj.wx.reserve(res); traj.wy.reserve(res); traj.wz.reserve(res);
    traj.qnorm.reserve(res); traj.T_rot.reserve(res);
    traj.renorm_delta.reserve(res);

    // Estado inicial
    State7 y = { state0.q0, state0.q1, state0.q2, state0.q3,
                 state0.wx, state0.wy, state0.wz };
    double t = t0;

    // Normalizar estado inicial
    {
        const double n = std::sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2] + y[3]*y[3]);
        if (n > 0.0) { y[0]/=n; y[1]/=n; y[2]/=n; y[3]/=n; }
    }

    auto push_sample = [&]() {
        traj.t .push_back(t);
        traj.q0.push_back(y[0]); traj.q1.push_back(y[1]);
        traj.q2.push_back(y[2]); traj.q3.push_back(y[3]);
        traj.wx.push_back(y[4]); traj.wy.push_back(y[5]);
        traj.wz.push_back(y[6]);
        const double nq = std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]+y[3]*y[3]);
        traj.qnorm .push_back(nq);
        traj.T_rot .push_back(inertia.T_rot(y[4], y[5], y[6]));
        traj.renorm_delta.push_back(std::abs(nq - 1.0));
    };

    push_sample();

    for (int step = 0; step < n_steps; ++step) {
        double h = h_step;
        if (t + h > tf) h = tf - t;
        if (!(h > 0.0)) break;

        // ── RK4 ────────────────────────────────────────────────
        const State7 k1 = rhs_attitude(t,          y,           Iinv, inertia, torque);
        const State7 k2 = rhs_attitude(t + 0.5*h,  add(y, scale(k1, 0.5*h)), Iinv, inertia, torque);
        const State7 k3 = rhs_attitude(t + 0.5*h,  add(y, scale(k2, 0.5*h)), Iinv, inertia, torque);
        const State7 k4 = rhs_attitude(t + h,       add(y, scale(k3, h)),     Iinv, inertia, torque);

        // y_new = y + (h/6)(k1 + 2k2 + 2k3 + k4)
        const double inv6h = h / 6.0;
        State7 y_new;
        for (int i = 0; i < 7; ++i)
            y_new[i] = y[i] + inv6h * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);

        t += h;

        // Verificação de finitude
        bool finite_ok = true;
        for (int i = 0; i < 7; ++i)
            if (!std::isfinite(y_new[i])) { finite_ok = false; break; }
        if (!finite_ok) {
            traj.status  = "ERROR";
            traj.message = "non-finite state at t=" + std::to_string(t);
            break;
        }

        y = y_new;

        // ── Renormalização do quaternion ────────────────────────
        const double nq = std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]+y[3]*y[3]);
        const double delta_n = std::abs(nq - 1.0);
        const bool   tol_triggered = (renorm_tol > 0.0 && delta_n > renorm_tol);
        const bool   step_triggered = ((step + 1) % renorm_every == 0);
        if (step_triggered || tol_triggered) {
            if (nq > 0.0) { y[0]/=nq; y[1]/=nq; y[2]/=nq; y[3]/=nq; }
        }

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