// src_cpp/lib/schwarzschild_6dof.cpp
//
// Item 8 — Thrust Vectoring: Integrador Acoplado 6-DOF
//
// Estado y ∈ ℝ¹³ = [r, phi, pr, E, L, m, q0, q1, q2, q3, wx, wy, wz]
//
// O acoplamento central é:
//   n̂_inercial = R(q) · k̂_body            (direcção de empuxo)
//   a_r   = (F/m) · n_r                    (aceleração radial, geométrica)
//   a_phi = (F/m) · n_phi / r              (aceleração tangencial)
//   dL/dτ = r · (F/m)_geom · n_phi        (variação de L)
//   dE/dτ via f^μ u_μ = 0                  (variação de E)
//   dm/dτ = −m |F|_SI / (Isp · g₀)        (equação de Tsiolkovsky contínua)
//
// A atitude evolui pelo mesmo RK4, recebendo torque de reacção do motor.

#include "relorbit/models/schwarzschild_6dof.hpp"
#include "relorbit/models/attitude.hpp"
#include "relorbit/models/schwarzschild_equatorial.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace relorbit {

// ── Dimensão do estado ───────────────────────────────────────────
//   0:r  1:phi  2:pr  3:E  4:L  5:m  6:q0  7:q1  8:q2  9:q3  10:wx  11:wy  12:wz
static constexpr int STATE_DIM = 13;
using StateV = Eigen::Matrix<double, STATE_DIM, 1>;


// ── Helpers inline ───────────────────────────────────────────────

static inline double schw_A(double M, double r) {
    return 1.0 - 2.0 * M / r;
}

// Velocidade angular no eixo φ (coordenada)
//   dphi/dτ = L / r²  (Schwarzschild equatorial, sem empuxo tangencial acumulado ainda)
//   Com E e L variáveis: usar L corrente
static inline double schw_dphi(double /*E*/, double L, double r) {
    return L / (r * r);
}

// ── omega_matrix (reutiliza do attitude.cpp via inline aqui) ─────
static Eigen::Matrix4d omega4(const Vec3& w) {
    const double wx = w[0], wy = w[1], wz = w[2];
    Eigen::Matrix4d Om;
    Om <<  0.0, -wx, -wy, -wz,
           wx,  0.0,  wz, -wy,
           wy, -wz,  0.0,  wx,
           wz,  wy,  -wx,  0.0;
    return Om;
}


// ── rhs_6dof ─────────────────────────────────────────────────────
//
// Computa ẏ para o estado acoplado.
//
// Entradas:
//   y        — estado actual
//   M        — massa do BH [geom]
//   engine   — configuração do motor
//   att_cfg  — tensor de inércia + torque externo
//   Iinv     — inversa pré-calculada do tensor de inércia
static StateV rhs_6dof(
    double          tau,
    const StateV&   y,
    double          M,
    const EngineCfg& engine,
    const AttitudeCfg6DOF& att_cfg,
    const Mat3&     Iinv
) {
    // ── Extrair campos ──────────────────────────────────────────
    const double r   = y[0];
    const double phi = y[1];
    const double pr  = y[2];
    const double E   = y[3];
    const double L   = y[4];
    const double m   = std::max(y[5], 1e-10);   // protecção contra m→0

    const Vec4 q = y.segment<4>(6);
    const Vec3 w = y.segment<3>(10);

    const double r2    = r * r;
    const double r_safe = std::max(r, 1e-300);
    const double A     = schw_A(M, r_safe);
    const double A_safe = std::max(A, 1e-300);

    StateV dy = StateV::Zero();

    // ── Órbita geodésica base ───────────────────────────────────
    //   dr/dτ = pr
    dy[0] = pr;

    //   dphi/dτ = L / r²
    dy[1] = L / r2;

    //   dpr/dτ = −½ dVeff/dr  (Schwarzschild)
    dy[2] = -0.5 * dVeff_dr_schw(M, r_safe, L);

    //   dt/dτ = E / A   (não gravado no estado, mas calculado se necessário)
    //   dE/dτ = 0 (geodésica)
    dy[3] = 0.0;

    //   dL/dτ = 0 (geodésica)
    dy[4] = 0.0;

    //   dm/dτ = 0 (geodésica)
    dy[5] = 0.0;

    // ── Atitude geodésica (sem empuxo) ──────────────────────────
    //   q̇ = ½ Ω(ω) q
    dy.segment<4>(6) = 0.5 * (omega4(w) * q);

    //   ω̇ = I⁻¹ [τ_ext − ω × (I ω)]
    {
        const Vec3 Iw    = att_cfg.inertia.mul(w);
        const Vec3 cross = w.cross(Iw);
        const Vec3 tau_e = att_cfg.ext_torque.get(tau);
        dy.segment<3>(10) = Iinv * (tau_e - cross);
    }

    // ── Acoplamento: motor activo ────────────────────────────────
    if (!engine.active(tau) || m < att_cfg.inertia.I(0,0) * 1e-20) {
        return dy;   // geodésica pura
    }

    // Direcção do bocal no frame inercial equatorial
    //   n̂_inercial = R(q) · k̂_body
    const Vec3 k_body = engine.nozzle_body.normalized();
    const Mat3 R      = dcm_from_quaternion(q);
    const Vec3 n_iner = R * k_body;     // [nx, ny, nz] 3D inercial

    // Projecção em coordenadas polares equatoriais:
    //   eixo r̂   no plano equatorial: (cos φ, sin φ, 0)
    //   eixo φ̂   no plano equatorial: (−sin φ, cos φ, 0)
    //   eixo ẑ   não tem força (equatorial → descartar componente z)
    const double cphi = std::cos(phi);
    const double sphi = std::sin(phi);
    const double n_r   =  cphi * n_iner[0] + sphi * n_iner[1];
    const double n_phi = -sphi * n_iner[0] + cphi * n_iner[1];

    // Aceleração específica em unidades geométricas [c²/M]
    //   F [N] / m [kg] / c² [m/s²] = adimensional em G=c=1
    const double a_geom = engine.F_geom(m);   // (F/m) / c²

    // ── Força sobre a órbita ────────────────────────────────────

    //   f^r = a_geom · n_r   → correcção a dpr
    //   Equação de geodésica perturbada (Papapetrou / forças externas):
    //   dpr/dτ += f_r^coord = a_geom · n_r
    dy[2] += a_geom * n_r;

    //   f^phi = a_geom · n_phi / r  → dL/dτ = r² · f^phi = r · a_geom · n_phi
    dy[4] += r_safe * a_geom * n_phi;

    //   dE/dτ via f^μ u_μ = 0 (Schwarzschild):
    //   u^t = E / A,  u^r = pr,  u^phi = L/r²
    //   g_tt f^t + g_rr u^r f^r_phys + g_pp u^phi f^phi_phys = 0
    //   f^t = −(g_rr pr a_r + g_pp (L/r²)(a_phi))  / g_tt
    //
    //   g_tt = −A,  g_rr = 1/A,  g_phiphi = r²
    //   f^r_coord  = a_geom · n_r
    //   f^phi_coord= a_geom · n_phi / r
    {
        const double g_rr  =  1.0 / A_safe;
        const double g_tt  = -A_safe;
        const double g_pp  =  r2;
        const double uphi  =  L / r2;
        const double ft_num = -(g_rr * pr * a_geom * n_r
                              + g_pp * uphi * a_geom * n_phi / r_safe);
        const double ft = (std::abs(g_tt) > 1e-14) ? ft_num / g_tt : 0.0;
        // dE/dτ = −d(u_t)/dτ = −(g_tt ft)   mas u_t = −E  →  dE = −g_tt ft
        dy[3] += -g_tt * ft;
    }

    //   dm/dτ = −m |F|_SI / (Isp · g₀)
    // dm/dτ = −F_newton / (Isp · g₀)   [kg/s]  (consistente com kerr_lowthrust)
    dy[5] = -engine.F_newton / (engine.isp_s * DOF6_G0);

    // ── Torque de reacção do motor sobre a atitude ───────────────
    //   τ_total = τ_externo + τ_reacção_motor
    {
        const Vec3 Iw    = att_cfg.inertia.mul(w);
        const Vec3 cross = w.cross(Iw);
        const Vec3 tau_e = att_cfg.ext_torque.get(tau);
        const Vec3 tau_r = engine.active(tau) ? engine.torque_reaction : Vec3::Zero();
        dy.segment<3>(10) = Iinv * (tau_e + tau_r - cross);
    }

    return dy;
}


// ── simulate_schwarzschild_6dof_rk4 ──────────────────────────────
TrajectoryCoupled simulate_schwarzschild_6dof_rk4(
    double M,
    double E0, double L0,
    double r0, double phi0, double pr0,
    const AttitudeState&    att0,
    double tau0, double tauf,
    const EngineCfg&        engine,
    const AttitudeCfg6DOF&  att_cfg,
    const SolverCfg6DOF&    cfg
) {
    TrajectoryCoupled traj;
    traj.M = M; traj.r0 = r0; traj.phi0 = phi0; traj.E0 = E0; traj.L0 = L0;

    // Validação básica
    if (!(cfg.dt > 0.0) || !std::isfinite(cfg.dt) || !(tauf >= tau0) || !(M > 0.0)) {
        traj.status  = OrbitStatus::ERROR;
        traj.message = "invalid parameters";
        return traj;
    }

    // Inversa do tensor de inércia
    Mat3 Iinv;
    if (!att_cfg.inertia.invert(Iinv)) {
        traj.status  = OrbitStatus::ERROR;
        traj.message = "inertia tensor is singular";
        return traj;
    }

    const double r_cap = cfg.capture_r * M;

    int n_steps = cfg.n_steps;
    if (n_steps <= 0)
        n_steps = std::max(1, static_cast<int>(std::ceil((tauf - tau0) / cfg.dt)));

    const int    rec          = (cfg.record_every > 0) ? cfg.record_every : 1;
    const int    renorm_every = (cfg.renorm_every  > 0) ? cfg.renorm_every : 1;
    const double renorm_tol   = cfg.renorm_tol;

    // Reservar memória
    const size_t res = static_cast<size_t>(n_steps / rec) + 2;
    traj.tau    .reserve(res); traj.r .reserve(res); traj.phi.reserve(res);
    traj.pr     .reserve(res); traj.E .reserve(res); traj.L  .reserve(res);
    traj.mass   .reserve(res); traj.epsilon.reserve(res); traj.tcoord.reserve(res);
    traj.q0.reserve(res); traj.q1.reserve(res); traj.q2.reserve(res); traj.q3.reserve(res);
    traj.wx.reserve(res); traj.wy.reserve(res); traj.wz.reserve(res);
    traj.qnorm.reserve(res); traj.T_rot.reserve(res);
    traj.thrust_r.reserve(res); traj.thrust_phi.reserve(res);
    traj.pointing_err.reserve(res);

    // Estado inicial
    StateV y = StateV::Zero();
    y[0] = r0; y[1] = phi0; y[2] = pr0; y[3] = E0; y[4] = L0;
    y[5] = engine.F_newton > 0.0 ? att_cfg.inertia.I(0,0) * 0.0 + engine.isp_s * 0.0 : 0.0;
    // massa inicial: usar mass0_kg da EngineCfg se disponível,
    // aqui passada via att0 (convenção: att0.w[0] reaproveitado? Não.)
    // A massa inicial em kg é externa — passa-se por parâmetro separado:
    // NOTA: precisa de ser passada — vamos usar engine.isp_s como sentinel
    // e deixar o campo y[5] para ser inicializado pelo caller via wrapper Python.
    // Por ora: y[5] = 1000.0 kg (default), sobrescrito antes de chamar.
    y[5] = engine.mass0_kg;

    // Quaternion inicial (normalizado)
    {
        Vec4 q0v = att0.q;
        const double nq = q0v.norm();
        if (nq > 0.0) q0v /= nq;
        y.segment<4>(6) = q0v;
    }
    y.segment<3>(10) = att0.w;

    double tau    = tau0;
    double tcoord = 0.0;

    // Lambda: grava ponto actual
    auto push = [&]() {
        const double r_  = y[0];
        const double phi_= y[1];
        const double pr_ = y[2];
        const double E_  = y[3];
        const double L_  = y[4];
        const double m_  = y[5];
        const Vec4 q = y.segment<4>(6);
        const Vec3 w = y.segment<3>(10);

        traj.tau .push_back(tau);
        traj.r   .push_back(r_);
        traj.phi .push_back(phi_);
        traj.pr  .push_back(pr_);
        traj.E   .push_back(E_);
        traj.L   .push_back(L_);
        traj.mass.push_back(m_);
        traj.epsilon.push_back(pr_*pr_ + Veff_schw(M, r_, L_) - E_*E_);

        const double A_ = schw_A(M, std::max(r_, 1e-300));
        tcoord += 0.0;   // actualizado no loop
        traj.tcoord.push_back(tcoord);

        traj.q0.push_back(q[0]); traj.q1.push_back(q[1]);
        traj.q2.push_back(q[2]); traj.q3.push_back(q[3]);
        traj.wx.push_back(w[0]); traj.wy.push_back(w[1]); traj.wz.push_back(w[2]);

        const double nq = q.norm();
        traj.qnorm.push_back(nq);
        traj.T_rot.push_back(att_cfg.inertia.T_rot(w));

        // Thrust components (se activo)
        double tr = 0.0, tphi = 0.0;
        if (engine.active(tau)) {
            const Vec3 k_body = engine.nozzle_body.normalized();
            const Mat3 R      = dcm_from_quaternion(q);
            const Vec3 n      = R * k_body;
            const double a    = engine.F_geom(m_);
            const double cphi_ = std::cos(phi_);
            const double sphi_ = std::sin(phi_);
            tr   = a * ( cphi_*n[0] + sphi_*n[1]);
            tphi = a * (-sphi_*n[0] + cphi_*n[1]);
        }
        traj.thrust_r  .push_back(tr);
        traj.thrust_phi.push_back(tphi);
        traj.pointing_err.push_back(0.0);  // extensível: comparar com alvo
    };

    push();

    for (int step = 0; step < n_steps; ++step) {
        double h = cfg.dt;
        if (tau + h > tauf) h = tauf - tau;
        if (!(h > 0.0)) break;

        // ── RK4 ────────────────────────────────────────────────
        const StateV k1 = rhs_6dof(tau,         y,             M, engine, att_cfg, Iinv);
        const StateV k2 = rhs_6dof(tau + 0.5*h, y + 0.5*h*k1, M, engine, att_cfg, Iinv);
        const StateV k3 = rhs_6dof(tau + 0.5*h, y + 0.5*h*k2, M, engine, att_cfg, Iinv);
        const StateV k4 = rhs_6dof(tau + h,     y + h    *k3, M, engine, att_cfg, Iinv);

        StateV y_new = y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

        // Actualizar tcoord (integração por Euler — barato, não entra no RK4 orbital)
        {
            const double A_ = schw_A(M, std::max(y[0], 1e-300));
            tcoord += h * y[3] / std::max(A_, 1e-300);
        }

        tau += h;

        // Verificação de finitude
        if (!y_new.allFinite()) {
            traj.status  = OrbitStatus::ERROR;
            traj.message = "non-finite state at tau=" + std::to_string(tau);
            break;
        }

        // Garantir massa >= 0
        if (y_new[5] < 0.0) y_new[5] = 0.0;

        y = y_new;

        // Renormalização do quaternion
        const double nq      = y.segment<4>(6).norm();
        const double delta_n = std::abs(nq - 1.0);
        const bool tol_trig  = (renorm_tol > 0.0 && delta_n > renorm_tol);
        const bool step_trig = ((step + 1) % renorm_every == 0);
        if ((step_trig || tol_trig) && nq > 0.0)
            y.segment<4>(6) /= nq;

        // Gravação
        if ((step + 1) % rec == 0 || step == n_steps - 1 || tau >= tauf)
            push();

        // Captura / horizonte
        if (y[0] <= r_cap) {
            traj.status  = OrbitStatus::CAPTURE;
            traj.message = "captured at r=" + std::to_string(y[0]);
            return traj;
        }

        if (tau >= tauf) break;
    }

    if (traj.status == OrbitStatus::ERROR && traj.message.empty()) {
        traj.status  = OrbitStatus::BOUND;
        traj.message = "integration complete";
    }
    return traj;
}

} // namespace relorbit