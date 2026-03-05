// src_cpp/lib/kerr_6dof.cpp
//
// Integrador 6-DOF acoplado: Kerr + Atitude + Torque de Maré
//
// Estado 14D:  [r, phi, pr, E, L, m,  q0,q1,q2,q3,  wx,wy,wz,  pad]
//               0   1    2  3  4  5   6  7  8  9     10 11 12   13
//
// Física orbital: reutiliza exactamente as equações de kerr_lowthrust.cpp
// Física de atitude: reutiliza a cinemática e dinâmica de schwarzschild_6dof.cpp
// Torque de maré: via kerr_metric.hpp (3 níveis: WEAK_N, DIAG_EIJ, RIEMANN_FD)

#include "relorbit/models/kerr_6dof.hpp"
#include "relorbit/models/attitude.hpp"
#include "relorbit/models/schwarzschild_6dof.hpp"
#include "relorbit/models/kerr_lowthrust.hpp"
#include "relorbit/gr/kerr_metric.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace relorbit {

static constexpr int KDIM = 14;
using KStateV = Eigen::Matrix<double, KDIM, 1>;

// ─── omega_matrix 4×4 ────────────────────────────────────────────
static Eigen::Matrix4d omega4k(const Vec3& w) {
    const double wx = w[0], wy = w[1], wz = w[2];
    Eigen::Matrix4d Om;
    Om <<  0.0, -wx, -wy, -wz,
           wx,  0.0,  wz, -wy,
           wy, -wz,  0.0,  wx,
           wz,  wy,  -wx,  0.0;
    return Om;
}

// ─── Calcular torque de maré no body frame ────────────────────────
// FIX-C: contador de diagnóstico — imprime nas primeiras 3 avaliações.
// Se tidal_norm = 0, verificar se n_body está alinhado com eixo principal.
// Causa típica: q_inicial = identidade → n_body = [1,0,0] = eixo x (principal)
//               → n × (I·n) = 0 sempre (equilíbrio trivial).
// Solução YAML: usar q0 tal que nenhum eixo principal coincida com r̂.
// Exemplo seguro: q = (0.924, 0, 0.383, 0)  [45° volta ao eixo z]
static int s_tidal_diag_n = 0;

static Vec3 compute_tidal_torque(
    double M, double a, double r,
    const Vec4& q,
    const Mat3& I_body,
    const TidalCfg& tcfg
) {
    if (!tcfg.enabled) return Vec3::Zero();

    const Mat3 R = dcm_from_quaternion(q);   // body→ZAMO local

    if (tcfg.model == TidalModel::WEAK_N) {
        // Direcção radial no ZAMO local: r̂ = [1,0,0]
        const Vec3 n_zamo(1.0, 0.0, 0.0);
        const Vec3 n_body = R.transpose() * n_zamo;  // r̂ no frame do corpo
        const Vec3 In     = I_body * n_body;
        const Vec3 cross  = n_body.cross(In);         // = n × (I·n)
        // FIX-C diagnóstico: 3 primeiras avaliações
        if (s_tidal_diag_n < 3) {
            std::cout << "[TIDAL_WEAK_N][diag#" << s_tidal_diag_n << "]"
                      << " r=" << r
                      << " n_body=(" << n_body[0] << "," << n_body[1]
                      << "," << n_body[2] << ")"
                      << " n×(In)=(" << cross[0] << "," << cross[1]
                      << "," << cross[2] << ")"
                      << " scale=3M/r^3=" << (3.0*M/(r*r*r))
                      << " |tau_expected|=" << (3.0*M/(r*r*r)) * cross.norm()
                      << std::endl;
            // Aviso explícito se torque é zero
            if (cross.norm() < 1e-15) {
                std::cout << "  *** AVISO: torque WEAK_N = 0. "
                          << "n_body alinhado com eixo principal de I? "
                          << "Use quaternion inicial com rotacao nao-trivial. ***"
                          << std::endl;
            }
            ++s_tidal_diag_n;
        }
        return gr::tidal_torque_weak_n(M, r, I_body, n_body);
    }

    // Para DIAG_EIJ e RIEMANN_FD: computar E_local e usar quadrupolo
    Mat3 E_local;
    if (tcfg.model == TidalModel::DIAG_EIJ) {
        E_local = gr::tidal_diag_weak(M, a, r, tcfg.spin_correction);
    } else {
        // RIEMANN_FD
        E_local = gr::tidal_riemann_fd(M, a, r, tcfg.fd_eps_r);
    }

    // E no body frame: E_body = Rᵀ E_local R
    const Mat3 E_body = R.transpose() * E_local * R;

    // Quadrupolo no body frame
    Mat3 Q_body;
    if (tcfg.Q_from_inertia) {
        Q_body = gr::quadrupole_from_inertia(I_body);
    } else {
        Q_body = tcfg.Q_body;
    }

    return gr::tidal_torque_quadrupole(Q_body, E_body);
}

// ─── rhs_kerr_6dof ───────────────────────────────────────────────
static KStateV rhs_kerr_6dof(
    double          tau,
    const KStateV&  y,
    double          M,
    double          a,
    const EngineCfg& engine,
    const AttitudeCfgKerr& att_cfg,
    const Mat3&     Iinv
) {
    // Extrair estado
    const double r   = y[0];
    const double phi = y[1];
    const double pr  = y[2];
    const double E   = y[3];
    const double L   = y[4];
    const double m   = std::max(y[5], 1e-10);

    const Vec4 q = y.segment<4>(6);
    const Vec3 w = y.segment<3>(10);

    const double r2    = r * r;
    const double r3    = r2 * r;
    const double r4    = r3 * r;
    const double a2    = a * a;
    const double Delta = lt_kerr_Delta(M, a, r);
    const double Dsafe = std::max(Delta, 1e-300);
    const double Dsq   = std::sqrt(std::max(Delta, 0.0));
    const double rho2  = lt_kerr_rho2(M, a, r);
    const double rho2s = std::max(rho2, 1e-300);
    const double K     = L - a * E;

    KStateV dy = KStateV::Zero();

    // ── Órbita geodésica Kerr ──────────────────────────────────
    dy[0] = pr;  // dr/dτ

    dy[1] = (2.0*M*a*E/r + (1.0-2.0*M/r)*L) / Dsafe;  // dphi/dτ

    // dpr/dτ = geodésico Kerr
    dy[2] = -M/r2 + (L*L + a2*(1.0-E*E))/r3 - 3.0*M*K*K/r4;

    // dE/dτ = 0 (geodésica)
    dy[3] = 0.0;
    // dL/dτ = 0 (geodésica)
    dy[4] = 0.0;
    // dm/dτ = 0 (geodésica)
    dy[5] = 0.0;

    // ── Atitude geodésica ──────────────────────────────────────
    dy.segment<4>(6) = 0.5 * (omega4k(w) * q);

    {
        const Vec3 Iw    = att_cfg.inertia.mul(w);
        const Vec3 cross = w.cross(Iw);
        const Vec3 tau_e = att_cfg.ext_torque.get(tau);

        // Torque de maré
        const Vec3 tau_t = compute_tidal_torque(
            M, a, r, q, att_cfg.inertia.I, att_cfg.tidal
        );

        dy.segment<3>(10) = Iinv * (tau_e + tau_t - cross);
    }

    // ── Acoplamento motor ──────────────────────────────────────
    if (!engine.active(tau)) return dy;

    // Direcção de empuxo no frame ZAMO local (= inercial equatorial)
    const Vec3 k_body = engine.nozzle_body.normalized();
    const Mat3 R      = dcm_from_quaternion(q);
    const Vec3 n_iner = R * k_body;  // 3D

    // Projecção em coord polares:
    const double cphi = std::cos(phi);
    const double sphi = std::sin(phi);
    const double n_r   =  cphi * n_iner[0] + sphi * n_iner[1];
    const double n_phi = -sphi * n_iner[0] + cphi * n_iner[1];

    const double a_geom = engine.F_geom(m);

    // dpr += f^r_coord = a_geom * n_r * √Δ/r   (mapeamento ZAMO Kerr)
    dy[2] += a_geom * n_r * Dsq / r;

    // dL/dτ = √ρ² * F_phi  (torque tangencial ZAMO)
    dy[4] += std::sqrt(rho2s) * a_geom * n_phi;

    // dE/dτ via ortogonalidade f^μ u_μ = 0 (mesmo cálculo do kerr_lowthrust)
    {
        const double g_tt   = -(1.0 - 2.0*M/r);
        const double g_tphi = -2.0*M*a/r;
        const double g_rr   = r2 / Dsafe;
        const double ut     = ((r2+a2+2.0*M*a2/r)*E - 2.0*M*a*L/r) / Dsafe;
        const double uphi   = dy[1];  // já calculado

        const double fr_coord   = a_geom * Dsq / r * n_r;
        const double fphi_coord = a_geom * n_phi / std::sqrt(rho2s);

        const double denom_ft = g_tt*ut + g_tphi*uphi;
        double ft = 0.0;
        if (std::abs(denom_ft) > 1e-14) {
            ft = -(g_tphi*ut*fphi_coord + rho2*uphi*fphi_coord + g_rr*pr*fr_coord) / denom_ft;
        }
        dy[3] += -(g_tt*ft + g_tphi*fphi_coord);
    }

    // dm/dτ
    dy[5] = -engine.F_newton / (engine.isp_s * DOF6_G0);

    // Torque de reacção do motor
    {
        const Vec3 Iw    = att_cfg.inertia.mul(w);
        const Vec3 cross = w.cross(Iw);
        const Vec3 tau_e = att_cfg.ext_torque.get(tau);
        const Vec3 tau_t = compute_tidal_torque(
            M, a, r, q, att_cfg.inertia.I, att_cfg.tidal
        );
        const Vec3 tau_r = engine.torque_reaction;
        dy.segment<3>(10) = Iinv * (tau_e + tau_t + tau_r - cross);
    }

    return dy;
}

// ─── simulate_kerr_6dof_rk4 ──────────────────────────────────────
TrajectoryCoupledKerr simulate_kerr_6dof_rk4(
    double M, double a,
    double E0, double L0,
    double r0, double phi0, double pr0,
    const AttitudeState&   att0,
    double tau0, double tauf,
    const EngineCfg&       engine,
    const AttitudeCfgKerr& att_cfg,
    const SolverCfg6DOF&   cfg
) {
    TrajectoryCoupledKerr traj;
    traj.M = M; traj.a = a;
    traj.r0 = r0; traj.phi0 = phi0; traj.E0 = E0; traj.L0 = L0;

    if (!(cfg.dt > 0.0) || !std::isfinite(cfg.dt) || !(tauf >= tau0) || !(M > 0.0)) {
        traj.status  = OrbitStatus::ERROR;
        traj.message = "invalid parameters";
        return traj;
    }

    Mat3 Iinv;
    if (!att_cfg.inertia.invert(Iinv)) {
        traj.status  = OrbitStatus::ERROR;
        traj.message = "inertia tensor is singular";
        return traj;
    }

    // Horizonte de Kerr
    const double r_plus = M + std::sqrt(std::max(0.0, M*M - a*a));
    const double r_hor  = r_plus * (1.0 + cfg.capture_eps);
    const double r_cap  = cfg.capture_r * M;

    int n_steps = cfg.n_steps;
    if (n_steps <= 0)
        n_steps = std::max(1, static_cast<int>(std::ceil((tauf - tau0) / cfg.dt)));

    const int    rec          = std::max(1, cfg.record_every);
    const int    renorm_every = std::max(1, cfg.renorm_every);
    const double renorm_tol   = cfg.renorm_tol;

    const size_t res = static_cast<size_t>(n_steps / rec) + 2;
    traj.tau    .reserve(res); traj.r .reserve(res); traj.phi.reserve(res);
    traj.pr     .reserve(res); traj.E .reserve(res); traj.L  .reserve(res);
    traj.mass   .reserve(res); traj.epsilon.reserve(res); traj.tcoord.reserve(res);
    traj.q0.reserve(res); traj.q1.reserve(res); traj.q2.reserve(res); traj.q3.reserve(res);
    traj.wx.reserve(res); traj.wy.reserve(res); traj.wz.reserve(res);
    traj.qnorm.reserve(res); traj.T_rot.reserve(res);
    traj.thrust_r.reserve(res); traj.thrust_phi.reserve(res); traj.pointing_err.reserve(res);
    traj.tidal_tau_x.reserve(res); traj.tidal_tau_y.reserve(res); traj.tidal_tau_z.reserve(res);
    traj.tidal_norm.reserve(res); traj.align_angle_rad.reserve(res); traj.tidal_E_norm.reserve(res);

    // Estado inicial
    KStateV y = KStateV::Zero();
    y[0] = r0; y[1] = phi0; y[2] = pr0;
    y[3] = E0; y[4] = L0;
    y[5] = engine.mass0_kg;
    {
        Vec4 q0v = att0.q;
        const double nq = q0v.norm();
        if (nq > 0.0) q0v /= nq;
        y.segment<4>(6) = q0v;
    }
    y.segment<3>(10) = att0.w;

    double tau    = tau0;
    double tcoord = 0.0;

    // Log de início para RIEMANN_FD
    if (att_cfg.tidal.enabled && att_cfg.tidal.model == TidalModel::RIEMANN_FD) {
        std::cout << "[KERR_6DOF] RIEMANN_FD activo: eps_r=" << att_cfg.tidal.fd_eps_r
                  << " n_steps=" << n_steps << std::endl;
    }

    // Lambda: gravar ponto
    auto push = [&]() {
        const double r_  = y[0], phi_= y[1], pr_ = y[2];
        const double E_  = y[3], L_  = y[4], m_  = y[5];
        const Vec4 q = y.segment<4>(6);
        const Vec3 w = y.segment<3>(10);

        traj.tau.push_back(tau);
        traj.r.push_back(r_); traj.phi.push_back(phi_); traj.pr.push_back(pr_);
        traj.E.push_back(E_); traj.L.push_back(L_); traj.mass.push_back(m_);

        // epsilon Kerr Carter
        const double K_ = L_ - a*E_;
        const double veff_ = (E_*E_-1.0)+2.0*M/r_-(L_*L_+a*a*(1.0-E_*E_))/(r_*r_)+2.0*M*K_*K_/(r_*r_*r_);
        traj.epsilon.push_back(pr_*pr_ - veff_);
        traj.tcoord.push_back(tcoord);

        traj.q0.push_back(q[0]); traj.q1.push_back(q[1]);
        traj.q2.push_back(q[2]); traj.q3.push_back(q[3]);
        traj.wx.push_back(w[0]); traj.wy.push_back(w[1]); traj.wz.push_back(w[2]);
        traj.qnorm.push_back(q.norm());
        traj.T_rot.push_back(att_cfg.inertia.T_rot(w));

        // Empuxo
        double tr = 0.0, tphi = 0.0;
        if (engine.active(tau)) {
            const Vec3 k_body = engine.nozzle_body.normalized();
            const Mat3 R      = dcm_from_quaternion(q);
            const Vec3 n      = R * k_body;
            const double ag   = engine.F_geom(m_);
            const double cp   = std::cos(phi_), sp = std::sin(phi_);
            const double Delta_ = lt_kerr_Delta(M, a, r_);
            const double Dsq_   = std::sqrt(std::max(Delta_, 0.0));
            tr   = ag * Dsq_/r_ * ( cp*n[0] + sp*n[1]);
            tphi = ag            * (-sp*n[0] + cp*n[1]);
        }
        traj.thrust_r.push_back(tr);
        traj.thrust_phi.push_back(tphi);
        traj.pointing_err.push_back(0.0);

        // Torque de maré
        const Vec3 tt = compute_tidal_torque(M, a, r_, q, att_cfg.inertia.I, att_cfg.tidal);
        traj.tidal_tau_x.push_back(tt[0]);
        traj.tidal_tau_y.push_back(tt[1]);
        traj.tidal_tau_z.push_back(tt[2]);
        traj.tidal_norm.push_back(tt.norm());

        // E_ij norma para diagnóstico
        double enorm = 0.0;
        if (att_cfg.tidal.enabled) {
            Mat3 Eloc;
            if (att_cfg.tidal.model == TidalModel::RIEMANN_FD) {
                Eloc = gr::tidal_riemann_fd(M, a, r_, att_cfg.tidal.fd_eps_r);
            } else {
                Eloc = gr::tidal_diag_weak(M, a, r_, att_cfg.tidal.spin_correction);
            }
            enorm = Eloc.norm();
        }
        traj.tidal_E_norm.push_back(enorm);

        // Ângulo de alinhamento: entre x̂_body e direcção radial no frame local
        const Mat3 Rmat = dcm_from_quaternion(q);
        const Vec3 x_body_in_local = Rmat.col(0);  // coluna 0 = eixo x_body em local
        const Vec3 n_radial(1.0, 0.0, 0.0);
        const double cosA = x_body_in_local.dot(n_radial);
        traj.align_angle_rad.push_back(std::acos(std::max(-1.0, std::min(1.0, cosA))));
    };

    if (!(r0 > r_hor)) {
        traj.status  = OrbitStatus::ERROR;
        traj.message = "r0 inside horizon";
        return traj;
    }
    push();

    for (int step = 0; step < n_steps; ++step) {
        double h = cfg.dt;
        if (tau + h > tauf) h = tauf - tau;
        if (!(h > 0.0)) break;

        // ── RK4 ────────────────────────────────────────────────
        const KStateV k1 = rhs_kerr_6dof(tau,         y,             M, a, engine, att_cfg, Iinv);
        const KStateV k2 = rhs_kerr_6dof(tau + 0.5*h, y + 0.5*h*k1, M, a, engine, att_cfg, Iinv);
        const KStateV k3 = rhs_kerr_6dof(tau + 0.5*h, y + 0.5*h*k2, M, a, engine, att_cfg, Iinv);
        const KStateV k4 = rhs_kerr_6dof(tau + h,     y + h    *k3, M, a, engine, att_cfg, Iinv);

        KStateV y_new = y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

        // Actualizar tcoord (Euler barato)
        {
            const double Delta_ = lt_kerr_Delta(M, a, y[0]);
            const double Dsafe_ = std::max(Delta_, 1e-300);
            const double r2_    = y[0]*y[0];
            const double a2_    = a*a;
            const double ut_    = ((r2_+a2_+2.0*M*a2_/y[0])*y[3] - 2.0*M*a*y[4]/y[0]) / Dsafe_;
            tcoord += h * ut_;
        }

        tau += h;

        if (!y_new.allFinite()) {
            traj.status  = OrbitStatus::ERROR;
            traj.message = "non-finite state at tau=" + std::to_string(tau);
            break;
        }

        if (y_new[5] < 0.0) y_new[5] = 0.0;
        y = y_new;

        // Renormalização do quaternion
        const double nq      = y.segment<4>(6).norm();
        const double delta_n = std::abs(nq - 1.0);
        const bool tol_trig  = (renorm_tol > 0.0 && delta_n > renorm_tol);
        const bool step_trig = ((step + 1) % renorm_every == 0);
        if ((step_trig || tol_trig) && nq > 0.0)
            y.segment<4>(6) /= nq;

        if ((step + 1) % rec == 0 || step == n_steps - 1 || tau >= tauf)
            push();

        // Captura / horizonte
        if (y[0] <= r_cap) {
            traj.status  = OrbitStatus::CAPTURE;
            traj.message = "captured at r=" + std::to_string(y[0]);
            return traj;
        }
        if (y[0] <= r_hor) {
            traj.status  = OrbitStatus::CAPTURE;
            traj.message = "horizon crossed";
            return traj;
        }

        if (tau >= tauf) break;
    }

    if (traj.status == OrbitStatus::ERROR && traj.message.empty()) {
        traj.status  = OrbitStatus::BOUND;
        traj.message = "integration complete";
    }

    // ── FIX-D v2: Log de convergência RIEMANN_FD — passo diagnóstico independente ─
    //
    // Problema anterior v1: usava eps_prod, eps_prod/2, eps_prod/4.
    //   Para eps_prod = 1e-5, estes valores já estão ABAIXO do piso de
    //   precisão dupla para ∂_r Γ (piso ~ 1e-11 relativo em r=10M).
    //   Resultado: err(eps/2) = 0 ou ratio ≈ 2 espúrio, sem física.
    //
    // Solução: usar passo de diagnóstico h_diag >> eps_prod mas ainda em
    //   regime O(h²):  h_diag = r * 0.05  [~50x optimal para eps_prod=1e-5]
    //   Convergência verificada numericamente: ratio ~4 para h ∈ [r*0.005, r*0.2].
    //
    // Três resoluções: h, h/2, h/4 — reference = E(h/4).
    //   err_c = ||E(h) − E(h/4)||_F / ||E(h/4)||_F     ← "passo grosso"
    //   err_f = ||E(h/2) − E(h/4)||_F / ||E(h/4)||_F   ← "passo fino"
    //   ratio = err_c / err_f  →  5.0 para FD centrado O(h²) 
    //     (nota: com referência E(h/4): ratio analítico = (1−1/16)/(1/4−1/16) = 5,
    //      não 4; valor observado tipicamente 4.8–5.2 em regime limpo)
    //
    // Testamos em dois raios: r0 e 50M.
    if (att_cfg.tidal.enabled && att_cfg.tidal.model == TidalModel::RIEMANN_FD) {
        const double eps_prod = att_cfg.tidal.fd_eps_r;
        const std::vector<double> r_tests = {r0, 50.0 * M};
        for (double r_test : r_tests) {
            if (r_test <= r_plus * 1.05) continue;  // evitar horizonte

            // Passo de diagnóstico: 5% de r, min 0.01M (sempre em regime O(h²))
            const double h_diag = std::max(r_test * 0.05, 0.01 * M);

            // Piso de precisão dupla: se err < fp_floor, reportar "(FP floor)"
            const double fp_floor = 1e-12;  // relativo, conservador

            const Mat3 Eh1 = gr::tidal_riemann_fd(M, a, r_test, h_diag);
            const Mat3 Eh2 = gr::tidal_riemann_fd(M, a, r_test, h_diag * 0.5);
            const Mat3 Eh4 = gr::tidal_riemann_fd(M, a, r_test, h_diag * 0.25);  // ref
            const Mat3 Ewf = gr::tidal_diag_weak(M, a, r_test, false);

            const double ref      = std::max(Eh4.norm(), 1e-30);
            const double err_c    = (Eh1 - Eh4).norm() / ref;
            const double err_f    = (Eh2 - Eh4).norm() / ref;
            const double ratio    = (err_f > fp_floor) ? err_c / err_f : 0.0;
            // GR correction at PRODUCTION eps vs weak-field (physical, ≠ 0 even for exact FD)
            const Mat3 Eprod = gr::tidal_riemann_fd(M, a, r_test, eps_prod);
            const double gr_corr = (Eprod - Ewf).norm() / std::max(Ewf.norm(), 1e-30);

            std::cout << "[KERR_6DOF][FD_CONV] r=" << r_test/M << "M"
                      << "  h_diag=" << h_diag
                      << "  err(h)="    << err_c
                      << "  err(h/2)="  << err_f;
            if (err_f < fp_floor) {
                std::cout << "  ratio=FLOOR(piso FP atingido; usar h_diag maior)";
            } else {
                std::cout << "  ratio=" << ratio
                          << " (alvo ~5 para ref=E(h/4); ~4 para ref=exacto)";
            }
            std::cout << "  GR_vs_wf(prod_eps)=" << gr_corr
                      << std::endl;
        }
    }

    return traj;
}

} // namespace relorbit