// src_cpp/include/relorbit/models/kerr_6dof.hpp
//
// Kerr 6-DOF acoplado: Órbita Kerr equatorial + Corpo rígido (quaternion) + Torque de maré
//
// ═══════════════════════════════════════════════════════════════
// MODELO:
//   Órbita Kerr Boyer-Lindquist equatorial (frame ZAMO para empuxo)
//   + dinâmica de atitude quaternion (corpo rígido)
//   + torque de maré: 3 níveis de fidelidade
//
// ESTADO: y ∈ ℝ¹⁴
//   idx   campo       descrição
//   0     r           coordenada radial  [M]
//   1     phi         ângulo azimutal    [rad]
//   2     pr          dr/dτ  (momento radial)
//   3     E           energia específica (covariante, varia com empuxo)
//   4     L           momento angular específico
//   5     m_kg        massa da nave      [kg]
//   6..9  q0,q1,q2,q3 quaternion (body→inercial ZAMO local)
//   10..12 wx,wy,wz   velocidade angular body frame [rad/s]
//   13    (reservado)
//
// TORQUE DE MARÉ (TidalModel):
//
//   WEAK_N:
//     τ = 3(M/r³) n_body × (I n_body)
//     n_body = Rᵀ(q) [1,0,0]ᵀ  (direcção radial no body frame)
//
//   DIAG_EIJ:
//     E_local = diag(−2M/r³, M/r³, M/r³)  (+ correcção de spin opcional)
//     E_body = Rᵀ E_local R
//     Q_body = I − (trI/3) I₃
//     τ = −axial(Q E − E Q)
//
//   RIEMANN_FD:
//     E_ij calculado por Riemann FD numérico em r  (modo monstro)
//     Projectado na tetrada ZAMO → E_body → τ via quadrupolo
//
// ═══════════════════════════════════════════════════════════════
#pragma once

#include <Eigen/Dense>
#include "relorbit/models/attitude.hpp"
#include "relorbit/models/schwarzschild_6dof.hpp"   // reutiliza EngineCfg, SolverCfg6DOF
#include "relorbit/models/kerr_lowthrust.hpp"        // helpers Kerr
#include "relorbit/gr/kerr_metric.hpp"
#include "relorbit/types.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace relorbit {

// ── Enum: modelo de maré ──────────────────────────────────────────
enum class TidalModel {
    NONE       = 0,   ///< desligado
    WEAK_N     = 1,   ///< gravity-gradient clássico (cross product)
    DIAG_EIJ   = 2,   ///< E_ij diagonal (campo fraco/analítico) + quadrupolo
    RIEMANN_FD = 3    ///< E_ij via Riemann numérico FD (modo monstro)
};

// ── TidalCfg ─────────────────────────────────────────────────────
struct TidalCfg {
    bool        enabled         = false;
    TidalModel  model           = TidalModel::WEAK_N;
    double      fd_eps_r        = 1e-5;   ///< passo FD para RIEMANN_FD
    bool        Q_from_inertia  = true;   ///< deriva Q do tensor de inércia
    Mat3        Q_body          = Mat3::Zero();  ///< quadrupolo custom (se não Q_from_inertia)
    bool        spin_correction = false;  ///< correcção de spin no DIAG_EIJ
};

// ── AttitudeCfg6DOF_Kerr ─────────────────────────────────────────
// Extende AttitudeCfg6DOF com TidalCfg
struct AttitudeCfgKerr {
    InertiaTensor inertia;
    TorqueCfg     ext_torque;
    int           renorm_every = 1;
    double        renorm_tol   = 1e-9;
    TidalCfg      tidal;
};

// ── TrajectoryCoupledKerr ─────────────────────────────────────────
struct TrajectoryCoupledKerr {
    // Órbita
    std::vector<double> tau, r, phi, pr;
    std::vector<double> E, L, mass;
    std::vector<double> epsilon;
    std::vector<double> tcoord;

    // Atitude
    std::vector<double> q0, q1, q2, q3;
    std::vector<double> wx, wy, wz;
    std::vector<double> qnorm;
    std::vector<double> T_rot;

    // Empuxo
    std::vector<double> thrust_r, thrust_phi;
    std::vector<double> pointing_err;

    // Torque de maré
    std::vector<double> tidal_tau_x, tidal_tau_y, tidal_tau_z;
    std::vector<double> tidal_norm;
    std::vector<double> align_angle_rad;   ///< ângulo entre eixo body_x e direcção radial
    std::vector<double> tidal_E_norm;      ///< ||E_ij|| de Frobenius (diagnóstico)

    // Meta
    double M   = 1.0;
    double a   = 0.0;
    double r0  = 0.0;
    double phi0 = 0.0;
    double E0  = 0.0;
    double L0  = 0.0;

    OrbitStatus status  = OrbitStatus::ERROR;
    std::string message;
};

// ── Função principal de integração ──────────────────────────────
TrajectoryCoupledKerr simulate_kerr_6dof_rk4(
    double M, double a,
    double E0, double L0,
    double r0, double phi0, double pr0,
    const AttitudeState&    att0,
    double tau0, double tauf,
    const EngineCfg&        engine,
    const AttitudeCfgKerr&  att_cfg,
    const SolverCfg6DOF&    cfg
);

} // namespace relorbit