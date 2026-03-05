// src_cpp/include/relorbit/models/schwarzschild_6dof.hpp
//
// Item 8 — Thrust Vectoring: Acoplamento 6-DOF Órbita + Atitude
//
// ═══════════════════════════════════════════════════════════════
// MODELO: Schwarzschild equatorial + corpo rígido (quaternion)
//
// ESTADO: y ∈ ℝ¹⁴
//   idx   campo       descrição
//   0     r           coordenada radial  [M]
//   1     phi         ângulo azimutal    [rad]
//   2     pr          momento radial     dr/dτ
//   3     E           energia específica (varia com empuxo)
//   4     L           momento angular específico (varia com empuxo)
//   5     m_kg        massa da nave      [kg]
//   6     q0          quaternion escalar
//   7     q1          quaternion vec x
//   8     q2          quaternion vec y
//   9     q3          quaternion vec z
//   10    wx          veloc. angular x   [rad/s]
//   11    wy          veloc. angular y
//   12    wz          veloc. angular z
//   (13 reservado / padding)
//
// ── ACOPLAMENTO ──────────────────────────────────────────────────
//
//   A direcção do empuxo no frame inercial é:
//     n̂_inercial = R(q) · k̂_body
//
//   onde k̂_body é o eixo do bocal no body frame (por omissão ẑ = [0,0,1]).
//
//   Componentes de coordenada em Schwarzschild equatorial:
//     f^r   = a_geom · n_r                           [unidades geométricas]
//     f^phi = a_geom · n_phi / r
//     f^t   determinado por ortogonalidade f^μ u_μ = 0
//
//   onde n_r, n_phi são as projecções de n̂_inercial nos eixos (r, φ).
//
//   Dinâmica dos conservados:
//     dL/dτ = r² · f^phi  = r · a_geom · n_phi
//     dE/dτ = A(r)⁻¹ · E · f^t       (A = 1 − 2M/r)
//     dm/dτ = −F_newton / (Isp · g₀)
//
//   Dinâmica de atitude (acoplada de volta):
//     o torque de reacção do motor actua sobre ω — configurável via
//     `engine_torque_body` (por omissão zero: motor alinhado com CM).
//
// ── CRITÉRIOS DE VALIDAÇÃO ───────────────────────────────────────
//
//   (a) ‖q‖ = 1  — renormalização por passo
//   (b) ε = pr² + V_eff − E²  ≈ 0  quando motor desligado
//   (c) T_rot constante quando torques = 0
//
// ═══════════════════════════════════════════════════════════════

#pragma once

#include <Eigen/Dense>
#include "relorbit/models/attitude.hpp"
#include "relorbit/models/schwarzschild_equatorial.hpp"
#include "relorbit/types.hpp"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace relorbit {

// ── Constantes físicas ───────────────────────────────────────────
static constexpr double DOF6_C_MS = 2.99792458e8;   // [m/s]
static constexpr double DOF6_G0   = 9.80665;         // [m/s²]


// ── EngineCfg ────────────────────────────────────────────────────
//
// Configuração do motor de empuxo vectorial.
//
// F_newton : magnitude do empuxo  [N]
// isp_s    : impulso específico   [s]
// nozzle_body: direcção do bocal no body frame (normalizada internamente)
// tau_on/off : janela de activação
// torque_reaction: torque de reacção do motor no body frame [N·m]
//
// ── NOTA DE ESCALA (bug histórico) ──────────────────────────────
//
// F_geom = F/(m·c²) usa c_SI = 3×10⁸ m/s.  O integrador opera em
// unidades geométricas (c=G=1), logo a_geom fica ~9×10¹⁶ vezes menor
// do que o valor fisicamente correcto para c=1.
//
// Solução: definir a_geom_override > 0 para bypassa F/(m·c²):
//   a_geom_override = F_newton / mass0_kg   [c=1, unidades geom]
//
// Quando a_geom_override > 0:
//   F_geom()  devolve a_geom_override (independente de m_kg)
//   active()  dispara mesmo quando F_newton = 0
//
// Exemplo para validação (c=1):
//   engine.F_newton        = 30.0;
//   engine.a_geom_override = 30.0 / 1000.0;  // = 0.03 M⁻¹
//
struct EngineCfg {
    double F_newton  = 0.0;                     // empuxo total [N]
    double isp_s     = 3000.0;                  // Isp [s]
    double tau_on    = 0.0;
    double tau_off   = 1e18;

    // Eixo do bocal no body frame (normalizado ao usar)
    Vec3 nozzle_body = Vec3(0.0, 0.0, 1.0);    // default: eixo +z

    // Torque de reacção do motor no body frame
    Vec3 torque_reaction = Vec3::Zero();

    double mass0_kg    = 1000.0;   // massa inicial da nave [kg]
    double dry_mass_kg = 300.0;    // massa seca [kg]

    // a_geom_override > 0: bypassa F/(m·c²).
    // Usar quando o integrador opera em unidades geométricas c=1.
    // Valor típico: F_newton / mass0_kg  (aceleração específica c=1).
    // Quando 0 (default): usa F_geom = F/(m·c²) (comportamento original).
    double a_geom_override = 0.0;

    // active() = true se tau está dentro da janela E se há força efectiva.
    // Considera a_geom_override além de F_newton para não silenciar o motor
    // quando F_newton é mantido só para dm/dτ mas a_geom vem do override.
    bool active(double tau) const {
        return tau >= tau_on && tau <= tau_off
               && (F_newton > 0.0 || a_geom_override > 0.0);
    }

    // F_geom: aceleração específica [M⁻¹] em unidades geométricas.
    // Se a_geom_override > 0, ignora c_SI e devolve o override directamente.
    // Caso contrário: a = F / (m · c²)  [fórmula SI — correcta apenas se
    // o integrador usa c_SI internamente, incorrecto para c=1 puro].
    double F_geom(double m_kg) const {
        if (a_geom_override > 0.0) return a_geom_override;
        return (m_kg > 0.0) ? (F_newton / (m_kg * DOF6_C_MS * DOF6_C_MS)) : 0.0;
    }
};


// ── AttitudeCfg6DOF ──────────────────────────────────────────────
//
// Reutiliza InertiaTensor e TorqueCfg do módulo de atitude.
// Acrescenta renorm_every e renorm_tol.
struct AttitudeCfg6DOF {
    InertiaTensor inertia;
    TorqueCfg     ext_torque;   // torque externo (reacção, perturbações)
    int    renorm_every = 1;
    double renorm_tol   = 1e-9;
};


// ── TrajectoryCoupled ────────────────────────────────────────────
struct TrajectoryCoupled {
    // Órbita
    std::vector<double> tau, r, phi, pr;
    std::vector<double> E, L, mass;
    std::vector<double> epsilon;
    std::vector<double> tcoord;     // t coordenada (integrada)

    // Atitude
    std::vector<double> q0, q1, q2, q3;
    std::vector<double> wx, wy, wz;
    std::vector<double> qnorm;
    std::vector<double> T_rot;

    // Empuxo
    std::vector<double> thrust_r;       // componente radial do accel. [geom]
    std::vector<double> thrust_phi;     // componente tangencial
    std::vector<double> pointing_err;   // |n̂_actual − n̂_target| [rad, 0 se sem alvo]

    // Meta
    double M   = 1.0;
    double r0  = 0.0;
    double phi0 = 0.0;
    double E0  = 0.0;
    double L0  = 0.0;

    OrbitStatus status  = OrbitStatus::ERROR;
    std::string message;
};


// ── SolverCfg6DOF ────────────────────────────────────────────────
struct SolverCfg6DOF {
    double dt           = 0.005;
    int    n_steps      = 0;
    int    record_every = 10;
    int    renorm_every = 1;
    double renorm_tol   = 1e-9;
    double capture_r    = 2.0;
    double capture_eps  = 1e-12;
};


// ── Função principal de integração ──────────────────────────────
TrajectoryCoupled simulate_schwarzschild_6dof_rk4(
    double M,
    double E0, double L0,
    double r0, double phi0, double pr0,
    const AttitudeState&   att0,
    double tau0, double tauf,
    const EngineCfg&       engine,
    const AttitudeCfg6DOF& att_cfg,
    const SolverCfg6DOF&   cfg
);

} // namespace relorbit