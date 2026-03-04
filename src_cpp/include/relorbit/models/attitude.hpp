// src_cpp/include/relorbit/models/attitude.hpp
//
// Módulo C — Item 7: Dinâmica de Atitude 6-DOF com Quaternions
//
// ═══════════════════════════════════════════════════════════════
// ESTADO: y = (q0, q1, q2, q3, ωx, ωy, ωz)   dim = 7
//
//  q = [q0, qv]  quaternion unitário  (q0: escalar, qv: vetor)
//  ω = [ωx, ωy, ωz]  velocidade angular no body frame  [rad/s]
//
// ── CINEMÁTICA (equação do quaternion) ──────────────────────────
//
//  q̇ = ½ Ω(ω) q
//
//  Ω(ω) =  ⎡  0  -ωx -ωy -ωz ⎤
//           ⎢ ωx   0   ωz -ωy ⎥
//           ⎢ ωy  -ωz   0   ωx⎥
//           ⎣ ωz   ωy -ωx   0 ⎦
//
//  Convenção: q leva vetores do body frame para o inercial.
//
// ── DINÂMICA (Euler do corpo rígido, body frame) ─────────────────
//
//  I ω̇ + ω × (I ω) = τ
//  => ω̇ = I⁻¹ [ τ − ω × (I ω) ]
//
//  I: tensor de inércia 3×3, constante no body frame.
//  τ: torque externo no body frame [N·m (SI) ou adimensional].
//
// ── CRITÉRIOS DE VALIDAÇÃO ───────────────────────────────────────
//
//  (a) Norma do quaternion: ‖q‖ = 1
//      Renormalização controlada a cada passo (ou quando ‖q‖−1 > ε).
//
//  (b) Energia cinética rotacional (sem torque):
//      T_rot = ½ ωᵀ I ω  = constante quando τ = 0.
//
// ═══════════════════════════════════════════════════════════════

#pragma once
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace relorbit {

// ───────────────────────────────────────────────────────────────
// AttitudeCfg — parâmetros de integração e renormalização
// ───────────────────────────────────────────────────────────────
struct AttitudeCfg {
    double dt           = 0.01;   // passo de integração [s ou adim.]
    int    n_steps      = 0;      // 0 => calculado a partir de (tf−t0)/dt
    int    record_every = 1;      // gravar a cada N passos

    // Renormalização do quaternion:
    //   renorm_every > 0  => forçar a cada N passos
    //   renorm_tol   > 0  => forçar quando | ‖q‖ − 1 | > tol
    // Ambos podem estar activos simultaneamente.
    int    renorm_every = 1;      // renormalizar em todo passo (default seguro)
    double renorm_tol   = 1e-9;   // tolerância alternativa
};

// ───────────────────────────────────────────────────────────────
// TorqueCfg — torque externo simples (constante por janela de tempo)
// ───────────────────────────────────────────────────────────────
struct TorqueCfg {
    double tx     = 0.0;    // componente x do torque no body frame
    double ty     = 0.0;    // componente y
    double tz     = 0.0;    // componente z
    double t_on   = 0.0;    // início da aplicação
    double t_off  = 1e18;   // fim da aplicação

    bool   active(double t) const { return t >= t_on && t <= t_off; }
    double get_x (double t) const { return active(t) ? tx : 0.0; }
    double get_y (double t) const { return active(t) ? ty : 0.0; }
    double get_z (double t) const { return active(t) ? tz : 0.0; }
};

// ───────────────────────────────────────────────────────────────
// AttitudeState — estado completo num instante
// ───────────────────────────────────────────────────────────────
struct AttitudeState {
    double q0, q1, q2, q3;   // quaternion  (escalar + vetor)
    double wx, wy, wz;        // velocidade angular no body frame

    // Normaliza o quaternion in-place; devolve a norma anterior.
    double renormalize() {
        const double n = std::sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
        if (n > 0.0) { q0 /= n; q1 /= n; q2 /= n; q3 /= n; }
        return n;
    }

    double qnorm() const {
        return std::sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    }
};

// ───────────────────────────────────────────────────────────────
// TrajectoryAttitude — resultado da integração
// ───────────────────────────────────────────────────────────────
struct TrajectoryAttitude {
    // Séries temporais
    std::vector<double> t;             // tempo
    std::vector<double> q0, q1, q2, q3;   // quaternion
    std::vector<double> wx, wy, wz;    // velocidade angular
    std::vector<double> qnorm;         // ‖q‖  (deve ser ≈ 1)
    std::vector<double> T_rot;         // energia cinética rotacional ½ ωᵀ I ω
    std::vector<double> renorm_delta;  // | ‖q‖ − 1 | antes de renormalizar

    // Parâmetros guardados
    double Ixx = 1.0, Iyy = 1.0, Izz = 1.0;  // momentos principais de inércia
    double Ixy = 0.0, Ixz = 0.0, Iyz = 0.0;  // termos fora da diagonal

    std::string status;   // "OK" | "ERROR"
    std::string message;
};

// ───────────────────────────────────────────────────────────────
// InertiaTensor — wrapper para o tensor 3×3 simétrico no body frame
// ───────────────────────────────────────────────────────────────
//  Convenção de armazenamento (row-major, 0-indexed):
//    I[0]=Ixx  I[1]=Ixy  I[2]=Ixz
//    I[3]=Ixy  I[4]=Iyy  I[5]=Iyz
//    I[6]=Ixz  I[7]=Iyz  I[8]=Izz
// ───────────────────────────────────────────────────────────────
struct InertiaTensor {
    std::array<double, 9> I = {1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0};

    // Constrói tensor diagonal (corpo com eixos principais alinhados ao body)
    static InertiaTensor diagonal(double Ixx, double Iyy, double Izz) {
        InertiaTensor it;
        it.I = {Ixx,  0.0,  0.0,
                0.0,  Iyy,  0.0,
                0.0,  0.0,  Izz};
        return it;
    }

    // Constrói tensor completo simétrico
    static InertiaTensor full(double Ixx, double Iyy, double Izz,
                               double Ixy, double Ixz, double Iyz) {
        InertiaTensor it;
        it.I = {Ixx, Ixy, Ixz,
                Ixy, Iyy, Iyz,
                Ixz, Iyz, Izz};
        return it;
    }

    // Produto  v_out = I * v
    std::array<double,3> mul(double vx, double vy, double vz) const {
        return { I[0]*vx + I[1]*vy + I[2]*vz,
                 I[3]*vx + I[4]*vy + I[5]*vz,
                 I[6]*vx + I[7]*vy + I[8]*vz };
    }

    // Energia cinética rotacional  T = ½ ωᵀ I ω
    double T_rot(double wx, double wy, double wz) const {
        auto Iw = mul(wx, wy, wz);
        return 0.5 * (wx*Iw[0] + wy*Iw[1] + wz*Iw[2]);
    }

    // Inversa analítica do tensor 3×3 simétrico.
    // Devolve false se o determinante for degenerado (|det| < tol).
    bool invert(std::array<double,9>& Iinv, double tol = 1e-18) const {
        // cofactores
        const double c00 = I[4]*I[8] - I[5]*I[7];
        const double c01 = I[5]*I[6] - I[3]*I[8];
        const double c02 = I[3]*I[7] - I[4]*I[6];
        const double det = I[0]*c00 + I[1]*c01 + I[2]*c02;
        if (std::abs(det) < tol) return false;
        const double inv_det = 1.0 / det;
        Iinv[0] = c00 * inv_det;
        Iinv[1] = (I[2]*I[7] - I[1]*I[8]) * inv_det;
        Iinv[2] = (I[1]*I[5] - I[2]*I[4]) * inv_det;
        Iinv[3] = c01 * inv_det;
        Iinv[4] = (I[0]*I[8] - I[2]*I[6]) * inv_det;
        Iinv[5] = (I[2]*I[3] - I[0]*I[5]) * inv_det;
        Iinv[6] = c02 * inv_det;
        Iinv[7] = (I[1]*I[6] - I[0]*I[7]) * inv_det;
        Iinv[8] = (I[0]*I[4] - I[1]*I[3]) * inv_det;
        return true;
    }
};

// ───────────────────────────────────────────────────────────────
// DCM a partir do quaternion (body → inercial)
//
//   R(q) = (q0²−‖qv‖²) I₃  +  2 qv qvᵀ  +  2 q0 [qv]×
//
//   Devolve array row-major 9 elementos.
// ───────────────────────────────────────────────────────────────
inline std::array<double, 9> dcm_from_quaternion(
    double q0, double q1, double q2, double q3)
{
    const double s  = q0*q0 - q1*q1 - q2*q2 - q3*q3;
    std::array<double,9> R;
    R[0] = s + 2.0*q1*q1;     R[1] = 2.0*(q1*q2 - q0*q3); R[2] = 2.0*(q1*q3 + q0*q2);
    R[3] = 2.0*(q1*q2 + q0*q3); R[4] = s + 2.0*q2*q2;     R[5] = 2.0*(q2*q3 - q0*q1);
    R[6] = 2.0*(q1*q3 - q0*q2); R[7] = 2.0*(q2*q3 + q0*q1); R[8] = s + 2.0*q3*q3;
    return R;
}

// ───────────────────────────────────────────────────────────────
// Função principal de integração
// ───────────────────────────────────────────────────────────────
TrajectoryAttitude simulate_attitude_rk4(
    const AttitudeState&  state0,     // estado inicial (q, ω)
    const InertiaTensor&  inertia,    // tensor de inércia no body frame
    const TorqueCfg&      torque,     // torque externo
    double t0, double tf,             // intervalo de tempo
    const AttitudeCfg&    cfg         // configuração do integrador
);

} // namespace relorbit