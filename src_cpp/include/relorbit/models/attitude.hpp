// src_cpp/include/relorbit/models/attitude.hpp
//
// Módulo C — Item 7: Dinâmica de Atitude 6-DOF com Quaternions
//
// ═══════════════════════════════════════════════════════════════
// ESTADO: y ∈ ℝ⁷ = (q0, q1, q2, q3, ωx, ωy, ωz)
//
//  q = [q0, qv]  quaternion unitário  (q0: escalar, qv: vector)
//  ω  velocidade angular no body frame  [rad/s]
//
// ── CINEMÁTICA ───────────────────────────────────────────────────
//
//  q̇ = ½ Ω(ω) q
//
//  Ω(ω) =  |  0  -wx -wy -wz |
//           | wx   0   wz -wy |
//           | wy  -wz   0   wx|
//           | wz   wy -wx   0 |
//
//  Convenção: q leva vectores do body frame para o inercial.
//
// ── DINÂMICA (Euler, body frame) ─────────────────────────────────
//
//  I w' + w x (I w) = tau
//  =>  w' = I^{-1} [ tau - w x (I w) ]
//
//  I  : Eigen::Matrix3d simétrico positivo-definido (constante, body frame)
//  tau: torque externo [N·m]
//
// ── CRITÉRIOS DE VALIDAÇÃO ───────────────────────────────────────
//
//  (a)  ||q|| = 1        via renormalização controlada
//  (b)  T_rot constante  T = 1/2 w^T I w  quando tau = 0
//
// ═══════════════════════════════════════════════════════════════

#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <string>
#include <vector>

namespace relorbit {

// ── Aliases ──────────────────────────────────────────────────────
using Vec3   = Eigen::Vector3d;
using Vec4   = Eigen::Vector4d;
using Mat3   = Eigen::Matrix3d;
using State7 = Eigen::Matrix<double, 7, 1>;   // [q0 q1 q2 q3 wx wy wz]


// ── AttitudeCfg ──────────────────────────────────────────────────
struct AttitudeCfg {
    double dt           = 0.01;
    int    n_steps      = 0;        // 0 => calculado internamente
    int    record_every = 1;
    int    renorm_every = 1;
    double renorm_tol   = 1e-9;
};


// ── TorqueCfg ────────────────────────────────────────────────────
struct TorqueCfg {
    Vec3   tau   = Vec3::Zero();
    double t_on  = 0.0;
    double t_off = 1e18;

    bool active(double t)  const { return t >= t_on && t <= t_off; }
    Vec3 get(double t)     const { return active(t) ? tau : Vec3::Zero(); }

    // Acessores escalares (compatibilidade pybind / YAML)
    double get_x(double t) const { return active(t) ? tau.x() : 0.0; }
    double get_y(double t) const { return active(t) ? tau.y() : 0.0; }
    double get_z(double t) const { return active(t) ? tau.z() : 0.0; }

    double tx() const { return tau.x(); }
    double ty() const { return tau.y(); }
    double tz() const { return tau.z(); }
    void set_tx(double v) { tau.x() = v; }
    void set_ty(double v) { tau.y() = v; }
    void set_tz(double v) { tau.z() = v; }
};


// ── AttitudeState ────────────────────────────────────────────────
struct AttitudeState {
    Vec4 q = Vec4(1.0, 0.0, 0.0, 0.0);   // [q0, q1, q2, q3]
    Vec3 w = Vec3::Zero();                 // velocidade angular

    // Acessores escalares (compatibilidade pybind / YAML)
    double q0() const { return q[0]; }  void set_q0(double v) { q[0] = v; }
    double q1() const { return q[1]; }  void set_q1(double v) { q[1] = v; }
    double q2() const { return q[2]; }  void set_q2(double v) { q[2] = v; }
    double q3() const { return q[3]; }  void set_q3(double v) { q[3] = v; }
    double wx() const { return w[0]; }  void set_wx(double v) { w[0] = v; }
    double wy() const { return w[1]; }  void set_wy(double v) { w[1] = v; }
    double wz() const { return w[2]; }  void set_wz(double v) { w[2] = v; }

    double renormalize() {
        const double n = q.norm();
        if (n > 0.0) q /= n;
        return n;
    }
    double qnorm() const { return q.norm(); }
};


// ── TrajectoryAttitude ───────────────────────────────────────────
struct TrajectoryAttitude {
    std::vector<double> t;
    std::vector<double> q0, q1, q2, q3;
    std::vector<double> wx, wy, wz;
    std::vector<double> qnorm;
    std::vector<double> T_rot;
    std::vector<double> renorm_delta;

    double Ixx = 1.0, Iyy = 1.0, Izz = 1.0;
    double Ixy = 0.0, Ixz = 0.0, Iyz = 0.0;

    std::string status;
    std::string message;
};


// ── InertiaTensor ────────────────────────────────────────────────
//
// Wrapper sobre Eigen::Matrix3d.
// A inversa usa Cholesky (LLT) para tensores SPD,
// com fallback para FullPivLU em casos degenerados.
struct InertiaTensor {
    Mat3 I = Mat3::Identity();

    static InertiaTensor diagonal(double Ixx, double Iyy, double Izz) {
        InertiaTensor it;
        it.I = Mat3::Zero();
        it.I(0,0) = Ixx;
        it.I(1,1) = Iyy;
        it.I(2,2) = Izz;
        return it;
    }

    static InertiaTensor full(double Ixx, double Iyy, double Izz,
                               double Ixy, double Ixz, double Iyz) {
        InertiaTensor it;
        it.I << Ixx, Ixy, Ixz,
                Ixy, Iyy, Iyz,
                Ixz, Iyz, Izz;
        return it;
    }

    // I w  (produto matriz-vector — Eigen vectorizado)
    Vec3 mul(const Vec3& w) const { return I * w; }

    // T_rot = 1/2 w^T I w
    double T_rot(const Vec3& w) const { return 0.5 * w.dot(I * w); }

    // Inversa: Cholesky (SPD) ou LU (fallback)
    bool invert(Mat3& Iinv) const {
        Eigen::LLT<Mat3> llt(I);
        if (llt.info() == Eigen::Success) {
            Iinv = llt.solve(Mat3::Identity());
            return true;
        }
        Eigen::FullPivLU<Mat3> lu(I);
        if (!lu.isInvertible()) return false;
        Iinv = lu.inverse();
        return true;
    }

    // Acesso por índice (compatibilidade pybind)
    double coeff(int i, int j) const { return I(i, j); }
};


// ── DCM a partir do quaternion (body → inercial) ─────────────────
//
// Delega para Eigen::Quaterniond::toRotationMatrix().
inline Mat3 dcm_from_quaternion(double q0, double q1, double q2, double q3) {
    return Eigen::Quaterniond(q0, q1, q2, q3).normalized().toRotationMatrix();
}

inline Mat3 dcm_from_quaternion(const Vec4& q) {
    return dcm_from_quaternion(q[0], q[1], q[2], q[3]);
}


// ── Função principal de integração ──────────────────────────────
TrajectoryAttitude simulate_attitude_rk4(
    const AttitudeState&  state0,
    const InertiaTensor&  inertia,
    const TorqueCfg&      torque,
    double t0, double tf,
    const AttitudeCfg&    cfg
);

} // namespace relorbit