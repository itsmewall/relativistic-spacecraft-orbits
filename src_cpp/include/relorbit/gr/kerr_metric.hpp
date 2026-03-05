// src_cpp/include/relorbit/gr/kerr_metric.hpp
//
// Helpers de métrica Kerr em Boyer-Lindquist EQUATORIAL (θ = π/2)
// ─────────────────────────────────────────────────────────────────
// Convenção de sinal: (−,+,+,+)
// Coordenadas: x^μ = (t, r, θ, φ)  →  aqui θ=π/2 fixo, sin θ = 1
//
// Tetrada ZAMO (Zero Angular Momentum Observer) ortonormal:
//   e_(t̂)^μ = (1/√|g_tt^ZAMO|, 0, 0, 0)   com g_tt^ZAMO = g_tt − g_tφ²/g_φφ
//   e_(r̂)^μ = (0, 1/√g_rr, 0, 0)
//   e_(φ̂)^μ = (0, 0, 0,  1/√g_φφ)  — já inclui arrasto de quadro via Ω_ZAMO
//   e_(θ̂)^μ = (0, 0, 1/r, 0)         (equatorial: g_θθ = r²)
//
// Componentes de E_ij (tensor de maré) na tetrada ZAMO:
//   E_{îĵ} = R_{î t̂ ĵ t̂}
// ─────────────────────────────────────────────────────────────────
#pragma once
#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace relorbit {
namespace gr {

using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Vec3 = Eigen::Vector3d;

// ─── Estruturas de métrica ────────────────────────────────────────

/// Componentes covariantes da métrica Kerr equatorial g_{μν}
struct KerrMetricComp {
    double gtt, gtr, gtphi;     // gtr=0 no BL
    double grr, gthth, gphiphi;
    // Inversa
    double gtt_up, gtphi_up, grr_up, gthth_up, gphiphi_up;
};

/// Calcula g_{μν} e g^{μν} em (M,a,r) equatorial
inline KerrMetricComp kerr_metric(double M, double a, double r) {
    const double r2   = r * r;
    const double a2   = a * a;
    const double Delta = r2 - 2.0 * M * r + a2;
    const double Sigma = r2;              // equatorial: sin²θ=1
    const double rho2  = r2 + a2 + 2.0 * M * a2 / r;   // = g_φφ

    KerrMetricComp g{};
    g.gtt     = -(1.0 - 2.0 * M / r);
    g.gtr     = 0.0;
    g.gtphi   = -2.0 * M * a / r;
    g.grr     = r2 / Delta;
    g.gthth   = r2;                  // Sigma = r² equatorial
    g.gphiphi = rho2;

    // Inversa (2x2 bloco t-φ, mais g^rr, g^θθ independentes)
    const double det_tphi = g.gtt * g.gphiphi - g.gtphi * g.gtphi;
    g.gtt_up    =  g.gphiphi / det_tphi;
    g.gtphi_up  = -g.gtphi   / det_tphi;
    g.gphiphi_up =  g.gtt    / det_tphi;
    g.grr_up    = Delta / r2;
    g.gthth_up  = 1.0 / r2;

    return g;
}

// ─── Símbolos de Christoffel (componentes não-nulas no equatorial) ─
//
// Calculados via Γ^μ_{αβ} = ½ g^{μσ}(∂_α g_{βσ} + ∂_β g_{αρ} − ∂_σ g_{αβ})
// No equatorial só ∂_r contribui (campo estacionário e axialmente simétrico).
// ∂_θ → zero aqui (θ fixo).
//
// Saída: array 4×4×4 de double (Gamma[mu][alpha][beta])
using Gamma444 = std::array<std::array<std::array<double, 4>, 4>, 4>;

/// ∂_r g_{μν} analítico em Boyer-Lindquist equatorial
inline std::array<double, 6> dg_dr_kerr(double M, double a, double r) {
    // dg_tt/dr, dg_tphi/dr, dg_rr/dr, dg_thth/dr, dg_phiphi/dr
    const double r2   = r * r;
    const double a2   = a * a;
    const double Delta = r2 - 2.0 * M * r + a2;
    // g_tt = −1 + 2M/r  →  dg_tt/dr = −2M/r²
    const double dg_tt    = -2.0 * M / r2;
    // g_tφ = −2Ma/r    →  dg_tphi/dr = 2Ma/r²
    const double dg_tphi  =  2.0 * M * a / r2;
    // g_rr = r²/Δ      →  dg_rr/dr = (2rΔ − r²(2r−2M)) / Δ²
    const double dDelta   = 2.0 * r - 2.0 * M;
    const double dg_rr    = (2.0 * r * Delta - r2 * dDelta) / (Delta * Delta);
    // g_θθ = r²        →  dg_thth/dr = 2r
    const double dg_thth  = 2.0 * r;
    // g_φφ = r² + a² + 2Ma²/r  →  dg_phiphi/dr = 2r − 2Ma²/r²
    const double dg_phiphi = 2.0 * r - 2.0 * M * a2 / r2;

    return { dg_tt, dg_tphi, dg_rr, dg_thth, dg_phiphi, 0.0 };
}

/// Christoffel Γ^μ_{αβ} em (M,a,r) equatorial via derivada analítica ∂_r g
/// Retorna array[mu][alpha][beta] 4×4×4 (μ,α,β ∈ {t=0,r=1,θ=2,φ=3})
Gamma444 christoffel_kerr_eq(double M, double a, double r);

/// Derivada ∂_r Γ^μ_{αβ} por diferença finita centrada em r, passo eps
Gamma444 dGamma_dr_fd(double M, double a, double r, double eps);

/// Tensor de Riemann R^μ_{ναβ} por diferença finita
/// R^μ_{ναβ} = ∂_α Γ^μ_{βν} − ∂_β Γ^μ_{αν} + Γ^μ_{ασ}Γ^σ_{βν} − Γ^μ_{βσ}Γ^σ_{αν}
/// (só termos com derivadas em r contribuem no equatorial estacionário; outras ∂ → 0)
Mat4 riemann_Et0t_kerr_eq(double M, double a, double r, double eps_r);

// ─── Tetrada ZAMO ─────────────────────────────────────────────────
//
// Frame ortonormal (ĝ_{ab} = η_{ab} = diag(-1,+1,+1,+1)):
//   e_(t̂)^μ = (e^t̂_t, 0, 0, 0)        — observador ZAMO
//   e_(r̂)^μ = (0, e^r̂_r, 0, 0)
//   e_(θ̂)^μ = (0, 0, e^θ̂_θ, 0)
//   e_(φ̂)^μ = (0, e^φ̂_t, 0, e^φ̂_φ)   — inclui arrasto de quadro
//
// Índices espaciais da tetrada: 0=r̂, 1=θ̂, 2=φ̂  (ordem usada em E_ij)

struct ZAMOTetrad {
    // Componentes contra-variantes e_(a)^μ  para a = t̂,r̂,θ̂,φ̂
    double et_t;            // e_(t̂)^t
    double er_r;            // e_(r̂)^r
    double eth_th;          // e_(θ̂)^θ
    double ephi_t, ephi_phi; // e_(φ̂)^t, e_(φ̂)^φ
    double Omega_ZAMO;      // velocidade angular de arrasto
};

/// Calcula a tetrada ZAMO para Kerr equatorial
inline ZAMOTetrad zamo_tetrad(double M, double a, double r) {
    const auto g  = kerr_metric(M, a, r);
    const double r2  = r * r;
    const double a2  = a * a;
    const double Delta = r2 - 2.0 * M * r + a2;

    // Ω_ZAMO = −g_{tφ} / g_{φφ}
    const double Omega = -g.gtphi / g.gphiphi;

    // e_(t̂)^t = 1/√(−g_tt + g_{tφ}² / g_{φφ})
    //          = 1/√(−g_tt − Ω g_{tφ})
    const double alpha2 = -(g.gtt + Omega * g.gtphi); // lapse²
    const double alpha  = std::sqrt(std::max(alpha2, 1e-300));

    // e_(r̂)^r = 1/√g_rr = √(Δ)/r
    const double er_r  = std::sqrt(std::max(Delta, 0.0)) / r;

    // e_(θ̂)^θ = 1/√g_{θθ} = 1/r
    const double eth_th = 1.0 / r;

    // e_(φ̂)^φ = 1/√g_{φφ}
    const double ephi_phi = 1.0 / std::sqrt(std::max(g.gphiphi, 1e-300));

    // e_(φ̂)^t  = −Ω_ZAMO * e_(φ̂)^φ  (arrasto de quadro)
    const double ephi_t = -Omega * ephi_phi;

    ZAMOTetrad tet{};
    tet.et_t      = 1.0 / alpha;
    tet.er_r      = er_r;
    tet.eth_th    = eth_th;
    tet.ephi_t    = ephi_t;
    tet.ephi_phi  = ephi_phi;
    tet.Omega_ZAMO = Omega;
    return tet;
}

// ─── Tensor de maré E_ij (3×3, frame ZAMO espacial) ─────────────
//
// E_{îĵ} = R_{î 0̂ ĵ 0̂}  (projecção na tetrada, 0̂ = t̂)
// Índices î,ĵ ∈ {r̂=0, θ̂=1, φ̂=2}
//
// Modo DIAG_EIJ (campo fraco / analítico):
//   E_local = diag(−2M/r³, +M/r³, +M/r³)   [ordem r̂, θ̂, φ̂]
// Mais correções de spin Kerr de ordem M*a/r⁴ (activáveis por flag).
//
// Modo RIEMANN_FD:
//   Calcula Riemann por FD, projecta em tetrada, extrai E_ij.

/// Tensor de maré fraco: diag(−2M/r³, M/r³, M/r³) + correcção spin Kerr
/// spin_correction = true adiciona termos de ordem a·M/r⁴ que diferenciam Kerr de Schwarzschild
inline Mat3 tidal_diag_weak(double M, double a, double r, bool spin_correction = false) {
    const double r3 = r * r * r;
    Mat3 E = Mat3::Zero();
    E(0, 0) = -2.0 * M / r3;   // E_{r̂r̂}
    E(1, 1) =  M / r3;         // E_{θ̂θ̂}
    E(2, 2) =  M / r3;         // E_{φ̂φ̂}
    if (spin_correction && std::abs(a) > 1e-14) {
        // Heurístico: correcção Lense-Thirring de 1ª ordem
        // E_{rφ} ~ −3Ma/r⁴  (componente off-diagonal de interacção spin-órbita)
        const double r4 = r3 * r;
        const double sc = -3.0 * M * a / r4;
        E(0, 2) = sc;
        E(2, 0) = sc;
    }
    return E;
}

/// Tensor de maré completo via Riemann FD
/// Retorna E_ij em frame ZAMO, índices [r̂,θ̂,φ̂]
Mat3 tidal_riemann_fd(double M, double a, double r, double eps_r);

// ─── Torque de maré no body frame ─────────────────────────────────
//
// τ_body = −axial( Q_body * E_body − E_body * Q_body )
//
// onde axial(A) converte (A−Aᵀ)/2 em vetor:
//   axial(A)[0] = A[2,1] − A[1,2]  /  2
//   axial(A)[1] = A[0,2] − A[2,0]  /  2
//   axial(A)[2] = A[1,0] − A[0,1]  /  2

inline Vec3 axial(const Mat3& A) {
    return Vec3(
        0.5 * (A(2, 1) - A(1, 2)),
        0.5 * (A(0, 2) - A(2, 0)),
        0.5 * (A(1, 0) - A(0, 1))
    );
}

/// τ_body = −axial(Q E − E Q)  (torque de maré via acoplamento quadrupolo-tidal)
inline Vec3 tidal_torque_quadrupole(const Mat3& Q_body, const Mat3& E_body) {
    const Mat3 comm = Q_body * E_body - E_body * Q_body;
    return -axial(comm);
}

/// τ_body = 3*(M/r³) * (n_body × (I * n_body))  (gravity-gradient clássico)
inline Vec3 tidal_torque_weak_n(double M, double r, const Mat3& I_body, const Vec3& n_body) {
    const double c = 3.0 * M / (r * r * r);
    return c * n_body.cross(I_body * n_body);
}

/// Quadrupolo de massa de um corpo rígido derivado de I:
///   Q = I − (tr(I)/3) * I₃
inline Mat3 quadrupole_from_inertia(const Mat3& I) {
    return I - (I.trace() / 3.0) * Mat3::Identity();
}

} // namespace gr
} // namespace relorbit