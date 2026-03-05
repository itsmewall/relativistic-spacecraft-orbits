// src_cpp/include/relorbit/gr/kerr_null_geodesic.hpp
//
// Integrador de geodésicas nulas em Kerr equatorial (θ = π/2)
// para ray tracing de telemetria.
//
// ══════════════════════════════════════════════════════════════════════════════
// FÍSICA
//
// Equações de movimento (E=1, b = L/E = parâmetro de impacto):
//
//   r²  dr/dλ  = σ √R(r)
//   r² dφ/dλ  = (b−a) + a(r²+a²−ab)/Δ
//   r²  dt/dλ  = (r²+a²)(r²+a²−ab)/Δ + a(a−b)
//
//   R(r) = (r²+a²−ab)² − Δ(b−a)²     [potencial radial]
//   Δ(r) = r²−2Mr+a²
//
//   σ = +1 (afastando) ou −1 (aproximando)
//   Inversão (ponto de retorno): R(r*)=0 → σ muda de sinal
//   Captura: r ≤ r₊·(1+ε₀)   r₊ = M + √(M²−a²)
//
// ALGORITMO LUT (Look-Up Table)
//   1. Pré-computa N_lut raios para r_s fixo: O(N_lut × N_steps)
//   2. Query por bisecção no array ordenado b → Δφ: O(log N_lut)
//
// ALGORITMO BISSECÇÃO
//   Encontra b* tal que Δφ(b*) = Δφ_alvo em N_bisect iterações.
//   Usa LUT como bracket inicial.
//
// REDSHIFT
//   1+z = (k_μ u^μ)_emissão / (k_μ u^μ)_recepção
//   Para nave em órbita circular e receptor estático em r >> M.
//
// ══════════════════════════════════════════════════════════════════════════════

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace relorbit {
namespace gr {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;


// ── Parâmetros do espaço-tempo e do integrador ────────────────────────────────

struct NullGeodesicConfig {
    double M            = 1.0;       ///< massa BH [geom]
    double a            = 0.0;       ///< spin específico [M]
    double r_obs        = 1000.0;    ///< raio receptor [M]
    int    n_lut        = 1000;      ///< raios na LUT
    int    n_steps      = 12000;     ///< passos RK4 por raio
    double dl_coarse    = 0.5;       ///< passo longe do BH [M]
    double dl_fine      = 0.05;      ///< passo perto do BH [M]
    double r_switch     = 20.0;      ///< raio de transição [M]
    int    n_bisect     = 50;        ///< iterações de bissecção

    double r_horizon() const {
        return M + std::sqrt(std::max(M*M - a*a, 0.0));
    }
    double b_crit_approx() const {
        if (std::abs(a) < 1e-10) return 3.0 * std::sqrt(3.0) * M;
        return 3.0 * std::sqrt(3.0) * M * (1.0 - 0.4 * a / M);
    }
};


// ── Resultado de um raio ──────────────────────────────────────────────────────

struct NullRayResult {
    double b;               ///< parâmetro de impacto [M]
    double dphi;            ///< deflexão total Δφ [rad]
    double t_coord;         ///< tempo coordenado de voo [M]
    bool   captured;        ///< caiu no horizonte?
    int    n_turns;         ///< inversões radiais (winding)
};


// ── Ponto de telemetria ───────────────────────────────────────────────────────

struct TelemetrySignal {
    double tau_s;           ///< tempo próprio de emissão [M]
    double r_s;             ///< raio da nave [M]
    double phi_s;           ///< ângulo da nave [rad]
    bool   visible;         ///< alguma geodésica nula chega ao receptor?
    int    n_images;        ///< número de imagens (1=directo, 2+=lensado)

    struct Image {
        double b;           ///< parâmetro de impacto [M]
        double dphi;        ///< deflexão [rad]
        double t_coord;     ///< tempo de voo [M]
        double redshift_z;  ///< z gravitacional + Doppler
        double time_delay;  ///< atraso vs linha recta [M]
    };
    std::vector<Image> images;
};


// ── Potencial radial nulo de Kerr ─────────────────────────────────────────────

/// R(r) = (r²+a²−ab)² − Δ(b−a)²
inline double kerr_null_R(double M, double a, double b, double r) {
    const double D = r*r - 2.0*M*r + a*a;
    const double T = r*r + a*a - a*b;
    return T*T - D*(b - a)*(b - a);
}

/// Δ(r) = r²−2Mr+a²
inline double kerr_Delta(double M, double a, double r) {
    return r*r - 2.0*M*r + a*a;
}


// ── Integrador de raio único (RK4) ────────────────────────────────────────────

/// Integra uma geodésica nula de r_s até r_obs (ou captura).
/// Retorna NullRayResult com Δφ, Δt, status.
NullRayResult integrate_null_ray(
    const NullGeodesicConfig& cfg,
    double b,
    double r_s,
    double dl       = 0.5    ///< passo de integração
);


// ── LUT (Look-Up Table) ───────────────────────────────────────────────────────

struct NullGeodesicLUT {
    std::vector<double> b_arr;     ///< parâmetros de impacto
    std::vector<double> phi_arr;   ///< deflexões Δφ (rad)
    std::vector<double> t_arr;     ///< tempos de voo
    std::vector<bool>   cap_arr;   ///< capturado?
    std::vector<int>    wind_arr;  ///< winding numbers
    double r_s = 0.0;

    int n_arrived() const;
    int n_captured() const;

    /// Interpola b* para um dado dphi_target.
    /// Retorna (b_star, t_star) ou {nan, nan} se fora do intervalo.
    std::pair<double, double> query_phi(double dphi_target,
                                        int winding = 0) const;
};

/// Constrói a LUT: dispara cfg.n_lut raios e preenche tabelas.
NullGeodesicLUT build_null_geodesic_lut(
    const NullGeodesicConfig& cfg,
    double r_s
);


// ── Bissecção de b* ───────────────────────────────────────────────────────────

/// Encontra b* tal que Δφ(b*) ≈ dphi_target (com winding adicional).
/// Usa LUT como bracket inicial.
/// Retorna nullopt se não encontrado.
std::optional<NullRayResult> bisect_impact_parameter(
    const NullGeodesicConfig& cfg,
    double r_s,
    double phi_s,
    double phi_obs,
    int    winding  = 0,
    const NullGeodesicLUT* lut = nullptr  ///< bracket inicial
);


// ── Redshift ──────────────────────────────────────────────────────────────────

/// Velocidade angular da órbita circular Kerr.
/// Ω = √M / (r^{3/2} ± a√M)  (+ prograde, − retrograde)
inline double circular_orbit_omega(double M, double a, double r, bool prograde = true) {
    const double sgn = prograde ? 1.0 : -1.0;
    const double den = std::pow(r, 1.5) + sgn * a * std::sqrt(M);
    return (std::abs(den) < 1e-30) ? 0.0 : std::sqrt(M) / den;
}

/// Factor 1+z combinado (gravitacional + Doppler) para geodésica com impacto b.
/// omega_s = velocidade angular da nave [rad/M] (0 = estática)
double compute_redshift_kerr(
    double M,
    double a,
    double b,
    double r_s,
    double r_obs,
    double omega_s  = 0.0
);


// ── Ray tracer de trajectória completa ────────────────────────────────────────

struct RayTracerOptions {
    double receiver_r   = 1000.0;    ///< raio do receptor [M]
    double receiver_phi = 0.0;       ///< ângulo do receptor [rad]
    int    n_images_max = 2;         ///< imagens gravitacionais máximas
    bool   compute_redshift = true;  ///< calcular redshift?
    bool   compute_delay    = true;  ///< calcular atraso de tempo?
};

/// Ray tracer para arrays de posição (τ_arr, r_arr, φ_arr).
/// Constrói LUT uma vez e consulta para cada ponto.
std::vector<TelemetrySignal> raytrace_trajectory(
    const NullGeodesicConfig& cfg,
    const RayTracerOptions&   opts,
    const std::vector<double>& tau_arr,
    const std::vector<double>& r_arr,
    const std::vector<double>& phi_arr,
    const std::vector<double>& omega_arr  ///< velocidade angular da nave
);

} // namespace gr
} // namespace relorbit