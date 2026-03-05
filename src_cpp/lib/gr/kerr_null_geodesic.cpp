// src_cpp/lib/gr/kerr_null_geodesic.cpp
//
// Implementação do integrador de geodésicas nulas em Kerr equatorial.
// Ver header para documentação completa.

// MSVC não define M_PI por defeito — tem de vir antes de qualquer <cmath>
#if defined(_MSC_VER) && !defined(_USE_MATH_DEFINES)
#  define _USE_MATH_DEFINES
#endif

#include "relorbit/gr/kerr_null_geodesic.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace relorbit {
namespace gr {

// ── Helpers internos ──────────────────────────────────────────────────────────

namespace {

struct RHS {
    double dr, dphi, dt;
};

/// RHS das equações de movimento para uma geodésica nula Kerr equatorial.
inline RHS kerr_null_rhs(double M, double a, double b, double r, double sigma) {
    const double D  = std::max(r*r - 2.0*M*r + a*a, 1e-30);
    const double T  = r*r + a*a - a*b;
    const double Rv = T*T - D*(b - a)*(b - a);
    const double r2 = r*r;

    const double sqR = (Rv > 0.0) ? std::sqrt(Rv) : 0.0;
    RHS rhs;
    rhs.dr   = sigma * sqR / r2;
    rhs.dphi = (b - a + a*T/D) / r2;
    rhs.dt   = ((r2 + a*a)*T/D + a*(a - b)) / r2;
    return rhs;
}

/// RK4 para um passo de geodésica nula.
/// Retorna (r_new, phi_new, t_new).
inline std::tuple<double,double,double>
rk4_null_step(double M, double a, double b,
              double r, double phi, double t, double sigma, double dl) {
    auto k1 = kerr_null_rhs(M, a, b, r, sigma);
    auto k2 = kerr_null_rhs(M, a, b, r + 0.5*dl*k1.dr, sigma);
    auto k3 = kerr_null_rhs(M, a, b, r + 0.5*dl*k2.dr, sigma);
    auto k4 = kerr_null_rhs(M, a, b, r +     dl*k3.dr, sigma);

    return {
        r   + dl/6.0*(k1.dr   + 2*k2.dr   + 2*k3.dr   + k4.dr),
        phi + dl/6.0*(k1.dphi + 2*k2.dphi + 2*k3.dphi + k4.dphi),
        t   + dl/6.0*(k1.dt   + 2*k2.dt   + 2*k3.dt   + k4.dt),
    };
}

} // anonymous namespace


// ── Integrador de raio único ──────────────────────────────────────────────────

NullRayResult integrate_null_ray(
    const NullGeodesicConfig& cfg,
    double b,
    double r_s,
    double dl
) {
    const double M = cfg.M, a = cfg.a;
    const double r_hor = cfg.r_horizon();
    const double r_cap = r_hor * 1.005;

    double r = r_s, phi = 0.0, t = 0.0;
    double sigma = +1.0;
    int    n_turns = 0;

    for (int step = 0; step < cfg.n_steps; ++step) {
        // Verificar potencial radial
        const double D  = std::max(r*r - 2.0*M*r + a*a, 1e-30);
        const double T  = r*r + a*a - a*b;
        const double Rv = T*T - D*(b - a)*(b - a);

        if (Rv < 0.0) {
            if (sigma > 0.0) {
                sigma = -1.0;   // inversão de direcção (ponto de retorno)
                ++n_turns;
            } else {
                return {b, phi, t, true, n_turns};   // capturado
            }
        }

        // Passo adaptativo: mais fino perto do BH
        const double h = (r < cfg.r_switch) ? cfg.dl_fine : dl;

        auto [r_new, phi_new, t_new] = rk4_null_step(M, a, b, r, phi, t, sigma, h);

        r   = r_new;
        phi = phi_new;
        t   = t_new;

        // Captura por horizonte
        if (r <= r_cap) {
            return {b, phi, t, true, n_turns};
        }

        // Chegada ao receptor
        if (sigma > 0.0 && r >= cfg.r_obs) {
            return {b, phi, t, false, n_turns};
        }
    }

    return {b, phi, t, false, n_turns};
}


// ── LUT ───────────────────────────────────────────────────────────────────────

int NullGeodesicLUT::n_arrived() const {
    return static_cast<int>(
        std::count(cap_arr.begin(), cap_arr.end(), false));
}

int NullGeodesicLUT::n_captured() const {
    return static_cast<int>(
        std::count(cap_arr.begin(), cap_arr.end(), true));
}

std::pair<double,double> NullGeodesicLUT::query_phi(
    double dphi_target, int winding
) const {
    const double target = dphi_target + 2.0*M_PI*winding;

    // Construir view ordenada das entradas não-capturadas
    // (pré-computada em build_null_geodesic_lut via _phi_ok/_b_ok)
    // Aqui fazemos inline para simplicidade (perf: O(N) — optimizável com cache)
    std::vector<std::pair<double,double>> ok;   // (phi, b)
    ok.reserve(phi_arr.size());
    for (size_t i = 0; i < b_arr.size(); ++i) {
        if (!cap_arr[i]) ok.emplace_back(phi_arr[i], b_arr[i]);
    }
    std::sort(ok.begin(), ok.end());

    // Busca binária
    auto it = std::lower_bound(ok.begin(), ok.end(),
                               std::make_pair(target, -1e300));
    if (it == ok.begin() || it == ok.end())
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};

    const auto [phi_hi, b_hi] = *it;
    const auto [phi_lo, b_lo] = *std::prev(it);

    // Interpolação linear
    const double denom = phi_hi - phi_lo;
    if (std::abs(denom) < 1e-30)
        return {b_lo, 0.0};
    const double alpha = (target - phi_lo) / denom;

    // Interpolar também t_arr  (associar por index — simplificação aqui)
    const double b_interp = b_lo + alpha*(b_hi - b_lo);

    // Para t: busca índices correspondentes em b_arr
    // (em produção: guardar arrays ordenados separadamente)
    auto find_t = [&](double b_val) -> double {
        for (size_t i = 0; i < b_arr.size(); ++i)
            if (std::abs(b_arr[i] - b_val) < 1e-6) return t_arr[i];
        return 0.0;
    };
    const double t_interp = find_t(b_lo) + alpha*(find_t(b_hi) - find_t(b_lo));

    return {b_interp, t_interp};
}


NullGeodesicLUT build_null_geodesic_lut(
    const NullGeodesicConfig& cfg,
    double r_s
) {
    const double M = cfg.M, a = cfg.a;
    const double r_hor   = cfg.r_horizon();
    const double b_min   = r_hor * 1.02;
    const double b_max   = std::max(cfg.r_obs * 5.0, 200.0 * M);
    const double b_crit  = cfg.b_crit_approx();

    const int n     = cfg.n_lut;
    const int n_den = 2 * n / 3;
    const int n_far = n - n_den;

    // Grelha: densa perto de b_crit, esparsa longe
    std::vector<double> b_arr;
    b_arr.reserve(n);
    for (int i = 0; i < n_den; ++i) {
        b_arr.push_back(b_min + (b_crit * 2.5 - b_min) * i / (n_den - 1));
    }
    const double log_lo = std::log(b_crit * 2.5);
    const double log_hi = std::log(b_max);
    for (int i = 1; i <= n_far; ++i) {
        b_arr.push_back(std::exp(log_lo + (log_hi - log_lo) * i / n_far));
    }
    // Remover duplicados
    std::sort(b_arr.begin(), b_arr.end());
    b_arr.erase(std::unique(b_arr.begin(), b_arr.end()), b_arr.end());

    NullGeodesicLUT lut;
    lut.r_s = r_s;
    lut.b_arr.reserve(b_arr.size());
    lut.phi_arr.reserve(b_arr.size());
    lut.t_arr.reserve(b_arr.size());
    lut.cap_arr.reserve(b_arr.size());
    lut.wind_arr.reserve(b_arr.size());

    for (double b : b_arr) {
        auto res = integrate_null_ray(cfg, b, r_s, cfg.dl_coarse);
        lut.b_arr.push_back(res.b);
        lut.phi_arr.push_back(res.dphi);
        lut.t_arr.push_back(res.t_coord);
        lut.cap_arr.push_back(res.captured);
        lut.wind_arr.push_back(res.n_turns);
    }

    std::cout << "[KERR_NULL] LUT pronta: " << lut.n_arrived()
              << " chegaram, " << lut.n_captured() << " capturados"
              << " (r_s=" << r_s << "M, N=" << b_arr.size() << ")"
              << std::endl;

    return lut;
}


// ── Bissecção ─────────────────────────────────────────────────────────────────

std::optional<NullRayResult> bisect_impact_parameter(
    const NullGeodesicConfig& cfg,
    double r_s,
    double phi_s,
    double phi_obs,
    int    winding,
    const NullGeodesicLUT* lut
) {
    const double M = cfg.M, a = cfg.a;
    const double target = std::fmod(phi_obs - phi_s, 2.0*M_PI)
                          + 2.0*M_PI * (winding + (phi_obs < phi_s ? 1 : 0));

    // Bracket inicial
    double b_lo, b_hi;
    if (lut != nullptr) {
        // Usar LUT como bracket (preciso e rápido)
        auto [b_g, _t] = lut->query_phi(target, 0);
        if (std::isnan(b_g)) return std::nullopt;
        b_lo = b_g * 0.96;
        b_hi = b_g * 1.04;
    } else {
        // Scan grosso
        const double r_hor = cfg.r_horizon();
        const int N_scan = 50;
        std::vector<double> b_scan, phi_scan;
        for (int i = 0; i < N_scan; ++i) {
            double b = r_hor*1.05 + (200.0*cfg.M - r_hor*1.05) * i / (N_scan-1);
            auto res = integrate_null_ray(cfg, b, r_s, cfg.dl_coarse);
            if (!res.captured) {
                b_scan.push_back(b); phi_scan.push_back(res.dphi);
            }
        }
        if (b_scan.size() < 2) return std::nullopt;
        b_lo = -1; b_hi = -1;
        for (size_t i = 0; i+1 < b_scan.size(); ++i) {
            if ((phi_scan[i] - target)*(phi_scan[i+1] - target) < 0) {
                b_lo = b_scan[i]; b_hi = b_scan[i+1]; break;
            }
        }
        if (b_lo < 0) return std::nullopt;
    }

    // Bissecção
    double phi_lo_val = integrate_null_ray(cfg, b_lo, r_s, cfg.dl_fine).dphi;

    for (int iter = 0; iter < cfg.n_bisect; ++iter) {
        if (std::abs(b_hi - b_lo) < 1e-10) break;
        const double b_m = 0.5*(b_lo + b_hi);
        auto res_m = integrate_null_ray(cfg, b_m, r_s, cfg.dl_fine);
        if (res_m.captured) { b_lo = b_m; continue; }
        if ((phi_lo_val - target)*(res_m.dphi - target) < 0) {
            b_hi = b_m;
        } else {
            b_lo = b_m;
            phi_lo_val = res_m.dphi;
        }
    }

    const double b_star = 0.5*(b_lo + b_hi);
    auto res = integrate_null_ray(cfg, b_star, r_s, cfg.dl_fine);
    if (res.captured) return std::nullopt;
    return res;
}


// ── Redshift ──────────────────────────────────────────────────────────────────

double compute_redshift_kerr(
    double M, double a, double b, double r_s, double r_obs, double omega_s
) {
    const double gtt_s     = -(1.0 - 2.0*M/r_s);
    const double gtphi_s   = -2.0*M*a/r_s;
    const double gphiphi_s = r_s*r_s + a*a + 2.0*M*a*a/r_s;

    const double norm2 = -(gtt_s + 2.0*gtphi_s*omega_s + gphiphi_s*omega_s*omega_s);
    if (norm2 <= 0.0) return 1.0;
    const double ut   = 1.0 / std::sqrt(norm2);
    const double uphi = omega_s * ut;

    // k_t = -1, k_φ = b
    const double ku_emit = -ut + b * uphi;

    // Receptor estático
    const double gtt_obs = -(1.0 - 2.0*M/r_obs);
    if (-gtt_obs <= 0.0) return 1.0;
    const double ku_obs = -1.0 / std::sqrt(-gtt_obs);

    if (std::abs(ku_emit) < 1e-30) return 1.0;
    const double freq_ratio = ku_obs / ku_emit;
    return (freq_ratio > 0.0) ? (1.0 / freq_ratio) : 1.0;
}


// ── Ray tracer de trajectória ─────────────────────────────────────────────────

std::vector<TelemetrySignal> raytrace_trajectory(
    const NullGeodesicConfig& cfg,
    const RayTracerOptions&   opts,
    const std::vector<double>& tau_arr,
    const std::vector<double>& r_arr,
    const std::vector<double>& phi_arr,
    const std::vector<double>& omega_arr
) {
    // ── Construir LUT para r_s mediano ────────────────────────────────────────
    const size_t N = tau_arr.size();
    if (N == 0) return {};

    std::vector<double> r_sorted = r_arr;
    std::sort(r_sorted.begin(), r_sorted.end());
    const double r_med = r_sorted[N/2];

    const auto lut = build_null_geodesic_lut(cfg, r_med);

    // ── Query para cada ponto ─────────────────────────────────────────────────
    std::vector<TelemetrySignal> signals;
    signals.reserve(N);

    const double phi_obs   = opts.receiver_phi;
    const double r_obs     = opts.receiver_r;
    const double t_straight = r_obs;

    for (size_t i = 0; i < N; ++i) {
        const double tau_i  = tau_arr[i];
        const double r_i    = r_arr[i];
        const double phi_i  = phi_arr[i];
        const double omega_i = (i < omega_arr.size()) ? omega_arr[i] : 0.0;

        TelemetrySignal sig;
        sig.tau_s   = tau_i;
        sig.r_s     = r_i;
        sig.phi_s   = phi_i;
        sig.visible = false;
        sig.n_images = 0;

        for (int winding = 0; winding < opts.n_images_max; ++winding) {
            const double dphi_target = std::fmod(phi_obs - phi_i, 2.0*M_PI)
                                       + 2.0*M_PI * winding;
            // Garantir dphi_target >= 0
            const double dpt = (dphi_target < 0) ? dphi_target + 2.0*M_PI : dphi_target;

            auto [b_star, t_fly] = lut.query_phi(dpt, 0);
            if (std::isnan(b_star)) continue;

            TelemetrySignal::Image img;
            img.b        = b_star;
            img.dphi     = dpt;
            img.t_coord  = t_fly;
            img.time_delay = t_fly - t_straight;
            img.redshift_z = opts.compute_redshift
                ? (compute_redshift_kerr(cfg.M, cfg.a, b_star, r_i, r_obs, omega_i) - 1.0)
                : 0.0;

            sig.images.push_back(img);
            sig.visible = true;
            ++sig.n_images;
        }

        signals.push_back(sig);
    }

    const int n_vis = static_cast<int>(
        std::count_if(signals.begin(), signals.end(),
                      [](const TelemetrySignal& s){ return s.visible; }));
    std::cout << "[KERR_NULL] Trajectória: " << n_vis << "/" << N
              << " pontos visíveis." << std::endl;

    return signals;
}

} // namespace gr
} // namespace relorbit