// src_cpp/lib/gr/kerr_metric.cpp
//
// Implementação: métrica Kerr, Christoffel, Riemann por FD, tensor de maré ZAMO
//
// CONVENÇÕES:
//   índices coordenados: t=0, r=1, θ=2, φ=3
//   assinatura: (-,+,+,+)
//   equatorial: θ = π/2 (fixo)
//
// APIs conforme o header:
//   - christoffel_kerr_eq
//   - dGamma_dr_fd
//   - riemann_Et0t_kerr_eq   (NOVO: estava declarado no .hpp e faltava aqui)
//   - tidal_riemann_fd
//
#include "relorbit/gr/kerr_metric.hpp"

#include <algorithm>
#include <cmath>

namespace relorbit {
namespace gr {

// ───────────────────────────────────────────────────────────────
// Christoffel analítico Kerr equatorial (apenas ∂_r g entra)
// Γ^μ_{αβ} = 1/2 g^{μσ} (∂_α g_{βσ} + ∂_β g_{ασ} - ∂_σ g_{αβ})
// No equatorial estacionário/axissimétrico: ∂_t=∂_φ=0, θ fixo, logo só ∂_r.
// ───────────────────────────────────────────────────────────────
Gamma444 christoffel_kerr_eq(double M, double a, double r) {
    Gamma444 G{}; // zero-init

    const auto g  = kerr_metric(M, a, r);
    const auto dg = dg_dr_kerr(M, a, r);

    const double dg_tt     = dg[0];
    const double dg_tphi   = dg[1];
    const double dg_rr     = dg[2];
    const double dg_thth   = dg[3];
    const double dg_phiphi = dg[4];

    const double guu_tt     = g.gtt_up;
    const double guu_tphi   = g.gtphi_up;
    const double guu_phiphi = g.gphiphi_up;
    const double guu_rr     = g.grr_up;
    const double guu_thth   = g.gthth_up;

    // ── Γ^t_{αβ}
    // Γ^t_{tr} = 1/2 (g^{tt} ∂_r g_tt + g^{tφ} ∂_r g_tφ)
    G[0][0][1] = G[0][1][0] = 0.5 * (guu_tt * dg_tt + guu_tphi * dg_tphi);

    // Γ^t_{φr} = 1/2 (g^{tt} ∂_r g_tφ + g^{tφ} ∂_r g_φφ)
    G[0][3][1] = G[0][1][3] = 0.5 * (guu_tt * dg_tphi + guu_tphi * dg_phiphi);

    // (em BL: g^{tr}=0 e ∂_t=0 ⇒ Γ^t_{rr} = 0)
    G[0][1][1] = 0.0;

    // ── Γ^r_{αβ}
    // Γ^r_{tt} = -1/2 g^{rr} ∂_r g_tt
    G[1][0][0] = -0.5 * guu_rr * dg_tt;

    // Γ^r_{tφ} = -1/2 g^{rr} ∂_r g_tφ
    G[1][0][3] = G[1][3][0] = -0.5 * guu_rr * dg_tphi;

    // Γ^r_{rr} =  1/2 g^{rr} ∂_r g_rr
    G[1][1][1] =  0.5 * guu_rr * dg_rr;

    // Γ^r_{θθ} = -1/2 g^{rr} ∂_r g_θθ
    G[1][2][2] = -0.5 * guu_rr * dg_thth;

    // Γ^r_{φφ} = -1/2 g^{rr} ∂_r g_φφ
    G[1][3][3] = -0.5 * guu_rr * dg_phiphi;

    // ── Γ^θ_{αβ}
    // No equatorial (Σ=r²): Γ^θ_{θr} = 1/r
    G[2][2][1] = G[2][1][2] = 1.0 / r;
    // Γ^θ_{φφ} tem factor sinθ cosθ → 0 em θ=π/2
    G[2][3][3] = 0.0;

    // ── Γ^φ_{αβ}
    // Γ^φ_{tr} = 1/2 (g^{φt} ∂_r g_tt + g^{φφ} ∂_r g_tφ)
    G[3][0][1] = G[3][1][0] = 0.5 * (guu_tphi * dg_tt + guu_phiphi * dg_tphi);

    // Γ^φ_{φr} = 1/2 (g^{φt} ∂_r g_tφ + g^{φφ} ∂_r g_φφ)
    G[3][3][1] = G[3][1][3] = 0.5 * (guu_tphi * dg_tphi + guu_phiphi * dg_phiphi);

    return G;
}

// ───────────────────────────────────────────────────────────────
// ∂_r Γ por FD centrado
// ───────────────────────────────────────────────────────────────
Gamma444 dGamma_dr_fd(double M, double a, double r, double eps) {
    const double h = (eps > 0.0) ? eps : 1e-5;
    const double den = 2.0 * h;

    const auto Gp = christoffel_kerr_eq(M, a, r + h);
    const auto Gm = christoffel_kerr_eq(M, a, r - h);

    Gamma444 dG{};
    for (int mu = 0; mu < 4; ++mu)
        for (int al = 0; al < 4; ++al)
            for (int be = 0; be < 4; ++be)
                dG[mu][al][be] = (Gp[mu][al][be] - Gm[mu][al][be]) / den;

    return dG;
}

// ───────────────────────────────────────────────────────────────
// R_{μ t ρ t} (covariante) em coordenadas BL equatorial.
// Retorna uma Mat4 A tal que A(mu, rho) = R_{μ t ρ t}.
//
// Implementação: usa
// R^λ_{t ρ t} = ∂_ρ Γ^λ_{tt} - ∂_t Γ^λ_{ρt} + Γ^λ_{ρσ}Γ^σ_{tt} - Γ^λ_{tσ}Γ^σ_{ρt}
// Com ∂_t=0. No equatorial, só ∂_r != 0, então:
//  - se ρ=r(1): ∂_ρ Γ^λ_{tt} = ∂_r Γ^λ_{tt} (via dGamma)
//  - se ρ=θ(2) ou φ(3) ou t(0): termo derivada = 0, ficam só os quadráticos.
//
// Depois baixa índice: R_{μ t ρ t} = g_{μλ} R^λ_{t ρ t}.
// ───────────────────────────────────────────────────────────────
Mat4 riemann_Et0t_kerr_eq(double M, double a, double r, double eps_r) {
    const double eps = (eps_r > 0.0) ? eps_r : 1e-5;

    const auto G  = christoffel_kerr_eq(M, a, r);
    const auto dG = dGamma_dr_fd(M, a, r, eps);
    const auto gm = kerr_metric(M, a, r);

    auto R_up = [&](int lam, int rho) -> double {
        // R^lam_{t rho t}
        double val = 0.0;

        // derivada só quando rho == r(1)
        if (rho == 1) val += dG[lam][0][0]; // ∂_r Γ^lam_{tt}

        // quadráticos: + Γ^lam_{rho σ} Γ^σ_{tt} − Γ^lam_{t σ} Γ^σ_{rho t}
        for (int sig = 0; sig < 4; ++sig) {
            val += G[lam][rho][sig] * G[sig][0][0];
            val -= G[lam][0][sig]   * G[sig][rho][0];
        }
        return val;
    };

    auto lower_mu = [&](int mu, const double Rup[4]) -> double {
        // contrai g_{mu lam} Rup[lam]
        // g_{tφ} é o único off-diagonal relevante.
        if (mu == 0) {
            return gm.gtt * Rup[0] + gm.gtphi * Rup[3];
        } else if (mu == 1) {
            return gm.grr * Rup[1];
        } else if (mu == 2) {
            return gm.gthth * Rup[2];
        } else { // mu==3
            return gm.gtphi * Rup[0] + gm.gphiphi * Rup[3];
        }
    };

    Mat4 A = Mat4::Zero();
    for (int rho = 0; rho < 4; ++rho) {
        double Rup[4] = {0,0,0,0};
        for (int lam = 0; lam < 4; ++lam) Rup[lam] = R_up(lam, rho);

        for (int mu = 0; mu < 4; ++mu) {
            A(mu, rho) = lower_mu(mu, Rup);
        }
    }

    return A; // A(mu,rho) = R_{μ t ρ t}
}

// ───────────────────────────────────────────────────────────────
// Tensor de maré E_ij em tetrada ZAMO, i,j ∈ {r̂=0, θ̂=1, φ̂=2}
// E_{îĵ} = R_{î t̂ ĵ t̂}
// Projecta R_{μ t ρ t} com e_(t̂)^t e com pernas espaciais ZAMO.
// ───────────────────────────────────────────────────────────────
Mat3 tidal_riemann_fd(double M, double a, double r, double eps_r) {
    const double eps = (eps_r > 0.0) ? eps_r : 1e-5;

    const auto tet = zamo_tetrad(M, a, r);

    // A(mu,rho) = R_{μ t ρ t}
    const Mat4 A = riemann_Et0t_kerr_eq(M, a, r, eps);

    // E_{îĵ} = (e_(t̂)^t)^2 * e_(î)^μ e_(ĵ)^ρ * R_{μ t ρ t}
    const double et2 = tet.et_t * tet.et_t;

    auto leg = [&](int ihat, int mu) -> double {
        // e_(î)^μ
        // r̂: só mu=r(1)
        if (ihat == 0) return (mu == 1) ? tet.er_r : 0.0;
        // θ̂: só mu=θ(2)
        if (ihat == 1) return (mu == 2) ? tet.eth_th : 0.0;
        // φ̂: mu=t(0) e mu=φ(3)
        return (mu == 0) ? tet.ephi_t : (mu == 3) ? tet.ephi_phi : 0.0;
    };

    Mat3 E = Mat3::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int mu = 0; mu < 4; ++mu)
                for (int rho = 0; rho < 4; ++rho)
                    s += leg(i, mu) * leg(j, rho) * A(mu, rho);
            E(i, j) = et2 * s;
        }
    }

    return E;
}

} // namespace gr
} // namespace relorbit