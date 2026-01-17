# units.py
# -*- coding: utf-8 -*-
"""
Unidades, constantes e conversões (SI + geométricas) para o projeto RelOrbit.

Princípios (não quebre isso):
1) Núcleo relativístico (GR) em unidades geométricas: G = c = 1.
2) Conversões SI <-> geométricas apenas nas camadas de entrada/saída (mission/report/plots).
3) A massa do buraco negro define as escalas:
   - comprimento geométrico:  M_len  = G M / c^2   [m]
   - tempo geométrico:        M_time = G M / c^3   [s]
   - raio de Schwarzschild:   r_s    = 2 G M / c^2 = 2*M_len

Convenções recomendadas:
- Schwarzschild: horizonte em r = 2M (geom).
- ISCO (Schwarzschild): r = 6M (geom).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# =========================
# Constantes fundamentais (SI)
# =========================
C: Final[float] = 2.997_924_58e8            # [m/s] exata
G: Final[float] = 6.674_30e-11              # [m^3/(kg*s^2)] (CODATA 2018)
K_B: Final[float] = 1.380_649e-23           # [J/K] exata
H: Final[float] = 6.626_070_15e-34          # [J*s] exata
HBAR: Final[float] = 1.054_571_817e-34      # [J*s]
E_CHARGE: Final[float] = 1.602_176_634e-19  # [C] exata
N_A: Final[float] = 6.022_140_76e23         # [1/mol] exata

# Nota: após a redefinição SI (2019), mu0 e eps0 não são “exatas” por definição.
# Mantemos valores de referência úteis em engenharia.
EPS0: Final[float] = 8.854_187_8128e-12     # [F/m]
MU0: Final[float] = 1.256_637_062_12e-6     # [N/A^2]


# =========================
# Constantes astronômicas (SI)
# =========================
M_SUN: Final[float] = 1.988_47e30           # [kg]
R_SUN: Final[float] = 6.957_00e8            # [m]

M_EARTH: Final[float] = 5.972_2e24          # [kg]
R_EARTH: Final[float] = 6.371_0e6           # [m]

AU: Final[float] = 1.495_978_707e11         # [m] (IAU)
PC: Final[float] = 3.085_677_581_491_367e16 # [m]
KPC: Final[float] = 1.0e3 * PC              # [m]
MPC: Final[float] = 1.0e6 * PC              # [m]
LY: Final[float] = 9.460_730_472_580_8e15   # [m]


# =========================
# Tempo
# =========================
SECOND: Final[float] = 1.0e0
MINUTE: Final[float] = 6.0e1
HOUR: Final[float] = 3.6e3
DAY: Final[float] = 8.640_0e4
YEAR_JULIAN: Final[float] = 3.155_76e7      # 365.25 dias


# =========================
# Matemática / numérico
# =========================
PI: Final[float] = 3.141_592_653_589_793e0
TWO_PI: Final[float] = 6.283_185_307_179_586e0
HALF_PI: Final[float] = 1.570_796_326_794_8966e0

DEG2RAD: Final[float] = PI / 1.8e2
RAD2DEG: Final[float] = 1.8e2 / PI

ARCSEC2RAD: Final[float] = (PI / 6.48e5)    # 1 arcsec = pi/(180*3600)
RAD2ARCSEC: Final[float] = 1.0e0 / ARCSEC2RAD

# Tolerâncias genéricas (devem ser usadas conscientemente, não “no escuro”)
EPS: Final[float] = 1.0e-12
HORIZON_EPS_GEOM: Final[float] = 1.0e-9      # margem em unidades geométricas para evento r <= 2M + eps


# =========================
# Propulsão / missão (SI)
# =========================
G0: Final[float] = 9.806_65e0                # gravidade padrão [m/s^2] (definida)


# =========================
# Conversões SI <-> geométricas
# =========================
def gravitational_parameter(mass_kg: float) -> float:
    """Parâmetro gravitacional mu = G*M em SI [m^3/s^2]."""
    return G * mass_kg


def mass_to_M_length(mass_kg: float) -> float:
    """M_len = G*M/c^2 [m]."""
    return (G * mass_kg) / (C * C)


def mass_to_M_time(mass_kg: float) -> float:
    """M_time = G*M/c^3 [s]."""
    return (G * mass_kg) / (C * C * C)


def M_length_to_mass(M_len_m: float) -> float:
    """M = M_len*c^2/G [kg]."""
    return (M_len_m * C * C) / G


def M_time_to_mass(M_time_s: float) -> float:
    """M = M_time*c^3/G [kg]."""
    return (M_time_s * C * C * C) / G


def schwarzschild_radius(mass_kg: float) -> float:
    """r_s = 2GM/c^2 [m]."""
    return 2.0e0 * (G * mass_kg) / (C * C)


def isco_radius_schwarzschild(mass_kg: float) -> float:
    """r_ISCO = 6GM/c^2 [m]."""
    return 6.0e0 * (G * mass_kg) / (C * C)


def geom_r_from_SI_r(bh_mass_kg: float, r_m: float) -> float:
    """r [m] -> r/M (geom comprimento), dado M do BH."""
    M_len = mass_to_M_length(bh_mass_kg)
    return r_m / M_len


def SI_r_from_geom_r(bh_mass_kg: float, r_over_M: float) -> float:
    """r/M (geom) -> r [m], dado M do BH."""
    M_len = mass_to_M_length(bh_mass_kg)
    return r_over_M * M_len


def geom_t_from_SI_t(bh_mass_kg: float, t_s: float) -> float:
    """t [s] -> t/M (geom tempo), dado M do BH."""
    M_time = mass_to_M_time(bh_mass_kg)
    return t_s / M_time


def SI_t_from_geom_t(bh_mass_kg: float, t_over_M: float) -> float:
    """t/M (geom) -> t [s], dado M do BH."""
    M_time = mass_to_M_time(bh_mass_kg)
    return t_over_M * M_time


def geom_v_from_SI_v(v_mps: float) -> float:
    """v [m/s] -> v/c (adimensional, geom)."""
    return v_mps / C


def SI_v_from_geom_v(v_over_c: float) -> float:
    """v/c -> v [m/s]."""
    return v_over_c * C


# =========================
# Estrutura de parâmetros de BH (SI + escalas)
# =========================
@dataclass(frozen=True)
class BlackHoleSI:
    """
    Buraco negro parametrizado por massa SI.

    Derivados:
    - mu     = GM                 [m^3/s^2]
    - M_len  = GM/c^2             [m]
    - M_time = GM/c^3             [s]
    - r_s    = 2GM/c^2 = 2*M_len  [m]
    - r_isco = 6GM/c^2            [m]
    """
    mass_kg: float

    @property
    def mu(self) -> float:
        return gravitational_parameter(self.mass_kg)

    @property
    def M_len(self) -> float:
        return mass_to_M_length(self.mass_kg)

    @property
    def M_time(self) -> float:
        return mass_to_M_time(self.mass_kg)

    @property
    def r_s(self) -> float:
        return 2.0e0 * self.M_len

    @property
    def r_isco(self) -> float:
        return 6.0e0 * self.M_len

    def to_geom_length(self, x_m: float) -> float:
        """x [m] -> x/M (geom comprimento)."""
        return x_m / self.M_len

    def from_geom_length(self, x_over_M: float) -> float:
        """x/M -> x [m]."""
        return x_over_M * self.M_len

    def to_geom_time(self, t_s: float) -> float:
        """t [s] -> t/M (geom tempo)."""
        return t_s / self.M_time

    def from_geom_time(self, t_over_M: float) -> float:
        """t/M -> t [s]."""
        return t_over_M * self.M_time


# =========================
# Helpers
# =========================
def fmt_sci(x: float, sig: int = 6) -> str:
    """Notação científica com 'sig' dígitos após o ponto."""
    return f"{x:.{sig}e}"


def example_bh_solar_mass(multiples_of_solar_mass: float) -> BlackHoleSI:
    """BH com massa = k * M_sun."""
    return BlackHoleSI(mass_kg=multiples_of_solar_mass * M_SUN)


__all__ = [
    # fundamentais
    "C", "G", "K_B", "H", "HBAR", "E_CHARGE", "N_A", "EPS0", "MU0",
    # astro
    "M_SUN", "R_SUN", "M_EARTH", "R_EARTH", "AU", "PC", "KPC", "MPC", "LY",
    # tempo
    "SECOND", "MINUTE", "HOUR", "DAY", "YEAR_JULIAN",
    # math/num
    "PI", "TWO_PI", "HALF_PI", "DEG2RAD", "RAD2DEG", "ARCSEC2RAD", "RAD2ARCSEC",
    "EPS", "HORIZON_EPS_GEOM",
    # propulsao
    "G0",
    # conversoes
    "gravitational_parameter", "mass_to_M_length", "mass_to_M_time",
    "M_length_to_mass", "M_time_to_mass",
    "schwarzschild_radius", "isco_radius_schwarzschild",
    "geom_r_from_SI_r", "SI_r_from_geom_r", "geom_t_from_SI_t", "SI_t_from_geom_t",
    "geom_v_from_SI_v", "SI_v_from_geom_v",
    # BH struct
    "BlackHoleSI",
    # helpers
    "fmt_sci", "example_bh_solar_mass",
]


if __name__ == "__main__":
    # Sanity check (não é teste formal)
    bh = example_bh_solar_mass(1.0e0)
    print("BH = 1 M_sun")
    print("mu (m^3/s^2)      :", fmt_sci(bh.mu))
    print("M_len=GM/c^2 (m)   :", fmt_sci(bh.M_len))
    print("M_time=GM/c^3 (s)  :", fmt_sci(bh.M_time))
    print("r_s=2GM/c^2 (m)    :", fmt_sci(bh.r_s))
    print("r_ISCO=6GM/c^2 (m) :", fmt_sci(bh.r_isco))
