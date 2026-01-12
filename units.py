# units.py
# -*- coding: utf-8 -*-
"""
Unidades e constantes físicas para o projeto:
Órbitas relativísticas (Schwarzschild/Kerr opcional) + comparação Newtoniana.

Convenção recomendada:
- Núcleo relativístico em unidades geométricas: G = c = 1
- Conversões para SI apenas nas camadas de entrada/saída (plots/mission/report).

Notas:
- Em unidades geométricas, a massa vira um comprimento: M_len = G*M/c^2  (metros)
- E também vira um tempo: M_time = G*M/c^3  (segundos)
- Raio de Schwarzschild: r_s = 2*G*M/c^2 = 2*M_len
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# =========================
# Constantes fundamentais SI
# =========================
C: Final[float] = 2.997_924_58e8          # m/s (exata)
G: Final[float] = 6.674_30e-11            # m^3/(kg*s^2) (CODATA 2018; uso comum)
K_B: Final[float] = 1.380_649e-23         # J/K (exata)
H: Final[float] = 6.626_070_15e-34        # J*s (exata)
HBAR: Final[float] = 1.054_571_817e-34    # J*s
EPS0: Final[float] = 8.854_187_8128e-12   # F/m
MU0: Final[float] = 1.256_637_062_12e-6   # N/A^2

# =========================
# Constantes astronômicas SI
# =========================
M_SUN: Final[float] = 1.988_47e30         # kg
R_SUN: Final[float] = 6.957_00e8          # m
M_EARTH: Final[float] = 5.972_2e24        # kg
R_EARTH: Final[float] = 6.371_0e6         # m

AU: Final[float] = 1.495_978_707e11       # m (definição IAU)
PC: Final[float] = 3.085_677_581_491_367e16  # m
KPC: Final[float] = 1.0e3 * PC            # m
MPC: Final[float] = 1.0e6 * PC            # m
LY: Final[float] = 9.460_730_472_580_8e15 # m

DAY: Final[float] = 8.640_0e4             # s
YEAR_JULIAN: Final[float] = 3.155_76e7    # s (365.25 dias)

# =========================
# Números úteis (dimensionais)
# =========================
PI: Final[float] = 3.141_592_653_589_793e0
TWO_PI: Final[float] = 6.283_185_307_179_586e0

# Tolerâncias numéricas típicas
EPS: Final[float] = 1.0e-12               # genérico
HORIZON_EPS_M: Final[float] = 1.0e-9      # margem mínima (em unidades de M geométrico) para evento de horizonte


# ==========================================
# Conversões para unidades geométricas (G=c=1)
# ==========================================
def mass_to_M_length(mass_kg: float) -> float:
    """Converte massa (kg) para 'M' como comprimento (metros): M_len = G*M/c^2."""
    return (G * mass_kg) / (C * C)


def mass_to_M_time(mass_kg: float) -> float:
    """Converte massa (kg) para 'M' como tempo (segundos): M_time = G*M/c^3."""
    return (G * mass_kg) / (C * C * C)


def M_length_to_mass(M_len_m: float) -> float:
    """Converte 'M' (comprimento, em m) para massa (kg): M = M_len*c^2/G."""
    return (M_len_m * C * C) / G


def M_time_to_mass(M_time_s: float) -> float:
    """Converte 'M' (tempo, em s) para massa (kg): M = M_time*c^3/G."""
    return (M_time_s * C * C * C) / G


def schwarzschild_radius(mass_kg: float) -> float:
    """Raio de Schwarzschild em metros: r_s = 2GM/c^2."""
    return 2.0e0 * (G * mass_kg) / (C * C)


def gravitational_parameter(mass_kg: float) -> float:
    """Parâmetro gravitacional mu = GM em SI (m^3/s^2). Útil no Newtoniano."""
    return G * mass_kg


# =========================
# Estrutura de parâmetros BH
# =========================
@dataclass(frozen=True)
class BlackHoleSI:
    """
    Representa um buraco negro por sua massa em SI e provê conversões úteis.

    Campos derivados:
    - mu: GM (m^3/s^2)
    - M_len: GM/c^2 (m)    [= 'M' geométrico como comprimento]
    - M_time: GM/c^3 (s)   [= 'M' geométrico como tempo]
    - r_s: 2GM/c^2 (m)
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

    def to_geom_length(self, length_m: float) -> float:
        """Converte comprimento SI (m) para unidades de M (geométrico, comprimento): x/M."""
        return length_m / self.M_len

    def to_geom_time(self, time_s: float) -> float:
        """Converte tempo SI (s) para unidades de M (geométrico, tempo): t/M."""
        return time_s / self.M_time

    def from_geom_length(self, x_over_M: float) -> float:
        """Converte comprimento em unidades de M para metros."""
        return x_over_M * self.M_len

    def from_geom_time(self, t_over_M: float) -> float:
        """Converte tempo em unidades de M para segundos."""
        return t_over_M * self.M_time


# =========================
# Atalhos práticos p/ relatório
# =========================
def fmt_sci(x: float, sig: int = 6) -> str:
    """
    Formata número em notação científica com casas decimais controladas.
    Ex.: 1.23456e+03
    """
    return f"{x:.{sig}e}"


def example_bh_solar_mass(multiples_of_solar_mass: float) -> BlackHoleSI:
    """Cria um BH com massa em múltiplos de M_sun (ex.: 4e6 para Sgr A* aproximado)."""
    return BlackHoleSI(mass_kg=multiples_of_solar_mass * M_SUN)


if __name__ == "__main__":
    # Sanity check rápido (não é teste formal)
    bh = example_bh_solar_mass(1.0e0)
    print("BH = 1 M_sun")
    print("mu (m^3/s^2):", fmt_sci(bh.mu))
    print("M_len = GM/c^2 (m):", fmt_sci(bh.M_len))
    print("M_time = GM/c^3 (s):", fmt_sci(bh.M_time))
    print("r_s = 2GM/c^2 (m):", fmt_sci(bh.r_s))
