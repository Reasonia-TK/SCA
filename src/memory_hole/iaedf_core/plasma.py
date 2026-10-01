"""プラズマ導出量（BKM 1D/2Dノートブックのグローバル定義を関数化）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import AMU, EPS0, ME, QE
from .schemas import PlasmaConfig


@dataclass(frozen=True)
class PlasmaDerived:
    ion_mass: float
    te: float
    n_s: float
    f_rf: float
    omega: float
    rf_period: float
    bohm_speed: float
    j_ion: float
    j_e_sat: float
    debye_length: float
    floating_drop: float


def derive_plasma(config: PlasmaConfig) -> PlasmaDerived:
    ion_mass = config.ion_mass_amu * AMU
    te = config.electron_temperature_eV
    n_s = config.sheath_edge_density_m3
    f_rf = config.frequency_Hz
    bohm_speed = float(np.sqrt(QE * te / ion_mass))
    j_ion = QE * n_s * bohm_speed
    j_e_sat = QE * n_s * float(np.sqrt(QE * te / (2.0 * np.pi * ME)))
    debye_length = float(np.sqrt(EPS0 * te / (QE * n_s)))
    floating_drop = te * float(np.log(j_e_sat / j_ion))
    return PlasmaDerived(
        ion_mass=ion_mass,
        te=te,
        n_s=n_s,
        f_rf=f_rf,
        omega=2.0 * np.pi * f_rf,
        rf_period=1.0 / f_rf,
        bohm_speed=bohm_speed,
        j_ion=j_ion,
        j_e_sat=j_e_sat,
        debye_length=debye_length,
        floating_drop=floating_drop,
    )


def child_langmuir_width(voltage_V, derived: PlasmaDerived):
    voltage = np.maximum(np.asarray(voltage_V, dtype=float), 1.0e-9)
    prefactor = (4.0 / 9.0) * EPS0 * np.sqrt(2.0 * QE / derived.ion_mass)
    width = np.sqrt(prefactor * voltage**1.5 / derived.j_ion)
    return np.maximum(width, derived.debye_length)


def sheath_capacitance(voltage_V, derived: PlasmaDerived, capacitance_factor: float):
    return capacitance_factor * EPS0 / child_langmuir_width(voltage_V, derived)


def conduction_current(vs, derived: PlasmaDerived):
    return derived.j_ion - derived.j_e_sat * np.exp(-np.minimum(np.maximum(vs, 0.0) / derived.te, 100.0))
