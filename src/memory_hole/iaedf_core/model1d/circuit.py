"""1D 自己無撞着プラズマ電位（2シースKCL、ノートブックと同一定式化）。"""

from __future__ import annotations

import warnings

import numpy as np

from ..plasma import PlasmaDerived, conduction_current, sheath_capacitance
from ..schemas import Circuit1DConfig


def solve_periodic_plasma_potential(
    voltage_func, dvoltage_func, derived: PlasmaDerived, circuit: Circuit1DConfig
):
    n_phase = int(circuit.phase_points)
    dphase = 2.0 * np.pi / n_phase
    phases = np.arange(n_phase) * dphase
    area_ratio = circuit.powered_to_grounded_area_ratio
    v_ground = circuit.grounded_electrode_voltage_V
    cap_factor = circuit.capacitance_factor
    omega = derived.omega

    def dvp_dphase(phase, vp):
        ve = float(voltage_func(phase))
        vsp = max(vp - ve, 0.0)
        vsg = max(vp - v_ground, 0.0)
        cp = float(sheath_capacitance(vsp, derived, cap_factor))
        cg = float(sheath_capacitance(vsg, derived, cap_factor))
        jp = float(conduction_current(vsp, derived))
        jg = float(conduction_current(vsg, derived))
        dve = float(dvoltage_func(phase))
        return (area_ratio * cp * dve - (area_ratio * jp + jg) / omega) / (area_ratio * cp + cg)

    vp = v_ground + derived.floating_drop
    values = np.empty(n_phase)
    err = np.inf
    for cycle in range(1, int(circuit.max_cycles) + 1):  # noqa: B007 - returned below
        start = vp
        for i, ph in enumerate(phases):
            values[i] = vp
            k1 = dvp_dphase(ph, vp)
            k2 = dvp_dphase(ph + 0.5 * dphase, vp + 0.5 * dphase * k1)
            k3 = dvp_dphase(ph + 0.5 * dphase, vp + 0.5 * dphase * k2)
            k4 = dvp_dphase(ph + dphase, vp + dphase * k3)
            vp += dphase * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        err = abs(vp - start)
        if err < circuit.periodic_tolerance_V:
            break
    else:
        warnings.warn("Vpが周期定常状態へ収束しませんでした。", stacklevel=2)
    ve = np.asarray(voltage_func(phases), dtype=float)
    return {
        "phase": phases,
        "V_e": ve,
        "V_p": values.copy(),
        "V_sp": np.maximum(values - ve, 0.0),
        "V_sg": np.maximum(values - v_ground, 0.0),
        "cycles": cycle,
        "periodic_error_V": err,
    }
