"""2D 多電極自己無撞着プラズマ電位（ウェハ/リング/接地の3シースKCL）。"""

from __future__ import annotations

import warnings

import numpy as np

from ..plasma import PlasmaDerived, conduction_current, sheath_capacitance
from ..schemas import Circuit2DConfig, ElectrodesConfig


def make_dvp_dphase(
    wafer_func,
    ring_func,
    dwafer_func,
    dring_func,
    derived: PlasmaDerived,
    circuit: Circuit2DConfig,
    electrodes: ElectrodesConfig,
):
    area_w = electrodes.wafer_to_ground_area_ratio
    area_r = electrodes.ring_to_ground_area_ratio
    v_ground = electrodes.ground_voltage_V
    cap_factor = circuit.capacitance_factor
    omega = derived.omega

    def dvp_dphase(phase, vp):
        vw = float(wafer_func(phase))
        vr = float(ring_func(phase))
        vsw = max(vp - vw, 0.0)
        vsr = max(vp - vr, 0.0)
        vsg = max(vp - v_ground, 0.0)
        cw = float(sheath_capacitance(vsw, derived, cap_factor))
        cr = float(sheath_capacitance(vsr, derived, cap_factor))
        cg = float(sheath_capacitance(vsg, derived, cap_factor))
        jw = float(conduction_current(vsw, derived))
        jr = float(conduction_current(vsr, derived))
        jg = float(conduction_current(vsg, derived))
        numerator = area_w * cw * float(dwafer_func(phase)) + area_r * cr * float(dring_func(phase))
        numerator -= (area_w * jw + area_r * jr + jg) / omega
        return numerator / (area_w * cw + area_r * cr + cg)

    return dvp_dphase


def solve_periodic_plasma_potential(
    wafer_func,
    ring_func,
    dwafer_func,
    dring_func,
    derived: PlasmaDerived,
    circuit: Circuit2DConfig,
    electrodes: ElectrodesConfig,
):
    dvp_dphase = make_dvp_dphase(wafer_func, ring_func, dwafer_func, dring_func, derived, circuit, electrodes)
    n = int(circuit.phase_points)
    dp = 2.0 * np.pi / n
    phase = np.arange(n) * dp
    v_ground = electrodes.ground_voltage_V
    vp = v_ground + derived.floating_drop
    values = np.empty(n)
    error = np.inf
    for cycle in range(1, int(circuit.max_cycles) + 1):  # noqa: B007 - returned below
        start = vp
        for i, p in enumerate(phase):
            values[i] = vp
            k1 = dvp_dphase(p, vp)
            k2 = dvp_dphase(p + dp / 2, vp + dp * k1 / 2)
            k3 = dvp_dphase(p + dp / 2, vp + dp * k2 / 2)
            k4 = dvp_dphase(p + dp, vp + dp * k3)
            vp += dp * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        error = abs(vp - start)
        if error < circuit.periodic_tolerance_V:
            break
    else:
        warnings.warn("Vpが周期定常状態へ収束しませんでした。", stacklevel=2)
    vw = np.asarray(wafer_func(phase))
    vr = np.asarray(ring_func(phase))
    return {
        "phase": phase,
        "V_p": values.copy(),
        "V_w": vw,
        "V_r": vr,
        "V_sw": np.maximum(values - vw, 0.0),
        "V_sr": np.maximum(values - vr, 0.0),
        "V_sg": np.maximum(values - v_ground, 0.0),
        "cycles": cycle,
        "periodic_error_V": error,
    }


def circuit_residual(
    solution,
    wafer_func,
    ring_func,
    dwafer_func,
    dring_func,
    derived: PlasmaDerived,
    circuit: Circuit2DConfig,
    electrodes: ElectrodesConfig,
):
    dvp_dphase = make_dvp_dphase(wafer_func, ring_func, dwafer_func, dring_func, derived, circuit, electrodes)
    area_w = electrodes.wafer_to_ground_area_ratio
    area_r = electrodes.ring_to_ground_area_ratio
    cap_factor = circuit.capacitance_factor
    omega = derived.omega
    p = solution["phase"]
    vp = solution["V_p"]
    dvp = omega * np.array([dvp_dphase(a, b) for a, b in zip(p, vp, strict=True)])
    dvw = omega * dwafer_func(p)
    dvr = omega * dring_func(p)
    cw = sheath_capacitance(solution["V_sw"], derived, cap_factor)
    cr = sheath_capacitance(solution["V_sr"], derived, cap_factor)
    cg = sheath_capacitance(solution["V_sg"], derived, cap_factor)
    jw = conduction_current(solution["V_sw"], derived)
    jr = conduction_current(solution["V_sr"], derived)
    jg = conduction_current(solution["V_sg"], derived)
    residual = area_w * (jw + cw * (dvp - dvw)) + area_r * (jr + cr * (dvp - dvr)) + (jg + cg * dvp)
    scale = max(float(np.max(np.abs(np.r_[jw, jr, jg]))), derived.j_ion)
    return float(np.max(np.abs(residual)) / scale)
