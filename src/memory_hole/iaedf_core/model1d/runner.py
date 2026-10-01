"""BKM 1D ランナー: 回路解 -> TPMC -> 検証（ノートブックの実行フローを関数化）。"""

from __future__ import annotations

import time

import numpy as np

from ..constants import QE
from ..plasma import child_langmuir_width, derive_plasma
from ..schemas import Config1D
from ..waveform import make_derivative_function, make_waveform_function
from ..xsec import build_cross_sections
from .circuit import solve_periodic_plasma_potential
from .tpmc import run_tpmc


def run_1d(config: Config1D, *, xsec_text: str | None = None, progress_cb=None):
    """1Dベストモデル一式を実行し、結果と検証値を返す。

    progress_cb(fraction: 0..1, text: str) は任意。
    """

    def report(frac, text):
        if progress_cb is not None:
            progress_cb(float(frac), text)

    derived = derive_plasma(config.plasma)
    sigma_cx, sigma_el = build_cross_sections(config.gas, xsec_text)
    voltage_func = make_waveform_function(config.waveform, derived.omega)
    dvoltage_func = make_derivative_function(voltage_func)

    report(0.02, "プラズマ電位（KCL）を解いています")
    circuit_solution = solve_periodic_plasma_potential(voltage_func, dvoltage_func, derived, config.circuit)
    vsp_table = circuit_solution["V_sp"]
    vsp_max = float(np.max(vsp_table))
    s_max = float(child_langmuir_width(vsp_max, derived))
    tau_ion = 3.0 * s_max * np.sqrt(derived.ion_mass / (2.0 * QE * float(np.mean(vsp_table))))

    pressures = list(config.gas.pressures_mTorr)
    results = []
    t0 = time.perf_counter()
    for k, pressure in enumerate(pressures):
        base = 0.05 + 0.85 * k / len(pressures)
        span = 0.85 / len(pressures)
        report(base, f"TPMC実行中 p={pressure:g} mTorr ({k + 1}/{len(pressures)})")
        result = run_tpmc(
            pressure,
            derived=derived,
            sheath=config.sheath,
            gas=config.gas,
            tpmc=config.tpmc,
            sigma_cx=sigma_cx,
            sigma_el=sigma_el,
            vsp_table=vsp_table,
            s_max=s_max,
            seed=config.tpmc.seed + 10_000 * k,
            magnetic=config.magnetic,
            progress_cb=lambda f, _b=base, _s=span: report(_b + _s * f, None),
        )
        results.append(result)
    tpmc_seconds = time.perf_counter() - t0

    report(0.92, "数値検証を実行しています")
    validation = validate_1d(config, derived, circuit_solution, results, sigma_cx, sigma_el, s_max, tau_ion)
    report(1.0, "完了")
    return {
        "derived": derived,
        "circuit_solution": circuit_solution,
        "s_max": s_max,
        "tau_ion": tau_ion,
        "results": results,
        "tpmc_seconds": tpmc_seconds,
        "validation": validation,
    }


def validate_1d(config: Config1D, derived, circuit_solution, results, sigma_cx, sigma_el, s_max, tau_ion):
    """ノートブックの検証セルと同一のチェック。"""
    vsp_table = circuit_solution["V_sp"]
    vsp_max = float(np.max(vsp_table))
    static_v = vsp_max
    static_check = run_tpmc(
        0.0,
        derived=derived,
        sheath=config.sheath,
        gas=config.gas,
        tpmc=config.tpmc,
        sigma_cx=sigma_cx,
        sigma_el=sigma_el,
        vsp_table=np.full_like(vsp_table, static_v),
        s_max=s_max,
        seed=config.tpmc.seed + 999_999,
        n_particles=min(3000, config.tpmc.n_particles),
        magnetic=config.magnetic,
    )
    ok = static_check["reached"]
    initial_ke = (
        0.5
        * derived.ion_mass
        * (derived.bohm_speed**2 + 3.0 * QE * config.tpmc.ion_temperature_eV / derived.ion_mass)
        / QE
    )
    gain = float(np.mean(static_check["energy_eV"][ok]) - initial_ke)
    energy_rel_error = abs(gain - static_v) / static_v

    largest_p = max(r["max_p_collision"] for r in results)
    collision_ok = largest_p <= config.tpmc.max_recommended_collision_probability

    omega = derived.omega
    v_tilde_eff = 0.5 * (vsp_table.max() - vsp_table.min())
    de_riley = 2.0 * v_tilde_eff / np.sqrt(1.0 + (omega * tau_ion / 4.0) ** 2)

    magnetic_checks = magnetic_validation(
        config.magnetic,
        derived,
        dt=derived.rf_period / config.tpmc.steps_per_rf_period,
        v_char=np.sqrt(2.0 * QE * vsp_max / derived.ion_mass),
        transit_s=tau_ion,
        sheath_scale=s_max,
    )

    return {
        **magnetic_checks,
        "static_expected_gain_eV": static_v,
        "static_tpmc_gain_eV": gain,
        "energy_conservation_rel_error": float(energy_rel_error),
        "energy_conservation_ok": bool(energy_rel_error < 1.0e-3),
        "max_step_collision_probability": float(largest_p),
        "collision_probability_ok": bool(collision_ok),
        "periodic_error_V": float(circuit_solution["periodic_error_V"]),
        "periodic_cycles": int(circuit_solution["cycles"]),
        "riley_delta_E_eV": float(de_riley),
        "riley_v_tilde_eff_V": float(v_tilde_eff),
        "omega_tau_ion_over_4": float(omega * tau_ion / 4.0),
        "passed": bool(
            energy_rel_error < 1.0e-3 and collision_ok and magnetic_checks.get("gyration_resolution_ok", True)
        ),
    }


def magnetic_validation(magnetic, derived, *, dt, v_char, transit_s, sheath_scale):
    """静磁場有効時の検証項目（無効時は空dictを返す）。

    Borisは大きなω_ci·dtでも安定だが、精度確保の目安として0.3未満を要求する。
    """
    if magnetic is None or not magnetic.enabled:
        return {}
    omega_ci = QE * magnetic.magnitude / derived.ion_mass
    gyroradius = v_char / omega_ci
    return {
        "magnetic_field_T": float(magnetic.magnitude),
        "omega_ci_dt": float(omega_ci * dt),
        "gyration_resolution_ok": bool(omega_ci * dt < 0.3),
        "ion_gyroradius_m": float(gyroradius),
        "gyroradius_to_sheath_ratio": float(gyroradius / sheath_scale),
        "magnetic_deflection_deg": float(np.degrees(omega_ci * transit_s)),
    }
