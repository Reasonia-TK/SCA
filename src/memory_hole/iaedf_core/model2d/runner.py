"""BKM 2D ランナー: 回路 -> Laplace基底 -> 空間電荷 -> TPMC -> 検証。"""

from __future__ import annotations

import time

import numpy as np

from ..plasma import child_langmuir_width, derive_plasma
from ..schemas import Config2D
from ..waveform import make_derivative_function, make_waveform_function
from ..xsec import build_cross_sections
from .analysis import edge_summary
from .circuit import circuit_residual, solve_periodic_plasma_potential
from .field import build_basis, max_surface_height
from .space_charge import build_space_charge
from .tpmc import run_tpmc_2d


def run_2d(config: Config2D, *, xsec_text: str | None = None, progress_cb=None, log_cb=None):
    def report(frac, text):
        if progress_cb is not None:
            progress_cb(float(frac), text)

    def log(msg):
        if log_cb is not None:
            log_cb(msg)

    derived = derive_plasma(config.plasma)
    sigma_cx, sigma_el = build_cross_sections(config.gas, xsec_text)
    wafer_func = make_waveform_function(config.wafer_waveform, derived.omega)
    ring_func = make_waveform_function(config.ring_waveform, derived.omega, wafer_func=wafer_func)
    dwafer_func = make_derivative_function(wafer_func)
    dring_func = make_derivative_function(ring_func)

    report(0.02, "多電極プラズマ電位（KCL）を解いています")
    circuit_solution = solve_periodic_plasma_potential(
        wafer_func, ring_func, dwafer_func, dring_func, derived, config.circuit, config.electrodes
    )
    kcl_residual = circuit_residual(
        circuit_solution,
        wafer_func,
        ring_func,
        dwafer_func,
        dring_func,
        derived,
        config.circuit,
        config.electrodes,
    )

    vsw_max = float(np.max(circuit_solution["V_sw"]))
    vsr_max = float(np.max(circuit_solution["V_sr"]))
    sheath_scale = float(child_langmuir_width(max(vsw_max, vsr_max), derived))
    geo = config.geometry
    domain_top = max_surface_height(geo) + geo.top_clearance_factor * sheath_scale

    report(0.08, "2D Laplace 3基底を解いています")
    t0 = time.perf_counter()
    basis, partition_error = build_basis(geo, config.field2d, domain_top)
    basis_seconds = time.perf_counter() - t0
    log(f"3基底の解時間 {basis_seconds:.1f} s, partition-of-unity誤差 {partition_error:.3e}")

    pressures = list(config.gas.pressures_mTorr)
    results = []
    for k, pressure in enumerate(pressures):
        base = 0.20 + 0.72 * k / len(pressures)
        span = 0.72 / len(pressures)
        t0 = time.perf_counter()
        common = dict(
            derived=derived,
            geo=geo,
            f2d=config.field2d,
            gas=config.gas,
            tpmc=config.tpmc,
            sigma_cx=sigma_cx,
            sigma_el=sigma_el,
            basis=basis,
            circuit_solution=circuit_solution,
            domain_top=domain_top,
            magnetic=config.magnetic,
        )
        if config.space_charge.enabled:
            report(base, f"空間電荷反復中 p={pressure:g} mTorr")
            sc = build_space_charge(
                pressure,
                config.tpmc.seed + 100_000 * (k + 1),
                sc=config.space_charge,
                log_cb=log,
                progress_cb=lambda f, _b=base, _s=span: report(_b + 0.7 * _s * f, None),
                **common,
            )
            correction = {"Ex": sc["Ex"], "Ey": sc["Ey"]}
        else:
            sc = None
            correction = None
        report(base + 0.7 * span, f"TPMC実行中 p={pressure:g} mTorr")
        run = run_tpmc_2d(
            pressure,
            n_particles=config.tpmc.n_particles,
            seed=config.tpmc.seed + 10_000 * k,
            correction=correction,
            progress_cb=lambda f, _b=base + 0.7 * span, _s=0.3 * span: report(_b + _s * f, None),
            **common,
        )
        elapsed = time.perf_counter() - t0
        log(
            f"p={pressure:g} mTorr: TPMC到達 {run['energy_eV'].size}"
            f"/{config.tpmc.n_particles} (未了 {run['n_lost']}), "
            f"合計 {elapsed:.1f} s"
        )
        results.append(
            {
                "pressure": pressure,
                "run": run,
                "sc": sc,
                "summary": edge_summary(run, geo, config.analysis),
                "elapsed_s": elapsed,
            }
        )

    report(0.95, "数値検証を実行しています")
    validation = validate_2d(
        config,
        circuit_solution,
        kcl_residual,
        partition_error,
        basis,
        results,
        derived=derived,
        sheath_scale=sheath_scale,
    )
    report(1.0, "完了")
    return {
        "derived": derived,
        "circuit_solution": circuit_solution,
        "kcl_residual": kcl_residual,
        "basis": basis,
        "partition_error": partition_error,
        "domain_top": domain_top,
        "sheath_scale": sheath_scale,
        "results": results,
        "validation": validation,
    }


def validate_2d(
    config: Config2D,
    circuit_solution,
    kcl_residual,
    partition_error,
    basis,
    results,
    *,
    derived,
    sheath_scale,
):
    """ノートブックの検証セルと同一のチェック（+静磁場有効時の追加項目）。"""
    from ..constants import QE
    from ..model1d.runner import magnetic_validation

    largest_p = max(res["run"]["max_p_collision"] for res in results)
    collision_ok = largest_p <= config.tpmc.max_recommended_collision_probability
    sc_histories = {}
    sc_ok = True
    if config.space_charge.enabled:
        for res in results:
            if res["sc"] is not None:
                sc_histories[str(res["pressure"])] = res["sc"]["history"]
                if res["sc"]["history"][-1] > 5.0:
                    sc_ok = False
    min_sheath = float(min(circuit_solution["V_sw"].min(), circuit_solution["V_sr"].min()))
    reverse_sheath = min_sheath <= 0.0
    basis_residuals = {k: float(v["residual"]) for k, v in basis.items()}
    vs_max = max(float(circuit_solution["V_sw"].max()), float(circuit_solution["V_sr"].max()))
    vs_mean = max(float(np.mean(circuit_solution["V_sw"])), 1e-9)
    magnetic_checks = magnetic_validation(
        config.magnetic,
        derived,
        dt=derived.rf_period / config.tpmc.steps_per_rf_period,
        v_char=np.sqrt(2.0 * QE * vs_max / derived.ion_mass),
        transit_s=3.0 * sheath_scale * np.sqrt(derived.ion_mass / (2.0 * QE * vs_mean)),
        sheath_scale=sheath_scale,
    )
    passed = bool(
        collision_ok
        and sc_ok
        and not reverse_sheath
        and partition_error < 1.0e-3
        and magnetic_checks.get("gyration_resolution_ok", True)
    )
    return {
        **magnetic_checks,
        "periodic_error_V": float(circuit_solution["periodic_error_V"]),
        "periodic_cycles": int(circuit_solution["cycles"]),
        "kcl_max_relative_residual": float(kcl_residual),
        "partition_of_unity_error": float(partition_error),
        "basis_scaled_residuals": basis_residuals,
        "space_charge_histories": sc_histories,
        "space_charge_converged": bool(sc_ok),
        "max_step_collision_probability": float(largest_p),
        "collision_probability_ok": bool(collision_ok),
        "min_sheath_voltage_V": min_sheath,
        "reverse_sheath_warning": bool(reverse_sheath),
        "passed": passed,
    }
