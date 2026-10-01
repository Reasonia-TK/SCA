"""実行結果からプロット用JSONとNPZ保存データを作る。"""

from __future__ import annotations

import numpy as np

from .model2d.field import MAT_INSULATOR, MAT_RING
from .model2d.field import wafer_ranges as _wafer_ranges
from .plasma import child_langmuir_width
from .schemas import Config1D, Config2D


def _downsample(array, n=512):
    array = np.asarray(array)
    if array.size <= n:
        return array
    idx = np.linspace(0, array.size - 1, n).astype(int)
    return array[idx]


def _finite_list(a):
    return [float(v) for v in np.asarray(a, dtype=float)]


def _angle_bins(default_bins: int, step_deg) -> int:
    """角度刻み[deg]指定時はビン数(180/刻み)へ変換する（指定時優先）。"""
    if step_deg is not None and step_deg > 0:
        return int(np.clip(round(180.0 / step_deg), 10, 2000))
    return default_bins


# ---------------- 1D ----------------


def build_plots_1d(output: dict, config: Config1D) -> dict:
    circuit = output["circuit_solution"]
    results = output["results"]
    plot_cfg = config.plot

    deg = np.degrees(circuit["phase"])
    vp_plot = {
        "phase_deg": _finite_list(_downsample(deg)),
        "V_e": _finite_list(_downsample(circuit["V_e"])),
        "V_p": _finite_list(_downsample(circuit["V_p"])),
        "V_sp": _finite_list(_downsample(circuit["V_sp"])),
        "V_sg": _finite_list(_downsample(circuit["V_sg"])),
    }

    # シース厚さ波形: TPMCの電子フロント s_e(t) と回路容量のChild瞬時幅
    derived = output["derived"]
    s_max = float(output["s_max"])
    vsp = np.asarray(circuit["V_sp"])
    vsp_max = float(np.max(vsp))
    if config.sheath.model == "moving_front":
        s_e = s_max * (np.maximum(vsp, 1e-9) / vsp_max) ** config.sheath.front_width_exponent
    else:
        s_e = np.full_like(vsp, s_max)
    sheath_plot = {
        "phase_deg": _finite_list(_downsample(deg)),
        "s_e_mm": _finite_list(_downsample(s_e * 1e3)),
        "s_cl_powered_mm": _finite_list(_downsample(child_langmuir_width(vsp, derived) * 1e3)),
        "s_cl_ground_mm": _finite_list(
            _downsample(child_langmuir_width(np.asarray(circuit["V_sg"]), derived) * 1e3)
        ),
        "s_max_mm": s_max * 1e3,
    }

    all_energy = np.concatenate([r["energy_eV"][r["reached"]] for r in results])
    if plot_cfg.energy_max_eV is not None:
        energy_max = float(plot_cfg.energy_max_eV)
    else:
        energy_max = float(np.percentile(all_energy, 99.7) * 1.05) if all_energy.size else 1.0

    iedf, iadf, iaedf, summary_rows = [], [], [], []
    rf_period = output["derived"].rf_period
    for r in results:
        ok = r["reached"]
        E = r["energy_eV"][ok]
        A = np.abs(r["signed_angle_deg"][ok])
        summary_rows.append(
            {
                "pressure_mTorr": r["pressure_mTorr"],
                "hit_percent": float(100 * np.mean(ok)),
                "mean_energy_eV": float(np.mean(E)) if E.size else None,
                "e05_eV": float(np.percentile(E, 5)) if E.size else None,
                "e95_eV": float(np.percentile(E, 95)) if E.size else None,
                "mean_abs_angle_deg": float(np.mean(A)) if A.size else None,
                "angle95_deg": float(np.percentile(A, 95)) if A.size else None,
                "mean_transit_over_T": float(np.mean(r["transit_s"][ok]) / rf_period) if E.size else None,
                "cx_percent": float(100 * np.mean(r["n_cx"][ok] > 0)) if E.size else None,
                "escaped": int(np.sum(r["escaped"])),
                "cutoff": int(np.sum(r["cutoff"])),
            }
        )
        hist, edges = np.histogram(E, bins=plot_cfg.energy_bins, range=(0, energy_max), density=True)
        iedf.append(
            {
                "pressure_mTorr": r["pressure_mTorr"],
                "edges_eV": _finite_list(edges),
                "density": _finite_list(hist),
            }
        )
        ahist, aedges = np.histogram(
            r["signed_angle_deg"][ok],
            bins=_angle_bins(2 * plot_cfg.angle_bins, plot_cfg.angle_step_deg),
            range=(-90, 90),
            density=True,
        )
        centers = 0.5 * (aedges[:-1] + aedges[1:])
        iadf.append(
            {
                "pressure_mTorr": r["pressure_mTorr"],
                "angle_deg": _finite_list(centers),
                "density": _finite_list(ahist),
            }
        )
        h2, a_edges, e_edges = np.histogram2d(
            r["signed_angle_deg"][ok],
            E,
            bins=(_angle_bins(plot_cfg.angle_bins, plot_cfg.angle_step_deg), plot_cfg.energy_bins // 2),
            range=((-90, 90), (0, energy_max)),
            density=True,
        )
        iaedf.append(
            {
                "pressure_mTorr": r["pressure_mTorr"],
                "angle_edges_deg": _finite_list(a_edges),
                "energy_edges_eV": _finite_list(e_edges),
                "density": [[float(v) for v in row] for row in h2.T],
            }
        )

    return {
        "model": "1d",
        "vp_waveform": vp_plot,
        "sheath_width": sheath_plot,
        "energy_max_eV": energy_max,
        "summary_rows": summary_rows,
        "iedf": iedf,
        "iadf": iadf,
        "iaedf": iaedf,
        "scalars": {
            "s_max_mm": output["s_max"] * 1e3,
            "tau_ion_ns": output["tau_ion"] * 1e9,
            "tau_ion_over_T": output["tau_ion"] * output["derived"].f_rf,
            "debye_mm": output["derived"].debye_length * 1e3,
            "floating_drop_V": output["derived"].floating_drop,
            "vsp_min_V": float(np.min(circuit["V_sp"])),
            "vsp_max_V": float(np.max(circuit["V_sp"])),
            "tpmc_seconds": output["tpmc_seconds"],
        },
    }


def save_npz_1d(path, output: dict):
    circuit = output["circuit_solution"]
    arrays = {
        "phase": circuit["phase"],
        "V_e": circuit["V_e"],
        "V_p": circuit["V_p"],
        "V_sp": circuit["V_sp"],
        "V_sg": circuit["V_sg"],
    }
    for i, r in enumerate(output["results"]):
        for key in (
            "energy_eV",
            "signed_angle_deg",
            "transit_s",
            "impact_phase_deg",
            "n_cx",
            "n_elastic",
            "reached",
            "escaped",
            "cutoff",
            "vx_m_s",
            "vy_m_s",
            "vz_m_s",
        ):
            arrays[f"p{i}_{key}"] = r[key]
        arrays[f"p{i}_pressure_mTorr"] = np.array(r["pressure_mTorr"])
    arrays["velocity_frame"] = np.array("sca_surface_local_xyz")
    np.savez_compressed(path, **arrays)


# ---------------- 2D ----------------


def build_plots_2d(output: dict, config: Config2D) -> dict:
    circuit = output["circuit_solution"]
    geo = config.geometry
    analysis = config.analysis
    length = geo.domain_length_m

    deg = np.degrees(circuit["phase"])
    vp_plot = {
        "phase_deg": _finite_list(_downsample(deg)),
        "V_w": _finite_list(_downsample(circuit["V_w"])),
        "V_r": _finite_list(_downsample(circuit["V_r"])),
        "V_p": _finite_list(_downsample(circuit["V_p"])),
        "V_sw": _finite_list(_downsample(circuit["V_sw"])),
        "V_sr": _finite_list(_downsample(circuit["V_sr"])),
    }

    # シース厚さ波形（回路容量と同じChild瞬時幅。2Dは電子フロントを持たない）
    derived = output["derived"]
    sheath_plot = {
        "phase_deg": _finite_list(_downsample(deg)),
        "s_cl_wafer_mm": _finite_list(
            _downsample(child_langmuir_width(np.asarray(circuit["V_sw"]), derived) * 1e3)
        ),
        "s_cl_ring_mm": _finite_list(
            _downsample(child_langmuir_width(np.asarray(circuit["V_sr"]), derived) * 1e3)
        ),
        "s_max_mm": float(output["sheath_scale"]) * 1e3,
    }

    surface = output["basis"]["plasma"]
    points = np.asarray(geo.points_m, dtype=float)
    geometry_plot = {
        "x_mm": _finite_list(np.asarray(surface["x"]) * 1e3),
        "surface_mm": _finite_list(np.asarray(surface["surface"]) * 1e3),
        "points_mm": [[float(p[0] * 1e3), float(p[1] * 1e3)] for p in points],
        "segment_materials": list(geo.segment_materials),
        "wafer_ranges_mm": [[float(a * 1e3), float(b * 1e3)] for a, b in _wafer_ranges(geo)],
    }

    profiles, iedf, iaedf, summary_rows, phi_sc_maps = [], [], [], [], []
    bin_w = 0.2e-3
    edges = np.arange(0.0, length + bin_w, bin_w)
    centers_x = 0.5 * (edges[:-1] + edges[1:])
    for res in output["results"]:
        run = res["run"]
        pressure = res["pressure"]
        counts, _ = np.histogram(run["impact_x_m"], bins=edges)
        e_sum, _ = np.histogram(run["impact_x_m"], bins=edges, weights=run["energy_eV"])
        a_sum, _ = np.histogram(run["impact_x_m"], bins=edges, weights=run["angle_deg"])
        n_total = max(run["energy_eV"].size, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_e = e_sum / np.maximum(counts, 1)
            mean_a = a_sum / np.maximum(counts, 1)
        profiles.append(
            {
                "pressure_mTorr": pressure,
                "x_mm": _finite_list(centers_x * 1e3),
                "flux": _finite_list(counts / bin_w / n_total * 1e-3),
                "mean_energy_eV": _finite_list(mean_e),
                "mean_angle_deg": _finite_list(mean_a),
            }
        )

        e_all = run["energy_eV"]
        e_max = float(np.percentile(e_all, 99.7)) * 1.05 if e_all.size else 1.0
        onw = run["on_wafer"]
        material = run["impact_material"]
        wafer_hist, wedges = np.histogram(
            e_all[onw], bins=analysis.energy_bins, range=(0, e_max), density=True
        )
        entry = {
            "pressure_mTorr": pressure,
            "edges_eV": _finite_list(wedges),
            "wafer_density": _finite_list(wafer_hist),
        }
        ring_mask = material == MAT_RING
        if np.any(ring_mask):
            ring_hist, _ = np.histogram(
                e_all[ring_mask], bins=analysis.energy_bins, range=(0, e_max), density=True
            )
            entry["ring_density"] = _finite_list(ring_hist)
        ins_mask = material == MAT_INSULATOR
        if np.any(ins_mask):
            ins_hist, _ = np.histogram(
                e_all[ins_mask], bins=analysis.energy_bins, range=(0, e_max), density=True
            )
            entry["insulator_density"] = _finite_list(ins_hist)
        iedf.append(entry)

        if np.any(onw):
            ew = e_all[onw]
            e_max_w = float(np.percentile(ew, 99.7)) * 1.05
            h2, a_edges, e_edges = np.histogram2d(
                run["angle_deg"][onw],
                ew,
                bins=(_angle_bins(analysis.angle_bins, analysis.angle_step_deg), analysis.energy_bins // 2),
                range=((-90, 90), (0, e_max_w)),
                density=True,
            )
            iaedf.append(
                {
                    "pressure_mTorr": pressure,
                    "angle_edges_deg": _finite_list(a_edges),
                    "energy_edges_eV": _finite_list(e_edges),
                    "density": [[float(v) for v in row] for row in h2.T],
                }
            )

        summary = dict(res["summary"])
        summary["pressure_mTorr"] = pressure
        summary["n_reached"] = int(run["energy_eV"].size)
        summary["n_lost"] = int(run["n_lost"])
        summary["elapsed_s"] = res["elapsed_s"]
        summary_rows.append(summary)

        if res["sc"] is not None:
            phi = np.asarray(res["sc"]["phi_sc"])
            step_y = max(1, phi.shape[0] // 60)
            step_x = max(1, phi.shape[1] // 128)
            sub = phi[::step_y, ::step_x]
            phi_sc_maps.append(
                {
                    "pressure_mTorr": pressure,
                    "x_mm": _finite_list(np.asarray(surface["x"])[::step_x] * 1e3),
                    "y_mm": _finite_list(np.asarray(surface["y"])[::step_y] * 1e3),
                    "phi_sc_V": [[float(v) for v in row] for row in sub],
                }
            )

    return {
        "model": "2d",
        "vp_waveform": vp_plot,
        "sheath_width": sheath_plot,
        "geometry": geometry_plot,
        "profiles": profiles,
        "iedf": iedf,
        "iaedf": iaedf,
        "phi_sc": phi_sc_maps,
        "summary_rows": summary_rows,
        "scalars": {
            "kcl_residual": output["kcl_residual"],
            "partition_error": output["partition_error"],
            "domain_top_mm": output["domain_top"] * 1e3,
            "sheath_scale_mm": output["sheath_scale"] * 1e3,
            "vsw_max_V": float(np.max(circuit["V_sw"])),
            "vsr_max_V": float(np.max(circuit["V_sr"])),
        },
    }


def save_npz_2d(path, output: dict):
    circuit = output["circuit_solution"]
    arrays = {
        "phase": circuit["phase"],
        "V_p": circuit["V_p"],
        "V_w": circuit["V_w"],
        "V_r": circuit["V_r"],
        "V_sw": circuit["V_sw"],
        "V_sr": circuit["V_sr"],
        "V_sg": circuit["V_sg"],
    }
    for i, res in enumerate(output["results"]):
        run = res["run"]
        for key in (
            "energy_eV",
            "angle_deg",
            "impact_x_m",
            "on_wafer",
            "impact_segment",
            "impact_material",
            "vx_m_s",
            "vy_m_s",
            "vz_m_s",
            "impact_phase_deg",
        ):
            arrays[f"p{i}_{key}"] = run[key]
        arrays[f"p{i}_pressure_mTorr"] = np.array(res["pressure"])
        if res["sc"] is not None:
            arrays[f"p{i}_phi_sc"] = res["sc"]["phi_sc"]
    arrays["velocity_frame"] = np.array("sca_surface_local_xyz")
    np.savez_compressed(path, **arrays)
