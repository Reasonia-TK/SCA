from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import time
from pathlib import Path

import h5py
import numpy as np

from . import __version__
from .config import CaseConfig
from .field import FieldSolver, build_mesh
from .geometry import derive
from .inlet import sample_distribution, verify_metadata
from .saturation import SaturationMonitor
from .storage import atomic_json, permission_message, replace_file
from .transport import AMU, E_CHARGE, ELECTRON_MASS, boundary_maps, sample_generated, trace, wilson_interval

SPACE_CHARGE_WARNING = "空間電荷を省略。5%感度検証は未実施。"


def code_id():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def gpu_info():
    try:
        from .cuda_runtime import prepare_cuda_runtime

        prepare_cuda_runtime()
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        free, total = cp.cuda.runtime.memGetInfo()
        return {
            "available": True,
            "name": props["name"].decode(),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "free_bytes": free,
            "total_bytes": total,
            "cupy": cp.__version__,
        }
    except Exception as error:
        return {"available": False, "reason": str(error)}


def provenance(config, derived, distributions):
    deps = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "gmsh", "numba", "h5py", "fastapi")
    }
    return {
        "app_version": __version__,
        "code_sha256": code_id(),
        "config_sha256": config.fingerprint(),
        "geometry_id": derived["geometry_id"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": deps,
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
        if Path("uv.lock").exists()
        else None,
        "gpu": gpu_info(),
        "input_files": [meta for _, meta in distributions.values()],
        "units": {"length": "m", "energy": "eV", "charge": "C", "time": "s"},
        "run_until": config.numerics.run_until,
        "wafer_capacitance_role": "記録のみ。局所電流からウェハ電位を再計算しない。",
    }


def new_stats():
    return dict(
        injected_weight=0.0,
        weight_squared=0.0,
        bottom_weight=0.0,
        bottom_x=0.0,
        bottom_y=0.0,
        bottom_energy=0.0,
        bottom_energy_squared=0.0,
        bottom_weight_squared=0.0,
        escaped_weight=0.0,
        conductor_weight=0.0,
        oxide_weight=0.0,
        unresolved_weight=0.0,
        samples=0,
    )


def new_ledger():
    return dict(
        injected_c=0.0,
        injected_absolute_c=0.0,
        escaped_c=0.0,
        conductor_c=0.0,
        deposited_oxide_c=0.0,
        unresolved_c=0.0,
        unresolved_absolute_c=0.0,
        emitted_c=0.0,
        leaked_c=0.0,
    )


def summarize_stats(stats):
    output = {}
    for name, s in stats.items():
        ci = wilson_interval(s["bottom_weight"], s["injected_weight"], s["weight_squared"])
        w = s["bottom_weight"]
        ess = w * w / s["bottom_weight_squared"] if s["bottom_weight_squared"] else 0
        mean = s["bottom_energy"] / w if w else None
        variance = max(0, s["bottom_energy_squared"] / w - mean * mean) if w else None
        # Unbiased weighted sample variance and normal approximation; single events have no CI.
        half = 1.95996398454 * math.sqrt(variance / (ess - 1)) if ess > 1 else None
        output[name] = {
            **s,
            "bottom_arrival": ci,
            "bottom_centroid_nm": [s["bottom_x"] / w * 1e9, s["bottom_y"] / w * 1e9] if w else None,
            "mean_bottom_energy_ev": mean,
            "energy_ci_half_ev": half,
            "bottom_effective_samples": ess,
        }
    return output


def save_checkpoint(path, state, sigma, histogram, depth_azimuth, meta):
    temp = path.with_suffix(".tmp.h5")
    with h5py.File(temp, "w") as h5:
        h5.attrs["state_json"] = json.dumps(state, ensure_ascii=False, allow_nan=False)
        h5.attrs["metadata_json"] = json.dumps(meta, ensure_ascii=False)
        h5.create_dataset("sigma_c_m2", data=sigma)
        h5.create_dataset("surface_impact_weight", data=histogram)
        h5.create_dataset("depth_azimuth_weight", data=depth_azimuth)
    replace_file(temp, path)


def load_checkpoint(path):
    with h5py.File(path) as h5:
        return (
            json.loads(h5.attrs["state_json"]),
            h5["sigma_c_m2"][:],
            h5["surface_impact_weight"][:],
            h5["depth_azimuth_weight"][:],
            json.loads(h5.attrs["metadata_json"]),
        )


def field_preview(mesh, phi, field):
    sample = np.linspace(0, len(mesh.nodes) - 1, min(6000, len(mesh.nodes)), dtype=int)
    gas_ids = np.flatnonzero(mesh.material == 0)
    vector_ids = gas_ids[np.linspace(0, len(gas_ids) - 1, min(1500, len(gas_ids)), dtype=int)]
    return {
        "field_samples": {"xyz_nm": (mesh.nodes[sample] * 1e9).tolist(), "potential_v": phi[sample].tolist()},
        "field_vectors": {
            "centers_nm": (mesh.nodes[mesh.tets[vector_ids]].mean(axis=1) * 1e9).tolist(),
            "electric_field_v_m": field[vector_ids].tolist(),
        },
    }


def write_trajectory_data(h5, trajectories, trajectory_fields):
    for index, path_data in enumerate(trajectories):
        dataset = h5.create_dataset(f"trajectories/{index}", data=np.array(path_data["points_nm"]) * 1e-9)
        dataset.attrs["species"] = path_data["species"]
        if "times_s" in path_data:
            h5.create_dataset(f"trajectory_times/{index}", data=path_data["times_s"])
            for key in ("field_time_s", "rf_phase_deg", "phase_index", "source_species"):
                dataset.attrs[key] = path_data[key]
    for key, values in (trajectory_fields or {}).items():
        group = h5.create_group(f"trajectory_fields/{key}")
        group.attrs["phase_deg"] = values["phase_deg"]
        group.attrs["field_time_s"] = values["field_time_s"]
        for name, array in {
            "sample_positions_m": np.asarray(values["field_samples"]["xyz_nm"]) * 1e-9,
            "sample_potential_v": values["field_samples"]["potential_v"],
            "vector_centers_m": np.asarray(values["field_vectors"]["centers_nm"]) * 1e-9,
            "electric_field_v_m": values["field_vectors"]["electric_field_v_m"],
            "surface_sigma_c_m2": values["surface"]["sigma_c_m2"],
        }.items():
            group.create_dataset(name, data=array, compression="gzip", compression_opts=1)


def write_results(
    directory,
    mesh,
    phi,
    field,
    sigma,
    histogram,
    depth_azimuth,
    state,
    meta,
    derived,
    save_snapshot=True,
    trajectory_fields=None,
):
    path = directory / "results.tmp.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps(meta, ensure_ascii=False)
        h5.attrs["state_json"] = json.dumps(state, ensure_ascii=False)
        for name, array in {
            "mesh/nodes_m": mesh.nodes,
            "mesh/tetrahedra": mesh.tets,
            "mesh/material": mesh.material,
            "field/potential_v": phi,
            "field/electric_field_v_m": field,
            "surface/faces": mesh.surface_faces,
            "surface/area_m2": mesh.surface_area,
            "surface/centers_m": mesh.surface_centers,
            "surface/sigma_c_m2": sigma,
            "surface/impact_weight": histogram,
            "statistics/depth_azimuth_weight": depth_azimuth,
        }.items():
            h5.create_dataset(name, data=array, compression="gzip", compression_opts=1)
        h5.create_dataset(
            "statistics/collision_energy_angle_weight", data=state["collision_energy_angle_weight"]
        )
        write_trajectory_data(h5, state["trajectories"], trajectory_fields)
    replace_file(path, directory / "results.h5")
    preview = {
        "statistics": summarize_stats(state["stats"]),
        "ledger": state["ledger"],
        "history": state["history"],
        "trajectories": state["trajectories"],
        "metadata": meta,
        "derived": derived,
        **field_preview(mesh, phi, field),
        "trajectory_fields": trajectory_fields or {},
        "surface": {
            "centers_nm": (mesh.surface_centers * 1e9).tolist(),
            "sigma_c_m2": sigma.tolist(),
            "area_m2": mesh.surface_area.tolist(),
            "impact_weight": histogram.tolist(),
        },
        "depth_azimuth_weight": depth_azimuth.tolist(),
        "time_s": state["time_s"],
        "run_until": meta.get("run_until", "time"),
        "termination_reason": state.get("termination_reason"),
        "saturation": state.get("saturation"),
    }
    preview["collision_energy_angle_weight"] = state["collision_energy_angle_weight"]
    preview["asymmetry"] = {}
    azimuth = (np.arange(36) + 0.5) * 2 * math.pi / 36
    selected_angle = math.radians(derived["dummies"][0]["angle_deg"])
    for label, values in zip(meta["collision_labels"], depth_azimuth, strict=True):
        sums = values.sum(axis=0)
        total = float(sums.sum())
        side = float(sums[np.cos(azimuth - selected_angle) >= 0].sum())
        opposite = total - side
        preview["asymmetry"][label] = {
            "selected_side_weight": side,
            "opposite_side_weight": opposite,
            "side_to_opposite_ratio": side / opposite if opposite else None,
            "first_harmonic_x": float(np.dot(sums, np.cos(azimuth)) / total) if total else None,
            "first_harmonic_y": float(np.dot(sums, np.sin(azimuth)) / total) if total else None,
        }
    atomic_json(directory / "preview.json", preview)
    if save_snapshot:
        snapshots = directory / "snapshots"
        snapshots.mkdir(exist_ok=True)
        atomic_json(snapshots / f"{state['step']:05}.json", preview)
        with h5py.File(snapshots / f"{state['step']:05}.tmp.h5", "w") as h5:
            h5.attrs["time_s"] = state["time_s"]
            h5.attrs["step"] = state["step"]
            h5.create_dataset("potential_v", data=phi, compression="gzip", compression_opts=1)
            h5.create_dataset("electric_field_v_m", data=field, compression="gzip", compression_opts=1)
            h5.create_dataset("sigma_c_m2", data=sigma, compression="gzip", compression_opts=1)
            write_trajectory_data(h5, state["trajectories"], trajectory_fields)
        replace_file(snapshots / f"{state['step']:05}.tmp.h5", snapshots / f"{state['step']:05}.h5")


def combine_time_series(directory):
    import shutil

    temporary = directory / "complete.tmp.h5"
    shutil.copyfile(directory / "results.h5", temporary)
    with h5py.File(temporary, "a") as output:
        if "time_series" in output:
            del output["time_series"]
        history = output.create_group("time_series")
        for path in sorted((directory / "snapshots").glob("*.h5")):
            with h5py.File(path) as snapshot:
                target = history.create_group(path.stem)
                target.attrs["time_s"] = snapshot.attrs["time_s"]
                for key in snapshot:
                    snapshot.copy(key, target)
    replace_file(temporary, directory / "results.h5")


class Paused(Exception):
    pass


def report_failure(directory, error, progress):
    # Details remain local to the job; no credentials or input payload are logged.
    import traceback

    details = traceback.format_exc()
    try:
        (directory / "error.log").write_text(details, encoding="utf-8")
    except OSError:
        logging.getLogger(__name__).error(
            "ジョブ %s のエラーログを保存できません。\n%s", directory.name, details
        )
    stage = permission_message(error, directory) if isinstance(error, PermissionError) else str(error)
    try:
        progress("failed", stage, error_type=type(error).__name__)
    except OSError as status_error:
        logging.getLogger(__name__).error(
            "ジョブ %s の失敗状態を保存できません。\n%s", directory.name, details
        )
        raise error from status_error


def run_job(directory: str, distribution_root: str):
    directory = Path(directory)
    started = time.perf_counter()

    def progress(status, stage, **extra):
        atomic_json(
            directory / "status.json",
            dict(status=status, stage=stage, elapsed_s=time.perf_counter() - started, **extra),
        )

    try:
        config = CaseConfig.model_validate_json((directory / "config.json").read_text(encoding="utf-8"))
        derived = derive(config)
        distributions = {}
        for sid in {s.distribution_id for s in config.ions if s.distribution_id} | (
            {config.electron.distribution_id} if config.electron.distribution_id else set()
        ):
            if not sid.isalnum() or len(sid) != 32:
                raise ValueError("分布IDが不正です。")
            root = Path(distribution_root) / sid
            meta = json.loads(root.with_suffix(".json").read_text(encoding="utf-8"))
            verify_metadata(config, meta)
            with np.load(root.with_suffix(".npz"), allow_pickle=False) as raw:
                data = {k: raw[k] for k in raw.files}
                if data["phases"].size == 0:
                    data["phases"] = None
            distributions[sid] = (data, meta)
        if config.numerics.backend == "gpu" and not gpu_info()["available"]:
            raise ValueError("GPUが利用できません。CPUを選ぶかuv sync --extra gpuを実行してください。")
        progress("running", "境界適合メッシュを生成")
        mesh = build_mesh(config)
        mesh_hash = hashlib.sha256(mesh.nodes.tobytes() + mesh.tets.tobytes()).hexdigest()
        progress(
            "running", "FEM行列と境界電位の基底解を生成", nodes=len(mesh.nodes), tetrahedra=len(mesh.tets)
        )
        solver = FieldSolver(mesh, config)
        face_kind, face_patch = boundary_maps(mesh)
        metadata = provenance(config, derived, distributions)
        metadata["mesh_id"] = mesh_hash
        metadata["warnings"] = [
            *derived["warnings"],
            "固定形状・凍結RF位相の平均場モデル。実デバイス予測と形状異常の発生確率は未検証。",
            "統計区間は固定電場での条件付き推定。帯電を含む区間には独立ジョブ比較が必要。",
            "表面係数は材料・エネルギー・角度に依存しない未校正の定数。",
            SPACE_CHARGE_WARNING,
        ]
        if config.validation_case:
            metadata["warnings"].append("検証用仮値：工程条件の推奨には使用できません。")
        for _, m in distributions.values():
            metadata["warnings"].extend(m["warnings"])
        if config.surface.secondary_yield:
            metadata["warnings"].append("二次電子は法線方向の単一エネルギー近似です。")
        metadata["mesh"] = dict(
            nodes=len(mesh.nodes), tetrahedra=len(mesh.tets), surface_patches=len(mesh.surface_faces)
        )
        rng = np.random.default_rng(config.numerics.seed)
        names = [s.name for s in config.ions] + (["electron"] if config.electron.enabled else [])
        collision_labels = names + (["secondary_electron"] if config.surface.secondary_yield else [])
        metadata["collision_labels"] = collision_labels
        energy_edges = np.r_[0, np.geomspace(0.01, 1000, 48), np.inf]
        angle_edges = np.linspace(0, 90, 19)
        metadata["collision_energy_edges_ev"] = [float(e) if np.isfinite(e) else None for e in energy_edges]
        metadata["collision_angle_edges_deg"] = angle_edges.tolist()
        metadata["inlet_potential_mode"] = config.inlet_potential_mode
        sigma = np.full(
            len(mesh.surface_faces), 0 if config.mode == "uncharged" else config.initial_sigma_c_m2
        )
        histogram = np.zeros((len(collision_labels), len(sigma)))
        depth_azimuth = np.zeros((len(collision_labels), 24, 36))
        state = dict(
            time_s=0.0,
            step=0,
            stats={name: new_stats() for name in names},
            ledger=new_ledger(),
            history=[],
            trajectories=[],
            rng=rng.bit_generator.state,
            collision_energy_angle_weight=np.zeros((len(collision_labels), 49, 18)).tolist(),
        )
        checkpoint = directory / "checkpoint.h5"
        if checkpoint.exists():
            state, sigma, histogram, depth_azimuth, previous = load_checkpoint(checkpoint)
            if (
                previous["config_sha256"] != config.fingerprint()
                or previous["mesh_id"] != mesh_hash
                or previous["code_sha256"] != code_id()
            ):
                raise ValueError(
                    "設定・メッシュ・コード版が変わったチェックポイントは同一計算として再開できません。"
                )
            rng.bit_generator.state = state["rng"]
        saturation_mode = config.numerics.run_until == "saturation"
        criteria = config.numerics.saturation
        time_limit = criteria.max_time_s if saturation_mode else config.numerics.duration_s
        update_limit = criteria.max_updates if saturation_mode else 10000
        monitor = None
        if saturation_mode:
            restored = state.get("saturation_state")
            reference_sigma = np.array(restored["reference_sigma_c_m2"]) if restored else sigma
            reference_phi, _, _ = solver.solve(reference_sigma, config.waveform.voltage(0))
            monitor = SaturationMonitor(
                criteria, state["time_s"], reference_sigma, reference_phi, mesh.surface_area, restored
            )
            state["saturation_state"] = monitor.state
            state["saturation"] = monitor.summary()
        save_checkpoint(checkpoint, state, sigma, histogram, depth_azimuth, metadata)
        nominal_dt = (
            criteria.window_s / criteria.steps_per_window
            if saturation_mode
            else config.numerics.duration_s / config.numerics.charging_steps
            if config.mode == "self_consistent"
            else config.numerics.duration_s
        )
        transport_function = trace
        if config.numerics.backend == "gpu":
            from .gpu import GPUTracer

            transport_function = GPUTracer(mesh, face_kind, face_patch)
        while state["time_s"] < time_limit * (1 - 1e-12) and not state.get("termination_reason"):
            if state["step"] >= update_limit:
                raise RuntimeError("帯電更新が10000回を超えました。時間・許容電位変化を見直してください。")
            rng_before = copy.deepcopy(rng.bit_generator.state)
            physical_dt = min(nominal_dt, time_limit - state["time_s"])
            if monitor:
                physical_dt = min(physical_dt, monitor.remaining_window_s(state["time_s"]))
            step_stats = {name: new_stats() for name in names}
            ledger = new_ledger()
            deposited = np.zeros(len(sigma))
            impacts = np.zeros_like(histogram)
            depth_impacts = np.zeros_like(depth_azimuth)
            energy_angle = np.zeros((len(collision_labels), 49, 18))
            paths = []
            trajectory_fields = {}
            max_flight = 0.0
            total_samples = config.numerics.samples_per_species * len(names)
            done = 0
            field_start = time.perf_counter()
            fields = []
            for phase_index in range(config.numerics.phase_bins):
                phase = (phase_index + 0.5) * 360 / config.numerics.phase_bins
                fields.append(solver.solve(sigma, config.waveform.voltage(phase)))
            if config.numerics.backend == "gpu":
                transport_function.begin_step([f[1] for f in fields])
            field_seconds = time.perf_counter() - field_start
            transport_start = time.perf_counter()
            for species_index, name in enumerate(names):
                species = config.ions[species_index] if species_index < len(config.ions) else None
                mass = species.mass_amu * AMU if species else ELECTRON_MASS
                q = species.charge_number * E_CHARGE if species else -E_CHARGE
                flux = species.flux_m2_s if species else config.electron.flux_m2_s
                distribution_id = species.distribution_id if species else config.electron.distribution_id
                for offset in range(0, config.numerics.samples_per_species, config.numerics.batch_size):
                    if (directory / "pause.request").exists():
                        rng.bit_generator.state = rng_before
                        state["rng"] = rng_before
                        save_checkpoint(checkpoint, state, sigma, histogram, depth_azimuth, metadata)
                        raise Paused()
                    count = min(config.numerics.batch_size, config.numerics.samples_per_species - offset)
                    p, v, phases = sample_generated(rng, count, config, derived, species)
                    if distribution_id:
                        data, meta = distributions[distribution_id]
                        v, phases = sample_distribution(rng, count, data, meta, mass / AMU)
                    weight = np.full(
                        count,
                        flux
                        * derived["injection_area_m2"]
                        * physical_dt
                        / config.numerics.samples_per_species,
                    )
                    if species is None and config.electron.phase_flux_model == "boltzmann":
                        wafer = np.array([config.waveform.voltage(phase) for phase in phases])
                        if config.electron.plasma_potential_samples:
                            phase_table = np.array(
                                [p.phase_deg for p in config.electron.plasma_potential_samples]
                            )
                            potential_table = np.array(
                                [p.potential_v for p in config.electron.plasma_potential_samples]
                            )
                            plasma = np.interp(phases, phase_table, potential_table, period=360)
                        else:
                            plasma = config.electron.plasma_potential_v
                        weight *= np.exp(np.minimum(0, (wafer - plasma) / config.electron.temperature_ev))
                    step_stats[name]["injected_weight"] += float(weight.sum())
                    step_stats[name]["weight_squared"] += float(np.dot(weight, weight))
                    step_stats[name]["samples"] += count
                    ledger["injected_c"] += float(q * weight.sum())
                    ledger["injected_absolute_c"] += float(abs(q) * weight.sum())
                    starts = solver.locate(p)
                    for bin_index in range(config.numerics.phase_bins):
                        indexes = np.flatnonzero(
                            np.floor(phases * config.numerics.phase_bins / 360).astype(int) == bin_index
                        )
                        if not len(indexes):
                            continue
                        active_p, active_v, active_tet, active_weight = (
                            p[indexes],
                            v[indexes],
                            starts[indexes],
                            weight[indexes],
                        )
                        active_q = np.full(len(indexes), q)
                        active_mass = np.full(len(indexes), mass)
                        original = np.ones(len(indexes), bool)
                        flight = np.zeros(len(indexes))
                        electric_field = fields[bin_index][1]
                        for _event in range(config.surface.max_events):
                            quota, remainder = divmod(config.numerics.representative_trajectories, len(names))
                            path_count = max(
                                0,
                                quota
                                + (species_index < remainder)
                                - sum(path["source_species"] == name for path in paths),
                            )
                            output = transport_function(
                                active_p,
                                active_v,
                                active_tet,
                                active_q,
                                active_mass,
                                electric_field,
                                mesh.inverse,
                                mesh.gradients,
                                mesh.neighbors,
                                face_kind,
                                face_patch,
                                config.numerics.max_particle_steps,
                                config.numerics.particle_dt_s,
                                mesh.min_cell_m,
                                config.numerics.cell_fraction,
                                path_count,
                            )
                            (
                                end,
                                out_v,
                                status,
                                patches,
                                tets,
                                times,
                                normals,
                                angles,
                                trajectories,
                                lengths,
                                trajectory_times,
                            ) = output
                            flight_before = flight.copy()
                            flight += times
                            max_flight = max(max_flight, float(flight.max(initial=0)))
                            for j, length in enumerate(lengths):
                                if length < 2:
                                    continue
                                paths.append(
                                    dict(
                                        species=name if original[j] else "secondary_electron",
                                        source_species=name,
                                        points_nm=(trajectories[j, :length] * 1e9).tolist(),
                                        times_s=(trajectory_times[j, :length] + flight_before[j]).tolist(),
                                        field_time_s=state["time_s"],
                                        rf_phase_deg=(bin_index + 0.5) * 360 / config.numerics.phase_bins,
                                        phase_index=bin_index,
                                    )
                                )
                                key = str(bin_index)
                                if key not in trajectory_fields:
                                    trajectory_fields[key] = {
                                        **field_preview(mesh, fields[bin_index][0], electric_field),
                                        "surface": {
                                            "centers_nm": (mesh.surface_centers * 1e9).tolist(),
                                            "sigma_c_m2": sigma.tolist(),
                                            "area_m2": mesh.surface_area.tolist(),
                                        },
                                        "phase_deg": paths[-1]["rf_phase_deg"],
                                        "field_time_s": state["time_s"],
                                    }
                            energy = 0.5 * active_mass * np.sum(out_v * out_v, axis=1) / E_CHARGE
                            surface_hit = status == 2
                            wall = (status == 1) | (status == 2)
                            groups = [(species_index, original)]
                            if len(collision_labels) > len(names):
                                groups.append((len(collision_labels) - 1, ~original))
                            for group_index, group_mask in groups:
                                oxide_group = surface_hit & group_mask
                                np.add.at(
                                    impacts[group_index], patches[oxide_group], active_weight[oxide_group]
                                )
                                wall_group = wall & group_mask
                                if np.any(wall_group):
                                    zbin = np.clip(
                                        (end[wall_group, 2] / (derived["depth_nm"] * 1e-9) * 24).astype(int),
                                        0,
                                        23,
                                    )
                                    abin = np.floor(
                                        (np.arctan2(end[wall_group, 1], end[wall_group, 0]) % (2 * math.pi))
                                        / (2 * math.pi)
                                        * 36
                                    ).astype(int)
                                    np.add.at(
                                        depth_impacts[group_index], (zbin, abin), active_weight[wall_group]
                                    )
                                    eh, _, _ = np.histogram2d(
                                        energy[wall_group],
                                        angles[wall_group],
                                        bins=(energy_edges, angle_edges),
                                        weights=active_weight[wall_group],
                                    )
                                    energy_angle[group_index] += eh
                            # Reflected charge is carried by the outgoing particle; neutral ions deposit q.
                            reflect_e = (
                                wall
                                & (active_q < 0)
                                & (rng.random(len(status)) < config.surface.electron_reflection)
                            )
                            neutral = (
                                wall
                                & (active_q > 0)
                                & (rng.random(len(status)) < config.surface.ion_neutralization_reflection)
                            )
                            emitted = (
                                wall
                                & (active_q > 0)
                                & (rng.random(len(status)) < config.surface.secondary_yield)
                            )
                            deposit_q = active_q.copy()
                            deposit_q[reflect_e] = 0
                            deposit_q[emitted] += E_CHARGE
                            oxide = status == 2
                            np.add.at(deposited, patches[oxide], deposit_q[oxide] * active_weight[oxide])
                            ledger["deposited_oxide_c"] += float(
                                np.dot(deposit_q[oxide], active_weight[oxide])
                            )
                            ledger["conductor_c"] += float(
                                np.dot(deposit_q[status == 1], active_weight[status == 1])
                            )
                            ledger["escaped_c"] += float(
                                np.dot(active_q[status == 0], active_weight[status == 0])
                            )
                            ledger["unresolved_c"] += float(
                                np.dot(active_q[status == 3], active_weight[status == 3])
                            )
                            ledger["unresolved_absolute_c"] += float(
                                np.dot(np.abs(active_q[status == 3]), active_weight[status == 3])
                            )
                            ledger["emitted_c"] -= E_CHARGE * float(active_weight[emitted].sum())
                            continuing = reflect_e | neutral
                            finished_original = original & ~continuing
                            for outcome, label in (
                                (0, "escaped"),
                                (1, "conductor"),
                                (2, "oxide"),
                                (3, "unresolved"),
                            ):
                                chosen = finished_original & (status == outcome)
                                step_stats[name][label + "_weight"] += float(active_weight[chosen].sum())
                            bottom = (
                                finished_original
                                & wall
                                & np.isclose(end[:, 2], derived["depth_nm"] * 1e-9, atol=1e-14, rtol=0)
                            )
                            bw = active_weight[bottom]
                            s = step_stats[name]
                            s["bottom_weight"] += float(bw.sum())
                            s["bottom_weight_squared"] += float(np.dot(bw, bw))
                            s["bottom_x"] += float(np.dot(bw, end[bottom, 0]))
                            s["bottom_y"] += float(np.dot(bw, end[bottom, 1]))
                            s["bottom_energy"] += float(np.dot(bw, energy[bottom]))
                            s["bottom_energy_squared"] += float(np.dot(bw, energy[bottom] ** 2))
                            if not continuing.any() and not emitted.any():
                                break
                            reflection_v = (
                                out_v[continuing]
                                - 2
                                * np.sum(out_v[continuing] * normals[continuing], axis=1)[:, None]
                                * normals[continuing]
                            )
                            reflected_q = active_q[continuing].copy()
                            reflected_q[neutral[continuing]] = 0
                            secondary_v = -normals[emitted] * math.sqrt(
                                2 * config.surface.secondary_energy_ev * E_CHARGE / ELECTRON_MASS
                            )
                            active_p = np.concatenate(
                                [
                                    end[continuing] - normals[continuing] * 1e-16,
                                    end[emitted] - normals[emitted] * 1e-16,
                                ]
                            )
                            active_v = np.concatenate([reflection_v, secondary_v])
                            active_tet = np.r_[tets[continuing], tets[emitted]]
                            active_weight = np.r_[active_weight[continuing], active_weight[emitted]]
                            active_q = np.r_[reflected_q, np.full(emitted.sum(), -E_CHARGE)]
                            active_mass = np.r_[
                                active_mass[continuing], np.full(emitted.sum(), ELECTRON_MASS)
                            ]
                            original = np.r_[original[continuing], np.zeros(emitted.sum(), bool)]
                            flight = np.r_[flight[continuing], flight[emitted]]
                        else:
                            ledger["unresolved_c"] += float(np.dot(active_q, active_weight))
                            ledger["unresolved_absolute_c"] += float(np.dot(np.abs(active_q), active_weight))
                            step_stats[name]["unresolved_weight"] += float(active_weight[original].sum())
                    done += count
                    progress(
                        "running",
                        "粒子輸送・表面相互作用",
                        step=state["step"],
                        time_s=state["time_s"],
                        progress=min(
                            0.99,
                            max(
                                (state["time_s"] + physical_dt * done / total_samples) / time_limit,
                                (state["step"] + done / total_samples) / update_limit if monitor else 0,
                            ),
                        ),
                        saturation=monitor.summary() if monitor else None,
                        samples_completed=done,
                        samples_total=total_samples,
                        nodes=len(mesh.nodes),
                        backend=config.numerics.backend,
                        particles_per_s=done / max(time.perf_counter() - transport_start, 1e-9),
                    )
            transport_seconds = time.perf_counter() - transport_start
            accepted_dt = physical_dt
            factor = 1.0
            delta_voltage = 0.0
            if config.mode == "self_consistent":
                delta_sigma = deposited / mesh.surface_area
                charge_phi, _, _ = solver.solve(delta_sigma, 0)
                zero_phi, _, _ = solver.solve(np.zeros_like(sigma), 0)
                delta_voltage = float(np.max(np.abs(charge_phi - zero_phi)))
                factor = min(1, config.numerics.max_voltage_change_v / max(delta_voltage, 1e-30))
                accepted_dt *= factor
                if accepted_dt < min(nominal_dt, time_limit) * 1e-10:
                    raise RuntimeError("帯電時間刻みが過小です。流束・電位変化目標・形状を確認してください。")
                for key, value in ledger.items():
                    ledger[key] = value * factor
                for s in step_stats.values():
                    for key in s:
                        if key != "samples":
                            s[key] *= factor**2 if key.endswith("squared") else factor
                deposited *= factor
                impacts *= factor
                depth_impacts *= factor
                energy_angle *= factor
                sigma += deposited / mesh.surface_area
                if config.leakage_tau_s:
                    after = sigma * np.exp(-accepted_dt / config.leakage_tau_s)
                    ledger["leaked_c"] = float(np.dot(sigma - after, mesh.surface_area))
                    sigma = after
            for name in names:
                for key, value in step_stats[name].items():
                    state["stats"][name][key] += value
            for key, value in ledger.items():
                state["ledger"][key] += value
            histogram += impacts
            depth_azimuth += depth_impacts
            state["collision_energy_angle_weight"] = (
                np.array(state["collision_energy_angle_weight"]) + energy_angle
            ).tolist()
            state["trajectories"] = paths
            state["time_s"] += accepted_dt
            state["step"] += 1
            state["rng"] = rng.bit_generator.state
            phi, field, field_diag = solver.solve(sigma, config.waveform.voltage(0))
            if monitor:
                monitor.observe(
                    state["time_s"],
                    sigma,
                    phi,
                    ledger["injected_absolute_c"],
                    ledger["unresolved_absolute_c"],
                )
                state["saturation"] = monitor.summary()
            if monitor and monitor.state["saturated"]:
                state["termination_reason"] = "saturated"
            elif state["time_s"] >= time_limit * (1 - 1e-12):
                state["termination_reason"] = "time_limit" if monitor else "time_completed"
            elif monitor and state["step"] >= update_limit:
                state["termination_reason"] = "update_limit"
            balance = state["ledger"]
            residual = (
                balance["injected_c"]
                - balance["escaped_c"]
                - balance["conductor_c"]
                - balance["deposited_oxide_c"]
                - balance["unresolved_c"]
            )
            normalized_residual = abs(residual) / max(balance["injected_absolute_c"], 1e-300)
            rf_ratio = max_flight * config.waveform.frequency_hz if config.waveform.amplitude_v else 0
            collision_probability = None
            if config.collision_cross_section_m2 is not None:
                gas_n = config.gas_pressure_pa / (1.380649e-23 * config.gas_temperature_k)
                collision_probability = 1 - math.exp(
                    -gas_n * config.collision_cross_section_m2 * derived["depth_nm"] * 1e-9
                )
            state["history"].append(
                dict(
                    step=state["step"],
                    time_s=state["time_s"],
                    dt_s=accepted_dt,
                    surface_charge_c=float(np.dot(sigma, mesh.surface_area)),
                    max_sigma_c_m2=float(np.max(abs(sigma), initial=0)),
                    max_voltage_change_v=delta_voltage * factor,
                    charge_residual=normalized_residual,
                    field_seconds=field_seconds,
                    transport_seconds=transport_seconds,
                    max_flight_s=max_flight,
                    flight_to_rf_period=rf_ratio,
                    gas_collision_probability_lower_estimate=collision_probability,
                    frozen_rf_validated=rf_ratio < 0.01,
                    **({"saturation": monitor.summary()} if monitor else {}),
                    **field_diag,
                )
            )
            if (
                rf_ratio >= 0.01
                and "RF飛行時間が周期の1%以上です。凍結RF位相近似を再評価してください。"
                not in metadata["warnings"]
            ):
                metadata["warnings"].append(
                    "RF飛行時間が周期の1%以上です。凍結RF位相近似を再評価してください。"
                )
            write_results(
                directory,
                mesh,
                phi,
                field,
                sigma,
                histogram,
                depth_azimuth,
                state,
                metadata,
                derived,
                state["step"] % config.numerics.save_every_steps == 0
                or bool(state.get("termination_reason")),
                trajectory_fields,
            )
            # A resumable accepted update has complete outputs, including its final snapshot.
            save_checkpoint(checkpoint, state, sigma, histogram, depth_azimuth, metadata)
            if monitor:
                progress(
                    "running",
                    "帯電更新・飽和判定",
                    step=state["step"],
                    time_s=state["time_s"],
                    progress=min(0.99, max(state["time_s"] / time_limit, state["step"] / update_limit)),
                    saturation=monitor.summary(),
                    backend=config.numerics.backend,
                )
        combine_time_series(directory)
        reason = state.get("termination_reason", "time_completed")
        stages = {
            "saturated": "飽和判定成立（指定許容差内）",
            "time_limit": "最大時間で終了（未飽和）",
            "update_limit": "最大更新回数で終了（未飽和）",
            "time_completed": "計算完了",
        }
        progress(
            "completed",
            stages[reason],
            progress=min(1, max(state["time_s"] / time_limit, state["step"] / update_limit))
            if monitor
            else 1,
            time_s=state["time_s"],
            steps=state["step"],
            backend=config.numerics.backend,
            termination_reason=reason,
            saturation=monitor.summary() if monitor else None,
        )
    except Paused:
        try:
            progress(
                "paused",
                "保存済みの帯電更新点で停止（途中のバッチは再開時に再計算）",
                time_s=state["time_s"],
                step=state["step"],
                saturation=state.get("saturation"),
            )
        except OSError as error:
            report_failure(directory, error, progress)
    except Exception as error:
        report_failure(directory, error, progress)
