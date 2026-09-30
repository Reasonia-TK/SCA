"""Reproducible convergence and independent-seed checks, without COMSOL.

Fixed-field cases share exactly the same particles. Coupled cases keep the
total particle budget constant as the charging step is halved. Failed physical
acceptance criteria are reported separately from execution errors.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import t as student_t

from memory_hole.config import CaseConfig
from memory_hole.engine import atomic_json, code_id, gpu_info, run_job
from memory_hole.field import EPS0, FieldSolver, build_mesh
from memory_hole.geometry import derive
from memory_hole.transport import AMU, E_CHARGE, boundary_maps, sample_generated, trace

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260930, 314159, 271828, 161803, 141421]


def relative_difference(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1e-30))


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "mean": mean,
        "low": mean - half,
        "high": mean + half,
        "half_width": half,
        "relative_half_width": half / abs(mean) if mean else None,
        "replicates": len(values),
        "method": "Student t, independent whole-run seeds; df = replicates - 1",
        "values": values.tolist(),
    }


def probe_points(config):
    d = derive(config)
    points = []
    for z in np.linspace(config.geometry.carbon_thickness_nm + 150, d["depth_nm"] - 150, 12):
        radius = np.interp(z, [p["depth_nm"] for p in d["profile"]], [p["radius_nm"] for p in d["profile"]])
        points.append([0, 0, z])
        for fraction in (0.25, 0.5, 0.75):
            for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                points.append([fraction * radius * math.cos(angle), fraction * radius * math.sin(angle), z])
    return np.asarray(points) * 1e-9


def interpolate(mesh, phi, points, tets):
    if np.any(tets < 0):
        raise RuntimeError("Probe or injection point could not be located in the gas mesh")
    bary = np.einsum("ni,nij->nj", np.column_stack([np.ones(len(points)), points / 1e-9]), mesh.inverse[tets])
    return np.einsum("ni,ni->n", phi[mesh.tets[tets]], bary)


def summarize_trace(output, config):
    end, v, status = output[:3]
    bottom = (status == 1) & (abs(end[:, 2] - derive(config)["depth_nm"] * 1e-9) < 1e-12)
    energy = 0.5 * config.ions[0].mass_amu * AMU * np.sum(v**2, axis=1) / E_CHARGE
    return {
        "bottom_rate": float(bottom.mean()),
        "mean_bottom_energy_ev": float(energy[bottom].mean()) if bottom.any() else None,
        "unresolved": int(np.sum(status == 3)),
        "bottom_centroid_nm": (end[bottom].mean(axis=0)[:2] * 1e9).tolist() if bottom.any() else None,
    }


def fixed_checks(particles):
    from memory_hole.gpu import GPUTracer

    base = CaseConfig()
    base.mode = "fixed_charge"
    base.initial_sigma_c_m2 = 2e-5
    base.waveform.dc_v = -100
    base.numerics.max_particle_steps = 20000
    positions, velocities, _ = sample_generated(
        np.random.default_rng(SEEDS[0]), particles, base, derive(base), base.ions[0]
    )
    probes = probe_points(base)
    cases = []
    configs = []
    for factor in (1, 0.8, 0.64, 0.512):
        c = base.model_copy(deep=True)
        c.geometry.mesh_size_nm *= factor
        c.geometry.interface_mesh_size_nm *= factor
        configs.append((f"mesh_{factor:g}", c))
    for margin in (600, 900, 1500, 2400):
        c = base.model_copy(deep=True)
        c.geometry.lateral_margin_nm = margin
        configs.append((f"margin_{margin}", c))
    c = base.model_copy(deep=True)
    c.outer_boundary = "neumann"
    configs.append(("neumann", c))
    baseline = None
    movement = None
    space_charge = None
    for name, c in configs:
        start = time.perf_counter()
        mesh = build_mesh(c)
        solver = FieldSolver(mesh, c)
        kinds, patches = boundary_maps(mesh)
        sigma = np.full(len(mesh.surface_faces), c.initial_sigma_c_m2)
        phi, field, diag = solver.solve(sigma, c.waveform.dc_v)
        starts = solver.locate(positions)
        if np.any(starts < 0):
            raise RuntimeError("Injection point outside gas mesh")
        gpu = GPUTracer(mesh, kinds, patches)

        def arguments(
            dt=2e-12,
            fraction=0.2,
            electric=field,
            mesh=mesh,
            config=c,
            starts=starts,
            kinds=kinds,
            patches=patches,
        ):
            return (
                positions,
                velocities,
                starts,
                np.full(particles, E_CHARGE),
                np.full(particles, config.ions[0].mass_amu * AMU),
                electric,
                mesh.inverse,
                mesh.gradients,
                mesh.neighbors,
                kinds,
                patches,
                config.numerics.max_particle_steps,
                dt,
                mesh.min_cell_m,
                fraction,
                0,
            )

        output = gpu(*arguments())
        potential = interpolate(mesh, phi, probes, solver.locate(probes)) - c.waveform.dc_v
        case = {
            "name": name,
            "config": c.model_dump(),
            "nodes": len(mesh.nodes),
            "tetrahedra": len(mesh.tets),
            "surface_patches": len(mesh.surface_faces),
            "field_residual": diag["field_residual"],
            "potential_probes_relative_to_wafer_v": potential.tolist(),
            **summarize_trace(output, c),
        }
        if name == "mesh_1":
            cpu = trace(*arguments())
            initial_phi = interpolate(mesh, phi, positions, starts)
            final_phi = interpolate(mesh, phi, output[0], output[4])
            initial_ke = 0.5 * c.ions[0].mass_amu * AMU * np.sum(velocities**2, axis=1) / E_CHARGE
            final_ke = 0.5 * c.ions[0].mass_amu * AMU * np.sum(output[1] ** 2, axis=1) / E_CHARGE
            energy_error = float(
                np.max(abs(final_ke + final_phi - initial_ke - initial_phi)) / c.ions[0].energy_ev
            )
            variants = []
            for dt, fraction in ((1e-12, 0.2), (1e-12, 0.1), (0.5e-12, 0.05)):
                variant = gpu(*arguments(dt, fraction))
                stats = summarize_trace(variant, c)
                variants.append(
                    {
                        "max_dt_s": dt,
                        "cell_fraction": fraction,
                        **stats,
                        "status_match": bool(np.array_equal(output[2], variant[2])),
                        "patch_match": bool(np.array_equal(output[3], variant[3])),
                        "max_endpoint_difference_m": float(np.max(abs(output[0] - variant[0]))),
                        "bottom_rate_relative_difference": relative_difference(
                            stats["bottom_rate"], case["bottom_rate"]
                        ),
                    }
                )
            movement = {
                "variants": variants,
                "cpu_gpu_status_match": bool(np.array_equal(cpu[2], output[2])),
                "cpu_gpu_patch_match": bool(np.array_equal(cpu[3], output[3])),
                "cpu_gpu_max_endpoint_difference_m": float(np.max(abs(cpu[0] - output[0]))),
                "max_static_total_energy_relative_error": energy_error,
                "note": "The 2 ps ceiling can be inactive; cell_fraction is also halved to reduce actual flight segments.",
                "passed": energy_error < 0.01
                and np.array_equal(cpu[2], output[2])
                and np.array_equal(cpu[3], output[3])
                and all(
                    v["unresolved"] == 0 and v["bottom_rate_relative_difference"] < 0.02 for v in variants
                ),
            }
            # Preliminary ion-only homogeneous volume source inferred from residence.
            # This is a sensitivity experiment, not a coupled local-density model.
            gas = (mesh.material == 0) & (mesh.nodes[mesh.tets, 2].mean(axis=1) > 0)
            residence = float(output[5].mean())
            rho = (
                E_CHARGE
                * c.ions[0].flux_m2_s
                * derive(c)["injection_area_m2"]
                * residence
                / mesh.volumes[gas].sum()
            )
            rhs = np.zeros(len(mesh.nodes))
            np.add.at(rhs, mesh.tets[gas].ravel(), np.repeat(rho * mesh.volumes[gas] / (4 * EPS0 * 1e-9), 4))
            delta_phi = solver._solve(rhs, np.zeros(len(solver.fixed)))
            delta_field = -np.einsum("ti,tik->tk", delta_phi[mesh.tets], mesh.gradients)
            variants = []
            for sign in (-1, 1):
                stats = summarize_trace(gpu(*arguments(electric=field + sign * delta_field)), c)
                variants.append(
                    {
                        "source_sign": sign,
                        **stats,
                        "bottom_rate_relative_difference": relative_difference(
                            stats["bottom_rate"], case["bottom_rate"]
                        ),
                        "bottom_energy_relative_difference": relative_difference(
                            stats["mean_bottom_energy_ev"], case["mean_bottom_energy_ev"]
                        ),
                    }
                )
            space_charge = {
                "mean_ion_residence_s": residence,
                "homogeneous_density_c_m3": float(rho),
                "max_potential_change_v": float(np.max(abs(delta_phi))),
                "variants": variants,
                "preliminary_passed": all(
                    v["unresolved"] == 0
                    and v["bottom_rate_relative_difference"] < 0.05
                    and v["bottom_energy_relative_difference"] < 0.05
                    for v in variants
                ),
                "scope": "Generated 100 eV ions only, homogeneous pore density; local density, electrons and RF phase not validated.",
                "full_model_status": "not_validated",
            }
            baseline = copy.deepcopy(case)
        else:
            case["baseline_bottom_rate_relative_difference"] = relative_difference(
                case["bottom_rate"], baseline["bottom_rate"]
            )
            case["baseline_potential_l2_relative_difference"] = relative_difference(
                potential, baseline["potential_probes_relative_to_wafer_v"]
            )
        case["elapsed_s"] = time.perf_counter() - start
        cases.append(case)
        print(
            f"{name}: tets={len(mesh.tets)}, arrival={case['bottom_rate']:.6f}, elapsed={case['elapsed_s']:.1f}s",
            flush=True,
        )
        del gpu, solver, mesh
    mesh_differences = []
    mesh_cases = [case for case in cases if case["name"].startswith("mesh_")]
    for coarse, fine in zip(mesh_cases[:-1], mesh_cases[1:], strict=True):
        differences = {
            "pair": [coarse["name"], fine["name"]],
            "potential_l2_relative_difference": relative_difference(
                coarse["potential_probes_relative_to_wafer_v"], fine["potential_probes_relative_to_wafer_v"]
            ),
            "arrival_relative_difference": relative_difference(coarse["bottom_rate"], fine["bottom_rate"]),
            "energy_relative_difference": relative_difference(
                coarse["mean_bottom_energy_ev"], fine["mean_bottom_energy_ev"]
            ),
        }
        differences["passed"] = all(v < 0.02 for k, v in differences.items() if k.endswith("difference"))
        mesh_differences.append(differences)
    outer_pairs = []
    domain_cases = [cases[0], *[case for case in cases if case["name"].startswith("margin_")]]
    for a, b in zip(domain_cases[:-1], domain_cases[1:], strict=True):
        outer_pairs.append(
            {
                "pair": [a["name"], b["name"]],
                "potential_l2_relative_difference": relative_difference(
                    a["potential_probes_relative_to_wafer_v"], b["potential_probes_relative_to_wafer_v"]
                ),
                "arrival_relative_difference": relative_difference(a["bottom_rate"], b["bottom_rate"]),
            }
        )
    return {
        "scope": "Fixed uniform wall charge 2e-5 C/m2, -100 V, generated 100 eV ions; shared particles.",
        "particles": particles,
        "cases": cases,
        "mesh_pairs": mesh_differences,
        "mesh_passed": all(p["passed"] for p in mesh_differences)
        and all(c["unresolved"] == 0 for c in mesh_cases),
        "motion": movement,
        "outer_boundary": {
            "domain_pairs": outer_pairs,
            "domain_passed": all(
                p["potential_l2_relative_difference"] < 0.02 and p["arrival_relative_difference"] < 0.02
                for p in outer_pairs
            ),
            "neumann_vs_dirichlet_potential_relative_difference": cases[-1][
                "baseline_potential_l2_relative_difference"
            ],
            "note": "Boundary-condition change is model sensitivity; domain expansion also changes the generated mesh.",
        },
        "space_charge_preliminary": space_charge,
    }


def coupled_checks(output_root, budget, seed_count):
    base = CaseConfig.model_validate_json((ROOT / "data/preset.json").read_text(encoding="utf-8"))
    base.numerics.backend = "gpu"
    base.numerics.batch_size = 2000
    base.numerics.representative_trajectories = 0
    base.numerics.duration_s = 1e-4
    results = {}
    for steps in (1, 4, 8, 16):
        group = []
        for seed in SEEDS[:seed_count]:
            c = base.model_copy(deep=True)
            c.mode = "uncharged" if steps == 1 else "self_consistent"
            c.numerics.charging_steps = steps
            c.numerics.samples_per_species = budget // steps
            c.numerics.seed = seed
            c.name = f"数値検証 · {'無帯電' if steps == 1 else str(steps) + '帯電更新'} · seed {seed}"
            directory = output_root / "jobs" / f"steps_{steps}_seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            preview_path = directory / "preview.json"
            status_path = directory / "status.json"
            cached = False
            if preview_path.exists() and status_path.exists():
                preview = json.loads(preview_path.read_text(encoding="utf-8"))
                cached = (
                    json.loads(status_path.read_text(encoding="utf-8"))["status"] == "completed"
                    and preview["metadata"]["config_sha256"] == c.fingerprint()
                    and preview["metadata"]["code_sha256"] == code_id()
                )
            if not cached:
                if (directory / "checkpoint.h5").exists():
                    raise RuntimeError(
                        f"Stale validation checkpoint: choose a new output directory: {directory}"
                    )
                atomic_json(directory / "config.json", c.model_dump())
                run_job(str(directory), str(ROOT / "data/distributions"))
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status["status"] != "completed":
                    raise RuntimeError(f"Validation job failed: {directory}: {status['stage']}")
                preview = json.loads(preview_path.read_text(encoding="utf-8"))
            summary = {
                "seed": seed,
                "directory": str(directory.resolve()),
                "statistics": preview["statistics"],
                "final_surface_charge_c": preview["history"][-1]["surface_charge_c"],
                "max_charge_residual": max(h["charge_residual"] for h in preview["history"]),
                "max_flight_to_rf_period": max(h["flight_to_rf_period"] for h in preview["history"]),
                "accepted_steps": len(preview["history"]),
            }
            with h5py.File(directory / "results.h5") as h5:
                surface_charge = float(np.dot(h5["surface/sigma_c_m2"][:], h5["surface/area_m2"][:]))
                if not math.isclose(
                    surface_charge, summary["final_surface_charge_c"], rel_tol=1e-12, abs_tol=1e-30
                ):
                    raise RuntimeError("HDF5 surface charge differs from the saved history")
            group.append(summary)
            print(
                f"steps={steps}, seed={seed}: arrival={preview['statistics']['Ar+']['bottom_arrival']['rate']:.6f}",
                flush=True,
            )
        aggregate = {}
        for species in group[0]["statistics"]:
            rates = interval([r["statistics"][species]["bottom_arrival"]["rate"] for r in group])
            energy_values = [r["statistics"][species]["mean_bottom_energy_ev"] for r in group]
            energies = interval(energy_values) if all(v is not None for v in energy_values) else None
            aggregate[species] = {
                "arrival": rates,
                "energy_ev": energies,
                "arrival_precision_passed": rates["relative_half_width"] is not None
                and rates["relative_half_width"] <= 0.02,
                "energy_precision_passed": energies is not None
                and energies["relative_half_width"] is not None
                and energies["relative_half_width"] <= 0.02,
            }
        results[str(steps)] = {"runs": group, "independent_seed_statistics": aggregate}
        atomic_json(output_root / "coupled_partial.json", results)
    charging = []
    for coarse, fine in ((4, 8), (8, 16)):
        outputs = {}
        for species in results[str(coarse)]["independent_seed_statistics"]:
            for label, getter in (
                ("arrival", lambda r, sp=species: r["statistics"][sp]["bottom_arrival"]["rate"]),
                ("energy", lambda r, sp=species: r["statistics"][sp]["mean_bottom_energy_ev"]),
            ):
                a = [getter(r) for r in results[str(coarse)]["runs"]]
                b = [getter(r) for r in results[str(fine)]["runs"]]
                if any(v is None for v in a + b):
                    outputs[f"{species}_{label}"] = {"passed": False, "reason": "insufficient bottom events"}
                    continue
                paired = interval(np.asarray(a) - b)
                scale = abs(float(np.mean(b)))
                bound = (abs(paired["mean"]) + paired["half_width"]) / max(scale, 1e-30)
                outputs[f"{species}_{label}"] = {
                    "paired_difference": paired,
                    "relative_difference_upper_95": bound,
                    "passed": bound <= 0.02,
                }
        a = [r["final_surface_charge_c"] for r in results[str(coarse)]["runs"]]
        b = [r["final_surface_charge_c"] for r in results[str(fine)]["runs"]]
        paired = interval(np.asarray(a) - b)
        bound = float((abs(paired["mean"]) + paired["half_width"]) / max(abs(np.mean(b)), 1e-30))
        outputs["surface_charge"] = {
            "paired_difference": paired,
            "relative_difference_upper_95": bound,
            "passed": bound <= 0.02,
        }
        charging.append(
            {
                "pair": [coarse, fine],
                "outputs": outputs,
                "ion_and_charge_passed": all(
                    v["passed"] for k, v in outputs.items() if k.startswith("Ar+") or k == "surface_charge"
                ),
                "all_outputs_passed": all(v["passed"] for v in outputs.values()),
            }
        )
    return {
        "scope": "Supplied IAEDF pair, internal 10 mTorr, 16 frozen RF phases, 100 us duration.",
        "samples_per_species_per_run": budget,
        "seeds": SEEDS[:seed_count],
        "groups": results,
        "charging_step_pairs": charging,
        "max_charge_residual": max(r["max_charge_residual"] for g in results.values() for r in g["runs"]),
        "unresolved": {
            steps: {
                sp: max(r["statistics"][sp]["unresolved_weight"] for r in g["runs"])
                for sp in g["runs"][0]["statistics"]
            }
            for steps, g in results.items()
        },
        "frozen_rf_all_runs_passed": all(
            r["max_flight_to_rf_period"] < 0.01 for g in results.values() for r in g["runs"]
        ),
        "note": "Arrival/energy are integrated over the run; the final surface charge is compared separately. Electron intervals may remain insufficient. Input flux and projected-angle approximation remain uncalibrated.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("fixed", "coupled", "all"), default="all")
    parser.add_argument("--particles", type=int, default=20000)
    parser.add_argument("--samples", type=int, default=32000)
    parser.add_argument("--seeds", type=int, choices=(3, 4, 5), default=5)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/qa" / f"numerical_validation_{datetime.now():%Y%m%d}"
    )
    args = parser.parse_args()
    if args.samples < 160 or args.samples % 16:
        parser.error("--samples must be at least 160 and divisible by 16")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": code_id(),
            "gpu": gpu_info(),
            "validation_plan": {
                "analytic_and_regression": "pytest: plates, dielectric displacement, sheet charge, trajectory energy, charge balance, CPU/GPU, resume, supplied input",
                "fixed_field": f"4 mesh sizes; shared {args.particles} ions; max_dt and cell_fraction refinement; 5 lateral extents; Dirichlet/Neumann; static energy",
                "coupled": f"{args.seeds} independent seeds; {args.samples} particles per species per run; 4/8/16 charging updates; 95% Student t intervals",
                "targets": {
                    "analytic_relative": 0.01,
                    "charge_residual": 1e-6,
                    "convergence_relative": 0.02,
                    "statistical_relative_half_width": 0.02,
                },
                "comsol": "excluded by user request",
                "pending": [
                    "axisymmetric independent solver",
                    "RF time-dependent transport",
                    "local phase-resolved space charge",
                    "gas cross sections",
                    "measured calibration",
                ],
            },
        }
    )
    start = time.perf_counter()
    atomic_json(path, report)
    if args.section in {"fixed", "all"}:
        report["fixed"] = fixed_checks(args.particles)
        atomic_json(path, report)
    if args.section in {"coupled", "all"}:
        report["coupled"] = coupled_checks(args.output, args.samples, args.seeds)
        atomic_json(path, report)
    report["last_section_elapsed_s"] = time.perf_counter() - start
    atomic_json(path, report)
    print(f"Report: {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
