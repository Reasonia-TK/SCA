"""Warm CPU/GPU transport benchmark with identical particles, including transfers."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from memory_hole.config import CaseConfig
from memory_hole.engine import atomic_json, gpu_info
from memory_hole.field import FieldSolver, build_mesh
from memory_hole.geometry import derive
from memory_hole.gpu import GPUTracer
from memory_hole.transport import AMU, E_CHARGE, boundary_maps, sample_generated, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=100000)
    args = parser.parse_args()
    c = CaseConfig()
    mesh = build_mesh(c)
    solver = FieldSolver(mesh, c)
    kinds, patches = boundary_maps(mesh)
    p, v, _ = sample_generated(np.random.default_rng(5), args.particles, c, derive(c), c.ions[0])
    starts = solver.locate(p)
    _, field, _ = solver.solve(np.full(len(mesh.surface_faces), 2e-5), -100)

    def arguments(count):
        return (
            p[:count],
            v[:count],
            starts[:count],
            np.full(count, E_CHARGE),
            np.full(count, 40 * AMU),
            field,
            mesh.inverse,
            mesh.gradients,
            mesh.neighbors,
            kinds,
            patches,
            5000,
            2e-12,
            mesh.min_cell_m,
            0.2,
            0,
        )

    gpu = GPUTracer(mesh, kinds, patches)
    trace(*arguments(100))
    gpu(*arguments(100))
    measurements = []
    cpu = cuda = None
    for _ in range(3):
        start = time.perf_counter()
        cpu = trace(*arguments(args.particles))
        cpu_s = time.perf_counter() - start
        start = time.perf_counter()
        cuda = gpu(*arguments(args.particles))
        gpu_s = time.perf_counter() - start
        measurements.append({"cpu_s": cpu_s, "gpu_s": gpu_s, "speedup": cpu_s / gpu_s})
    result = {
        "gpu": gpu_info(),
        "particles": args.particles,
        "measurements": measurements,
        "status_match": bool(np.array_equal(cpu[2], cuda[2])),
        "patch_match": bool(np.array_equal(cpu[3], cuda[3])),
        "max_endpoint_difference_m": float(np.max(abs(cpu[0] - cuda[0]))),
        "unresolved": int(np.sum(cpu[2] == 3)),
        "timing_scope": "warm transport plus particle transfers and result copies; excludes meshing and field solve",
    }
    output = Path("data/transport_benchmark.json")
    output.parent.mkdir(exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
