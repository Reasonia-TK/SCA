import json
import multiprocessing
from pathlib import Path

import h5py
import numpy as np
import pytest

from memory_hole import engine
from memory_hole.config import CaseConfig
from memory_hole.field import FieldSolver, build_mesh
from memory_hole.geometry import derive
from memory_hole.transport import AMU, E_CHARGE, boundary_maps, sample_generated, trace


def cuda_spawn_worker(queue):
    from memory_hole.cuda_runtime import prepare_cuda_runtime

    prepare_cuda_runtime()
    import cupy as cp

    kernel = cp.RawKernel('extern "C" __global__ void smoke(double* x) { x[0] = sqrt(4.0); }', "smoke")
    value = cp.zeros(1)
    kernel((1,), (1,), (value,))
    queue.put(float(value.get()[0]))


def test_spawned_cuda_compiler():
    pytest.importorskip("cupy")
    if not engine.gpu_info()["available"]:
        pytest.skip("CUDA device unavailable")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=cuda_spawn_worker, args=(queue,))
    worker.start()
    worker.join(30)
    if worker.is_alive():
        worker.terminate()
        pytest.fail("CUDA spawned worker timed out")
    assert worker.exitcode == 0
    assert queue.get(timeout=2) == 2


@pytest.fixture(scope="module")
def reference():
    c = CaseConfig()
    mesh = build_mesh(c)
    solver = FieldSolver(mesh, c)
    maps = boundary_maps(mesh)
    return c, mesh, solver, maps


def test_mesh_excludes_dummy_particle_region_and_inlet_potential(reference):
    c, m, s, maps = reference
    gas_centers = m.nodes[m.tets[m.material == 0]].mean(axis=1) * 1e9
    d = derive(c)
    for dummy in d["dummies"]:
        radial = np.hypot(gas_centers[:, 0] - dummy["x_nm"], gas_centers[:, 1] - dummy["y_nm"])
        overlap = (
            (radial < dummy["diameter_nm"] / 2)
            & (gas_centers[:, 2] > dummy["top_nm"])
            & (gas_centers[:, 2] < dummy["bottom_nm"])
        )
        assert not overlap.any()
    assert len(m.surface_faces) > 0
    phi, _, diag = s.solve(np.zeros(len(m.surface_faces)), -170)
    assert np.max(abs(phi[s.inlet_nodes] + 170)) < 1e-10
    assert np.max(abs(phi[m.conductor_nodes] + 170)) < 1e-10
    assert diag["field_residual"] < 1e-8


def test_cpu_gpu_identical_particles(reference):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("CUDA device unavailable")
    from memory_hole.gpu import GPUTracer

    c, m, s, (kinds, patches) = reference
    c.geometry.injection_z_nm = 0
    p, v, _ = sample_generated(np.random.default_rng(4), 2048, c, derive(c), c.ions[0])
    starts = s.locate(p)
    assert np.all(starts >= 0)
    sigma = np.full(len(m.surface_faces), 2e-5)
    _, field, _ = s.solve(sigma, -100)
    args = (
        p,
        v,
        starts,
        np.full(len(p), E_CHARGE),
        np.full(len(p), 40 * AMU),
        field,
        m.inverse,
        m.gradients,
        m.neighbors,
        kinds,
        patches,
        5000,
        2e-12,
        m.min_cell_m,
        0.2,
        2,
    )
    cpu = trace(*args)
    gpu = GPUTracer(m, kinds, patches)(*args)
    assert np.array_equal(cpu[2], gpu[2])
    assert np.array_equal(cpu[3], gpu[3])
    assert np.allclose(cpu[0], gpu[0], rtol=1e-6, atol=1e-12)
    assert np.allclose(cpu[1], gpu[1], rtol=1e-6, atol=0.001)
    assert not np.any(cpu[2] == 3)
    assert np.array_equal(cpu[9], gpu[9])
    for index, length in enumerate(cpu[9]):
        assert cpu[10][index, 0] == 0
        assert np.all(np.diff(cpu[10][index, :length]) >= 0)
        assert cpu[10][index, length - 1] == pytest.approx(cpu[5][index])
        assert np.allclose(cpu[10][index, :length], gpu[10][index, :length], rtol=1e-6, atol=1e-20)


def test_point_locator_recovers_from_centroid_candidate_miss(reference, monkeypatch):
    _, mesh, solver, _ = reference
    points = np.array([[0, 0, 1200], [50, 20, 450], [99000, 99000, 1200]]) * 1e-9
    tree = solver.gas_tree
    wrong = int(np.argmax(np.linalg.norm(tree.data - points[0], axis=1)))

    class IncompleteNearestQuery:
        def query(self, positions, k):
            return np.zeros((len(positions), k)), np.full((len(positions), k), wrong)

        def query_ball_point(self, positions, radius):
            return tree.query_ball_point(positions, radius)

    monkeypatch.setattr(solver, "gas_tree", IncompleteNearestQuery())
    located = solver.locate(points)
    assert np.all(located[:2] >= 0)
    assert located[2] == -1
    for point, tet in zip(points[:2], located[:2], strict=True):
        bary = np.r_[1, point / 1e-9] @ mesh.inverse[tet]
        assert bary.min() >= -1e-8
        assert mesh.material[tet] == 0


def job(directory, config):
    directory.mkdir()
    engine.atomic_json(directory / "config.json", config.model_dump())
    engine.run_job(str(directory), str(directory.parent / "distributions"))
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status["status"] == "failed":
        pytest.fail((directory / "error.log").read_text(encoding="utf-8"))
    return status


def test_charge_balance_and_checkpoint_resume(tmp_path, monkeypatch):
    c = CaseConfig()
    c.mode = "self_consistent"
    c.numerics.samples_per_species = 40
    c.numerics.batch_size = 20
    c.numerics.phase_bins = 1
    c.numerics.charging_steps = 2
    c.surface.electron_reflection = 0.5
    c.surface.secondary_yield = 0.5
    c.surface.ion_neutralization_reflection = 0.5
    c.leakage_tau_s = 1e-5
    full = tmp_path / "full"
    interrupted = tmp_path / "interrupted"
    assert job(full, c)["status"] == "completed"
    original = engine.atomic_json

    def stop_after_first_step(path, value):
        original(path, value)
        if path.parent == interrupted and path.name == "status.json" and value.get("time_s", 0) > 0:
            (interrupted / "pause.request").touch()

    monkeypatch.setattr(engine, "atomic_json", stop_after_first_step)
    assert job(interrupted, c)["status"] == "paused"
    monkeypatch.setattr(engine, "atomic_json", original)
    (interrupted / "pause.request").unlink()
    engine.run_job(str(interrupted), str(tmp_path / "distributions"))
    assert json.loads((interrupted / "status.json").read_text(encoding="utf-8"))["status"] == "completed"
    with h5py.File(full / "checkpoint.h5") as a, h5py.File(interrupted / "checkpoint.h5") as b:
        assert np.array_equal(a["sigma_c_m2"][:], b["sigma_c_m2"][:])
        state_a, state_b = json.loads(a.attrs["state_json"]), json.loads(b.attrs["state_json"])
        assert state_a["stats"] == state_b["stats"]
        assert state_a["ledger"] == state_b["ledger"]
        assert max(h["charge_residual"] for h in state_a["history"]) < 1e-12
    with h5py.File(full / "results.h5") as h5:
        assert len(h5["time_series"]) == 2
        assert {h5[f"trajectories/{i}"].attrs["source_species"] for i in h5["trajectories"]} == {
            "Ar+",
            "electron",
        }
        for snapshot in h5["time_series"].values():
            for index, path in snapshot["trajectories"].items():
                times = snapshot[f"trajectory_times/{index}"][:]
                assert len(times) == len(path)
                assert np.all(np.diff(times) >= 0)
                assert times[-1] > 0
                phase = snapshot[f"trajectory_fields/{path.attrs['phase_index']}"]
                assert phase.attrs["phase_deg"] == path.attrs["rf_phase_deg"]
                assert phase.attrs["field_time_s"] == path.attrs["field_time_s"]
                assert path.attrs["field_time_s"] < snapshot.attrs["time_s"]


def test_supplied_pair():
    from memory_hole.iaedf import case_from_pair

    root = Path(__file__).resolve().parents[1]
    if not (root / "sample-20mTorr.npz").exists():
        pytest.skip("User sample not available")
    c, data, meta = case_from_pair(
        (root / "sample-20mTorr.npz").read_bytes(),
        (root / "sample-20mTorr.json").read_bytes(),
        "sample-20mTorr.npz",
        CaseConfig(),
        projection_policy="zero_out_of_plane",
    )
    assert c.waveform.dc_v == -170 and c.waveform.amplitude_v == 158
    assert meta["pressure_mtorr"] == 10
    assert len(data["values"]) == 299566
    assert data["phases"] is not None
    assert c.electron.phase_flux_model == "boltzmann"
    assert c.electron.flux_m2_s > c.ions[0].flux_m2_s
