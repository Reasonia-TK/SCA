import json
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memory_hole.config import CaseConfig, Waveform, WaveformSample
from memory_hole.iaedf_core.constants import QE
from memory_hole.iaedf_core.plasma import derive_plasma
from memory_hole.iaedf_core.schemas import Config1D, Config2D
from memory_hole.iaedf_core.waveform import make_waveform_function
from memory_hole.inlet import sample_distribution, verify_metadata, waveform_id
from memory_hole.storage import atomic_json, read_json
from memory_hole.upstream import (
    ASSETS,
    ApplyRequest,
    collector_length,
    connected_case,
    distribution_plot,
    run_upstream,
    validated_config,
)
from memory_hole.upstream_api import UpstreamService


def write_request(directory, model, config):
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "request.json", dict(name="検証IAEDF", model=model, config=config.model_dump()))
    atomic_json(directory / "status.json", dict(status="queued", progress=0))


@pytest.fixture(scope="module")
def completed_1d(tmp_path_factory):
    root = tmp_path_factory.mktemp("iaedf1d")
    config = Config1D()
    config.tpmc.n_particles = 600
    config.gas.pressures_mTorr = [0, 10]
    config.waveform.mode = "csv"
    config.waveform.csv_text = (ASSETS / "tailored_waveform_5harmonic.csv").read_text(encoding="utf-8-sig")
    write_request(root, "1d", config)
    run_upstream(str(root))
    return root, config


@pytest.fixture(scope="module")
def completed_2d(tmp_path_factory):
    root = tmp_path_factory.mktemp("iaedf2d")
    config = Config2D()
    config.field2d.nx, config.field2d.ny = 65, 40
    config.tpmc.n_particles = 1200
    config.gas.pressures_mTorr = [5]
    config.space_charge.deposition_particles = 800
    config.space_charge.outer_iterations = 2
    write_request(root, "2d", config)
    run_upstream(str(root))
    return root, config


def test_csv_waveform_periodic_interpolation_and_metadata(completed_1d):
    root, upstream = completed_1d
    c, data, meta = connected_case(root, ApplyRequest(config=CaseConfig(), pressure_index=1))
    reference = make_waveform_function(upstream.waveform, 2 * np.pi * upstream.plasma.frequency_Hz)
    angles = np.linspace(-720, 720, 1237)
    assert c.waveform.samples
    assert np.allclose([c.waveform.voltage(p) for p in angles], reference(np.radians(angles)), atol=1e-9)
    assert c.gas_pressure_pa == pytest.approx(1.33322368)
    assert c.geometry.injection_z_nm == c.inlet_evaluation_z_nm == 0
    assert c.electron.phase_flux_model == "boltzmann"
    assert len(c.electron.plasma_potential_samples) == upstream.circuit.phase_points
    verify_metadata(c, meta)
    altered = c.model_copy(deep=True)
    altered.waveform.samples[0].potential_v += 1
    with pytest.raises(ValueError, match="波形"):
        verify_metadata(altered, meta)
    with np.load(root / "raw.npz") as raw:
        mask = raw["p1_reached"]
        assert np.array_equal(data["phases"], raw["p1_impact_phase_deg"][mask])
        assert np.std(data["values"][:, 1]) > 1
        d = derive_plasma(upstream.plasma)
        assert np.allclose(
            0.5 * d.ion_mass * np.sum(data["values"] ** 2, axis=1) / QE, raw["p1_energy_eV"][mask]
        )
        assert c.ions[0].flux_m2_s == pytest.approx(d.n_s * d.bohm_speed * np.mean(mask))
    assert data["probability"].sum() == pytest.approx(1)


def test_2d_collector_flux_rotation_and_phases(completed_2d):
    root, upstream = completed_2d
    request = ApplyRequest(
        config=CaseConfig(), collector_min_m=3.2e-3, collector_max_m=12.8e-3, azimuth_deg=90
    )
    c, data, meta = connected_case(root, request)
    with np.load(root / "raw.npz") as raw:
        mask = raw["p0_on_wafer"] & (raw["p0_impact_x_m"] >= 3.2e-3) & (raw["p0_impact_x_m"] <= 12.8e-3)
        mask &= raw["p0_vz_m_s"] > 0
        assert data["values"][:, 0] == pytest.approx(-raw["p0_vy_m_s"][mask], abs=1e-8)
        assert data["values"][:, 1] == pytest.approx(raw["p0_vx_m_s"][mask], abs=1e-8)
        assert data["values"][:, 2] == pytest.approx(raw["p0_vz_m_s"][mask])
        assert data["phases"] == pytest.approx(raw["p0_impact_phase_deg"][mask])
        assert np.std(data["phases"]) > 30
        d = derive_plasma(upstream.plasma)
        expected = d.n_s * d.bohm_speed * mask.sum() / upstream.tpmc.n_particles * 16e-3 / 9.6e-3
        assert c.ions[0].flux_m2_s == pytest.approx(expected)
        assert c.waveform.voltage(0) == pytest.approx(
            upstream.wafer_waveform.sinusoid_dc_V + upstream.wafer_waveform.sinusoid_amplitude_V
        )
    verify_metadata(c, meta)
    graph = distribution_plot(root, 0, 3.2e-3, 12.8e-3)
    assert graph["sample_count"] == meta["sample_count"]
    assert len(graph["iaedf"]["density"]) == 80
    assert graph["mean_polar_angle_deg"] > 0


def test_collector_surface_area_includes_slope():
    geo = Config2D().geometry
    expected = 9.8e-3 + 2 * np.hypot(0.2e-3, 0.2e-3)
    assert collector_length(geo, 0, geo.domain_length_m) == pytest.approx(expected)


@pytest.mark.parametrize("bounds", [(None, None), (-1, 2), (12e-3, 3e-3), (0, 2e-3)])
def test_invalid_or_empty_collector_is_rejected(completed_2d, bounds):
    with pytest.raises(ValueError):
        connected_case(
            completed_2d[0],
            ApplyRequest(config=CaseConfig(), collector_min_m=bounds[0], collector_max_m=bounds[1]),
        )


@pytest.mark.parametrize(
    "path,value",
    [
        ("tpmc.n_particles", -5),
        ("plasma.frequency_Hz", 0),
        ("gas.pressures_mTorr", []),
        ("gas.xsec_csv_name", "../../other.csv"),
        ("plasma.ion_mass_amu", 4.0),
        ("plasma.electron_temperature_eV", float("nan")),
    ],
)
def test_invalid_model_input_is_rejected(path, value):
    config = Config1D().model_dump()
    section, key = path.split(".")
    config[section][key] = value
    with pytest.raises(ValueError):
        validated_config("1d", config)


def test_waveform_wraparound_and_legacy_hash():
    c = CaseConfig()
    import hashlib

    assert waveform_id(c) == hashlib.sha256(json.dumps([0, 0, 13.56e6, 0]).encode()).hexdigest()
    wave = Waveform(
        samples=[
            WaveformSample(phase_deg=p, potential_v=v) for p, v in [(30, 1), (120, 3), (210, -1), (300, -3)]
        ]
    )
    assert wave.voltage(0) == pytest.approx(-1 / 3)
    assert wave.voltage(360) == wave.voltage(0)
    with pytest.raises(ValueError):
        Waveform(samples=[WaveformSample(phase_deg=0, potential_v=1)] * 4)


def test_upstream_worker_is_spawn_safe_and_cancellable(tmp_path):
    config = Config1D()
    config.tpmc.n_particles = 100
    config.gas.pressures_mTorr = [0]
    write_request(tmp_path, "1d", config)
    (tmp_path / "cancel.request").touch()
    proc = multiprocessing.get_context("spawn").Process(target=run_upstream, args=(str(tmp_path),))
    proc.start()
    proc.join(30)
    if proc.is_alive():
        proc.terminate()
        pytest.fail("Spawned IAEDF worker timed out")
    assert proc.exitcode == 0
    assert read_json(tmp_path / "status.json")["status"] == "cancelled"
    assert not (tmp_path / "raw.npz").exists()


def test_api_apply_and_distribution_are_consumable(completed_1d, tmp_path):
    import shutil

    service = UpstreamService(tmp_path, tmp_path / "distributions", lambda p: read_json(p / "status.json"))
    jid = "a" * 32
    shutil.copytree(completed_1d[0], service.root / jid)
    app = FastAPI()
    app.include_router(service.router)
    client = TestClient(app)
    response = client.post(
        f"/api/iaedf/jobs/{jid}/apply", json=dict(config=CaseConfig().model_dump(), pressure_index=1)
    )
    assert response.status_code == 200
    body = response.json()
    c = CaseConfig.model_validate(body["config"])
    verify_metadata(c, body["metadata"])
    with np.load(service.distributions / (c.ions[0].distribution_id + ".npz")) as stored:
        data = {key: stored[key] for key in stored.files}
    velocities, phases = sample_distribution(
        np.random.default_rng(2), 300, data, body["metadata"], c.ions[0].mass_amu
    )
    assert velocities.shape == (300, 3) and np.all(velocities[:, 2] > 0)
    assert np.all((phases >= 0) & (phases < 360))
    assert client.get(f"/api/iaedf/jobs/{jid}/distribution?pressure_index=1").status_code == 200
    assert client.get(f"/api/iaedf/jobs/{jid}/export/raw").content[:2] == b"PK"
    assert client.get("/api/iaedf/jobs/../../x").status_code == 404
    assert client.post(f"/api/iaedf/jobs/{jid}/cancel").status_code == 409


def test_queue_budget_restart_and_cancel(tmp_path, monkeypatch):
    service = UpstreamService(tmp_path, tmp_path / "distributions", lambda p: read_json(p / "status.json"))
    service.start()
    jid = "b" * 32
    write_request(service.root / jid, "1d", Config1D())
    launches = []
    fake = SimpleNamespace(start=lambda: launches.append(True), is_alive=lambda: True)
    monkeypatch.setattr(
        "memory_hole.upstream_api.multiprocessing.get_context",
        lambda _: SimpleNamespace(Process=lambda **kwargs: fake),
    )
    service.tick(can_launch=False)
    assert not launches
    service.tick()
    assert launches == [True] and service.active()
    service.tick()
    assert launches == [True]
    service.workers.clear()
    atomic_json(service.root / jid / "status.json", dict(status="running"))
    service.start()
    service.tick(can_launch=False)
    assert service.read_status(service.root / jid)["status"] == "failed"


def test_api_rejects_unknown_fields_and_queued_apply(tmp_path):
    service = UpstreamService(tmp_path, tmp_path / "distributions", lambda p: read_json(p / "status.json"))
    app = FastAPI()
    app.include_router(service.router)
    client = TestClient(app)
    legacy = Config1D().model_dump()
    del legacy["magnetic"]
    normalized = client.post("/api/iaedf/validate", json=dict(model="1d", config=legacy))
    assert normalized.status_code == 200
    assert normalized.json()["config"]["magnetic"]["bz_T"] == 0
    response = client.post(
        "/api/iaedf/jobs", json=dict(name="queued", model="1d", config=Config1D().model_dump())
    )
    assert response.status_code == 200
    jid = response.json()["id"]
    assert (
        client.post(f"/api/iaedf/jobs/{jid}/apply", json=dict(config=CaseConfig().model_dump())).status_code
        == 409
    )
    assert client.post(f"/api/iaedf/jobs/{jid}/cancel").status_code == 200
    assert client.get(f"/api/iaedf/jobs/{jid}").json()["status"] == "cancelled"
    config = Config1D().model_dump()
    config["tpmc"]["unknown"] = 1
    with pytest.raises(ValueError):
        validated_config("1d", config)
