import math

import numpy as np
import pytest

from memory_hole.config import CaseConfig, Dummy, ProfilePoint
from memory_hole.field import EPS0, FieldSolver, Mesh, tetra_data
from memory_hole.geometry import derive
from memory_hole.inlet import parse_distribution
from memory_hole.transport import (
    E_CHARGE,
    ELECTRON_MASS,
    first_face_time,
    thermal_electrons,
    trace,
    wilson_interval,
)


def test_preset_and_dimension_changes():
    c = CaseConfig()
    d = derive(c)
    assert d["depth_nm"] == 2400
    assert d["envelope_diameter_nm"] == 740
    assert d["minimum_dummy_gap_nm"] == pytest.approx(190)
    assert d["dummies"][1]["y_nm"] == pytest.approx(268.4678751732)
    c.geometry.carbon_thickness_nm = 500
    c.geometry.oxide_thickness_nm = 3000
    c.geometry.dummy_count = 4
    c.geometry.gap_evaluation_depth_nm = 500
    d2 = derive(c)
    assert d2["depth_nm"] == 3500
    assert d2["dummies"][0]["top_nm"] == 500
    assert d2["dummies"][0]["length_nm"] == 3000
    assert d2["dummies"][1]["angle_deg"] == 90
    assert d2["geometry_id"] != d["geometry_id"]


def test_geometry_rejects_interference_and_disconnection():
    c = CaseConfig()
    c.geometry.dummies = [Dummy(gap_nm=2)]
    c.geometry.bottom_diameter_nm = 500
    with pytest.raises(ValueError, match="接触"):
        derive(c)
    c = CaseConfig()
    c.geometry.dummies = [Dummy(top_nm=301)]
    with pytest.raises(ValueError, match="接続"):
        derive(c)
    c = CaseConfig()
    c.geometry.profile = [ProfilePoint(depth_nm=0, radius_nm=200), ProfilePoint(depth_nm=100, radius_nm=200)]
    with pytest.raises(ValueError, match="プロファイル"):
        derive(c)


def plate_mesh(layered=False):
    grid = np.linspace(0, 100, 5)
    nodes = np.array([(x, y, z) for x in grid for y in grid for z in grid]) * 1e-9

    def ix(i, j, k):
        return i * 25 + j * 5 + k

    tets = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a, b, c, d, e, f, g, h = [
                    ix(i + x, j + y, k + z)
                    for x, y, z in (
                        (0, 0, 0),
                        (1, 0, 0),
                        (0, 1, 0),
                        (1, 1, 0),
                        (0, 0, 1),
                        (1, 0, 1),
                        (0, 1, 1),
                        (1, 1, 1),
                    )
                ]
                tets.extend(
                    [(a, b, d, h), (a, d, c, h), (a, c, g, h), (a, g, e, h), (a, e, f, h), (a, f, b, h)]
                )
    tets = np.array(tets)
    material = (
        (
            nodes[
                tets,
                :,
            ][..., 2].mean(axis=1)
            > 50e-9
        ).astype(np.int8)
        if layered
        else np.zeros(len(tets), np.int8)
    )
    inv, grad, vol = tetra_data(nodes, tets)
    return Mesh(
        nodes,
        tets,
        material,
        grad,
        vol,
        inv,
        np.full((len(tets), 4), -1),
        np.empty((0, 3), int),
        np.empty(0),
        np.empty((0, 3)),
        np.empty(0, int),
        np.flatnonzero(nodes[:, 2] == 0),
        np.flatnonzero(np.isclose(nodes[:, 2], 100e-9, atol=1e-15)),
        np.empty((0, 3), int),
        np.empty(0),
        25e-9,
        "plate",
    )


@pytest.mark.parametrize("layered", [False, True])
def test_fem_plates_and_dielectric_flux(layered):
    c = CaseConfig()
    c.oxide_relative_permittivity = 4
    m = plate_mesh(layered)
    solver = FieldSolver(m, c)
    phi, field, diag = solver.solve(np.empty(0), 1)
    z = m.nodes[:, 2] / 100e-9
    expected = 1 - z if not layered else np.where(z <= 0.5, 1 - 1.6 * z, 0.4 - 0.4 * z)
    assert np.max(abs(phi - expected)) < 1e-8
    if layered:
        displacement = field[:, 2] * np.where(m.material == 0, 1, 4) * EPS0
        assert np.std(displacement) / abs(np.mean(displacement)) < 1e-8
    assert diag["field_residual"] < 1e-8


def test_exact_accelerating_face_crossing():
    # b(t)=1-2t-t^2; the face is reached before t=1.
    assert first_face_time(1, -2, -1, 1) == pytest.approx(math.sqrt(2) - 1)


def test_known_interface_sheet_charge():
    c = CaseConfig()
    c.oxide_relative_permittivity = 4
    m = plate_mesh(layered=True)
    faces = m.tets[:, np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])].reshape(-1, 3)
    unique = np.unique(np.sort(faces, axis=1), axis=0)
    on_sheet = np.all(np.isclose(m.nodes[unique, 2], 50e-9, atol=1e-15, rtol=0), axis=1)
    m.surface_faces = unique[on_sheet]
    p = m.nodes[m.surface_faces]
    m.surface_area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
    sigma = 1e-4
    solver = FieldSolver(m, c)
    phi, field, _ = solver.solve(np.full(len(m.surface_faces), sigma), 0)
    height = 100e-9
    peak = sigma * height / (2 * EPS0 * (1 + 4))
    expected = peak * np.minimum(m.nodes[:, 2], height - m.nodes[:, 2]) / (height / 2)
    assert np.max(abs(phi - expected)) < 1e-9
    left_d = EPS0 * field[m.material == 0, 2].mean()
    right_d = EPS0 * 4 * field[m.material == 1, 2].mean()
    assert right_d - left_d == pytest.approx(sigma, rel=1e-8)


def test_uniform_field_trajectory_and_energy():
    nodes = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]) * 1e-9
    tets = np.array([[0, 1, 2, 3]])
    inv, grad, _ = tetra_data(nodes, tets)
    p = np.array([[10e-9, 10e-9, 10e-9]])
    v = np.array([[0.0, 0.0, 1e4]])
    q = np.array([E_CHARGE])
    mass = np.array([1e-26])
    field = np.array([[0.0, 0.0, 1e6]])
    out = trace(
        p,
        v,
        np.array([0]),
        q,
        mass,
        field,
        inv,
        grad,
        np.full((1, 4), -1, np.int64),
        np.full((1, 4), 1, np.int8),
        np.full((1, 4), -1, np.int64),
        10000,
        1e-12,
        1e-9,
        0.2,
        1,
    )
    end, velocity, status, _, _, time, *_ = out
    assert status[0] == 1
    assert end[0, 2] == pytest.approx(80e-9, rel=1e-7)
    assert end[0, 2] == pytest.approx(
        p[0, 2] + v[0, 2] * time[0] + 0.5 * q[0] / mass[0] * 1e6 * time[0] ** 2, rel=1e-7
    )
    delta_ke = 0.5 * mass[0] * (np.sum(velocity[0] ** 2) - np.sum(v[0] ** 2))
    assert delta_ke == pytest.approx(q[0] * 1e6 * (end[0, 2] - p[0, 2]), rel=1e-7)


def test_flux_weighted_electrons_and_rare_events():
    v = thermal_electrons(np.random.default_rng(2), 100000, 3)
    mean_ev = np.mean(0.5 * ELECTRON_MASS * np.sum(v * v, axis=1) / E_CHARGE)
    assert mean_ev == pytest.approx(6, rel=0.01)
    ci = wilson_interval(0, 1000, 1000)
    assert ci["rate"] == 0 and ci["high"] > 0 and ci["relative_half_width"] is None


def test_correlated_csv_and_legacy_npz_policy():
    c = CaseConfig()
    data, meta = parse_distribution(
        b"energy_ev,theta_deg,phi_deg,phase_deg,weight\n10,0,0,20,1\n100,30,45,90,3\n", "a.csv", c, 40
    )
    assert data["probability"].tolist() == [0.25, 0.75]
    assert data["phases"].tolist() == [20, 90]
    assert not meta["absolute_flux_from_distribution"]
    import io

    buf = io.BytesIO()
    np.savez(buf, p0_energy_eV=[10.0, 20.0], p0_reached=[True, False], p0_signed_angle_deg=[2.0, 4.0])
    with pytest.raises(ValueError, match="投影角"):
        parse_distribution(buf.getvalue(), "raw.npz", c, 40)
    data, meta = parse_distribution(buf.getvalue(), "raw.npz", c, 40, projection_policy="zero_out_of_plane")
    assert len(data["values"]) == 1 and len(meta["warnings"]) == 2
