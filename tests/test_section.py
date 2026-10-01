import json

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from memory_hole import api
from memory_hole.section import plane_frame, saved_section, section_arrays


def cube_solution():
    # Two conforming cubes with an interface at z=50 nm; six tets per cube.
    nodes = np.array([[x, y, z] for z in (0, 50, 100) for y in (0, 100) for x in (0, 100)], float)
    tets = []
    for offset in (0, 4):
        a, b, c, d, e, f, g, h = np.arange(8) + offset
        tets.extend([(a, b, d, h), (a, d, c, h), (a, c, g, h), (a, g, e, h), (a, e, f, h), (a, f, b, h)])
    tets = np.array(tets)
    material = np.repeat([0, 1], 6)
    potential = 1 + nodes @ np.array([0.01, -0.02, 0.03])
    electric = np.tile([-1e7, 2e7, -3e7], (len(tets), 1))
    faces = np.array([[4, 5, 7], [4, 7, 6]])
    return nodes, tets, material, potential, electric, faces, np.full(2, 1e-4)


@pytest.mark.parametrize(
    "plane,position,angle", [("xy", 37, 23), ("xz", 50, 0), ("yz", 50, 0), ("vertical", 0, 37)]
)
def test_sections_preserve_analytic_linear_potential_and_field(plane, position, angle):
    output = section_arrays(
        *cube_solution(), plane=plane, position_nm=position, angle_deg=angle, resolution=65
    )
    u, v = np.meshgrid(output["u_nm"], output["v_nm"])
    basis_u, basis_v, normal, *_ = plane_frame(plane, angle)
    xyz = u[..., None] * basis_u + v[..., None] * basis_v + position * normal
    expected = 1 + xyz @ np.array([0.01, -0.02, 0.03])
    valid = np.array(output["material"]) >= 0
    actual = np.array(output["values"]["potential_v"], dtype=float)
    assert valid.any()
    assert actual[valid] == pytest.approx(expected.ravel()[valid], abs=1e-12)
    for axis, value in zip("xyz", [-1e7, 2e7, -3e7], strict=True):
        assert np.array(output["values"][f"e_{axis}_v_m"], float)[valid] == pytest.approx(value)
    assert np.array(output["values"]["e_norm_v_m"], float)[valid] == pytest.approx(np.sqrt(14) * 1e7)
    assert output["axes"]["vertical_increases_down"] == (plane != "xy")
    assert output["angle_deg"] == (angle if plane == "vertical" else 0)
    json.dumps(output, allow_nan=False)


@pytest.mark.parametrize("position", [0, 50, 100])
def test_sections_through_shared_faces_have_no_gaps(position):
    output = section_arrays(*cube_solution(), plane="xy", position_nm=position, resolution=65)
    assert output["valid_points"] == output["width"] * output["height"]
    expected_material = 0 if position <= 50 else 1
    assert set(output["material"]) == {expected_material}
    # Internal tetrahedral edges are removed from the exterior outline.
    for a, b in output["boundary_segments"]:
        assert (a[0] == b[0] and a[0] in {0, 100}) or (a[1] == b[1] and a[1] in {0, 100})


def test_discontinuous_field_uses_each_material_without_smoothing():
    arrays = list(cube_solution())
    arrays[4][arrays[2] == 1] *= 0.25
    output = section_arrays(*arrays, plane="xz", position_nm=50, resolution=65)
    material = np.array(output["material"])
    field_z = np.array(output["values"]["e_z_v_m"])
    assert field_z[material == 0] == pytest.approx(-3e7)
    assert field_z[material == 1] == pytest.approx(-7.5e6)
    assert any(a[1] == b[1] == 50 for a, b in output["boundary_segments"])


def test_missing_volume_is_masked_instead_of_bridged():
    arrays = list(cube_solution())
    # Remove the upper cube while retaining original nodes and display bounds.
    arrays[1], arrays[2], arrays[4] = arrays[1][:6], arrays[2][:6], arrays[4][:6]
    output = section_arrays(*arrays, plane="xz", position_nm=50, resolution=65)
    heights = np.repeat(output["v_nm"], output["width"])
    above = heights > 50
    assert np.array(output["material"])[above].tolist() == [-1] * int(above.sum())
    assert np.array(output["values"]["potential_v"], object)[above].tolist() == [None] * int(above.sum())
    assert all(value is None for value in np.array(output["values"]["e_norm_v_m"], object)[above])


def test_sheet_charge_is_intersected_as_lines_or_coplanar_triangles():
    vertical = section_arrays(*cube_solution(), plane="xz", position_nm=50, resolution=65)
    assert vertical["surface_segments"]
    assert not vertical["surface_polygons"]
    assert all(p[1] == 50 for item in vertical["surface_segments"] for p in item["points_nm"])
    assert all(item["sigma_c_m2"] == 1e-4 for item in vertical["surface_segments"])
    horizontal = section_arrays(*cube_solution(), plane="xy", position_nm=50, resolution=65)
    assert not horizontal["surface_segments"]
    assert len(horizontal["surface_polygons"]) == 2
    outside = section_arrays(*cube_solution(), plane="xy", position_nm=25, resolution=65)
    assert not outside["surface_segments"] and not outside["surface_polygons"]


@pytest.mark.parametrize("position", [-1, 101, float("nan"), float("inf")])
def test_invalid_plane_positions_are_rejected(position):
    with pytest.raises(ValueError):
        section_arrays(*cube_solution(), plane="xy", position_nm=position, resolution=65)


def write_saved_solution(directory):
    nodes, tets, material, potential, electric, faces, sigma = cube_solution()
    with h5py.File(directory / "results.h5", "w") as h5:
        h5.attrs["metadata_json"] = json.dumps({"mesh_id": "analytic-cube"})
        h5.attrs["state_json"] = json.dumps({"step": 2, "time_s": 2e-6})
        for key, array in {
            "mesh/nodes_m": nodes * 1e-9,
            "mesh/tetrahedra": tets,
            "mesh/material": material,
            "field/potential_v": potential * 2,
            "field/electric_field_v_m": electric * 2,
            "surface/faces": faces,
            "surface/sigma_c_m2": sigma * 2,
        }.items():
            h5.create_dataset(key, data=array)
    snapshots = directory / "snapshots"
    snapshots.mkdir()
    with h5py.File(snapshots / "00001.h5", "w") as saved:
        saved.attrs["time_s"] = 1e-6
        for key, array in {
            "potential_v": potential,
            "electric_field_v_m": electric,
            "sigma_c_m2": sigma,
        }.items():
            saved.create_dataset(key, data=array)


def test_saved_times_read_snapshot_or_combined_hdf5_without_changing_job(tmp_path):
    write_saved_solution(tmp_path)
    latest = saved_section(tmp_path, plane="xz", position_nm=50, resolution=65)
    snapshot = saved_section(tmp_path, plane="xz", position_nm=50, resolution=65, step=1)
    assert latest["step"] == 2 and latest["time_s"] == 2e-6
    assert snapshot["step"] == 1 and snapshot["time_s"] == 1e-6
    assert np.array(latest["values"]["potential_v"]) == pytest.approx(
        np.array(snapshot["values"]["potential_v"]) * 2
    )
    assert latest["surface_segments"][0]["sigma_c_m2"] == snapshot["surface_segments"][0]["sigma_c_m2"] * 2
    with (
        h5py.File(tmp_path / "results.h5", "a") as h5,
        h5py.File(tmp_path / "snapshots/00001.h5", "r") as saved,
    ):
        group = h5.create_group("time_series/00001")
        group.attrs["time_s"] = saved.attrs["time_s"]
        for key in saved:
            saved.copy(key, group)
    (tmp_path / "snapshots/00001.h5").unlink()
    combined = saved_section(tmp_path, plane="xz", position_nm=50, resolution=65, step=1)
    assert combined == snapshot
    with pytest.raises(FileNotFoundError, match="保存されていません"):
        saved_section(tmp_path, step=999)


def test_slice_api_validation_and_missing_results(tmp_path, monkeypatch):
    jid = "a" * 32
    directory = tmp_path / jid
    directory.mkdir()
    monkeypatch.setattr(api, "JOBS", tmp_path)
    client = TestClient(api.app)
    assert client.get(f"/api/jobs/{jid}/slice").status_code == 404
    write_saved_solution(directory)
    response = client.get(f"/api/jobs/{jid}/slice?plane=xy&position_nm=37&resolution=65")
    assert response.status_code == 200
    assert response.json()["source"] == "full_fem"
    assert response.json()["field_phase_deg"] == 0
    assert client.get(f"/api/jobs/{jid}/slice?position_nm=101").status_code == 400
    for parameters in ["resolution=10000", "position_nm=nan", "plane=other", "step=-1", "angle_deg=361"]:
        assert client.get(f"/api/jobs/{jid}/slice?{parameters}").status_code == 422
    assert client.get(f"/api/jobs/{jid}/slice?step=999").status_code == 404
