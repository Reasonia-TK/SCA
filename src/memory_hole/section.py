"""Sections of the saved P1 tetrahedral solution; no interpolation across voids."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

from .storage import retry_permission

PLANES = {
    "xy": ([1, 0, 0], [0, 1, 0], [0, 0, 1], "x", "y", "z"),
    "xz": ([1, 0, 0], [0, 0, 1], [0, 1, 0], "x", "z", "y"),
    "yz": ([0, 1, 0], [0, 0, 1], [1, 0, 0], "y", "z", "x"),
}


def plane_frame(plane, angle_deg=0):
    if plane == "vertical":
        angle = math.radians(angle_deg)
        return (
            np.array([math.cos(angle), math.sin(angle), 0]),
            np.array([0, 0, 1]),
            np.array([-math.sin(angle), math.cos(angle), 0]),
            "s",
            "z",
            "n",
        )
    u, v, normal, *axes = PLANES[plane]
    return np.array(u), np.array(v), np.array(normal), *axes


def intersect(points, values, distances, u, v, tolerance):
    vertices = []
    for index in np.flatnonzero(np.abs(distances) <= tolerance):
        vertices.append((points[index], values[index]))
    for a, b in combinations(range(len(points)), 2):
        if (distances[a] < -tolerance and distances[b] > tolerance) or (
            distances[b] < -tolerance and distances[a] > tolerance
        ):
            fraction = distances[a] / (distances[a] - distances[b])
            vertices.append(
                (
                    points[a] + fraction * (points[b] - points[a]),
                    values[a] + fraction * (values[b] - values[a]),
                )
            )
    unique = {}
    for point, value in vertices:
        uv = np.array([point @ u, point @ v])
        unique.setdefault(tuple(np.round(uv, 7)), (uv, float(value)))
    if not unique:
        return np.empty((0, 2)), np.empty(0)
    uv = np.array([vertex[0] for vertex in unique.values()])
    data = np.array([vertex[1] for vertex in unique.values()])
    centered = uv - uv.mean(axis=0)
    order = np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))
    return uv[order], data[order]


def raster_triangle(points, phi, electric, material, grid_u, grid_v, output):
    lo_u, lo_v = points.min(axis=0)
    hi_u, hi_v = points.max(axis=0)
    left, right = np.searchsorted(grid_u, [lo_u - 1e-7, hi_u + 1e-7], side="left")
    top, bottom = np.searchsorted(grid_v, [lo_v - 1e-7, hi_v + 1e-7], side="left")
    right, bottom = min(right + 1, len(grid_u)), min(bottom + 1, len(grid_v))
    if right <= left or bottom <= top:
        return
    edges = (points[1:] - points[0]).T
    determinant = np.linalg.det(edges)
    if abs(determinant) < 1e-12:
        return
    x, y = np.meshgrid(grid_u[left:right], grid_v[top:bottom])
    delta = np.stack([x - points[0, 0], y - points[0, 1]], axis=-1)
    weights = delta @ np.linalg.inv(edges).T
    barycentric = np.dstack([1 - weights.sum(axis=-1), weights])
    inside = (barycentric.min(axis=-1) >= -1e-9) & (barycentric.max(axis=-1) <= 1 + 1e-9)
    section = np.s_[top:bottom, left:right]
    materials = output["material"][section]
    # On a shared face use the gas side, then a deterministic first element.
    chosen = inside & ((materials < 0) | (material < materials))
    materials[chosen] = material
    output["potential_v"][section][chosen] = (barycentric @ phi)[chosen]
    output["electric_field_v_m"][section][chosen] = electric


def section_arrays(
    nodes_nm,
    tets,
    material,
    potential,
    electric,
    faces,
    sigma,
    *,
    plane,
    position_nm,
    angle_deg=0,
    resolution=241,
):
    if plane not in {*PLANES, "vertical"}:
        raise ValueError("断面の向きが不正です。")
    if not math.isfinite(position_nm) or not math.isfinite(angle_deg) or not 65 <= resolution <= 401:
        raise ValueError("断面位置・角度・解像度が不正です。")
    if plane != "vertical":
        angle_deg = 0
    u, v, normal, u_axis, v_axis, normal_axis = plane_frame(plane, angle_deg)
    coordinates = nodes_nm @ np.array([u, v, normal]).T
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    tolerance = max(1e-7, np.ptp(coordinates[:, 2]) * 1e-10)
    if position_nm < lower[2] - tolerance or position_nm > upper[2] + tolerance:
        raise ValueError(f"断面位置は{lower[2]:g}～{upper[2]:g} nmの範囲で指定してください。")
    widths = upper[:2] - lower[:2]
    counts = np.maximum(2, np.ceil((resolution - 1) * widths / widths.max()).astype(int) + 1)
    grid_u, grid_v = [np.linspace(lower[i], upper[i], counts[i]) for i in range(2)]
    shape = (len(grid_v), len(grid_u))
    output = {
        "material": np.full(shape, -1, dtype=np.int8),
        "potential_v": np.full(shape, np.nan),
        "electric_field_v_m": np.full((*shape, 3), np.nan),
    }
    distances = coordinates[:, 2] - position_nm
    cell_distances = distances[tets]
    candidates = np.flatnonzero(
        (cell_distances.min(axis=1) <= tolerance) & (cell_distances.max(axis=1) >= -tolerance)
    )
    edge_map = {}
    polygons = set()
    intersected = 0
    for cell in candidates[np.argsort(material[candidates], kind="stable")]:
        vertices = tets[cell]
        points, values = intersect(
            nodes_nm[vertices], potential[vertices], distances[vertices], u, v, tolerance
        )
        if len(points) < 3:
            continue
        area = (
            abs(
                np.dot(points[:, 0], np.roll(points[:, 1], 1))
                - np.dot(points[:, 1], np.roll(points[:, 0], 1))
            )
            / 2
        )
        if area < 1e-12:
            continue
        key = tuple(sorted(tuple(point) for point in np.round(points, 7)))
        if key in polygons:
            continue
        polygons.add(key)
        intersected += 1
        for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
            key = tuple(sorted((tuple(np.round(start, 7)), tuple(np.round(end, 7)))))
            if key not in edge_map:
                edge_map[key] = [start.tolist(), end.tolist(), []]
            edge_map[key][2].append(int(material[cell]))
        for index in range(1, len(points) - 1):
            chosen = [0, index, index + 1]
            raster_triangle(
                points[chosen], values[chosen], electric[cell], material[cell], grid_u, grid_v, output
            )
    boundary = [edge[:2] for edge in edge_map.values() if len(edge[2]) == 1 or len(set(edge[2])) > 1]
    surface_segments, surface_polygons = [], []
    face_distances = distances[faces]
    face_candidates = np.flatnonzero(
        (face_distances.min(axis=1) <= tolerance) & (face_distances.max(axis=1) >= -tolerance)
    )
    for patch in face_candidates:
        vertices = faces[patch]
        points, _ = intersect(nodes_nm[vertices], np.zeros(3), distances[vertices], u, v, tolerance)
        item = {"points_nm": points.tolist(), "sigma_c_m2": float(sigma[patch]), "patch": int(patch)}
        if len(points) == 2:
            surface_segments.append(item)
        elif len(points) == 3:
            surface_polygons.append(item)
    valid = output["material"] >= 0

    def finite_list(values):
        return np.where(valid, values, None).ravel().tolist()

    return {
        "plane": plane,
        "position_nm": position_nm,
        "angle_deg": angle_deg,
        "axes": {
            "horizontal": u_axis,
            "vertical": v_axis,
            "normal": normal_axis,
            "vertical_increases_down": v_axis == "z",
        },
        "basis": {"u": u.tolist(), "v": v.tolist(), "normal": normal.tolist()},
        "bounds_nm": {
            "u": [float(lower[0]), float(upper[0])],
            "v": [float(lower[1]), float(upper[1])],
            "normal": [float(lower[2]), float(upper[2])],
        },
        "u_nm": grid_u.tolist(),
        "v_nm": grid_v.tolist(),
        "width": len(grid_u),
        "height": len(grid_v),
        "material": output["material"].ravel().tolist(),
        "values": {
            "potential_v": finite_list(output["potential_v"]),
            **{
                f"e_{axis}_v_m": finite_list(output["electric_field_v_m"][:, :, i])
                for i, axis in enumerate("xyz")
            },
            "e_norm_v_m": finite_list(np.linalg.norm(output["electric_field_v_m"], axis=-1)),
        },
        "boundary_segments": boundary,
        "surface_segments": surface_segments,
        "surface_polygons": surface_polygons,
        "intersected_elements": intersected,
        "valid_points": int(valid.sum()),
    }


@lru_cache(maxsize=3)
def load_mesh(path: str, mesh_id):
    def read():
        with h5py.File(path, "r") as h5:
            return (
                h5["mesh/nodes_m"][:] * 1e9,
                h5["mesh/tetrahedra"][:],
                h5["mesh/material"][:],
                h5["surface/faces"][:],
            )

    return retry_permission(read)


def saved_section(directory: Path, *, plane="xz", position_nm=0, angle_deg=0, resolution=241, step=None):
    path = directory / "results.h5"
    if not path.exists():
        raise FileNotFoundError("断面表示に必要なHDF5結果がまだありません。")

    def read_fields():
        with h5py.File(path, "r") as h5:
            metadata = json.loads(h5.attrs["metadata_json"])
            state = json.loads(h5.attrs["state_json"])
            selected_step, selected_time = state["step"], state["time_s"]
            group = h5["field"]
            sigma = h5["surface/sigma_c_m2"][:]
            if step is not None and step != selected_step:
                key = f"time_series/{step:05}"
                snapshot = directory / "snapshots" / f"{step:05}.h5"
                if key in h5:
                    group = h5[key]
                    selected_step, selected_time = step, float(group.attrs["time_s"])
                    sigma = group["sigma_c_m2"][:]
                elif snapshot.exists():
                    with h5py.File(snapshot, "r") as saved:
                        return (
                            metadata,
                            step,
                            float(saved.attrs["time_s"]),
                            saved["potential_v"][:],
                            saved["electric_field_v_m"][:],
                            saved["sigma_c_m2"][:],
                        )
                else:
                    raise FileNotFoundError("指定した帯電更新のHDF5結果が保存されていません。")
            return (
                metadata,
                selected_step,
                selected_time,
                group["potential_v"][:],
                group["electric_field_v_m"][:],
                sigma,
            )

    metadata, selected_step, selected_time, potential, electric, sigma = retry_permission(read_fields)
    mesh_id = metadata.get("mesh_id") or path.stat().st_mtime_ns
    nodes, tets, material, faces = load_mesh(str(path), mesh_id)
    result = section_arrays(
        nodes,
        tets,
        material,
        potential,
        electric,
        faces,
        sigma,
        plane=plane,
        position_nm=position_nm,
        angle_deg=angle_deg,
        resolution=resolution,
    )
    result.update(step=selected_step, time_s=selected_time, field_phase_deg=0, source="full_fem")
    return result
