"""Double-precision CPU reference: exact first tetra-face intersection.

The field is constant inside a P1 tetrahedron. Each flight segment is a
parabola, intersected with every face before a material boundary is crossed.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
ELECTRON_MASS = 9.1093837139e-31


@njit(cache=True)
def locate_points(positions, inverse, candidates):
    found = np.full(len(positions), -1, np.int64)
    for j in range(len(positions)):
        for tet in candidates[j]:
            bary = inverse[tet, 0].copy()
            for k in range(3):
                bary += positions[j, k] / 1e-9 * inverse[tet, k + 1]
            if bary.min() >= -1e-8:
                found[j] = tet
                break
    return found


@njit(cache=True)
def first_face_time(b, velocity, acceleration, limit):
    """First outward crossing of b(t)=b+v*t+a*t^2; b is barycentric."""
    b = max(b, 0.0)
    tiny = 1e-22
    if abs(acceleration) * limit < 1e-12 * max(abs(velocity), 1.0):
        if velocity < 0:
            t = -b / velocity
            if t > tiny and t <= limit:
                return t
            if t <= tiny and b < 1e-8:
                return tiny
        return math.inf
    discriminant = velocity * velocity - 4 * acceleration * b
    if discriminant < 0:
        return math.inf
    root = math.sqrt(discriminant)
    q = -0.5 * (velocity + math.copysign(root, velocity))
    t1 = q / acceleration
    t2 = b / q if q != 0 else math.inf
    best = math.inf
    for t in (t1, t2):
        if t >= 0 and t <= limit and velocity + 2 * acceleration * t < 0:
            best = min(best, max(t, tiny))
    return best


@njit(cache=True, parallel=True)
def trace(
    positions,
    velocities,
    starts,
    charges,
    masses,
    field,
    inverse,
    gradients,
    neighbors,
    face_kind,
    face_patch,
    max_steps,
    max_dt,
    cell_size,
    cell_fraction,
    path_count,
):
    n = len(positions)
    end = positions.copy()
    out_v = velocities.copy()
    status = np.full(n, 3, np.int8)  # 0 escaped, 1 conductor, 2 oxide, 3 unresolved
    patches = np.full(n, -1, np.int64)
    final_tets = starts.copy()
    times = np.zeros(n)
    normals = np.zeros((n, 3))
    angles = np.zeros(n)
    paths = np.full((min(path_count, n), 128, 3), np.nan)
    path_lengths = np.zeros(min(path_count, n), np.int64)
    path_times = np.full((min(path_count, n), 128), np.nan)
    for j in prange(n):
        p = positions[j].copy()
        v = velocities[j].copy()
        tet = starts[j]
        stored = 0
        if j < len(paths):
            paths[j, 0] = p
            path_times[j, 0] = 0.0
            stored = 1
        if tet < 0:
            continue
        elapsed = 0.0
        for step in range(max_steps):
            a = field[tet] * charges[j] / masses[j]
            speed = np.linalg.norm(v)
            acceleration = np.linalg.norm(a)
            dt = min(max_dt, cell_fraction * cell_size / max(speed, 1.0))
            if acceleration > 0:
                dt = min(dt, math.sqrt(2 * cell_fraction * cell_size / acceleration))
            bary = inverse[tet, 0].copy()
            for k in range(3):
                bary += p[k] / 1e-9 * inverse[tet, k + 1]
            hit_face = -1
            for face in range(4):
                gv = np.dot(gradients[tet, face], v)
                ga = 0.5 * np.dot(gradients[tet, face], a)
                t_hit = first_face_time(bary[face], gv, ga, dt)
                if t_hit <= dt:
                    dt = t_hit
                    hit_face = face
            p += v * dt + 0.5 * a * dt * dt
            v += a * dt
            elapsed += dt
            if j < len(paths) and (step % max(1, max_steps // 120) == 0) and stored < 126:
                paths[j, stored] = p
                path_times[j, stored] = elapsed
                stored += 1
            if hit_face >= 0:
                kind = face_kind[tet, hit_face]
                if kind < 0:
                    tet = neighbors[tet, hit_face]
                    continue
                status[j] = kind
                patches[j] = face_patch[tet, hit_face]
                normal = -gradients[tet, hit_face]
                normal /= np.linalg.norm(normal)
                normals[j] = normal
                cosine = min(1.0, max(0.0, np.dot(v, normal) / max(np.linalg.norm(v), 1e-30)))
                angles[j] = math.acos(cosine) * 180 / math.pi
                break
        end[j], out_v[j], times[j], final_tets[j] = p, v, elapsed, tet
        if j < len(paths):
            paths[j, stored] = p
            path_times[j, stored] = elapsed
            path_lengths[j] = stored + 1
    return end, out_v, status, patches, final_tets, times, normals, angles, paths, path_lengths, path_times


def boundary_maps(mesh):
    kinds = np.full((len(mesh.tets), 4), -1, dtype=np.int8)
    patches = np.full((len(mesh.tets), 4), -1, dtype=np.int64)
    lookup = {tuple(sorted(face)): i for i, face in enumerate(mesh.surface_faces)}
    outer_faces = {tuple(sorted(face)) for face in mesh.outer_faces}
    face_vertices = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    for tet in np.flatnonzero(mesh.material == 0):
        for side, local in enumerate(face_vertices):
            other = mesh.neighbors[tet, side]
            face = tuple(sorted(mesh.tets[tet, list(local)]))
            if other >= 0 and mesh.material[other] == 1:
                kinds[tet, side], patches[tet, side] = 2, lookup[face]
            elif other < 0:
                kinds[tet, side] = 0 if face in outer_faces else 1
    return kinds, patches


def thermal_electrons(rng, count, temperature_ev):
    """Normal-flux Maxwellian: Rayleigh normal speed and Gaussian tangentials."""
    scale = math.sqrt(temperature_ev * E_CHARGE / ELECTRON_MASS)
    v = rng.normal(0, scale, (count, 3))
    v[:, 2] = scale * np.sqrt(-2 * np.log(np.maximum(rng.random(count), 1e-300)))
    return v


def sample_generated(rng, count, config, derived, species=None):
    r = np.sqrt(rng.random(count)) * derived["injection_radius_nm"] * 1e-9
    theta = rng.uniform(0, 2 * math.pi, count)
    p = np.column_stack(
        [r * np.cos(theta), r * np.sin(theta), np.full(count, config.geometry.injection_z_nm * 1e-9 + 1e-16)]
    )
    phases = rng.uniform(0, 360, count)
    if species is None:
        v = thermal_electrons(rng, count, config.electron.temperature_ev)
    else:
        tangent = rng.normal(0, math.tan(math.radians(species.angular_sigma_deg)), (count, 2))
        direction = np.column_stack([tangent, np.ones(count)])
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        v = direction * math.sqrt(2 * species.energy_ev * E_CHARGE / (species.mass_amu * AMU))
    return p, v, phases


def wilson_interval(success_weight, total_weight, sum_weight_squared):
    if total_weight <= 0 or sum_weight_squared <= 0:
        return {"rate": None, "low": None, "high": None, "effective_samples": 0}
    n = total_weight**2 / sum_weight_squared
    p = min(1, max(0, success_weight / total_weight))
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "rate": p,
        "low": max(0, center - half),
        "high": min(1, center + half),
        "effective_samples": n,
        "relative_half_width": half / p if p else None,
        "method": "Wilson / Kish ESS; conditional on frozen-field samples",
    }
