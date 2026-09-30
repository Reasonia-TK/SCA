"""Boundary-conforming P1 tetrahedral FEM, including the gas/oxide interface.

Embedded metal and Carbon are excluded conductors with a common prescribed
potential. Interface charge enters the weak-form surface load exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import gmsh
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree

from .config import CaseConfig
from .geometry import NM, derive

EPS0 = 8.8541878128e-12
FACE_LOCAL = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


@dataclass
class Mesh:
    nodes: np.ndarray
    tets: np.ndarray
    material: np.ndarray  # 0 gas, 1 oxide
    gradients: np.ndarray
    volumes: np.ndarray
    inverse: np.ndarray
    neighbors: np.ndarray
    surface_faces: np.ndarray
    surface_area: np.ndarray
    surface_centers: np.ndarray
    surface_gas_tets: np.ndarray
    conductor_nodes: np.ndarray
    outer_nodes: np.ndarray
    outer_faces: np.ndarray
    outer_area: np.ndarray
    min_cell_m: float
    geometry_id: str


def tetra_data(nodes, tets):
    xyz = nodes[tets]
    matrices = np.concatenate([np.ones((*xyz.shape[:2], 1)), xyz / NM], axis=2)
    inverse = np.linalg.inv(matrices)
    gradients = inverse[:, 1:, :].transpose(0, 2, 1) / NM
    volumes = np.abs(np.linalg.det(xyz[:, 1:] - xyz[:, :1])) / 6
    return inverse, gradients, volumes


def build_mesh(config: CaseConfig) -> Mesh:
    d = derive(config)
    g = config.geometry
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", g.interface_mesh_size_nm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", g.mesh_size_nm)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.add("memory_hole")
        occ = gmsh.model.occ
        h = d["domain_half_width_nm"]
        upper = occ.addBox(-h, -h, -g.entrance_height_nm, 2 * h, 2 * h, g.entrance_height_nm)
        parts = [(3, upper)]
        profile = d["profile"]
        for a, b in pairwise(profile):
            height = b["depth_nm"] - a["depth_nm"]
            if abs(a["radius_nm"] - b["radius_nm"]) < 1e-8:
                tag = occ.addCylinder(0, 0, a["depth_nm"], 0, 0, height, a["radius_nm"])
            else:
                tag = occ.addCone(0, 0, a["depth_nm"], 0, 0, height, a["radius_nm"], b["radius_nm"])
            parts.append((3, tag))
        if len(parts) > 2:
            hole, _ = occ.fuse(parts[1:2], parts[2:])
        else:
            hole = parts[1:]
        # Preserve the z=0 collector plane; legacy inlet energies are evaluated there.
        gas = parts[:1] + hole
        ox = occ.addBox(-h, -h, g.carbon_thickness_nm, 2 * h, 2 * h, g.oxide_thickness_nm)
        oxide, _ = occ.cut([(3, ox)], gas, removeTool=False)
        metals = [
            (
                3,
                occ.addCylinder(
                    m["x_nm"], m["y_nm"], m["top_nm"], 0, 0, m["length_nm"], m["diameter_nm"] / 2
                ),
            )
            for m in d["dummies"]
        ]
        oxide, _ = occ.cut(oxide, metals)
        _, mapping = occ.fragment(gas, oxide)
        gas_tags = {tag for group in mapping[: len(gas)] for dim, tag in group if dim == 3}
        occ.synchronize()
        # A distance/threshold field refines the actual curved interfaces and narrow oxide gaps.
        surfaces = []
        max_radius = max(p["radius_nm"] for p in profile)
        for _, tag in gmsh.model.getBoundary([(3, t) for t in gas_tags], oriented=False):
            bounds = gmsh.model.getBoundingBox(2, tag)
            if max(abs(bounds[0]), abs(bounds[1]), abs(bounds[3]), abs(bounds[4])) < max_radius + 1:
                surfaces.append(tag)
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "SurfacesList", surfaces)
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 40)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", dist)
        fine = g.interface_mesh_size_nm
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", fine)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", g.mesh_size_nm)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", fine)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", fine + g.mesh_size_nm)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        lookup = np.full(int(max(tags)) + 1, -1, dtype=np.int64)
        lookup[tags] = np.arange(len(tags))
        nodes = np.asarray(coords).reshape(-1, 3) * NM
        tets, mats = [], []
        for _, volume in gmsh.model.getEntities(3):
            types, _, element_nodes = gmsh.model.mesh.getElements(3, volume)
            for typ, enodes in zip(types, element_nodes, strict=True):
                if typ != 4:
                    raise ValueError("一次四面体以外のメッシュは使用できません。")
                block = lookup[np.asarray(enodes).reshape(-1, 4)]
                tets.append(block)
                mats.append(np.full(len(block), 0 if volume in gas_tags else 1, dtype=np.int8))
        tets = np.concatenate(tets)
        material = np.concatenate(mats)
    finally:
        gmsh.finalize()
    inverse, gradients, volumes = tetra_data(nodes, tets)
    faces = tets[:, FACE_LOCAL].reshape(-1, 3)
    sorted_faces = np.sort(faces, axis=1)
    _, ids, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(ids, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    neighbors = np.full((len(tets), 4), -1, dtype=np.int64)
    surface, surface_gas, external = [], [], []
    for group, count in enumerate(counts):
        indexes = order[offsets[group] : offsets[group + 1]]
        a = int(indexes[0])
        if count == 2:
            b = int(indexes[1])
            neighbors[a // 4, a % 4] = b // 4
            neighbors[b // 4, b % 4] = a // 4
            if material[a // 4] != material[b // 4]:
                surface.append(faces[a])
                surface_gas.append(a // 4 if material[a // 4] == 0 else b // 4)
        elif count == 1:
            external.append(faces[a])
        else:
            raise ValueError("非多様体メッシュを検出しました。")
    surface = np.array(surface, dtype=np.int64).reshape(-1, 3)
    external = np.array(external, dtype=np.int64)
    center = nodes[external].mean(axis=1) / NM
    outer = (
        np.isclose(abs(center[:, 0]), h, atol=1e-5)
        | np.isclose(abs(center[:, 1]), h, atol=1e-5)
        | np.isclose(center[:, 2], -g.entrance_height_nm, atol=1e-5)
    )
    outer_faces = external[outer]
    conductor_nodes = np.unique(external[~outer])
    dirichlet_outer = (
        outer_faces
        if config.outer_boundary == "dirichlet"
        else external[np.isclose(center[:, 2], -g.entrance_height_nm, atol=1e-5)]
    )
    outer_nodes = np.setdiff1d(np.unique(dirichlet_outer), conductor_nodes)

    def areas(f):
        xyz = nodes[f]
        return np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1) / 2

    shortest = min(
        np.linalg.norm(nodes[tets[:, a]] - nodes[tets[:, b]], axis=1).min()
        for a in range(4)
        for b in range(a + 1, 4)
    )
    return Mesh(
        nodes,
        tets,
        material,
        gradients,
        volumes,
        inverse,
        neighbors,
        surface,
        areas(surface),
        nodes[surface].mean(axis=1),
        np.array(surface_gas),
        conductor_nodes,
        outer_nodes,
        outer_faces,
        areas(outer_faces),
        float(shortest),
        d["geometry_id"],
    )


class FieldSolver:
    def __init__(self, mesh: Mesh, config: CaseConfig):
        self.mesh, self.config = mesh, config
        eps_r = np.where(mesh.material == 0, 1, config.oxide_relative_permittivity)
        local = (
            np.einsum("tik,tjk->tij", mesh.gradients, mesh.gradients)
            * (eps_r * mesh.volumes / NM)[:, None, None]
        )
        rows = np.broadcast_to(mesh.tets[:, :, None], local.shape).ravel()
        cols = np.broadcast_to(mesh.tets[:, None, :], local.shape).ravel()
        self.matrix = coo_matrix(
            (local.ravel(), (rows, cols)), shape=(len(mesh.nodes), len(mesh.nodes))
        ).tocsr()
        self.fixed = np.r_[mesh.conductor_nodes, mesh.outer_nodes]
        inlet_nodes = np.empty(0, dtype=np.int64)
        if config.inlet_potential_mode != "unconstrained":
            gas_nodes = np.unique(mesh.tets[mesh.material == 0])
            p = mesh.nodes[gas_nodes] / NM
            inlet_nodes = gas_nodes[np.isclose(p[:, 2], 0, atol=1e-6)]
            inlet_nodes = np.setdiff1d(inlet_nodes, self.fixed)
        self.inlet_nodes = inlet_nodes
        self.fixed = np.r_[self.fixed, inlet_nodes]
        self.free = np.setdiff1d(np.arange(len(mesh.nodes)), self.fixed)
        self.reduced = self.matrix[self.free][:, self.free]
        self.coupling = self.matrix[self.free][:, self.fixed]
        diagonal = self.reduced.diagonal()
        self.preconditioner = LinearOperator(self.reduced.shape, matvec=lambda x: x / diagonal)
        if len(self.free) > 80000:
            import pyamg

            self.preconditioner = pyamg.smoothed_aggregation_solver(self.reduced).aspreconditioner()
        self.wafer_basis = self._solve(
            np.zeros(len(mesh.nodes)),
            np.r_[
                np.ones(len(mesh.conductor_nodes)),
                np.zeros(len(mesh.outer_nodes)),
                np.full(len(inlet_nodes), 1 if config.inlet_potential_mode == "wafer" else 0),
            ],
        )
        self.outer_basis = self._solve(
            np.zeros(len(mesh.nodes)),
            np.r_[
                np.zeros(len(mesh.conductor_nodes)),
                np.ones(len(mesh.outer_nodes)),
                np.zeros(len(inlet_nodes)),
            ],
        )
        self.inlet_basis = self._solve(
            np.zeros(len(mesh.nodes)),
            np.r_[np.zeros(len(mesh.conductor_nodes) + len(mesh.outer_nodes)), np.ones(len(inlet_nodes))],
        )
        self.gas_ids = np.flatnonzero(mesh.material == 0)
        gas_vertices = mesh.nodes[mesh.tets[self.gas_ids]]
        gas_centers = gas_vertices.mean(axis=1)
        self.gas_tree = cKDTree(gas_centers)
        self.gas_bounds_min = gas_vertices.min(axis=1)
        self.gas_bounds_max = gas_vertices.max(axis=1)
        self.gas_search_radius = float(np.linalg.norm(gas_vertices - gas_centers[:, None], axis=2).max())

    def _solve(self, rhs, prescribed):
        phi = np.zeros(len(self.mesh.nodes))
        phi[self.fixed] = prescribed
        solution, info = cg(
            self.reduced,
            rhs[self.free] - self.coupling @ prescribed,
            M=self.preconditioner,
            rtol=self.config.numerics.field_rtol,
            atol=0,
            maxiter=20000,
        )
        if info != 0:
            raise RuntimeError(f"電場反復解法が収束しませんでした（CG info={info}）。")
        phi[self.free] = solution
        return phi

    def solve(self, sigma: np.ndarray, wafer_v: float):
        rhs = np.zeros(len(self.mesh.nodes))
        load = sigma * self.mesh.surface_area / (3 * EPS0 * NM)
        np.add.at(rhs, self.mesh.surface_faces.ravel(), np.repeat(load, 3))
        if self.config.outer_boundary == "neumann":
            # outward D.n is prescribed; weak-form load is -D.n.
            external_load = (
                -self.config.outer_normal_displacement_c_m2 * self.mesh.outer_area / (3 * EPS0 * NM)
            )
            np.add.at(rhs, self.mesh.outer_faces.ravel(), np.repeat(external_load, 3))
        phi = (
            self._solve(rhs, np.zeros(len(self.fixed)))
            + wafer_v * self.wafer_basis
            + self.config.outer_potential_v * self.outer_basis
        )
        if self.config.inlet_potential_mode == "constant":
            phi += self.config.inlet_reference_potential_v * self.inlet_basis
        field = -np.einsum("ti,tik->tk", phi[self.mesh.tets], self.mesh.gradients)
        reaction = (self.matrix @ phi - rhs) * EPS0 * NM
        induced = float(reaction[self.mesh.conductor_nodes].sum())
        scale = max(np.linalg.norm(rhs[self.free]), np.linalg.norm(self.coupling @ phi[self.fixed]), 1e-20)
        residual = float(np.linalg.norm((self.matrix @ phi - rhs)[self.free]) / scale)
        return (
            phi,
            field,
            {
                "induced_conductor_charge_c": induced,
                "collector_boundary_charge_c": float(reaction[self.inlet_nodes].sum()),
                "field_residual": residual,
            },
        )

    def locate(self, positions):
        from .transport import locate_points

        _, indexes = self.gas_tree.query(positions, k=min(32, len(self.gas_ids)))
        if indexes.ndim == 1:
            indexes = indexes[:, None]
        candidates = self.gas_ids[indexes]
        found = locate_points(positions, self.mesh.inverse, candidates)
        missing = np.flatnonzero(found < 0)
        if len(missing):
            # A containing tetrahedron's centroid is within its maximum vertex
            # radius, even when it is absent from the 32 nearest centroids.
            tolerance = 1e-14
            nearby = self.gas_tree.query_ball_point(positions[missing], self.gas_search_radius + tolerance)
            for point_index, ids in zip(missing, nearby, strict=True):
                ids = np.asarray(ids, dtype=np.int64)
                point = positions[point_index]
                inside_bounds = np.all(
                    (point >= self.gas_bounds_min[ids] - tolerance)
                    & (point <= self.gas_bounds_max[ids] + tolerance),
                    axis=1,
                )
                candidates = self.gas_ids[ids[inside_bounds]]
                if len(candidates):
                    found[point_index] = locate_points(
                        positions[point_index : point_index + 1],
                        self.mesh.inverse,
                        candidates[None, :],
                    )[0]
        return found
