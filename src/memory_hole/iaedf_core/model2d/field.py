"""2D形状v2（非周期・鏡像対称壁）とLaplace 3基底（赤黒SOR）。

表面は制御点折れ線 s(x)。各セグメントの材質:
  wafer/ring = 導体（Dirichlet）、insulator = 非帯電誘電体（電位固定なし、
  固体内部もLaplace解に含める）。
左右壁と絶縁体下端はNeumann（鏡像）境界。
"""

from __future__ import annotations

import numpy as np

from ..schemas import Field2DConfig, GeometryConfig

MAT_WAFER, MAT_RING, MAT_INSULATOR = 0, 1, 2
_MAT_CODE = {"wafer": MAT_WAFER, "ring": MAT_RING, "insulator": MAT_INSULATOR}

_geo_cache: dict = {}


def _geo_key(geo: GeometryConfig):
    return (
        tuple(tuple(p) for p in geo.points_m),
        tuple(geo.segment_materials),
        geo.smoothing_m,
        geo.domain_length_m,
    )


def _nodes(geo: GeometryConfig):
    key = ("nodes",) + _geo_key(geo)
    cached = _geo_cache.get(key)
    if cached is None:
        points = np.asarray(geo.points_m, dtype=float)
        cached = (points[:, 0], points[:, 1])
        if len(_geo_cache) > 32:
            _geo_cache.clear()
        _geo_cache[key] = cached
    return cached


_SMOOTH_TABLE_N = 4096


def _smoothed_table(geo: GeometryConfig):
    key = ("table",) + _geo_key(geo)
    cached = _geo_cache.get(key)
    if cached is not None:
        return cached
    px, py = _nodes(geo)
    length = geo.domain_length_m
    grid = np.linspace(0.0, length, _SMOOTH_TABLE_N)
    table = np.interp(grid, px, py)
    sigma_cells = geo.smoothing_m / (length / (_SMOOTH_TABLE_N - 1))
    half = int(np.ceil(4.0 * sigma_cells))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_cells) ** 2)
    kernel /= kernel.sum()
    # 端は鏡像パディング（対称境界と整合）
    padded = np.r_[table[half:0:-1], table, table[-2 : -half - 2 : -1]]
    table = np.convolve(padded, kernel, mode="same")[half : half + _SMOOTH_TABLE_N]
    _geo_cache[key] = table
    return table


def surface_height(x, geo: GeometryConfig):
    q = np.clip(np.asarray(x, dtype=float), 0.0, geo.domain_length_m)
    if geo.smoothing_m > 0.0:
        table = _smoothed_table(geo)
        grid = np.linspace(0.0, geo.domain_length_m, _SMOOTH_TABLE_N)
        return np.interp(q, grid, table)
    px, py = _nodes(geo)
    return np.interp(q, px, py)


def segment_index_of(x, geo: GeometryConfig):
    """各xが属するセグメント番号（[x_i, x_{i+1})、末尾は最終セグメント）。"""
    px, _ = _nodes(geo)
    q = np.clip(np.asarray(x, dtype=float), 0.0, geo.domain_length_m)
    return np.clip(np.searchsorted(px, q, side="right") - 1, 0, len(px) - 2)


def material_codes(geo: GeometryConfig):
    return np.array([_MAT_CODE[m] for m in geo.segment_materials], dtype=np.int8)


def material_of(x, geo: GeometryConfig):
    """各xの表面材質コード（0=wafer, 1=ring, 2=insulator）。"""
    return material_codes(geo)[segment_index_of(x, geo)]


def wafer_ranges(geo: GeometryConfig):
    """wafer材質の連続x範囲 [(x0, x1), ...] を返す。"""
    px, _ = _nodes(geo)
    codes = material_codes(geo)
    ranges = []
    start = None
    for i, code in enumerate(codes):
        if code == MAT_WAFER and start is None:
            start = px[i]
        if code != MAT_WAFER and start is not None:
            ranges.append((start, px[i]))
            start = None
    if start is not None:
        ranges.append((start, px[-1]))
    return ranges


def max_surface_height(geo: GeometryConfig) -> float:
    sample = np.linspace(0.0, geo.domain_length_m, 2048)
    return float(np.max(surface_height(sample, geo)))


def _mirror_index(n: int):
    """鏡像（Neumann）境界の隣接ノード番号。端では1つ内側を参照する。"""
    lower = np.r_[1, np.arange(n - 1)]  # 西/南
    upper = np.r_[np.arange(1, n), n - 2]  # 東/北
    return lower, upper


def solve_basis(kind: str, geo: GeometryConfig, f2d: Field2DConfig, domain_top: float):
    nx, ny = int(f2d.nx), int(f2d.ny)
    length = geo.domain_length_m
    x = np.linspace(0.0, length, nx)
    s = surface_height(x, geo)
    y = np.linspace(0.0, domain_top, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    solid = Y <= s[None, :]
    materials = material_of(x, geo)
    wafer_cols = np.broadcast_to((materials == MAT_WAFER)[None, :], X.shape)
    ring_cols = np.broadcast_to((materials == MAT_RING)[None, :], X.shape)
    conductor = solid & (wafer_cols | ring_cols)
    plasma_top = np.zeros_like(conductor)
    plasma_top[-1] = True
    # 絶縁体の固体内部もLaplace解に含める（非帯電誘電体）
    active = ~conductor & ~plasma_top

    fixed = np.zeros_like(X)
    if kind == "plasma":
        fixed[-1] = 1.0
    elif kind == "wafer":
        fixed[conductor & wafer_cols] = 1.0
    elif kind == "ring":
        fixed[conductor & ring_cols] = 1.0
    else:
        raise ValueError(kind)

    psi = np.zeros_like(X)
    psi[~active] = fixed[~active]
    jj, ii = np.indices(psi.shape)
    red = active & (((ii + jj) & 1) == 0)
    black = active & ~red
    idx2, idy2 = 1.0 / dx**2, 1.0 / dy**2
    den = 2.0 * (idx2 + idy2)
    omega_sor = f2d.sor_omega
    west_i, east_i = _mirror_index(nx)
    south_j, north_j = _mirror_index(ny)

    iteration = 0
    for iteration in range(1, int(f2d.max_iterations) + 1):  # noqa: B007 - returned below
        update = 0.0
        for mask in (red, black):
            east = psi[:, east_i]
            west = psi[:, west_i]
            north = psi[north_j, :]
            south = psi[south_j, :]
            gs = ((east + west) * idx2 + (north + south) * idy2) / den
            new = (1.0 - omega_sor) * psi[mask] + omega_sor * gs[mask]
            update = max(update, float(np.max(np.abs(new - psi[mask]))))
            psi[mask] = new
        if update < f2d.tolerance:
            break

    lap = (psi[:, east_i] - 2 * psi + psi[:, west_i]) * idx2 + (
        psi[north_j, :] - 2 * psi + psi[south_j, :]
    ) * idy2
    residual = float(np.max(np.abs(lap[active])) * domain_top**2)
    ex = -np.gradient(psi, dx, axis=1, edge_order=2)
    ey = -np.gradient(psi, dy, axis=0, edge_order=2)
    ex[conductor] = 0.0
    ey[conductor] = 0.0
    return {
        "psi": psi,
        "Ex": ex,
        "Ey": ey,
        "x": x,
        "y": y,
        "surface": s,
        "conductor": conductor,
        "solid": solid,
        "active": active,
        "materials_x": materials,
        "dx": dx,
        "dy": dy,
        "iterations": iteration,
        "residual": residual,
    }


def build_basis(geo: GeometryConfig, f2d: Field2DConfig, domain_top: float):
    basis = {name: solve_basis(name, geo, f2d, domain_top) for name in ("plasma", "wafer", "ring")}
    field = basis["plasma"]
    partition = basis["plasma"]["psi"] + basis["wafer"]["psi"] + basis["ring"]["psi"]
    partition_error = float(np.max(np.abs(partition[field["active"]] - 1.0)))
    return basis, partition_error
