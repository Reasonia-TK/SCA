"""2D結果の要約解析v2（材質ベース: 端傾き・影響領域幅・材質別平均エネルギー）。

ウェハ端 = wafer材質範囲の境界のうち、領域壁(x=0, L)に接していないもの。
外向き = ウェハ内部から端へ向かう方向を正とする。
"""

from __future__ import annotations

import numpy as np

from ..schemas import AnalysisConfig, GeometryConfig
from .field import MAT_INSULATOR, MAT_RING, MAT_WAFER, wafer_ranges


def _real_edges(geo: GeometryConfig):
    """(位置, 向き) のリスト。向き+1=ウェハが左側にある端(外向き=+x)。"""
    length = geo.domain_length_m
    edges = []
    for x0, x1 in wafer_ranges(geo):
        if x0 > 1e-9:
            edges.append((x0, -1.0))  # ウェハは右側 -> 外向きは-x
        if x1 < length - 1e-9:
            edges.append((x1, +1.0))  # ウェハは左側 -> 外向きは+x
    return edges


def edge_summary(run, geo: GeometryConfig, analysis: AnalysisConfig):
    excl = analysis.edge_exclusion_m
    bin_w = analysis.bin_width_m
    edges_d = np.arange(excl, analysis.max_distance_m + bin_w, bin_w)
    centers_d = 0.5 * (edges_d[:-1] + edges_d[1:])

    material = run["impact_material"]
    energy = run["energy_eV"]

    def mean_energy(code):
        mask = material == code
        return float(np.mean(energy[mask])) if np.any(mask) else None

    summary = {
        "wafer_mean_energy_eV": mean_energy(MAT_WAFER),
        "ring_mean_energy_eV": mean_energy(MAT_RING),
        "insulator_mean_energy_eV": mean_energy(MAT_INSULATOR),
        "edge_outward_tilt_deg": None,
        "affected_width_m": 0.0,
        "tilt_profile_distance_m": centers_d.tolist(),
        "tilt_profile_deg": [None] * centers_d.size,
    }

    edges = _real_edges(geo)
    onw = material == MAT_WAFER
    if not edges or not np.any(onw):
        return summary
    xw = run["impact_x_m"][onw]
    aw = run["angle_deg"][onw]
    positions = np.array([e[0] for e in edges])
    signs = np.array([e[1] for e in edges])
    dist_all = np.abs(xw[:, None] - positions[None, :])
    nearest = np.argmin(dist_all, axis=1)
    dist = dist_all[np.arange(xw.size), nearest]
    # 外向き成分: 端の向き(+1=+xが外向き)に角度符号を合わせる
    outward = aw * signs[nearest]
    keep = dist >= excl
    dist, outward = dist[keep], outward[keep]

    prof = np.full(centers_d.size, np.nan)
    for i in range(centers_d.size):
        m = (dist >= edges_d[i]) & (dist < edges_d[i + 1])
        if np.sum(m) >= 30:
            prof[i] = float(np.mean(outward[m]))
    edge_mask = dist <= excl + analysis.edge_band_m
    if np.any(edge_mask):
        summary["edge_outward_tilt_deg"] = float(np.mean(outward[edge_mask]))
    above = np.isfinite(prof) & (np.abs(prof) > analysis.affected_threshold_deg)
    if np.any(above):
        summary["affected_width_m"] = float(centers_d[np.flatnonzero(above)[-1]])
    summary["tilt_profile_deg"] = [None if not np.isfinite(v) else float(v) for v in prof]
    return summary
