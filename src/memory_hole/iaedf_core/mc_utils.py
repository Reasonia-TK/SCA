"""TPMC共通ユーティリティ（1D/2Dノートブックで同一定義）。"""

from __future__ import annotations

import numpy as np


def periodic_table_at_time(table, time_s, rf_period: float):
    table = np.asarray(table)
    position = np.mod(np.asarray(time_s) / rf_period, 1.0) * table.size
    index = np.floor(position).astype(np.int64)
    fraction = position - index
    nxt = (index + 1) % table.size
    return table[index] + fraction * (table[nxt] - table[index])


def boris_push(vx, vy, vz, ex, ey, ez, bx, by, bz, qm_dt):
    """Borisプッシャーによる1ステップの速度更新（E+v×B、静磁場）。

    qm_dt = q*dt/m。E成分は配列可、B成分はスカラー[T]。
    半分のEキック -> 磁場回転（厳密回転なのでBは仕事をしない）-> 半分のEキック。
    """
    half = 0.5 * qm_dt
    vmx = vx + half * ex
    vmy = vy + half * ey
    vmz = vz + half * ez
    tx, ty, tz = half * bx, half * by, half * bz
    t2 = tx * tx + ty * ty + tz * tz
    sx, sy, sz = 2.0 * tx / (1.0 + t2), 2.0 * ty / (1.0 + t2), 2.0 * tz / (1.0 + t2)
    vpx = vmx + (vmy * tz - vmz * ty)
    vpy = vmy + (vmz * tx - vmx * tz)
    vpz = vmz + (vmx * ty - vmy * tx)
    vxn = vmx + (vpy * sz - vpz * sy) + half * ex
    vyn = vmy + (vpz * sx - vpx * sz) + half * ey
    vzn = vmz + (vpx * sy - vpy * sx) + half * ez
    return vxn, vyn, vzn


def isotropic_unit_vectors(rng, size):
    cosine = 2.0 * rng.random(size) - 1.0
    sine = np.sqrt(np.maximum(1.0 - cosine**2, 0.0))
    azimuth = 2.0 * np.pi * rng.random(size)
    return np.column_stack((cosine, sine * np.cos(azimuth), sine * np.sin(azimuth)))
