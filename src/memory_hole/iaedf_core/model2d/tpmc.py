"""2D3V TPMC v2（非周期・鏡像反射壁、LXCat衝突 + 補正電場 + 密度堆積）。"""

from __future__ import annotations

import numpy as np

from ..constants import KB, MTORR_TO_PA, QE
from ..mc_utils import boris_push, isotropic_unit_vectors, periodic_table_at_time
from ..plasma import PlasmaDerived
from ..schemas import Field2DConfig, GasConfig, GeometryConfig, MagneticFieldConfig, Tpmc2DConfig
from .field import MAT_WAFER, material_of, segment_index_of, surface_height


def bilinear(field, x, y, dx, dy, nx, ny):
    gx = np.clip(x / dx, 0.0, nx - 1.001)
    ix0 = np.floor(gx).astype(np.int64)
    ix1 = ix0 + 1
    fx = gx - ix0
    gy = np.clip(y / dy, 0.0, ny - 1.001)
    iy0 = np.floor(gy).astype(np.int64)
    iy1 = iy0 + 1
    fy = gy - iy0
    return (
        (1 - fx) * (1 - fy) * field[iy0, ix0]
        + fx * (1 - fy) * field[iy0, ix1]
        + (1 - fx) * fy * field[iy1, ix0]
        + fx * fy * field[iy1, ix1]
    )


def _reflect_walls(xt, vx, length):
    """鏡像対称壁での鏡面反射（面内をはみ出した分を折り返す）。"""
    left = xt < 0.0
    if np.any(left):
        xt[left] = -xt[left]
        vx[left] = -vx[left]
    right = xt > length
    if np.any(right):
        xt[right] = 2.0 * length - xt[right]
        vx[right] = -vx[right]
    np.clip(xt, 0.0, length, out=xt)
    return xt, vx


def run_tpmc_2d(
    pressure_mTorr,
    *,
    n_particles,
    seed,
    derived: PlasmaDerived,
    geo: GeometryConfig,
    f2d: Field2DConfig,
    gas: GasConfig,
    tpmc: Tpmc2DConfig,
    sigma_cx,
    sigma_el,
    basis,
    circuit_solution,
    domain_top,
    correction=None,
    collect_density=False,
    magnetic: MagneticFieldConfig | None = None,
    progress_cb=None,
):
    rng = np.random.default_rng(seed)
    field2d = basis["plasma"]
    nx, ny = int(f2d.nx), int(f2d.ny)
    dx, dy = field2d["dx"], field2d["dy"]
    length = geo.domain_length_m
    rf_period = derived.rf_period
    dt = rf_period / tpmc.steps_per_rf_period
    max_steps = int(tpmc.max_rf_periods * tpmc.steps_per_rf_period)
    gas_density = pressure_mTorr * MTORR_TO_PA / (KB * gas.gas_temperature_K)
    neutral_sigma = np.sqrt(KB * gas.gas_temperature_K / derived.ion_mass)
    ion_sigma = np.sqrt(QE * tpmc.ion_temperature_eV / derived.ion_mass)
    h_slope = min(dx * 0.2, geo.smoothing_m * 0.1 if geo.smoothing_m > 0 else dx * 0.2)
    bohm_speed = derived.bohm_speed
    # 静磁場（イオンのみ）。全成分0なら従来と完全同一の経路を通す
    b_on = magnetic is not None and magnetic.enabled
    qm_dt = QE / derived.ion_mass * dt

    gid = np.arange(n_particles)
    x = rng.uniform(0.0, length, n_particles)
    y = np.full(n_particles, domain_top - 0.2 * dy)
    vx = ion_sigma * rng.standard_normal(n_particles)
    vy = -np.maximum(bohm_speed + ion_sigma * rng.standard_normal(n_particles), 0.05 * bohm_speed)
    vz = ion_sigma * rng.standard_normal(n_particles)
    phase0 = rng.uniform(0.0, rf_period, n_particles)
    elapsed = np.zeros(n_particles)

    energy = np.full(n_particles, np.nan)
    angle = np.full(n_particles, np.nan)
    impact_x = np.full(n_particles, np.nan)
    impact_phase = np.full(n_particles, np.nan)
    # Local tangent / out-of-plane / inward-normal frame used by SCA.
    impact_velocity = np.full((n_particles, 3), np.nan)
    impact_segment = np.full(n_particles, -1, dtype=np.int32)
    impact_material = np.full(n_particles, -1, dtype=np.int8)
    deposit = np.zeros((ny, nx)) if collect_density else None
    max_p_collision = 0.0

    for step in range(max_steps):
        if gid.size == 0:
            break
        t_abs = phase0[gid] + elapsed
        vp = periodic_table_at_time(circuit_solution["V_p"], t_abs, rf_period)
        vw = periodic_table_at_time(circuit_solution["V_w"], t_abs, rf_period)
        vr = periodic_table_at_time(circuit_solution["V_r"], t_abs, rf_period)
        ex = (
            vp * bilinear(basis["plasma"]["Ex"], x, y, dx, dy, nx, ny)
            + vw * bilinear(basis["wafer"]["Ex"], x, y, dx, dy, nx, ny)
            + vr * bilinear(basis["ring"]["Ex"], x, y, dx, dy, nx, ny)
        )
        ey = (
            vp * bilinear(basis["plasma"]["Ey"], x, y, dx, dy, nx, ny)
            + vw * bilinear(basis["wafer"]["Ey"], x, y, dx, dy, nx, ny)
            + vr * bilinear(basis["ring"]["Ey"], x, y, dx, dy, nx, ny)
        )
        if correction is not None:
            ex = ex + bilinear(correction["Ex"], x, y, dx, dy, nx, ny)
            ey = ey + bilinear(correction["Ey"], x, y, dx, dy, nx, ny)
        if collect_density:
            ix = np.clip(np.floor(x / dx).astype(np.int64), 0, nx - 1)
            iy = np.clip(np.floor(y / dy).astype(np.int64), 0, ny - 1)
            np.add.at(deposit, (iy, ix), dt)
        if b_on:
            vx, vy, vz = boris_push(
                vx, vy, vz, ex, ey, 0.0, magnetic.bx_T, magnetic.by_T, magnetic.bz_T, qm_dt
            )
        else:
            vx += QE * ex / derived.ion_mass * dt
            vy += QE * ey / derived.ion_mass * dt

        if gas_density > 0.0:
            speed = np.sqrt(vx * vx + vy * vy + vz * vz)
            E_cm = 0.25 * derived.ion_mass * speed**2 / QE
            s_cx = sigma_cx(E_cm)
            s_el = sigma_el(E_cm)
            nu = gas_density * (s_cx + s_el) * speed
            p_coll = -np.expm1(-nu * dt)
            if p_coll.size:
                max_p_collision = max(max_p_collision, float(np.max(p_coll)))
            coll = np.flatnonzero(rng.random(gid.size) < p_coll)
            if coll.size:
                neutral_v = neutral_sigma * rng.standard_normal((coll.size, 3))
                p_cx = s_cx[coll] / np.maximum(s_cx[coll] + s_el[coll], 1e-300)
                is_cx_flag = rng.random(coll.size) < p_cx
                if np.any(is_cx_flag):
                    j = coll[is_cx_flag]
                    vx[j], vy[j], vz[j] = neutral_v[is_cx_flag].T
                if np.any(~is_cx_flag):
                    j = coll[~is_cx_flag]
                    ion_v = np.column_stack((vx[j], vy[j], vz[j]))
                    nv = neutral_v[~is_cx_flag]
                    v_cm = 0.5 * (ion_v + nv)
                    rel = np.linalg.norm(ion_v - nv, axis=1)
                    vx[j], vy[j], vz[j] = (v_cm + 0.5 * rel[:, None] * isotropic_unit_vectors(rng, j.size)).T

        xt = x + vx * dt
        yt = y + vy * dt
        xt, vx = _reflect_walls(xt, vx, length)
        d0 = y - surface_height(x, geo)
        d1 = yt - surface_height(xt, geo)
        hit = d1 <= 0.0
        back = yt >= domain_top
        if np.any(hit):
            gi = gid[hit]
            frac = np.clip(d0[hit] / np.maximum(d0[hit] - d1[hit], 1e-30), 0.0, 1.0)
            xh = np.clip(x[hit] + frac * (xt[hit] - x[hit]), 0.0, length)
            slope = (surface_height(xh + h_slope, geo) - surface_height(xh - h_slope, geo)) / (2 * h_slope)
            norm = np.sqrt(1.0 + slope * slope)
            vt = (vx[hit] + vy[hit] * slope) / norm
            vn = -(-vx[hit] * slope + vy[hit]) / norm
            energy[gi] = 0.5 * derived.ion_mass * (vx[hit] ** 2 + vy[hit] ** 2 + vz[hit] ** 2) / QE
            angle[gi] = np.degrees(np.arctan2(vt, np.abs(vn)))
            impact_x[gi] = xh
            impact_velocity[gi] = np.column_stack((vt, vz[hit], vn))
            hit_time = phase0[gi] + elapsed[hit] + frac * dt
            impact_phase[gi] = np.mod(hit_time / rf_period, 1.0) * 360.0
            impact_segment[gi] = segment_index_of(xh, geo)
            impact_material[gi] = material_of(xh, geo)
        keep = ~(hit | back)
        gid = gid[keep]
        x = xt[keep]
        y = yt[keep]
        vx, vy, vz = vx[keep], vy[keep], vz[keep]
        elapsed = elapsed[keep] + dt

        if progress_cb is not None and (step + 1) % tpmc.steps_per_rf_period == 0:
            progress_cb(1.0 - gid.size / n_particles)

    ok = np.isfinite(energy)
    out = {
        "energy_eV": energy[ok],
        "angle_deg": angle[ok],
        "vx_m_s": impact_velocity[ok, 0],
        "vy_m_s": impact_velocity[ok, 1],
        "vz_m_s": impact_velocity[ok, 2],
        "impact_phase_deg": impact_phase[ok],
        "impact_x_m": impact_x[ok],
        "impact_segment": impact_segment[ok],
        "impact_material": impact_material[ok],
        "on_wafer": impact_material[ok] == MAT_WAFER,
        "n_lost": int(np.sum(~ok)),
        "max_p_collision": max_p_collision,
    }
    if collect_density:
        weight = derived.n_s * bohm_speed * length / n_particles
        out["ion_density_m3"] = deposit * weight / (dx * dy)
    if progress_cb is not None:
        progress_cb(1.0)
    return out
