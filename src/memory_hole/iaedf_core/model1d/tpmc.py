"""1D3V TPMC（電子フロント運動シース + LXCat衝突、ノートブックと同一アルゴリズム）。"""

from __future__ import annotations

import numpy as np

from ..constants import KB, MTORR_TO_PA, QE
from ..mc_utils import boris_push, isotropic_unit_vectors, periodic_table_at_time
from ..plasma import PlasmaDerived
from ..schemas import GasConfig, MagneticFieldConfig, SheathConfig, TpmcConfig


def run_tpmc(
    pressure_mTorr,
    *,
    derived: PlasmaDerived,
    sheath: SheathConfig,
    gas: GasConfig,
    tpmc: TpmcConfig,
    sigma_cx,
    sigma_el,
    vsp_table,
    s_max,
    seed=None,
    n_particles=None,
    magnetic: MagneticFieldConfig | None = None,
    progress_cb=None,
):
    n_particles = int(n_particles or tpmc.n_particles)
    rng = np.random.default_rng(tpmc.seed if seed is None else seed)
    vsp_table = np.asarray(vsp_table)
    vsp_max_local = float(np.max(vsp_table))
    rf_period = derived.rf_period
    dt = rf_period / tpmc.steps_per_rf_period
    max_steps = int(np.ceil(tpmc.max_rf_periods * tpmc.steps_per_rf_period))
    gas_density = pressure_mTorr * MTORR_TO_PA / (KB * gas.gas_temperature_K)
    neutral_sigma = np.sqrt(KB * gas.gas_temperature_K / derived.ion_mass)
    ion_sigma = np.sqrt(QE * tpmc.ion_temperature_eV / derived.ion_mass)
    alpha = sheath.potential_exponent
    bohm_speed = derived.bohm_speed
    # 静磁場（イオンのみ）。全成分0なら従来と完全同一の経路を通す
    b_on = magnetic is not None and magnetic.enabled
    qm_dt = QE / derived.ion_mass * dt

    phase_time = rng.uniform(0.0, rf_period, n_particles)
    gid = np.arange(n_particles)
    x = np.zeros(n_particles)
    vx = np.maximum(bohm_speed + ion_sigma * rng.standard_normal(n_particles), 0.05 * bohm_speed)
    vy = ion_sigma * rng.standard_normal(n_particles)
    vz = ion_sigma * rng.standard_normal(n_particles)
    elapsed = np.zeros(n_particles)
    n_cx = np.zeros(n_particles, dtype=np.int32)
    n_el = np.zeros(n_particles, dtype=np.int32)

    energy = np.full(n_particles, np.nan)
    signed_angle = np.full(n_particles, np.nan)
    transit = np.full(n_particles, np.nan)
    impact_phase = np.full(n_particles, np.nan)
    # SCA frame: x/y are surface tangents; z points into the wafer.
    impact_velocity = np.full((n_particles, 3), np.nan)
    cx_out = np.zeros(n_particles, dtype=np.int32)
    el_out = np.zeros(n_particles, dtype=np.int32)
    escaped = np.zeros(n_particles, dtype=bool)
    max_p_collision = 0.0

    for step in range(max_steps):
        if gid.size == 0:
            break
        absolute_time = phase_time[gid] + elapsed
        vsp = periodic_table_at_time(vsp_table, absolute_time, rf_period)
        if sheath.model == "moving_front":
            s_e = s_max * (np.maximum(vsp, 1e-9) / vsp_max_local) ** sheath.front_width_exponent
            x_front = s_max - s_e
            xi = np.clip((x - x_front) / s_e, 0.0, 1.0)
            e_field = alpha * vsp / s_e * xi ** (alpha - 1.0)
        elif sheath.model == "static_width":
            xi = np.clip(x / s_max, 0.0, 1.0)
            e_field = alpha * vsp / s_max * xi ** (alpha - 1.0)
        else:
            raise ValueError("SHEATH['model']はmoving_front/static_width")
        if b_on:
            vx, vy, vz = boris_push(
                vx, vy, vz, e_field, 0.0, 0.0, magnetic.bx_T, magnetic.by_T, magnetic.bz_T, qm_dt
            )
        else:
            vx += QE * e_field / derived.ion_mass * dt

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
                is_cx = rng.random(coll.size) < p_cx
                if np.any(is_cx):
                    j = coll[is_cx]
                    vx[j], vy[j], vz[j] = neutral_v[is_cx].T
                    n_cx[j] += 1
                if np.any(~is_cx):
                    j = coll[~is_cx]
                    ion_v = np.column_stack((vx[j], vy[j], vz[j]))
                    nv = neutral_v[~is_cx]
                    v_cm = 0.5 * (ion_v + nv)
                    rel = np.linalg.norm(ion_v - nv, axis=1)
                    vx[j], vy[j], vz[j] = (v_cm + 0.5 * rel[:, None] * isotropic_unit_vectors(rng, j.size)).T
                    n_el[j] += 1

        x_new = x + vx * dt
        hit = x_new >= s_max
        back = x_new < 0.0
        if np.any(hit):
            j = gid[hit]
            fraction = np.clip((s_max - x[hit]) / np.maximum(x_new[hit] - x[hit], 1e-30), 0.0, 1.0)
            hit_time = elapsed[hit] + fraction * dt
            energy[j] = 0.5 * derived.ion_mass * (vx[hit] ** 2 + vy[hit] ** 2 + vz[hit] ** 2) / QE
            signed_angle[j] = np.degrees(np.arctan2(vy[hit], np.abs(vx[hit])))
            transit[j] = hit_time
            impact_velocity[j] = np.column_stack((vy[hit], vz[hit], vx[hit]))
            impact_phase[j] = np.mod((phase_time[j] + hit_time) / rf_period, 1.0) * 360.0
            cx_out[j] = n_cx[hit]
            el_out[j] = n_el[hit]
        if np.any(back):
            j = gid[back]
            escaped[j] = True
            cx_out[j] = n_cx[back]
            el_out[j] = n_el[back]
        keep = ~(hit | back)
        gid = gid[keep]
        x = x_new[keep]
        vx, vy, vz = vx[keep], vy[keep], vz[keep]
        elapsed = elapsed[keep] + dt
        n_cx, n_el = n_cx[keep], n_el[keep]

        if progress_cb is not None and (step + 1) % tpmc.steps_per_rf_period == 0:
            progress_cb(1.0 - gid.size / n_particles)

    cutoff = np.zeros(n_particles, dtype=bool)
    cutoff[gid] = True
    if progress_cb is not None:
        progress_cb(1.0)
    return {
        "pressure_mTorr": float(pressure_mTorr),
        "energy_eV": energy,
        "signed_angle_deg": signed_angle,
        "transit_s": transit,
        "impact_phase_deg": impact_phase,
        "vx_m_s": impact_velocity[:, 0],
        "vy_m_s": impact_velocity[:, 1],
        "vz_m_s": impact_velocity[:, 2],
        "n_cx": cx_out,
        "n_elastic": el_out,
        "reached": np.isfinite(energy),
        "escaped": escaped,
        "cutoff": cutoff,
        "max_p_collision": max_p_collision,
    }
