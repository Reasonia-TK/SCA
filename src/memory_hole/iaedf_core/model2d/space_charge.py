"""周期平均空間電荷Poisson補正（ノートブックと同一の緩和混合反復）。"""

from __future__ import annotations

import numpy as np

from ..constants import EPS0, QE
from ..plasma import PlasmaDerived
from ..schemas import Field2DConfig, GasConfig, GeometryConfig, SpaceChargeConfig, Tpmc2DConfig
from .field import _mirror_index
from .tpmc import run_tpmc_2d


def gaussian_smooth_2d(field, sigma):
    if sigma <= 0:
        return field
    half = int(np.ceil(4.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    sm = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, field)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, sm)


def poisson_sor(source, active, field2d, f2d: Field2DConfig, sc: SpaceChargeConfig):
    dx, dy = field2d["dx"], field2d["dy"]
    idx2, idy2 = 1.0 / dx**2, 1.0 / dy**2
    den = 2.0 * (idx2 + idy2)
    omega_sor = f2d.sor_omega
    phi = np.zeros_like(source)
    jj, ii = np.indices(phi.shape)
    red = active & (((ii + jj) & 1) == 0)
    black = active & ~red
    west_i, east_i = _mirror_index(phi.shape[1])
    south_j, north_j = _mirror_index(phi.shape[0])
    iteration = 0
    update = 0.0
    for iteration in range(1, int(sc.poisson_max_iterations) + 1):  # noqa: B007 - returned below
        update = 0.0
        for mask in (red, black):
            east = phi[:, east_i]
            west = phi[:, west_i]
            north = phi[north_j, :]
            south = phi[south_j, :]
            gs = ((east + west) * idx2 + (north + south) * idy2 + source) / den
            new = (1.0 - omega_sor) * phi[mask] + omega_sor * gs[mask]
            update = max(update, float(np.max(np.abs(new - phi[mask]))))
            phi[mask] = new
        if update < sc.poisson_tolerance_V:
            break
    return phi, iteration, update


def electron_density_avg(
    phi_sc, plasma_region, basis, circuit_solution, sc: SpaceChargeConfig, derived: PlasmaDerived
):
    idx = np.linspace(
        0, circuit_solution["phase"].size, int(sc.electron_phase_samples), endpoint=False
    ).astype(int)
    dens = np.zeros_like(phi_sc)
    for i in idx:
        vp = circuit_solution["V_p"][i]
        potential = (
            vp * basis["plasma"]["psi"]
            + circuit_solution["V_w"][i] * basis["wafer"]["psi"]
            + circuit_solution["V_r"][i] * basis["ring"]["psi"]
            + phi_sc
        )
        dens += np.exp(np.clip((potential - vp) / derived.te, -80.0, 0.0))
    dens *= derived.n_s / idx.size
    dens[~plasma_region] = 0.0
    return dens


def build_space_charge(
    pressure_mTorr,
    seed,
    *,
    derived: PlasmaDerived,
    geo: GeometryConfig,
    f2d: Field2DConfig,
    gas: GasConfig,
    tpmc: Tpmc2DConfig,
    sc: SpaceChargeConfig,
    sigma_cx,
    sigma_el,
    basis,
    circuit_solution,
    domain_top,
    magnetic=None,
    log_cb=None,
    progress_cb=None,
):
    field2d = basis["plasma"]
    conductor = field2d["conductor"]
    solid = field2d["solid"]
    # 電位を解く領域: 導体と上端以外（絶縁体の固体内部も含む）
    active = ~conductor
    active[-1] = False
    # 電荷が存在するプラズマ領域: 固体の外のみ
    plasma_region = ~solid
    plasma_region[-1] = False
    phi_sc = np.zeros_like(field2d["psi"])
    ex_sc = np.zeros_like(phi_sc)
    ey_sc = np.zeros_like(phi_sc)
    history = []
    ni = np.zeros_like(phi_sc)
    ne = np.zeros_like(phi_sc)
    n_outer = int(sc.outer_iterations)
    for outer in range(1, n_outer + 1):
        if progress_cb is not None:
            progress_cb((outer - 1) / n_outer)
        dep = run_tpmc_2d(
            pressure_mTorr,
            n_particles=sc.deposition_particles,
            seed=seed + outer,
            derived=derived,
            geo=geo,
            f2d=f2d,
            gas=gas,
            tpmc=tpmc,
            sigma_cx=sigma_cx,
            sigma_el=sigma_el,
            basis=basis,
            circuit_solution=circuit_solution,
            domain_top=domain_top,
            correction={"Ex": ex_sc, "Ey": ey_sc},
            collect_density=True,
            magnetic=magnetic,
        )
        ni = gaussian_smooth_2d(dep["ion_density_m3"], sc.density_smoothing_sigma_cells)
        ni = np.clip(ni, 0.0, sc.ion_density_clip_factor * derived.n_s)
        ni[~plasma_region] = 0.0
        ne = electron_density_avg(phi_sc, plasma_region, basis, circuit_solution, sc, derived)
        rho = QE * (ni - ne)
        rho[~plasma_region] = 0.0
        target, iters, update = poisson_sor(rho / EPS0, active, field2d, f2d, sc)
        peak = float(np.max(np.abs(target)))
        if peak > sc.max_abs_correction_V:
            target *= sc.max_abs_correction_V / peak
        relax = sc.under_relaxation
        mixed = (1.0 - relax) * phi_sc + relax * target
        outer_error = float(np.max(np.abs(mixed - phi_sc)))
        phi_sc = mixed
        ex_sc = -np.gradient(phi_sc, field2d["dx"], axis=1, edge_order=2)
        ey_sc = -np.gradient(phi_sc, field2d["dy"], axis=0, edge_order=2)
        ex_sc[conductor] = 0.0
        ey_sc[conductor] = 0.0
        history.append(outer_error)
        if log_cb is not None:
            log_cb(
                f"outer {outer}: Poisson {iters}回, "
                f"max|phi_sc|={float(np.max(np.abs(phi_sc))):.2f} V, "
                f"外部反復変化 {outer_error:.2f} V"
            )
    return {
        "phi_sc": phi_sc,
        "Ex": ex_sc,
        "Ey": ey_sc,
        "ion_density_m3": ni,
        "electron_density_m3": ne,
        "history": history,
    }
