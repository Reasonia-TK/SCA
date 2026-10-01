"""Self-contained IAEDF execution and the phase/velocity/flux contract with SCA."""

from __future__ import annotations

import hashlib
import math
import time
import traceback
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import CaseConfig, PhasePotential, Species, Waveform, WaveformSample
from .iaedf_core.constants import ME, MTORR_TO_PA, QE
from .iaedf_core.model1d.runner import run_1d
from .iaedf_core.model2d.field import surface_height
from .iaedf_core.model2d.runner import run_2d
from .iaedf_core.plasma import derive_plasma
from .iaedf_core.report import build_plots_1d, build_plots_2d, save_npz_1d, save_npz_2d
from .iaedf_core.schemas import Config1D, Config2D
from .iaedf_core.waveform import parse_csv_waveform
from .inlet import waveform_id
from .storage import atomic_json, read_json, retry_permission

CORE = Path(__file__).parent / "iaedf_core"
ASSETS = CORE / "assets"
XSECS = {"xsec_ar_ion_phelps_lxcat.csv": 39.948, "xsec_he_ion_phelps_lxcat.csv": 4.002602}


class UpstreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(default="IAEDF", min_length=1, max_length=120)
    model: Literal["1d", "2d"] = "1d"
    config: dict


class ApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    config: CaseConfig
    pressure_index: int = Field(default=0, ge=0)
    collector_min_m: float | None = None
    collector_max_m: float | None = None
    azimuth_deg: float = Field(default=0, ge=-180, le=180)


def defaults():
    return {
        "1d": Config1D().model_dump(),
        "2d": Config2D().model_dump(),
        "xsec_names": list(XSECS),
        "tailored_waveform_csv": (ASSETS / "tailored_waveform_5harmonic.csv").read_text(encoding="utf-8-sig"),
    }


def validated_config(model, data):
    if model not in {"1d", "2d"}:
        raise ValueError("IAEDFモデルは1dまたは2dを指定してください。")
    geometry = data.get("geometry", {})
    points = geometry.get("points_m")
    if model == "2d" and points is not None:
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points):
            raise ValueError("2D制御点は[x, y]の組で指定してください。")
        try:
            if any(float(b[0]) <= float(a[0]) for a, b in zip(points, points[1:], strict=False)):
                raise ValueError("2D制御点のx座標は重複しない昇順にしてください。")
        except (TypeError, IndexError) as error:
            raise ValueError("2D制御点の座標が不正です。") from error
    try:
        config = (Config1D if model == "1d" else Config2D).model_validate(data)
    except ValidationError as error:
        raise ValueError(str(error)) from error
    # Keep the imported numerical algorithms unchanged; guard their public entry here.
    for name in ("frequency_Hz", "electron_temperature_eV", "sheath_edge_density_m3", "ion_mass_amu"):
        if getattr(config.plasma, name) <= 0:
            raise ValueError(f"plasma.{name}は正の値にしてください。")
    gas, tpmc, circuit = config.gas, config.tpmc, config.circuit
    if not 1 <= len(gas.pressures_mTorr) <= 20 or any(p < 0 for p in gas.pressures_mTorr):
        raise ValueError("圧力は非負の値を1～20ケース指定してください。")
    if gas.gas_temperature_K <= 0 or gas.cross_section_scale < 0 or gas.elastic_to_cx_ratio < 0:
        raise ValueError("ガス温度は正、断面積倍率・弾性/CX比は非負にしてください。")
    if gas.cross_section_source == "lxcat_phelps":
        if gas.xsec_csv_name not in XSECS:
            raise ValueError("断面積CSVは同梱のArまたはHeを選んでください。")
        if not math.isclose(config.plasma.ion_mass_amu, XSECS[gas.xsec_csv_name], rel_tol=0.01):
            raise ValueError("イオン質量とAr/He断面積データが一致しません。")
    if not 10 <= tpmc.n_particles <= 2_000_000 or not 10 <= tpmc.steps_per_rf_period <= 10000:
        raise ValueError("粒子数は10～200万、RF周期の時間分割数は10～10000にしてください。")
    if not 0 < tpmc.max_rf_periods <= 1000 or tpmc.max_rf_periods * tpmc.steps_per_rf_period > 1_000_000:
        raise ValueError("最大追跡周期は0超～1000、総追跡ステップは100万以下にしてください。")
    if (
        tpmc.ion_temperature_eV < 0
        or tpmc.seed < 0
        or not 0 < tpmc.max_recommended_collision_probability <= 1
    ):
        raise ValueError("イオン温度・乱数シード・推奨衝突確率の指定が不正です。")
    if not 64 <= circuit.phase_points <= 8192 or not 1 <= circuit.max_cycles <= 1000:
        raise ValueError("回路位相点数は64～8192、最大反復周期は1～1000にしてください。")
    if circuit.capacitance_factor <= 0 or circuit.periodic_tolerance_V <= 0:
        raise ValueError("回路容量係数と周期収束許容差は正の値にしてください。")
    waves = [config.waveform] if model == "1d" else [config.wafer_waveform, config.ring_waveform]
    for i, wave in enumerate(waves):
        if wave.mode == "scaled_wafer" and (model == "1d" or i == 0):
            raise ValueError("scaled_waferは2Dリング波形専用です。")
        if wave.mode == "csv":
            if not wave.csv_text or len(wave.csv_text.encode()) > 1024 * 1024:
                raise ValueError("波形CSV本文を1MB以下で指定してください。")
            if len(wave.delimiter) != 1 or min(wave.x_column, wave.voltage_column, wave.skip_header_rows) < 0:
                raise ValueError("CSV区切り・列番号・ヘッダー行数の指定が不正です。")
            ph, vv = parse_csv_waveform(wave, 2 * math.pi * config.plasma.frequency_Hz)
            if not 4 <= len(ph) <= 20000 or not np.isfinite(vv).all():
                raise ValueError("CSV波形は重複除去後に4～20000点の有限値が必要です。")
    if model == "1d":
        if config.circuit.powered_to_grounded_area_ratio <= 0:
            raise ValueError("電極面積比は正の値にしてください。")
        if config.sheath.front_width_exponent <= 0 or config.sheath.potential_exponent < 1:
            raise ValueError("シースのフロント指数は正、電位指数は1以上にしてください。")
        if not 2 <= config.plot.energy_bins <= 1000 or not 2 <= config.plot.angle_bins <= 1000:
            raise ValueError("分布ビン数は2～1000にしてください。")
        if config.plot.energy_max_eV is not None and config.plot.energy_max_eV <= 0:
            raise ValueError("表示エネルギー上限は正の値にしてください。")
    else:
        f, sc, geo, analysis = config.field2d, config.space_charge, config.geometry, config.analysis
        if not (17 <= f.nx <= 1025 and 17 <= f.ny <= 1025 and f.nx * f.ny <= 1_000_000):
            raise ValueError("2D格子は各17～1025、総数100万以下にしてください。")
        if not 0 < f.sor_omega < 2 or f.tolerance <= 0 or not 1 <= f.max_iterations <= 100000:
            raise ValueError("2D SOR緩和係数・許容差・最大反復数が不正です。")
        if geo.top_clearance_factor <= 0 or geo.smoothing_m < 0:
            raise ValueError("上方余白係数は正、平滑化幅は非負にしてください。")
        if any(b[0] <= a[0] for a, b in zip(geo.points_m, geo.points_m[1:], strict=False)):
            raise ValueError("2D制御点のx座標は重複しない昇順にしてください。")
        if (
            min(config.electrodes.wafer_to_ground_area_ratio, config.electrodes.ring_to_ground_area_ratio)
            <= 0
        ):
            raise ValueError("ウェハ・リング電極面積比は正の値にしてください。")
        if not 1 <= sc.outer_iterations <= 100 or not 10 <= sc.deposition_particles <= 2_000_000:
            raise ValueError("空間電荷反復は1～100、堆積粒子数は10～200万にしてください。")
        if not 0 < sc.under_relaxation <= 1 or sc.density_smoothing_sigma_cells < 0:
            raise ValueError("空間電荷の緩和係数と平滑化幅が不正です。")
        if min(sc.ion_density_clip_factor, sc.max_abs_correction_V, sc.poisson_tolerance_V) <= 0:
            raise ValueError("空間電荷の密度・電位上限とPoisson許容差は正にしてください。")
        if not 1 <= sc.electron_phase_samples <= 2048 or not 1 <= sc.poisson_max_iterations <= 100000:
            raise ValueError("空間電荷の電子位相点数・Poisson反復数が不正です。")
        if (
            min(analysis.bin_width_m, analysis.max_distance_m, analysis.edge_band_m) <= 0
            or analysis.edge_exclusion_m < 0
        ):
            raise ValueError("2D解析の距離・幅の指定が不正です。")
        if analysis.max_distance_m / analysis.bin_width_m > 10000:
            raise ValueError("2D解析の距離ビン数は10000以下にしてください。")
        if not 2 <= analysis.energy_bins <= 1000 or not 2 <= analysis.angle_bins <= 1000:
            raise ValueError("2D分布ビン数は2～1000にしてください。")
    return config


def model_hash():
    digest = hashlib.sha256()
    for path in sorted(CORE.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".csv"}:
            digest.update(path.relative_to(CORE).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


class Cancelled(Exception):
    pass


def run_upstream(directory_str):
    directory = Path(directory_str)
    request = UpstreamRequest.model_validate(read_json(directory / "request.json"))
    config = validated_config(request.model, request.config)
    started = time.time()
    state = dict(status="running", stage="IAEDF起動中", progress=0, started_at=started)
    log = []
    last_write = 0.0

    def report(fraction, text):
        nonlocal last_write
        if (directory / "cancel.request").exists():
            raise Cancelled()
        state["progress"] = max(state["progress"], min(float(fraction), 0.99))
        if text:
            state["stage"] = text
        now = time.monotonic()
        if now - last_write >= 0.5 or text:
            state["elapsed_s"] = time.time() - started
            atomic_json(directory / "status.json", state)
            last_write = now

    try:
        report(0, "IAEDF計算を開始します")
        xsec = (
            (ASSETS / config.gas.xsec_csv_name).read_text(encoding="utf-8")
            if config.gas.cross_section_source == "lxcat_phelps"
            else None
        )
        if request.model == "1d":
            output = run_1d(config, xsec_text=xsec, progress_cb=report)
            plots = build_plots_1d(output, config)
            save_npz_1d(directory / "raw.partial.npz", output)
        else:
            output = run_2d(config, xsec_text=xsec, progress_cb=report, log_cb=log.append)
            plots = build_plots_2d(output, config)
            save_npz_2d(directory / "raw.partial.npz", output)
        report(0.99, "結果を保存しています")
        retry_permission(lambda: (directory / "raw.partial.npz").replace(directory / "raw.npz"))
        atomic_json(directory / "config.json", config.model_dump())
        atomic_json(directory / "plots.json", clean_json(plots))
        summary = dict(
            model=request.model,
            validation=output["validation"],
            rows=plots["summary_rows"],
            scalars=plots["scalars"],
            elapsed_s=time.time() - started,
            log=log,
            model_sha256=model_hash(),
            source_project="IAEDF-Sim",
            velocity_frame="local tangent, out-of-plane, inward normal (SCA x/y/z)",
        )
        atomic_json(directory / "summary.json", clean_json(summary))
        atomic_json(
            directory / "status.json",
            dict(state, status="completed", stage="完了", progress=1, elapsed_s=time.time() - started),
        )
    except Cancelled:
        atomic_json(directory / "status.json", dict(state, status="cancelled", stage="計算を中止しました"))
    except Exception:
        error = traceback.format_exc()
        retry_permission(lambda: (directory / "error.log").write_text(error, encoding="utf-8"))
        atomic_json(
            directory / "status.json",
            dict(state, status="failed", stage="IAEDF計算に失敗しました", error=error.splitlines()[-1]),
        )
        raise


def selected_mask(raw, model, index, config, minimum=None, maximum=None):
    if index >= len(config.gas.pressures_mTorr):
        raise ValueError("指定した圧力ケースがありません。")
    prefix = f"p{index}_"
    mask = (raw[prefix + "reached"] if model == "1d" else raw[prefix + "on_wafer"]).astype(bool)
    if model == "2d":
        if (
            minimum is None
            or maximum is None
            or not 0 <= minimum < maximum <= config.geometry.domain_length_m
        ):
            raise ValueError("2Dは計算領域内のコレクタx範囲を指定してください。")
        mask &= (raw[prefix + "impact_x_m"] >= minimum) & (raw[prefix + "impact_x_m"] <= maximum)
    mask &= raw[prefix + "vz_m_s"] > 0
    if not mask.any():
        raise ValueError("選択範囲にウェハへ入射する粒子がありません。範囲や粒子数を変更してください。")
    return prefix, mask


def collector_length(geo, minimum, maximum):
    """Absorbing wafer area per unit out-of-plane length, including surface slope."""
    length = 0.0
    for a, b, material in zip(geo.points_m, geo.points_m[1:], geo.segment_materials, strict=False):
        lo, hi = max(a[0], minimum), min(b[0], maximum)
        if material == "wafer" and hi > lo:
            x = np.linspace(lo, hi, 4097)
            y = surface_height(x, geo)
            length += float(np.hypot(np.diff(x), np.diff(y)).sum())
    if length <= 0:
        raise ValueError("コレクタ範囲にウェハ表面がありません。")
    return length


def connected_case(directory, request: ApplyRequest):
    upstream = UpstreamRequest.model_validate(read_json(directory / "request.json"))
    config = validated_config(upstream.model, read_json(directory / "config.json"))
    c = request.config.model_copy(deep=True)
    derived = derive_plasma(config.plasma)
    wave = config.waveform if upstream.model == "1d" else config.wafer_waveform
    if wave.mode == "sinusoid":
        c.waveform = Waveform(
            dc_v=wave.sinusoid_dc_V,
            amplitude_v=abs(wave.sinusoid_amplitude_V),
            frequency_hz=config.plasma.frequency_Hz,
            phase_origin_deg=90
            + wave.sinusoid_phase_offset_deg
            + (180 if wave.sinusoid_amplitude_V < 0 else 0),
            source="SCA内蔵IAEDF",
        )
    else:
        ph, vv = parse_csv_waveform(wave, derived.omega)
        c.waveform = Waveform(
            dc_v=float(np.mean(vv)),
            amplitude_v=float(np.ptp(vv) / 2),
            frequency_hz=config.plasma.frequency_Hz,
            source="SCA内蔵IAEDF / CSV周期補間",
            samples=[
                WaveformSample(phase_deg=float(p), potential_v=float(v))
                for p, v in zip(np.degrees(ph), vv, strict=True)
            ],
        )
    with np.load(directory / "raw.npz", allow_pickle=False) as raw:
        prefix, mask = selected_mask(
            raw,
            upstream.model,
            request.pressure_index,
            config,
            request.collector_min_m,
            request.collector_max_m,
        )
        values = np.column_stack([raw[prefix + key][mask] for key in ("vx_m_s", "vy_m_s", "vz_m_s")])
        phases = raw[prefix + "impact_phase_deg"][mask].copy()
        theta = math.radians(request.azimuth_deg)
        values[:, :2] = values[:, :2] @ np.array(
            [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
        )
        energy = raw[prefix + "energy_eV"][mask]
        reconstructed = 0.5 * derived.ion_mass * (values**2).sum(axis=1) / QE
        if (
            not np.isfinite(values).all()
            or not np.isfinite(phases).all()
            or not np.allclose(energy, reconstructed, rtol=1e-10)
        ):
            raise ValueError("IAEDF結果の速度・エネルギー・位相の整合性を確認できません。")
        pressure = float(raw[prefix + "pressure_mTorr"])
        vp, phase = raw["V_p"].copy(), np.degrees(raw["phase"])
        selected = int(mask.sum())
        incoming = config.tpmc.n_particles
        area_factor = (
            1.0
            if upstream.model == "1d"
            else config.geometry.domain_length_m
            / collector_length(config.geometry, request.collector_min_m, request.collector_max_m)
        )
        flux = derived.n_s * derived.bohm_speed * selected / incoming * area_factor
    c.name = f"{upstream.name} · {pressure:g} mTorr → ホール"
    c.validation_case = True
    c.geometry.injection_z_nm = c.inlet_evaluation_z_nm = 0
    c.inlet_potential_mode = "wafer"
    c.inlet_source = f"SCA内蔵IAEDF {directory.name} / p{request.pressure_index} / {pressure:g} mTorr"
    c.sheath_acceleration_included = True
    c.ions = [
        Species(
            name="Ar+" if abs(config.plasma.ion_mass_amu - 39.948) < 0.1 else "ion+",
            mass_amu=config.plasma.ion_mass_amu,
            flux_m2_s=flux,
            flux_source="IAEDFのn_s×Bohm速度×コレクタ到達率（モデル推定）",
        )
    ]
    c.electron.temperature_ev = config.plasma.electron_temperature_eV
    c.electron.flux_m2_s = float(
        derived.n_s * np.sqrt(QE * config.plasma.electron_temperature_eV / (2 * np.pi * ME))
    )
    c.electron.phase_flux_model = "boltzmann"
    c.electron.plasma_potential_samples = [
        PhasePotential(phase_deg=float(p), potential_v=float(v)) for p, v in zip(phase, vp, strict=True)
    ]
    c.electron.flux_source = "IAEDFシース端密度から熱流束を推定、RF位相別Boltzmann減衰"
    c.electron.distribution_id = None
    c.gas_pressure_pa = pressure * MTORR_TO_PA
    c.gas_temperature_k = config.gas.gas_temperature_K
    warnings = [
        "絶対流束はIAEDFのシース端密度に基づくモデル推定で、実測校正は未実施です。",
        "選択範囲の分布をホール入口の一様円盤へ適用します。範囲内の位置相関は省略します。",
    ]
    if upstream.model == "2d":
        warnings.append("2Dの速度は局所接線・面外・表面内向き法線の座標系で接続します。")
    if config.magnetic.enabled:
        warnings.append("静磁場はIAEDF側のイオン追跡に適用済みです。ホール内の磁場力は省略します。")
    validation = read_json(directory / "summary.json")["validation"]
    if not validation.get("passed", False):
        warnings.append("IAEDFの数値検証に未達の項目があります。検証結果を確認してください。")
    meta = dict(
        filename=f"IAEDF-{directory.name}.npz",
        mode="samples",
        sample_count=selected,
        mass_amu=config.plasma.ion_mass_amu,
        evaluation_z_nm=0,
        reference_potential_v=c.inlet_reference_potential_v,
        waveform_id=waveform_id(c),
        phase_origin_deg=c.waveform.phase_origin_deg,
        spatial_model="uniform_disk",
        absolute_flux_from_distribution=False,
        ion_flux_m2_s=flux,
        source_job_id=directory.name,
        source_model=upstream.model,
        pressure_case=f"p{request.pressure_index}",
        pressure_mTorr=pressure,
        collector_x_min_m=request.collector_min_m,
        collector_x_max_m=request.collector_max_m,
        azimuth_deg=request.azimuth_deg,
        velocity_frame="sca_surface_local_xyz",
        source_config_sha256=hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((directory / "raw.npz").read_bytes()).hexdigest(),
        warnings=warnings,
        upstream_validation=validation,
    )
    return c, {"values": values, "phases": phases, "probability": np.full(selected, 1 / selected)}, meta


def distribution_plot(directory, index, minimum=None, maximum=None):
    upstream = UpstreamRequest.model_validate(read_json(directory / "request.json"))
    config = validated_config(upstream.model, read_json(directory / "config.json"))
    with np.load(directory / "raw.npz", allow_pickle=False) as raw:
        prefix, mask = selected_mask(raw, upstream.model, index, config, minimum, maximum)
        energy = raw[prefix + "energy_eV"][mask]
        angle = np.degrees(np.arctan2(raw[prefix + "vx_m_s"][mask], raw[prefix + "vz_m_s"][mask]))
        polar = np.degrees(
            np.arctan2(
                np.hypot(raw[prefix + "vx_m_s"][mask], raw[prefix + "vy_m_s"][mask]),
                raw[prefix + "vz_m_s"][mask],
            )
        )
        emax = max(float(energy.max()) * 1.01, 1.0)
        eh, ee = np.histogram(energy, bins=100, range=(0, emax), density=True)
        ah, ae = np.histogram(angle, bins=180, range=(-90, 90), density=True)
        joint, _, je = np.histogram2d(
            angle, energy, bins=(180, 80), range=((-90, 90), (0, emax)), density=True
        )
        return dict(
            sample_count=int(mask.sum()),
            mean_energy_ev=float(energy.mean()),
            mean_polar_angle_deg=float(polar.mean()),
            pressure_mTorr=float(raw[prefix + "pressure_mTorr"]),
            iedf=dict(x=((ee[:-1] + ee[1:]) / 2).tolist(), y=eh.tolist()),
            iadf=dict(x=((ae[:-1] + ae[1:]) / 2).tolist(), y=ah.tolist()),
            iaedf=dict(density=joint.T.tolist(), angle_edges_deg=ae.tolist(), energy_edges_ev=je.tolist()),
        )
