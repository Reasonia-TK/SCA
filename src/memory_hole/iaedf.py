"""Adapter for the supplied IAEDF-Sim 1D sinusoidal NPZ + config pair."""

from __future__ import annotations

import hashlib
import io
import json
import math

import numpy as np

from .config import CaseConfig, PhasePotential, Species, Waveform
from .inlet import parse_distribution
from .transport import AMU, E_CHARGE, ELECTRON_MASS


def case_from_pair(
    payload: bytes,
    source: bytes,
    filename: str,
    template: CaseConfig,
    pressure_case="p0",
    projection_policy="reject",
):
    upstream = json.loads(source.decode("utf-8-sig"))
    plasma = upstream["plasma"]
    c = template.model_copy(deep=True)
    with np.load(io.BytesIO(payload), allow_pickle=False) as raw:
        if "V_e" not in raw or "phase" not in raw or "V_p" not in raw:
            raise ValueError("現在のペア取込には1Dのphase / V_e / V_pが必要です。")
        phase = np.asarray(raw["phase"], dtype=float)
        ve = np.asarray(raw["V_e"], dtype=float)
        vp = np.asarray(raw["V_p"], dtype=float)
        if (
            phase.ndim != 1
            or ve.shape != phase.shape
            or vp.shape != phase.shape
            or not np.isfinite([phase, ve, vp]).all()
        ):
            raise ValueError("位相・電位配列が不正です。")
        if len(phase) < 4 or np.any(np.diff(phase) <= 0) or phase[0] < 0 or phase[-1] >= 2 * math.pi:
            raise ValueError("RF位相は0以上2π未満の昇順ラジアンで指定してください。")
        if upstream["waveform"]["mode"] != "sinusoid":
            raise ValueError("任意波形の指定は後続対応です。現在は正弦波ペアを取り込めます。")
        basis = np.column_stack([np.ones(len(phase)), np.sin(phase), np.cos(phase)])
        dc, sin, cos = np.linalg.lstsq(basis, ve, rcond=None)[0]
        if np.max(abs(basis @ np.array([dc, sin, cos]) - ve)) > 1e-6:
            raise ValueError("NPZのV_eは単一正弦波として再現できません。")
        waveform = upstream["waveform"]
        if (
            abs(dc - waveform["sinusoid_dc_V"]) > 1e-6
            or abs(math.hypot(sin, cos) - waveform["sinusoid_amplitude_V"]) > 1e-6
        ):
            raise ValueError("NPZのV_eとJSONの波形パラメーターが一致しません。")
        phase_origin = math.degrees(math.atan2(cos, sin))
        if abs(phase_origin) < 1e-10:
            phase_origin = 0
        c.waveform = Waveform(
            dc_v=float(dc),
            amplitude_v=float(math.hypot(sin, cos)),
            frequency_hz=plasma["frequency_Hz"],
            phase_origin_deg=phase_origin,
            source=f"{filename}: phase / V_e",
        )
        pressure = float(raw[pressure_case + "_pressure_mTorr"])
        masses = plasma["ion_mass_amu"]
        density = plasma["sheath_edge_density_m3"]
        temperature = plasma["electron_temperature_eV"]
        bohm_speed = math.sqrt(temperature * E_CHARGE / (masses * AMU))
        reached = np.asarray(raw[pressure_case + "_reached"], dtype=bool)
        reached_fraction = float(np.mean(reached))
        # Bohm flux estimate is an explicit approximation, not a normalized-IAEDF conversion.
        c.ions = [
            Species(
                name="Ar+" if abs(masses - 39.948) < 0.001 else "ion",
                mass_amu=masses,
                flux_m2_s=density * bohm_speed * reached_fraction,
                flux_source="シース端密度×Bohm速度×到達割合による推定（粒子重みの上流検証は未完了）",
            )
        ]
        c.electron.temperature_ev = temperature
        c.electron.flux_m2_s = density * math.sqrt(temperature * E_CHARGE / (2 * math.pi * ELECTRON_MASS))
        c.electron.phase_flux_model = "boltzmann"
        c.electron.plasma_potential_samples = [
            PhasePotential(phase_deg=float(p * 180 / math.pi), potential_v=float(v))
            for p, v in zip(phase, vp, strict=True)
        ]
        c.electron.flux_source = "シース端密度・温度の熱流束×exp[-(V_p−V_e)/Te]による位相依存近似"
        c.gas_pressure_pa = pressure * 0.133322368
        c.gas_temperature_k = upstream["gas"]["gas_temperature_K"]
        c.geometry.injection_z_nm = 0
        c.inlet_evaluation_z_nm = 0
        c.inlet_potential_mode = "wafer"
        c.inlet_source = f"{filename} / {pressure_case} / {pressure:g} mTorr"
        c.validation_case = True
        c.name = f"IAEDF {pressure:g} mTorr · {pressure_case} · 検証"
        c.waveform.dc_v = round(c.waveform.dc_v, 12)
        c.waveform.amplitude_v = round(c.waveform.amplitude_v, 12)
    data, meta = parse_distribution(payload, filename, c, masses, pressure_case, projection_policy)
    meta.update(
        source_config_sha256=hashlib.sha256(source).hexdigest(),
        pressure_mtorr=pressure,
        ion_flux_estimate_m2_s=c.ions[0].flux_m2_s,
        reached_fraction=reached_fraction,
        evaluation_metadata_status="1D平坦ウェハ面 z=0。入口面電位をV_eへ拘束する検証モード",
    )
    meta["warnings"].extend(
        [
            "絶対イオン流束はBohm推定。上流の輸送重み・絶対流束の出力による検証が必要。",
            "平坦なウェハ面の入力を使用。入口上方のフリンジ輸送は評価対象外。",
        ]
    )
    if "20mTorr" in filename and pressure != 20:
        meta["warnings"].append(f"ファイル名と内部圧力が異なります。内部データの{pressure:g} mTorrを採用。")
    return c, data, meta
