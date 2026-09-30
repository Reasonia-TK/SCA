"""Explicit, correlated inlet contracts; never reinterpret projected angles as polar angles."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from .config import CaseConfig
from .transport import AMU, E_CHARGE

MAX_SAMPLES = 2_000_000


def waveform_id(config: CaseConfig):
    w = config.waveform
    return hashlib.sha256(
        json.dumps([w.dc_v, w.amplitude_v, w.frequency_hz, w.phase_origin_deg]).encode()
    ).hexdigest()


def parse_distribution(
    payload: bytes,
    filename: str,
    config: CaseConfig,
    mass_amu: float,
    pressure_case="p0",
    projection_policy="reject",
    metadata=None,
):
    warnings = []
    phases = None
    weights = None
    mode = "samples"
    if filename.lower().endswith(".npz"):
        with np.load(io.BytesIO(payload), allow_pickle=False) as raw:
            prefix = pressure_case + "_"
            key = prefix + "energy_eV"
            if key not in raw:
                raise ValueError(f"{key}がありません。圧力ケースを確認してください。")
            energy = np.asarray(raw[key], dtype=float)
            mask_key = prefix + "reached" if prefix + "reached" in raw else prefix + "on_wafer"
            if mask_key not in raw:
                raise ValueError("到達フラグがありません。選択コレクタ面への到達粒子が必要です。")
            mask = np.asarray(raw[mask_key], dtype=bool)
            if energy.ndim != 1 or mask.shape != energy.shape or len(energy) > MAX_SAMPLES:
                raise ValueError("粒子数または到達配列の形状が不正です。")
            if prefix + "on_wafer" in raw and prefix + "impact_x_m" in raw:
                if not metadata or "collector_x_min_m" not in metadata or "collector_x_max_m" not in metadata:
                    raise ValueError("2D出力にはコレクタのx範囲を明示してください。")
                x = raw[prefix + "impact_x_m"]
                mask &= (x >= metadata["collector_x_min_m"]) & (x <= metadata["collector_x_max_m"])
            keys = [prefix + name for name in ("vx_m_s", "vy_m_s", "vz_m_s")]
            if all(k in raw for k in keys):
                v = np.column_stack([np.asarray(raw[k])[mask] for k in keys])
                reconstructed = 0.5 * mass_amu * AMU * np.sum(v * v, axis=1) / E_CHARGE
                if not np.allclose(reconstructed, energy[mask], rtol=0.01, atol=1e-6):
                    raise ValueError("速度3成分から算出したエネルギーとenergy_eVが一致しません。")
            else:
                if projection_policy != "zero_out_of_plane":
                    raise ValueError(
                        "旧NPZの角度は投影角です。速度3成分を追加するか、検証用の面外速度ゼロ近似を明示してください。"
                    )
                angle_key = (
                    prefix + "signed_angle_deg"
                    if prefix + "signed_angle_deg" in raw
                    else prefix + "angle_deg"
                )
                angle = np.deg2rad(np.asarray(raw[angle_key])[mask])
                speed = np.sqrt(2 * energy[mask] * E_CHARGE / (mass_amu * AMU))
                v = np.column_stack([speed * np.sin(angle), np.zeros(len(speed)), speed * np.cos(angle)])
                warnings.append(
                    "投影角を保ち、面外速度をゼロとした検証用近似。3D極角分布の再現ではありません。"
                )
            if prefix + "impact_phase_deg" in raw:
                phases = np.asarray(raw[prefix + "impact_phase_deg"])[mask]
            else:
                warnings.append("到達RF位相が未保存です。RF位相は一様分布と仮定します。")
            if prefix + "weight" in raw:
                weights = np.asarray(raw[prefix + "weight"])[mask]
    elif filename.lower().endswith(".csv"):
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
        rows = list(reader)
        if not rows or len(rows) > MAX_SAMPLES:
            raise ValueError("CSVの行数が不正です。")
        columns = set(reader.fieldnames or [])

        def col(name):
            return np.array([float(row[name]) for row in rows])

        if {"energy_low_ev", "energy_high_ev", "theta_low_deg", "theta_high_deg", "probability"} <= columns:
            mode = "bins"
            bins = np.column_stack(
                [col(k) for k in ("energy_low_ev", "energy_high_ev", "theta_low_deg", "theta_high_deg")]
            )
            weights = col("probability")
            if (
                np.any(bins[:, 0] < 0)
                or np.any(bins[:, 1] <= bins[:, 0])
                or np.any(bins[:, 2] < 0)
                or np.any(bins[:, 3] > 90)
                or np.any(bins[:, 3] <= bins[:, 2])
            ):
                raise ValueError("エネルギー・極角のビン境界が不正です。")
            if not np.isfinite(bins).all():
                raise ValueError("ビン境界は有限値で指定してください。")
            v = bins
        elif {"vx_m_s", "vy_m_s", "vz_m_s"} <= columns:
            v = np.column_stack([col(k) for k in ("vx_m_s", "vy_m_s", "vz_m_s")])
        elif {"energy_ev", "theta_deg", "phi_deg"} <= columns:
            energy, theta, phi = col("energy_ev"), col("theta_deg"), col("phi_deg")
            if np.any(energy <= 0) or np.any(theta < 0) or np.any(theta >= 90):
                raise ValueError("入射エネルギーは正、3D極角は0以上90度未満としてください。")
            theta, phi = np.deg2rad(theta), np.deg2rad(phi)
            speed = np.sqrt(2 * energy * E_CHARGE / (mass_amu * AMU))
            v = speed[:, None] * np.column_stack(
                [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
            )
        else:
            raise ValueError("CSVには速度3成分、energy_ev/theta_deg/phi_deg、または相関ビン形式が必要です。")
        if "phase_deg" in columns:
            phases = col("phase_deg")
        if mode == "samples" and "weight" in columns:
            weights = col("weight")
    else:
        raise ValueError("入口分布にはCSVまたはNPZを指定してください。")
    if len(v) == 0 or not np.isfinite(v).all() or (mode == "samples" and np.any(v[:, 2] <= 0)):
        raise ValueError("入射サンプルが空、非有限、または法線方向速度が正ではありません。")
    if weights is None:
        weights = np.ones(len(v))
    if (
        weights.shape != (len(v),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("粒子重みまたはビン確率が不正です。")
    if phases is not None and (phases.shape != (len(v),) or not np.isfinite(phases).all()):
        raise ValueError("位相配列が不正です。")
    normalized = weights / weights.sum()
    meta = dict(
        filename=Path(filename).name,
        input_sha256=hashlib.sha256(payload).hexdigest(),
        mode=mode,
        sample_count=len(v),
        mass_amu=mass_amu,
        evaluation_z_nm=config.inlet_evaluation_z_nm,
        reference_potential_v=config.inlet_reference_potential_v,
        waveform_id=waveform_id(config),
        phase_origin_deg=config.waveform.phase_origin_deg,
        spatial_model="uniform_disk",
        absolute_flux_from_distribution=False,
        pressure_case=pressure_case,
        warnings=warnings,
        evaluation_metadata_status="利用者が指定したケース条件。上流との整合は未検証",
    )
    return {"values": v, "probability": normalized, "phases": phases}, meta


def sample_distribution(rng, count, data, meta, mass_amu):
    if abs(meta["mass_amu"] - mass_amu) > 1e-8:
        raise ValueError("入口分布の質量と粒子種が一致しません。")
    indexes = rng.choice(len(data["values"]), size=count, p=data["probability"])
    if meta["mode"] == "bins":
        bins = data["values"][indexes]
        energy = rng.uniform(bins[:, 0], bins[:, 1])
        theta = np.deg2rad(rng.uniform(bins[:, 2], bins[:, 3]))
        phi = rng.uniform(0, 2 * math.pi, count)
        speed = np.sqrt(2 * energy * E_CHARGE / (mass_amu * AMU))
        v = speed[:, None] * np.column_stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
    else:
        v = data["values"][indexes].copy()
    phases = data["phases"][indexes] if data["phases"] is not None else rng.uniform(0, 360, count)
    return v, phases % 360


def verify_metadata(config, meta):
    if abs(meta["evaluation_z_nm"] - config.geometry.injection_z_nm) > 1e-6:
        raise ValueError("入口分布と現在の注入面が一致しません。再取込してください。")
    if meta["waveform_id"] != waveform_id(config):
        raise ValueError(
            "入口分布のウェハ波形と現在の波形が一致しません。対応するIAEDFを再取込してください。"
        )
    if meta["reference_potential_v"] != config.inlet_reference_potential_v:
        raise ValueError("入口分布の基準電位が一致しません。")
