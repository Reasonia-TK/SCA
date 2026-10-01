"""駆動波形（CSV周期補間 / 正弦波 / ウェハ波形スケール）。

ノートブックの load_csv_waveform / make_waveform_function と同一の数値処理。
CSVはファイルパスではなく本文文字列（csv_text）から読む。
"""

from __future__ import annotations

import io

import numpy as np

from .schemas import WaveformConfig


def x_to_phase_rad(x_values, x_axis: str, omega: float):
    x = np.asarray(x_values, dtype=float)
    if x_axis == "phase_deg":
        return np.deg2rad(x)
    if x_axis == "phase_rad":
        return x
    scale = {"time_s": 1.0, "time_ns": 1.0e-9, "time_us": 1.0e-6}
    if x_axis in scale:
        return omega * x * scale[x_axis]
    raise ValueError("x_axisはphase_deg/phase_rad/time_s/time_ns/time_us")


def parse_csv_waveform(config: WaveformConfig, omega: float):
    """CSV本文から (phase[rad]昇順, voltage[V]) を作る。"""
    if not config.csv_text:
        raise ValueError("csvモードですが波形CSV本文が未設定です。")
    selected = np.genfromtxt(
        io.StringIO(config.csv_text),
        delimiter=config.delimiter,
        skip_header=int(config.skip_header_rows),
        usecols=(int(config.x_column), int(config.voltage_column)),
        dtype=float,
        invalid_raise=False,
    )
    selected = np.asarray(selected, dtype=float).reshape(-1, 2)
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    if selected.shape[0] < 4:
        raise ValueError("有効な波形点が4点未満です。")
    phase = np.mod(
        x_to_phase_rad(selected[:, 0], config.x_axis, omega) + np.deg2rad(config.phase_offset_deg),
        2.0 * np.pi,
    )
    voltage = selected[:, 1] * config.voltage_scale + config.voltage_offset_V
    phase[(2.0 * np.pi - phase) < 1e-9] = 0.0
    key = np.round(phase, 12)
    uniq, inverse = np.unique(key, return_inverse=True)
    v_mean = np.bincount(inverse, weights=voltage) / np.bincount(inverse)
    p_mean = np.bincount(inverse, weights=phase) / np.bincount(inverse)
    order = np.argsort(p_mean)
    return p_mean[order], v_mean[order]


def make_waveform_function(config: WaveformConfig, omega: float, wafer_func=None):
    """波形設定から 位相[rad]->電位[V] の周期関数を作る。"""
    mode = config.mode
    if mode == "csv":
        ph, vv = parse_csv_waveform(config, omega)
        ph_ext = np.r_[ph, ph[0] + 2.0 * np.pi]
        vv_ext = np.r_[vv, vv[0]]

        def voltage(phase_rad, _ph0=ph[0], _phe=ph_ext, _vve=vv_ext):
            p = np.mod(np.asarray(phase_rad, dtype=float), 2.0 * np.pi)
            p = np.where(p < _ph0, p + 2.0 * np.pi, p)
            return np.interp(p, _phe, _vve)

        return voltage
    if mode == "sinusoid":

        def voltage(phase_rad):
            return config.sinusoid_dc_V + config.sinusoid_amplitude_V * np.cos(
                np.asarray(phase_rad, dtype=float) + np.deg2rad(config.sinusoid_phase_offset_deg)
            )

        return voltage
    if mode == "scaled_wafer":
        if wafer_func is None:
            raise ValueError("mode='scaled_wafer'はリング波形でのみ使えます")

        def voltage(phase_rad):
            shifted = np.asarray(phase_rad, dtype=float) + np.deg2rad(config.wafer_phase_offset_deg)
            return config.wafer_scale * wafer_func(shifted) + config.dc_offset_V

        return voltage
    raise ValueError(f"未知のwaveform mode: {mode}")


def make_derivative_function(voltage_func, h: float = 1.0e-5):
    def derivative(phase_rad):
        p = np.asarray(phase_rad, dtype=float)
        return (voltage_func(p + h) - voltage_func(p - h)) / (2.0 * h)

    return derivative
