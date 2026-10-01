"""静磁場（Borisプッシャー）の検証。

解析解（純ジャイロ運動・E×Bドリフト）との一致、B=0での従来経路の完全一致、
1D/2D TPMCでの偏向の向きを確認する。
"""

import numpy as np
import pytest

from memory_hole.iaedf_core.mc_utils import boris_push
from memory_hole.iaedf_core.model1d.runner import run_1d
from memory_hole.iaedf_core.model1d.tpmc import run_tpmc
from memory_hole.iaedf_core.model2d.runner import run_2d
from memory_hole.iaedf_core.plasma import derive_plasma
from memory_hole.iaedf_core.schemas import Config1D, Config2D, MagneticFieldConfig, WaveformConfig
from memory_hole.iaedf_core.xsec import build_cross_sections


class TestBorisPush:
    def test_pure_gyration_conserves_speed_and_radius(self):
        # E=0, B=z軸: 速度は厳密回転（エネルギー保存は機械精度）、
        # 軌道半径はLarmor半径 v_perp/ω に一致する
        qm_dt = 1.0  # q=m=1, dt=1 の規格化
        omega = 0.05
        vx, vy, vz = np.array([1.0]), np.array([0.0]), np.array([0.3])
        speed0 = float(np.sqrt(vx**2 + vy**2 + vz**2)[0])
        x = y = 0.0
        xs, ys = [], []
        for _ in range(3 * int(round(2 * np.pi / omega))):
            vx, vy, vz = boris_push(vx, vy, vz, 0.0, 0.0, 0.0, 0.0, 0.0, omega, qm_dt)
            x += float(vx[0]) * qm_dt
            y += float(vy[0]) * qm_dt
            xs.append(x)
            ys.append(y)
        speed = float(np.sqrt(vx**2 + vy**2 + vz**2)[0])
        assert speed == pytest.approx(speed0, rel=1e-13)
        radius = 0.5 * (max(xs) - min(xs))
        assert radius == pytest.approx(1.0 / omega, rel=1e-3)
        assert float(vz[0]) == pytest.approx(0.3, abs=1e-14)

    def test_exb_drift(self):
        # E=(E0,0,0), B=(0,0,Bz): ドリフト速度は E×B/B² = (0, -E0/Bz, 0)
        e0, bz = 0.2, 1.0
        # 1ステップの回転角がちょうど2π/64になるdtを選び、整数周期で平均する
        dt = 2.0 * np.tan(np.pi / 64) / bz
        vx, vy, vz = np.array([0.0]), np.array([0.0]), np.array([0.0])
        vy_sum = 0.0
        n_steps = 64 * 20
        for _ in range(n_steps):
            vx, vy, vz = boris_push(vx, vy, vz, e0, 0.0, 0.0, 0.0, 0.0, bz, dt)
            vy_sum += float(vy[0])
        assert vy_sum / n_steps == pytest.approx(-e0 / bz, rel=2e-3)

    def test_zero_b_equals_plain_kick(self):
        vx, vy, vz = np.array([1.0, -2.0]), np.array([0.5, 0.0]), np.array([0.0, 3.0])
        ex = np.array([2.0, -1.0])
        nvx, nvy, nvz = boris_push(vx, vy, vz, ex, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25)
        np.testing.assert_array_equal(nvx, vx + 0.25 * ex)
        np.testing.assert_array_equal(nvy, vy)
        np.testing.assert_array_equal(nvz, vz)


class TestMagnetic1D:
    def test_zero_config_identical_to_none(self, xsec_text):
        # 全成分0のMagneticFieldConfigはmagnetic=Noneと完全一致（回帰ガード）
        config = Config1D()
        derived = derive_plasma(config.plasma)
        sigma_cx, sigma_el = build_cross_sections(config.gas, xsec_text)
        common = dict(
            derived=derived,
            sheath=config.sheath,
            gas=config.gas,
            tpmc=config.tpmc,
            sigma_cx=sigma_cx,
            sigma_el=sigma_el,
            vsp_table=np.full(64, 100.0),
            s_max=1.0e-3,
            n_particles=300,
            seed=7,
        )
        a = run_tpmc(5.0, magnetic=None, **common)
        b = run_tpmc(5.0, magnetic=MagneticFieldConfig(), **common)
        np.testing.assert_array_equal(a["energy_eV"], b["energy_eV"])
        np.testing.assert_array_equal(a["signed_angle_deg"], b["signed_angle_deg"])
        np.testing.assert_array_equal(a["n_cx"], b["n_cx"])

    def test_bz_deflects_iadf_and_validation(self, xsec_text):
        # 2D右手系でBz>0はvyを負に回す → 符号付き角度の平均が負にシフト。
        # Bは仕事をしないため静的エネルギー保存検証はそのまま合格する
        config = Config1D()
        config.waveform = WaveformConfig(mode="sinusoid")
        config.tpmc.n_particles = 2000
        config.gas.pressures_mTorr = [0.0]
        config.magnetic = MagneticFieldConfig(bz_T=0.5)
        output = run_1d(config, xsec_text=xsec_text)
        v = output["validation"]
        assert v["magnetic_field_T"] == pytest.approx(0.5)
        assert v["omega_ci_dt"] < 0.3
        assert v["gyration_resolution_ok"]
        assert v["energy_conservation_ok"]
        assert v["passed"]
        assert v["magnetic_deflection_deg"] > 0.0
        r = output["results"][0]
        mean_angle = float(np.mean(r["signed_angle_deg"][r["reached"]]))
        assert mean_angle < -3.0

    def test_no_magnetic_keys_when_disabled(self, xsec_text):
        config = Config1D()
        config.waveform = WaveformConfig(mode="sinusoid")
        config.tpmc.n_particles = 500
        config.gas.pressures_mTorr = [0.0]
        output = run_1d(config, xsec_text=xsec_text)
        assert "omega_ci_dt" not in output["validation"]


class TestMagnetic2D:
    def test_bz_shifts_wafer_angle(self, waveform_csv_text, xsec_text):
        # 2Dのz（面外）方向のB: vy<0で入射するイオンのvxが負に回り、
        # ウェハ上の平均入射角が負側へシフトする
        config = Config2D()
        config.wafer_waveform = WaveformConfig(mode="csv", csv_text=waveform_csv_text)
        config.ring_waveform = WaveformConfig(mode="csv", csv_text=waveform_csv_text)
        config.field2d.nx = 97
        config.field2d.ny = 48
        config.tpmc.n_particles = 1500
        config.tpmc.max_rf_periods = 40.0
        config.circuit.phase_points = 1024
        config.space_charge.enabled = False
        config.magnetic = MagneticFieldConfig(bz_T=0.5)
        output = run_2d(config, xsec_text=xsec_text)
        v = output["validation"]
        assert v["magnetic_field_T"] == pytest.approx(0.5)
        assert v["gyration_resolution_ok"]
        run = output["results"][0]["run"]
        on_wafer = run["on_wafer"]
        assert int(np.sum(on_wafer)) > 500
        mean_angle = float(np.mean(run["angle_deg"][on_wafer]))
        assert mean_angle < -2.0
