"""2D物理コアv2（非周期・鏡像壁・材質境界）の検証。

回路（KCL）はノートブック bkm_2d_wafer_edge_ring_tpmc.ipynb と同一定式化のため
既定設定でノートブック出力と比較する。ジオメトリv2は対称性・保存則で検証する。
"""

import numpy as np
import pytest

from memory_hole.iaedf_core.model2d.field import (
    MAT_INSULATOR,
    MAT_RING,
    MAT_WAFER,
    material_of,
    max_surface_height,
    surface_height,
    wafer_ranges,
)
from memory_hole.iaedf_core.model2d.runner import run_2d
from memory_hole.iaedf_core.report import build_plots_2d
from memory_hole.iaedf_core.schemas import Config2D, GeometryConfig, WaveformConfig


def reduced_config(waveform_csv_text) -> Config2D:
    """傾向確認用の縮小設定。"""
    config = Config2D()
    config.wafer_waveform = WaveformConfig(mode="csv", csv_text=waveform_csv_text)
    config.ring_waveform = WaveformConfig(mode="csv", csv_text=waveform_csv_text)
    config.field2d.nx = 97
    config.field2d.ny = 48
    config.tpmc.n_particles = 1500
    config.tpmc.max_rf_periods = 40.0
    config.space_charge.deposition_particles = 1000
    config.space_charge.outer_iterations = 2
    config.circuit.phase_points = 1024
    return config


@pytest.mark.slow
class TestCircuitParity:
    """回路解はジオメトリ非依存のため既定設定でノートブックと比較する。"""

    def test_kcl_residual_and_convergence(self, waveform_csv_text, xsec_text):
        config = Config2D()
        config.wafer_waveform = WaveformConfig(mode="csv", csv_text=waveform_csv_text)
        config.ring_waveform = WaveformConfig(mode="csv", csv_text=waveform_csv_text)
        config.field2d.nx = 65
        config.field2d.ny = 40
        config.space_charge.enabled = False
        config.tpmc.n_particles = 200
        output = run_2d(config, xsec_text=xsec_text)
        v = output["validation"]
        # ノートブック: 周期端点のVp差 4.586e-10 V, KCL残差 4.014e-15
        assert v["periodic_error_V"] == pytest.approx(4.586e-10, rel=0.01)
        assert v["kcl_max_relative_residual"] < 1.0e-12


class TestGeometryV2:
    def geo(self, materials=None) -> GeometryConfig:
        return GeometryConfig(
            points_m=[
                [0.0, 0.25e-3],
                [3.0e-3, 0.25e-3],
                [3.2e-3, 0.45e-3],
                [12.8e-3, 0.45e-3],
                [13.0e-3, 0.25e-3],
                [16.0e-3, 0.25e-3],
            ],
            segment_materials=materials or ["ring", "wafer", "wafer", "wafer", "ring"],
        )

    def test_surface_interpolation_and_clamp(self):
        geo = self.geo()
        xs = np.array([0.0, 3.0e-3, 3.1e-3, 8.0e-3, 16.0e-3, 20.0e-3])
        expected = np.array([0.25e-3, 0.25e-3, 0.35e-3, 0.45e-3, 0.25e-3, 0.25e-3])
        assert np.allclose(surface_height(xs, geo), expected, atol=1e-9)
        assert max_surface_height(geo) == pytest.approx(0.45e-3, abs=1e-9)

    def test_material_mapping(self):
        geo = self.geo()
        codes = material_of(np.array([1.0e-3, 3.1e-3, 8.0e-3, 14.0e-3]), geo)
        assert list(codes) == [MAT_RING, MAT_WAFER, MAT_WAFER, MAT_RING]
        assert wafer_ranges(geo) == [(3.0e-3, 13.0e-3)]

    def test_insulator_material(self):
        geo = self.geo(materials=["insulator", "wafer", "wafer", "wafer", "ring"])
        codes = material_of(np.array([1.0e-3, 14.0e-3]), geo)
        assert list(codes) == [MAT_INSULATOR, MAT_RING]

    def test_endpoint_auto_completion(self):
        geo = GeometryConfig(
            domain_length_m=10.0e-3,
            points_m=[[2.0e-3, 0.3e-3], [8.0e-3, 0.3e-3]],
            segment_materials=["wafer"],
        )
        assert geo.points_m[0][0] == 0.0
        assert geo.points_m[-1][0] == 10.0e-3
        assert len(geo.segment_materials) == 3

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            GeometryConfig(points_m=[[0.0, 1e-3], [16e-3, 1e-3]], segment_materials=["ring"])  # waferなし
        with pytest.raises(ValueError):
            GeometryConfig(
                points_m=[[0.0, 1e-3], [16e-3, 1e-3]], segment_materials=["wafer", "ring"]
            )  # 数不一致
        with pytest.raises(ValueError):
            GeometryConfig(points_m=[[0.0, 1e-3], [16e-3, -1e-3]], segment_materials=["wafer"])  # 負の高さ

    def test_legacy_step_migration(self):
        legacy = {
            "periodic_length_m": 16.0e-3,
            "wafer_left_m": 3.0e-3,
            "wafer_right_m": 13.0e-3,
            "wafer_height_m": 0.45e-3,
            "ring_height_m": 0.25e-3,
            "step_smoothing_width_m": 0.1e-3,
            "surface_mode": "step",
            "top_clearance_factor": 1.0,
        }
        geo = GeometryConfig.model_validate(legacy)
        assert geo.domain_length_m == 16.0e-3
        assert "wafer" in geo.segment_materials
        assert "ring" in geo.segment_materials
        ranges = wafer_ranges(geo)
        assert len(ranges) == 1
        assert ranges[0][0] == pytest.approx(2.9e-3, abs=1e-6)

    def test_legacy_profile_migration(self):
        legacy = {
            "periodic_length_m": 16.0e-3,
            "wafer_left_m": 3.0e-3,
            "wafer_right_m": 13.0e-3,
            "surface_mode": "profile",
            "profile_points_m": [[0.0, 0.25e-3], [3.0e-3, 0.25e-3], [3.2e-3, 0.45e-3], [13.0e-3, 0.25e-3]],
            "profile_smoothing_m": 0.1e-3,
        }
        geo = GeometryConfig.model_validate(legacy)
        assert geo.points_m[-1][0] == 16.0e-3
        assert geo.smoothing_m == 0.1e-3
        assert "wafer" in geo.segment_materials


class TestReducedRunV2:
    @pytest.fixture(scope="class")
    @classmethod
    def output(cls, waveform_csv_text, xsec_text):
        config = reduced_config(waveform_csv_text)
        return run_2d(config, xsec_text=xsec_text), config

    def test_partition_of_unity(self, output):
        result, _ = output
        assert result["partition_error"] < 1.0e-3

    def test_no_particle_loss_at_walls(self, output):
        """鏡像壁で粒子が消えないこと（吸収は表面のみ）。"""
        result, config = output
        run = result["results"][0]["run"]
        assert run["energy_eV"].size + run["n_lost"] == config.tpmc.n_particles
        assert run["energy_eV"].size > 0.9 * config.tpmc.n_particles

    def test_impacts_within_domain(self, output):
        result, config = output
        run = result["results"][0]["run"]
        assert np.all(run["impact_x_m"] >= 0.0)
        assert np.all(run["impact_x_m"] <= config.geometry.domain_length_m)
        assert np.all(run["impact_material"] >= 0)

    def test_symmetric_geometry_gives_symmetric_tilt(self, output):
        """左右対称形状では平均外向き傾きが左右で同符号・同程度になる。"""
        result, config = output
        run = result["results"][0]["run"]
        length = config.geometry.domain_length_m
        onw = run["on_wafer"]
        x, angle = run["impact_x_m"][onw], run["angle_deg"][onw]
        left_band = (x > 3.0e-3) & (x < 4.5e-3)
        right_band = (x > length - 4.5e-3) & (x < length - 3.0e-3)
        # 外向き: 左端では-x方向(角度は負)、右端では+x方向(角度は正)
        assert np.mean(angle[left_band]) < 0.5
        assert np.mean(angle[right_band]) > -0.5
        assert abs(np.mean(angle[left_band]) + np.mean(angle[right_band])) < 1.0

    def test_space_charge_history_decreases(self, output):
        result, _ = output
        history = result["results"][0]["sc"]["history"]
        assert history[-1] < history[0]

    def test_plots_json_serializable(self, output):
        import json

        result, config = output
        plots = build_plots_2d(result, config)
        text = json.dumps(plots)
        assert '"model": "2d"' in text
        assert plots["geometry"]["segment_materials"]
        assert plots["geometry"]["wafer_ranges_mm"]


class TestInsulatorRun:
    def test_insulator_segments_absorb_and_field_continues(self, waveform_csv_text, xsec_text):
        config = reduced_config(waveform_csv_text)
        # 右側リングを絶縁体に置き換える
        config.geometry = GeometryConfig(
            points_m=[
                [0.0, 0.25e-3],
                [3.0e-3, 0.25e-3],
                [3.2e-3, 0.45e-3],
                [12.8e-3, 0.45e-3],
                [13.0e-3, 0.25e-3],
                [16.0e-3, 0.25e-3],
            ],
            segment_materials=["ring", "wafer", "wafer", "wafer", "insulator"],
        )
        config.space_charge.enabled = False
        config.tpmc.n_particles = 1200
        output = run_2d(config, xsec_text=xsec_text)
        assert output["partition_error"] < 1.0e-3
        run = output["results"][0]["run"]
        from memory_hole.iaedf_core.model2d.field import MAT_INSULATOR as INS

        ins_hits = np.sum(run["impact_material"] == INS)
        assert ins_hits > 0, "絶縁セグメントにも粒子が到達すること"
        summary = output["results"][0]["summary"]
        assert summary["insulator_mean_energy_eV"] is not None
        assert summary["insulator_mean_energy_eV"] > 0
        plots = build_plots_2d(output, config)
        assert "insulator_density" in plots["iedf"][0]
