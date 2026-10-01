"""1D物理コアがノートブック bkm_1d_sheath_tpmc.ipynb と一致することの検証。

参照値はノートブックの実行済み出力（同一設定・同一シード）から取っている。
"""

import numpy as np
import pytest

from memory_hole.iaedf_core.model1d.runner import run_1d
from memory_hole.iaedf_core.plasma import child_langmuir_width, derive_plasma
from memory_hole.iaedf_core.schemas import Config1D, WaveformConfig


def default_config(waveform_csv_text) -> Config1D:
    config = Config1D()
    config.waveform = WaveformConfig(mode="csv", csv_text=waveform_csv_text, x_axis="time_s")
    return config


class TestDerivedQuantities:
    def test_notebook_printed_values(self):
        config = Config1D()
        derived = derive_plasma(config.plasma)
        assert derived.rf_period * 1e9 == pytest.approx(73.746, abs=1e-3)
        assert derived.debye_length * 1e3 == pytest.approx(0.1288, abs=1e-4)
        assert derived.floating_drop == pytest.approx(14.04, abs=0.01)

    def test_child_langmuir_width_floor(self):
        derived = derive_plasma(Config1D().plasma)
        width = child_langmuir_width(1e-12, derived)
        assert width == pytest.approx(derived.debye_length)


@pytest.mark.slow
class TestNotebookParity:
    """既定設定・既定シードでノートブックの出力と数値一致すること。"""

    @pytest.fixture(scope="class")
    @classmethod
    def output(cls, waveform_csv_text, xsec_text):
        config = default_config(waveform_csv_text)
        return run_1d(config, xsec_text=xsec_text)

    def test_circuit_vsp_max(self, output):
        vsp_max = float(np.max(output["circuit_solution"]["V_sp"]))
        assert vsp_max == pytest.approx(339.4817, abs=0.01)

    def test_riley_bridge(self, output):
        v = output["validation"]
        assert v["riley_delta_E_eV"] == pytest.approx(39.1, abs=0.1)
        assert v["riley_v_tilde_eff_V"] == pytest.approx(149.2, abs=0.1)
        assert v["omega_tau_ion_over_4"] == pytest.approx(7.57, abs=0.01)

    def test_energy_conservation(self, output):
        v = output["validation"]
        assert v["static_expected_gain_eV"] == pytest.approx(339.4817, abs=0.01)
        assert v["static_tpmc_gain_eV"] == pytest.approx(339.4711, abs=0.01)
        assert v["energy_conservation_rel_error"] < 1.0e-3

    def test_collision_probability(self, output):
        v = output["validation"]
        assert v["max_step_collision_probability"] == pytest.approx(1.246e-3, rel=0.01)
        assert v["collision_probability_ok"]

    def test_tpmc_particle_counts_match_notebook(self, output):
        # ノートブック出力: p=0: 30000到達/0脱出, p=5: 29973/27, p=20: 29859/141
        expected = [(30000, 0), (29973, 27), (29859, 141)]
        for r, (n_reach, n_escape) in zip(output["results"], expected, strict=True):
            assert int(np.sum(r["reached"])) == n_reach
            assert int(np.sum(r["escaped"])) == n_escape
            assert int(np.sum(r["cutoff"])) == 0

    def test_cx_rates_match_notebook(self, output):
        # ノートブック出力: CX経験率 0.0 / 19.3 / 57.7 %
        expected = [0.0, 19.3, 57.7]
        for r, cx in zip(output["results"], expected, strict=True):
            ok = r["reached"]
            assert 100 * np.mean(r["n_cx"][ok] > 0) == pytest.approx(cx, abs=0.05)

    def test_validation_passed(self, output):
        assert output["validation"]["passed"]


class TestSmoke:
    def test_reduced_run(self, waveform_csv_text, xsec_text):
        config = default_config(waveform_csv_text)
        config.tpmc.n_particles = 1500
        config.gas.pressures_mTorr = [0.0, 20.0]
        progress = []
        output = run_1d(config, xsec_text=xsec_text, progress_cb=lambda f, t: progress.append(f))
        assert output["validation"]["energy_conservation_ok"]
        assert progress[-1] == 1.0
        assert all(b >= a for a, b in zip(progress, progress[1:], strict=False))
        for r in output["results"]:
            assert np.sum(r["reached"]) > 0.9 * 1500

    def test_sinusoid_mode(self, xsec_text):
        config = Config1D()
        config.waveform = WaveformConfig(mode="sinusoid")
        config.tpmc.n_particles = 800
        config.gas.pressures_mTorr = [0.0]
        output = run_1d(config, xsec_text=xsec_text)
        assert output["validation"]["energy_conservation_ok"]

    def test_angle_step_overrides_bins(self, xsec_text):
        from memory_hole.iaedf_core.report import build_plots_1d

        config = Config1D()
        config.waveform = WaveformConfig(mode="sinusoid")
        config.tpmc.n_particles = 500
        config.gas.pressures_mTorr = [0.0]
        config.plot.angle_step_deg = 1.0  # -> 180ビン（angle_binsより優先）
        output = run_1d(config, xsec_text=xsec_text)
        plots = build_plots_1d(output, config)
        assert len(plots["iadf"][0]["density"]) == 180
        assert len(plots["iaedf"][0]["angle_edges_deg"]) == 181
        # 未指定なら従来通り 2*angle_bins / angle_bins
        config.plot.angle_step_deg = None
        plots = build_plots_1d(output, config)
        assert len(plots["iadf"][0]["density"]) == 240
        assert len(plots["iaedf"][0]["angle_edges_deg"]) == 121

    def test_sheath_width_plot(self, xsec_text):
        from memory_hole.iaedf_core.report import build_plots_1d

        config = Config1D()
        config.waveform = WaveformConfig(mode="sinusoid")
        config.tpmc.n_particles = 500
        config.gas.pressures_mTorr = [0.0]
        output = run_1d(config, xsec_text=xsec_text)
        sw = build_plots_1d(output, config)["sheath_width"]
        n = len(sw["phase_deg"])
        assert len(sw["s_e_mm"]) == n
        assert len(sw["s_cl_powered_mm"]) == n == len(sw["s_cl_ground_mm"])
        # moving_frontのs_eはVsp最大位相でs_maxに一致し、それ以下で振動する
        # （プロットは512点へ間引くため最大位相の点がわずかにずれ得る）
        assert max(sw["s_e_mm"]) == pytest.approx(sw["s_max_mm"], rel=1e-3)
        assert max(sw["s_e_mm"]) <= sw["s_max_mm"] * (1 + 1e-9)
        assert min(sw["s_e_mm"]) < 0.5 * sw["s_max_mm"]
        # 接地側シースはほぼ浮遊電位相当で薄い
        assert max(sw["s_cl_ground_mm"]) < min(sw["s_cl_powered_mm"])

    def test_approximation_xsec(self, waveform_csv_text):
        config = default_config(waveform_csv_text)
        config.gas.cross_section_source = "approximation"
        config.tpmc.n_particles = 500
        config.gas.pressures_mTorr = [5.0]
        output = run_1d(config)
        assert np.sum(output["results"][0]["reached"]) > 400
