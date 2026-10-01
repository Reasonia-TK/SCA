import numpy as np
import pytest
from pydantic import ValidationError

from memory_hole.config import CaseConfig, SaturationCriteria
from memory_hole.saturation import SaturationMonitor


def monitor():
    criteria = SaturationCriteria(
        window_s=1,
        min_windows=1,
        consecutive_windows=2,
        voltage_tolerance_v=0.1,
        charge_relative_tolerance=0,
        charge_absolute_tolerance_c_m2=0.01,
    )
    return SaturationMonitor(criteria, 0, np.zeros(2), np.zeros(2), np.ones(2))


def test_small_updates_do_not_count_as_physical_windows():
    m = monitor()
    for time in np.linspace(0.001, 0.999, 100):
        assert not m.observe(time, np.zeros(2), np.zeros(2), 1, 0)
    assert m.state["windows"] == 0
    assert not m.observe(1, np.zeros(2), np.zeros(2), 1, 0)
    assert m.state["consecutive"] == 1
    assert m.observe(2, np.zeros(2), np.zeros(2), 1, 0)


@pytest.mark.parametrize(
    "sigma,phi,unresolved",
    [
        ([1, -1], [0, 0], 0),  # Net total charge cancels while local patches change.
        ([0, 0], [0.2, 0], 0),
        ([0, 0], [0, 0], 0.2),
    ],
)
def test_unstable_or_unresolved_windows_cannot_saturate(sigma, phi, unresolved):
    m = monitor()
    m.observe(1, np.array(sigma), np.array(phi), 1, unresolved)
    assert not m.state["last_window"]["passed"]
    assert m.state["consecutive"] == 0


def test_returning_to_initial_charge_does_not_hide_an_excursion():
    m = monitor()
    m.observe(0.5, np.array([0.1, -0.1]), np.zeros(2), 1, 0)
    m.observe(1, np.zeros(2), np.zeros(2), 1, 0)
    assert m.state["last_window"]["density_change_c_m2"] == pytest.approx(0.1)
    assert not m.state["last_window"]["passed"]


def test_failed_window_resets_consecutive_stability():
    m = monitor()
    assert not m.observe(1, np.zeros(2), np.zeros(2), 1, 0)
    assert not m.observe(2, np.zeros(2), np.zeros(2), 1, 0.2)
    assert m.state["consecutive"] == 0
    assert not m.observe(3, np.zeros(2), np.zeros(2), 1, 0)
    assert m.observe(4, np.zeros(2), np.zeros(2), 1, 0)


def test_saturation_requires_self_consistent_charging():
    with pytest.raises(ValidationError, match="自己無撞着"):
        CaseConfig.model_validate({"numerics": {"run_until": "saturation"}})
    c = CaseConfig.model_validate({"mode": "self_consistent", "numerics": {"run_until": "saturation"}})
    assert c.numerics.saturation.max_time_s > 0
