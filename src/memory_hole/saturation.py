"""Numerical plateau detection over full physical-time windows.

Area-weighted absolute charge changes prevent cancellation between surface
patches. The maximum excursion inside a window also rejects oscillations that
return to the initial state at its end. This is a configurable numerical test,
not a confidence interval or a validation of the plasma/material model.
"""

from __future__ import annotations

import numpy as np

from .config import SaturationCriteria


class SaturationMonitor:
    def __init__(self, criteria: SaturationCriteria, time_s, sigma, phi, areas, restored=None):
        self.criteria = criteria
        self.areas = areas
        self.reference_phi = phi.copy()
        self.state = (
            restored
            if restored is not None
            else {
                "window_start_s": time_s,
                "reference_sigma_c_m2": sigma.tolist(),
                "windows": 0,
                "consecutive": 0,
                "saturated": False,
                "last_window": None,
                "max_voltage_change_v": 0.0,
                "max_density_change_c_m2": 0.0,
                "charge_scale_c_m2": self.mean_absolute_density(sigma),
                "injected_absolute_c": 0.0,
                "unresolved_absolute_c": 0.0,
            }
        )
        self.reference_sigma = np.array(self.state["reference_sigma_c_m2"])

    def mean_absolute_density(self, sigma):
        return float(np.dot(np.abs(sigma), self.areas) / self.areas.sum())

    def remaining_window_s(self, time_s):
        return max(0.0, self.state["window_start_s"] + self.criteria.window_s - time_s)

    def observe(self, time_s, sigma, phi, injected_absolute_c, unresolved_absolute_c):
        s, c = self.state, self.criteria
        s["max_voltage_change_v"] = max(
            s["max_voltage_change_v"], float(np.max(np.abs(phi - self.reference_phi), initial=0))
        )
        s["max_density_change_c_m2"] = max(
            s["max_density_change_c_m2"], self.mean_absolute_density(sigma - self.reference_sigma)
        )
        s["charge_scale_c_m2"] = max(s["charge_scale_c_m2"], self.mean_absolute_density(sigma))
        s["injected_absolute_c"] += injected_absolute_c
        s["unresolved_absolute_c"] += unresolved_absolute_c
        if time_s - s["window_start_s"] < c.window_s * (1 - 1e-10):
            return False
        s["windows"] += 1
        charge_limit = c.charge_absolute_tolerance_c_m2 + c.charge_relative_tolerance * s["charge_scale_c_m2"]
        unresolved_fraction = s["unresolved_absolute_c"] / max(s["injected_absolute_c"], 1e-300)
        passed = (
            s["max_voltage_change_v"] <= c.voltage_tolerance_v
            and s["max_density_change_c_m2"] <= charge_limit
            and unresolved_fraction <= c.max_unresolved_fraction
        )
        s["consecutive"] = s["consecutive"] + 1 if passed and s["windows"] >= c.min_windows else 0
        s["saturated"] = s["consecutive"] >= c.consecutive_windows
        s["last_window"] = {
            "start_s": s["window_start_s"],
            "end_s": time_s,
            "max_voltage_change_v": s["max_voltage_change_v"],
            "density_change_c_m2": s["max_density_change_c_m2"],
            "density_tolerance_c_m2": charge_limit,
            "unresolved_fraction": unresolved_fraction,
            "passed": passed,
        }
        s.update(
            window_start_s=time_s,
            reference_sigma_c_m2=sigma.tolist(),
            max_voltage_change_v=0.0,
            max_density_change_c_m2=0.0,
            charge_scale_c_m2=self.mean_absolute_density(sigma),
            injected_absolute_c=0.0,
            unresolved_absolute_c=0.0,
        )
        self.reference_sigma = sigma.copy()
        self.reference_phi = phi.copy()
        return s["saturated"]

    def summary(self):
        return {
            "windows": self.state["windows"],
            "consecutive": self.state["consecutive"],
            "required": self.criteria.consecutive_windows,
            "saturated": self.state["saturated"],
            "last_window": self.state["last_window"],
        }
