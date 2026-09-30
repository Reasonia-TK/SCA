"""Versioned input contract. UI lengths are nm; numerical geometry is always SI."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProfilePoint(InputModel):
    depth_nm: float = Field(ge=0)
    radius_nm: float = Field(gt=0)


class Dummy(InputModel):
    diameter_nm: float | None = Field(default=None, gt=0)
    gap_nm: float | None = Field(default=None, gt=0)
    angle_deg: float | None = None
    radius_nm: float | None = Field(default=None, gt=0)
    x_nm: float | None = None
    y_nm: float | None = None
    top_nm: float | None = Field(default=None, ge=0)
    bottom_nm: float | None = Field(default=None, gt=0)


class Geometry(InputModel):
    channel_diameter_nm: float = Field(default=400, gt=0)
    bottom_diameter_nm: float | None = Field(default=None, gt=0)
    carbon_thickness_nm: float = Field(default=300, gt=0)
    oxide_thickness_nm: float = Field(default=2100, gt=0)
    depth_mode: Literal["linked", "direct"] = "linked"
    depth_nm: float = Field(default=2400, gt=0)
    profile: list[ProfilePoint] = Field(default_factory=list)
    dummy_count: int = Field(default=6, ge=1, le=200)
    dummy_diameter_nm: float = Field(default=120, gt=0)
    gap_nm: float = Field(default=50, gt=0)
    reference_gap_nm: float = Field(default=50, gt=0)
    start_angle_deg: float = 0
    placement_mode: Literal["gap", "radial", "xy"] = "gap"
    gap_evaluation_depth_nm: float = Field(default=300, ge=0)
    dummies: list[Dummy] = Field(default_factory=list)
    lateral_margin_nm: float = Field(default=300, gt=0)
    domain_half_width_nm: float | None = Field(default=None, gt=0)
    entrance_height_nm: float = Field(default=300, gt=0)
    substrate_thickness_nm: float = Field(default=200, gt=0)
    injection_z_nm: float = Field(default=0, le=0)
    injection_radius_nm: float | None = Field(default=None, gt=0)
    mesh_size_nm: float = Field(default=110, gt=0)
    interface_mesh_size_nm: float = Field(default=35, gt=0)
    mesh_growth: float = Field(default=1.4, gt=1, le=3)


class Waveform(InputModel):
    dc_v: float = 0
    amplitude_v: float = Field(default=0, ge=0)
    frequency_hz: float = Field(default=13.56e6, gt=0)
    phase_origin_deg: float = 0
    source: str = "検証用仮値"

    def voltage(self, phase_deg: float) -> float:
        return self.dc_v + self.amplitude_v * math.sin(math.radians(phase_deg + self.phase_origin_deg))


class Species(InputModel):
    name: str = Field(default="Ar+", min_length=1, max_length=80)
    mass_amu: float = Field(default=39.948, gt=0)
    charge_number: int = Field(default=1, ge=-10, le=10)
    flux_m2_s: float = Field(default=1e19, ge=0)
    energy_ev: float = Field(default=100, gt=0)
    angular_sigma_deg: float = Field(default=2, ge=0, lt=60)
    distribution_id: str | None = None
    flux_source: str = "検証用仮値"


class PhasePotential(InputModel):
    phase_deg: float = Field(ge=0, lt=360)
    potential_v: float


class Electron(InputModel):
    enabled: bool = True
    temperature_ev: float = Field(default=3, gt=0)
    flux_m2_s: float = Field(default=1e19, ge=0)
    distribution_id: str | None = None
    phase_flux_model: Literal["constant", "boltzmann"] = "constant"
    plasma_potential_v: float = 0
    plasma_potential_samples: list[PhasePotential] = Field(default_factory=list)
    flux_source: str = "検証用仮値"


class Surface(InputModel):
    electron_reflection: float = Field(default=0, ge=0, le=1)
    ion_neutralization_reflection: float = Field(default=0, ge=0, le=1)
    secondary_yield: float = Field(default=0, ge=0, le=1)
    secondary_energy_ev: float = Field(default=2, gt=0)
    max_events: int = Field(default=4, ge=1, le=20)
    source: str = "未校正・検証用の定数係数"


class Numerics(InputModel):
    samples_per_species: int = Field(default=1000, ge=10, le=1000000)
    batch_size: int = Field(default=1000, ge=10, le=100000)
    seed: int = Field(default=20260930, ge=0)
    duration_s: float = Field(default=1e-6, gt=0)
    charging_steps: int = Field(default=4, ge=1, le=10000)
    max_particle_steps: int = Field(default=2500, ge=10, le=100000)
    particle_dt_s: float = Field(default=2e-12, gt=0)
    cell_fraction: float = Field(default=0.2, gt=0, le=0.5)
    phase_bins: int = Field(default=4, ge=1, le=64)
    representative_trajectories: int = Field(default=12, ge=0, le=100)
    relative_ci_target: float = Field(default=0.02, gt=0, lt=1)
    max_voltage_change_v: float = Field(default=10, gt=0)
    field_rtol: float = Field(default=1e-9, gt=0, lt=0.01)
    backend: Literal["cpu", "gpu"] = "cpu"
    save_every_steps: int = Field(default=1, ge=1, le=10000)


class CaseConfig(InputModel):
    schema_version: Literal["0.5.0"] = "0.5.0"
    name: str = Field(default="初期プリセット · CPU検証", min_length=1, max_length=160)
    validation_case: bool = True
    geometry: Geometry = Field(default_factory=Geometry)
    waveform: Waveform = Field(default_factory=Waveform)
    ions: list[Species] = Field(default_factory=lambda: [Species()])
    electron: Electron = Field(default_factory=Electron)
    surface: Surface = Field(default_factory=Surface)
    numerics: Numerics = Field(default_factory=Numerics)
    mode: Literal["uncharged", "fixed_charge", "self_consistent"] = "uncharged"
    oxide_relative_permittivity: float = Field(default=3.9, gt=0)
    initial_sigma_c_m2: float = 0
    leakage_tau_s: float | None = Field(default=None, gt=0)
    outer_boundary: Literal["dirichlet", "neumann"] = "dirichlet"
    outer_potential_v: float = 0
    outer_normal_displacement_c_m2: float = 0
    wafer_capacitance_pf: float = Field(default=4900, gt=0)
    metal_material: str = "理想導体（材種未指定）"
    substrate_material: str = "導体（材種未指定）"
    gas_pressure_pa: float = Field(default=0, ge=0)
    gas_temperature_k: float = Field(default=300, gt=0)
    collision_cross_section_m2: float | None = Field(default=None, ge=0)
    inlet_source: str = "検証用生成分布"
    sheath_acceleration_included: bool = True
    inlet_reference_potential_v: float = 0
    inlet_potential_mode: Literal["wafer", "constant", "unconstrained"] = "wafer"
    inlet_evaluation_z_nm: float = 0
    inlet_flux_definition: Literal["normal_number_flux"] = "normal_number_flux"

    @model_validator(mode="after")
    def validate_species(self):
        if not self.ions or any(s.charge_number <= 0 for s in self.ions):
            raise ValueError("正の電荷を持つイオン種を1種以上指定してください。")
        if len({s.name for s in self.ions}) != len(self.ions):
            raise ValueError("イオン種の名前は重複できません。")
        if abs(self.inlet_evaluation_z_nm - self.geometry.injection_z_nm) > 1e-6:
            raise ValueError("分布の評価面と注入面が一致しません。入口分布を更新してください。")
        if self.geometry.injection_z_nm != 0 and self.inlet_potential_mode != "unconstrained":
            raise ValueError(
                "指定評価面電位は現在Carbon上面の検証モードで利用できます。入口上方では外部評価面の契約が必要です。"
            )
        phase = [p.phase_deg for p in self.electron.plasma_potential_samples]
        if phase and (len(phase) < 2 or any(b <= a for a, b in pairwise(phase))):
            raise ValueError("プラズマ電位の位相サンプルは昇順で指定してください。")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()


def default_config() -> CaseConfig:
    return CaseConfig()
