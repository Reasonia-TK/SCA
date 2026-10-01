"""BKM 1D/2D設定スキーマ（ノートブックの設定セルと同一構造・同一既定値）。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class WaveformConfig(CoreModel):
    """駆動波形設定。csvモードはAPI層でwaveform_id -> csv_textに解決される。"""

    mode: Literal["csv", "sinusoid", "scaled_wafer"] = "sinusoid"
    # --- csvモード ---
    waveform_id: Optional[int] = None  # DB登録波形への参照（アプリ用）
    csv_text: Optional[str] = None  # 解決済みCSV本文（runnerが使用）
    delimiter: str = ","
    skip_header_rows: int = 1
    x_column: int = 0
    voltage_column: int = 1
    x_axis: Literal["phase_deg", "phase_rad", "time_s", "time_ns", "time_us"] = "time_s"
    voltage_scale: float = 1.0
    voltage_offset_V: float = 0.0
    phase_offset_deg: float = 0.0
    # --- sinusoidモード: Ve = dc + amp*cos(phase + offset) ---
    sinusoid_dc_V: float = -170.0
    sinusoid_amplitude_V: float = 158.0
    sinusoid_phase_offset_deg: float = 0.0
    # --- scaled_waferモード（2Dリング専用） ---
    wafer_scale: float = 1.0
    wafer_phase_offset_deg: float = 0.0
    dc_offset_V: float = 0.0


class PlasmaConfig(CoreModel):
    frequency_Hz: float = 13.56e6
    electron_temperature_eV: float = 3.0
    sheath_edge_density_m3: float = 1.0e16
    ion_mass_amu: float = 39.948


class Circuit1DConfig(CoreModel):
    powered_to_grounded_area_ratio: float = 0.30
    grounded_electrode_voltage_V: float = 0.0
    phase_points: int = 2048
    max_cycles: int = 200
    periodic_tolerance_V: float = 1.0e-8
    capacitance_factor: float = 1.0


class SheathConfig(CoreModel):
    model: Literal["moving_front", "static_width"] = "moving_front"
    front_width_exponent: float = 0.75
    potential_exponent: float = 4.0 / 3.0


class GasConfig(CoreModel):
    gas_temperature_K: float = 300.0
    pressures_mTorr: list[float] = Field(default_factory=lambda: [0.0, 5.0, 20.0])
    cross_section_source: Literal["lxcat_phelps", "approximation"] = "lxcat_phelps"
    xsec_csv_name: str = "xsec_ar_ion_phelps_lxcat.csv"  # data/ 内のファイル名
    elastic_to_cx_ratio: float = 0.5
    cross_section_scale: float = 1.0


class TpmcConfig(CoreModel):
    n_particles: int = 30_000
    ion_temperature_eV: float = 0.05
    steps_per_rf_period: int = 400
    max_rf_periods: float = 60.0
    seed: int = 20260821
    max_recommended_collision_probability: float = 0.05


class MagneticFieldConfig(CoreModel):
    """静磁場 [T]（イオンにのみ作用。全成分0で無効=従来と同一経路）。

    座標系:
      1D: x=シース深さ方向（イオン入射方向）、y=面内横方向（IADF角度の基準）、z=第3軸
      2D: x=横方向（ジオメトリのx）、y=鉛直上向き、z=面外（紙面手前）
    """

    bx_T: float = 0.0
    by_T: float = 0.0
    bz_T: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.bx_T != 0.0 or self.by_T != 0.0 or self.bz_T != 0.0

    @property
    def magnitude(self) -> float:
        return float((self.bx_T**2 + self.by_T**2 + self.bz_T**2) ** 0.5)


class PlotConfig(CoreModel):
    energy_bins: int = 200
    angle_bins: int = 120
    # 角度刻み [deg]。指定時はangle_binsより優先（ビン数 = 180/刻み）
    angle_step_deg: Optional[float] = None
    energy_max_eV: Optional[float] = None


class Config1D(CoreModel):
    waveform: WaveformConfig = Field(default_factory=WaveformConfig)
    plasma: PlasmaConfig = Field(default_factory=PlasmaConfig)
    circuit: Circuit1DConfig = Field(default_factory=Circuit1DConfig)
    sheath: SheathConfig = Field(default_factory=SheathConfig)
    gas: GasConfig = Field(default_factory=GasConfig)
    tpmc: TpmcConfig = Field(default_factory=TpmcConfig)
    magnetic: MagneticFieldConfig = Field(default_factory=MagneticFieldConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)


# ---------------- 2D ----------------


class ElectrodesConfig(CoreModel):
    wafer_to_ground_area_ratio: float = 0.24
    ring_to_ground_area_ratio: float = 0.06
    ground_voltage_V: float = 0.0


class Circuit2DConfig(CoreModel):
    phase_points: int = 2048
    max_cycles: int = 200
    periodic_tolerance_V: float = 1.0e-8
    capacitance_factor: float = 1.0


MATERIALS = ("wafer", "ring", "insulator")


def _default_geometry_points() -> list[list[float]]:
    return [
        [0.0, 0.25e-3],
        [2.9e-3, 0.25e-3],
        [3.1e-3, 0.45e-3],
        [12.9e-3, 0.45e-3],
        [13.1e-3, 0.25e-3],
        [16.0e-3, 0.25e-3],
    ]


def _default_geometry_materials() -> list[str]:
    return ["ring", "wafer", "wafer", "wafer", "ring"]


class GeometryConfig(CoreModel):
    """ジオメトリv2: 非周期領域 [0, domain_length]、左右壁は鏡像対称。

    表面は制御点折れ線 s(x)。隣接制御点間の各セグメントに境界材質を割り当てる:
      wafer     = ウェハ電位 V_w(t) のDirichlet
      ring      = エッジリング電位 V_r(t) のDirichlet
      insulator = 非帯電誘電体（電位固定なし、粒子は吸収・記録）
    """

    domain_length_m: float = 16.0e-3
    # 制御点 [[x_m, y_m], ...] x昇順。先頭はx=0、末尾はx=domain_length（自動補完）
    points_m: list[list[float]] = Field(default_factory=_default_geometry_points)
    # セグメント材質（len == len(points_m) - 1）
    segment_materials: list[str] = Field(default_factory=_default_geometry_materials)
    # 表面平滑化幅 [m]（0で平滑化なし）
    smoothing_m: float = 0.0
    top_clearance_factor: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data):
        """旧周期ジオメトリ（step/profileモード）を自動変換する。"""
        if not isinstance(data, dict) or "points_m" in data:
            return data
        if "periodic_length_m" not in data and "wafer_left_m" not in data:
            return data
        length = float(data.get("periodic_length_m", 16.0e-3))
        left = float(data.get("wafer_left_m", 3.0e-3))
        right = float(data.get("wafer_right_m", 13.0e-3))
        wafer_h = float(data.get("wafer_height_m", 0.45e-3))
        ring_h = float(data.get("ring_height_m", 0.25e-3))
        width = max(float(data.get("step_smoothing_width_m", 0.1e-3)), 0.02e-3)
        mode = data.get("surface_mode", "step")
        if mode == "profile" and data.get("profile_points_m"):
            points = sorted([[float(p[0]), float(p[1])] for p in data["profile_points_m"]])
            if points[0][0] > 0.0:
                points.insert(0, [0.0, points[0][1]])
            if points[-1][0] < length:
                points.append([length, points[-1][1]])
            smoothing = float(data.get("profile_smoothing_m", 0.0))
        else:
            points = [
                [0.0, ring_h],
                [max(left - width, 0.0), ring_h],
                [min(left + width, length), wafer_h],
                [max(right - width, 0.0), wafer_h],
                [min(right + width, length), ring_h],
                [length, ring_h],
            ]
            points = sorted({(p[0], p[1]) for p in points})
            points = [[p[0], p[1]] for p in points]
            smoothing = 0.0
        materials = []
        for i in range(len(points) - 1):
            mid = 0.5 * (points[i][0] + points[i + 1][0])
            materials.append("wafer" if left <= mid <= right else "ring")
        return {
            "domain_length_m": length,
            "points_m": points,
            "segment_materials": materials,
            "smoothing_m": smoothing,
            "top_clearance_factor": float(data.get("top_clearance_factor", 1.0)),
        }

    @model_validator(mode="after")
    def _normalize(self) -> "GeometryConfig":
        length = self.domain_length_m
        if length <= 0:
            raise ValueError("domain_length_mは正の値にしてください。")
        if len(self.points_m) < 2:
            raise ValueError("制御点は2点以上必要です。")
        points = sorted([[float(p[0]), float(p[1])] for p in self.points_m], key=lambda p: p[0])
        for point in points:
            if len(point) != 2:
                raise ValueError("points_mは[x, y]の組で指定してください。")
            if not (0.0 <= point[0] <= length):
                raise ValueError("制御点xは[0, domain_length_m]の範囲で指定してください。")
            if point[1] <= 0.0:
                raise ValueError("制御点の高さyは正の値にしてください。")
        materials = list(self.segment_materials)
        # 端点をx=0/Lへ自動補完（材質は端のセグメントを引き継ぐ）
        if points[0][0] > 0.0:
            points.insert(0, [0.0, points[0][1]])
            materials.insert(0, materials[0] if materials else "ring")
        if points[-1][0] < length:
            points.append([length, points[-1][1]])
            materials.append(materials[-1] if materials else "ring")
        if len(materials) != len(points) - 1:
            raise ValueError(
                f"segment_materialsはセグメント数({len(points) - 1})と"
                f"同数にしてください（現在{len(materials)}個）。"
            )
        for material in materials:
            if material not in MATERIALS:
                raise ValueError(f"未知の材質: {material}（wafer/ring/insulator）")
        if "wafer" not in materials:
            raise ValueError("waferセグメントが少なくとも1つ必要です。")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "segment_materials", materials)
        return self


class Field2DConfig(CoreModel):
    nx: int = 384
    ny: int = 176
    sor_omega: float = 1.94
    tolerance: float = 1.0e-7
    max_iterations: int = 30_000


class SpaceChargeConfig(CoreModel):
    enabled: bool = True
    outer_iterations: int = 5
    deposition_particles: int = 25_000
    under_relaxation: float = 0.35
    density_smoothing_sigma_cells: float = 1.5
    ion_density_clip_factor: float = 2.0
    electron_phase_samples: int = 32
    max_abs_correction_V: float = 250.0
    poisson_tolerance_V: float = 5.0e-3
    poisson_max_iterations: int = 20_000


class Tpmc2DConfig(CoreModel):
    n_particles: int = 40_000
    ion_temperature_eV: float = 0.05
    steps_per_rf_period: int = 400
    max_rf_periods: float = 40.0
    seed: int = 20260821
    max_recommended_collision_probability: float = 0.05


class Gas2DConfig(GasConfig):
    pressures_mTorr: list[float] = Field(default_factory=lambda: [15.0])


class AnalysisConfig(CoreModel):
    edge_exclusion_m: float = 0.25e-3
    edge_band_m: float = 0.60e-3
    bin_width_m: float = 0.25e-3
    max_distance_m: float = 5.0e-3
    affected_threshold_deg: float = 1.0
    energy_bins: int = 160
    angle_bins: int = 120
    # 角度刻み [deg]。指定時はangle_binsより優先（ビン数 = 180/刻み）
    angle_step_deg: Optional[float] = None


class Config2D(CoreModel):
    wafer_waveform: WaveformConfig = Field(default_factory=lambda: WaveformConfig())
    ring_waveform: WaveformConfig = Field(default_factory=lambda: WaveformConfig())
    plasma: PlasmaConfig = Field(default_factory=PlasmaConfig)
    electrodes: ElectrodesConfig = Field(default_factory=ElectrodesConfig)
    circuit: Circuit2DConfig = Field(default_factory=Circuit2DConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    field2d: Field2DConfig = Field(default_factory=Field2DConfig)
    space_charge: SpaceChargeConfig = Field(default_factory=SpaceChargeConfig)
    gas: Gas2DConfig = Field(default_factory=Gas2DConfig)
    tpmc: Tpmc2DConfig = Field(default_factory=Tpmc2DConfig)
    magnetic: MagneticFieldConfig = Field(default_factory=MagneticFieldConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
