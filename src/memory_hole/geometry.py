from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .config import CaseConfig

NM = 1e-9


def derive(config: CaseConfig) -> dict:
    g = config.geometry
    substrate = g.carbon_thickness_nm + g.oxide_thickness_nm
    depth = substrate if g.depth_mode == "linked" else g.depth_nm
    if depth > substrate + 1e-7:
        raise ValueError("Channel深さが基板上面を超えています。")
    if g.profile:
        z = np.array([p.depth_nm for p in g.profile])
        r = np.array([p.radius_nm for p in g.profile])
        if len(z) < 2 or z[0] != 0 or abs(z[-1] - depth) > 1e-6 or np.any(np.diff(z) <= 0):
            raise ValueError("半径プロファイルは入口0からChannel深さまで、深さを昇順で指定してください。")
        if abs(r[0] * 2 - g.channel_diameter_nm) > 1e-6:
            raise ValueError("プロファイルの入口半径とChannel径が一致しません。")
    else:
        z = np.array([0, depth])
        r = np.array([g.channel_diameter_nm / 2, (g.bottom_diameter_nm or g.channel_diameter_nm) / 2])
    if g.gap_evaluation_depth_nm > depth:
        raise ValueError("間隔の評価深さがChannelの外です。")
    if len(g.dummies) > g.dummy_count:
        raise ValueError("個別dummy設定の数がdummy個数を超えています。")
    eval_r = float(np.interp(g.gap_evaluation_depth_nm, z, r))
    metal_z = z[(z >= g.carbon_thickness_nm) & (z <= substrate)]
    max_r = max([float(np.interp(g.carbon_thickness_nm, z, r)), *np.interp(metal_z, z, r)])
    dummies = []
    for k in range(g.dummy_count):
        ov = g.dummies[k] if k < len(g.dummies) else None
        diameter = ov.diameter_nm if ov and ov.diameter_nm is not None else g.dummy_diameter_nm
        angle = (
            ov.angle_deg if ov and ov.angle_deg is not None else g.start_angle_deg + k * 360 / g.dummy_count
        )
        gap = ov.gap_nm if ov and ov.gap_nm is not None else g.gap_nm
        radius = eval_r + diameter / 2 + gap
        if g.placement_mode == "radial":
            if ov is None or ov.radius_nm is None:
                raise ValueError(f"dummy {k + 1}: 中心距離を指定してください。")
            radius = ov.radius_nm
        if g.placement_mode == "xy":
            if ov is None or ov.x_nm is None or ov.y_nm is None:
                raise ValueError(f"dummy {k + 1}: x,y座標を指定してください。")
            x, y = ov.x_nm, ov.y_nm
            radius, angle = math.hypot(x, y), math.degrees(math.atan2(y, x))
        else:
            x, y = radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))
        top = ov.top_nm if ov and ov.top_nm is not None else g.carbon_thickness_nm
        bottom = ov.bottom_nm if ov and ov.bottom_nm is not None else substrate
        if abs(top - g.carbon_thickness_nm) > 1e-6 or abs(bottom - substrate) > 1e-6:
            raise ValueError(f"dummy {k + 1}: Carbon下面と基板へ接続していません。浮遊導体は未対応です。")
        minimum_gap = radius - diameter / 2 - max_r
        if minimum_gap <= 1e-6:
            raise ValueError(f"dummy {k + 1}: Channelと接触・重複しています。最小間隔を正にしてください。")
        actual_gap = radius - diameter / 2 - eval_r
        dummies.append(
            dict(
                index=k,
                x_nm=x,
                y_nm=y,
                diameter_nm=diameter,
                angle_deg=angle,
                radius_nm=radius,
                gap_nm=actual_gap,
                minimum_gap_nm=minimum_gap,
                displacement_nm=actual_gap - g.reference_gap_nm,
                top_nm=top,
                bottom_nm=bottom,
                length_nm=bottom - top,
                conductor_id="wafer",
            )
        )
    min_pair = math.inf
    for i, a in enumerate(dummies):
        for b in dummies[i + 1 :]:
            clearance = (
                math.hypot(a["x_nm"] - b["x_nm"], a["y_nm"] - b["y_nm"])
                - (a["diameter_nm"] + b["diameter_nm"]) / 2
            )
            if clearance <= 1e-6:
                raise ValueError(f"dummy {a['index'] + 1}と{b['index'] + 1}が接触・重複しています。")
            min_pair = min(min_pair, clearance)
    extent = max(max(r), *[math.hypot(d["x_nm"], d["y_nm"]) + d["diameter_nm"] / 2 for d in dummies])
    half_width = g.domain_half_width_nm or extent + g.lateral_margin_nm
    for d in dummies:
        if max(abs(d["x_nm"]), abs(d["y_nm"])) + d["diameter_nm"] / 2 >= half_width:
            raise ValueError("dummyが計算領域の外側境界に接触しています。領域を広げてください。")
    if max(r) >= half_width or g.injection_z_nm < -g.entrance_height_nm:
        raise ValueError("Channelまたは注入面が計算領域に含まれていません。")
    injection_r = g.injection_radius_nm or r[0]
    if injection_r >= half_width or (g.injection_z_nm == 0 and injection_r > r[0] + 1e-6):
        raise ValueError("注入面の半径が粒子領域を超えています。")
    min_gap = min([d["minimum_gap_nm"] for d in dummies] + ([min_pair] if math.isfinite(min_pair) else []))
    warnings = []
    if g.interface_mesh_size_nm > min_gap / 2:
        warnings.append("最小SiO₂間隔に対して界面メッシュが粗い設定です。メッシュ収束を確認してください。")
    result = dict(
        depth_nm=depth,
        carbon_bottom_nm=g.carbon_thickness_nm,
        substrate_top_nm=substrate,
        bottom_material="substrate"
        if abs(depth - substrate) < 1e-6
        else ("carbon" if depth <= g.carbon_thickness_nm else "oxide"),
        profile=[dict(depth_nm=float(a), radius_nm=float(b)) for a, b in zip(z, r, strict=True)],
        dummies=dummies,
        domain_half_width_nm=half_width,
        envelope_diameter_nm=2 * extent,
        minimum_dummy_gap_nm=min_pair if math.isfinite(min_pair) else None,
        minimum_clearance_nm=min_gap,
        injection_radius_nm=float(injection_r),
        injection_area_m2=math.pi * (injection_r * NM) ** 2,
        warnings=warnings,
    )
    result["geometry_id"] = hashlib.sha256(
        json.dumps({"input": g.model_dump(), "derived": result}, sort_keys=True).encode()
    ).hexdigest()
    return result
