"""イオン-中性衝突断面積（LXCat Phelps評価済みデータ / 従来近似）。"""

from __future__ import annotations

import io

import numpy as np

from .schemas import GasConfig


def sigma_cx_approx_factory(scale: float):
    def sigma_cx_approx(E_cm_eV):
        energy = np.clip(np.asarray(E_cm_eV, dtype=float), 1.0e-3, 1.0e5)
        diameter_A = np.maximum(8.79 - 0.50 * np.log(energy), 0.0)
        return scale * (diameter_A * 1.0e-10) ** 2

    return sigma_cx_approx


def load_lxcat_ion_xsec_text(text: str):
    """LXCat形式CSV本文から backscat/isotropic のlog-log補間関数を作る。"""
    proc, en, sg = [], [], []
    for line in io.StringIO(text):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("process"):
            continue
        p, e, s = line.split(",")
        e, s = float(e), float(s)
        if e > 0.0 and s > 0.0:
            proc.append(p)
            en.append(e)
            sg.append(s)
    proc, en, sg = np.asarray(proc), np.asarray(en), np.asarray(sg)
    funcs = {}
    for kind in ("backscat", "isotropic"):
        mask = proc == kind
        if not np.any(mask):
            raise ValueError(f"断面積データに process={kind} がありません。")
        order = np.argsort(en[mask])
        log_e = np.log(en[mask][order])
        log_s = np.log(sg[mask][order])
        e_lo, e_hi = en[mask][order][0], en[mask][order][-1]

        def sigma(E_eV, _le=log_e, _ls=log_s, _lo=e_lo, _hi=e_hi):
            E = np.clip(np.asarray(E_eV, dtype=float), _lo, _hi)
            return np.exp(np.interp(np.log(E), _le, _ls))

        funcs[kind] = sigma
    return funcs


def build_cross_sections(gas: GasConfig, xsec_text: str | None):
    """設定に応じて (SIGMA_CX, SIGMA_EL) を返す。"""
    if gas.cross_section_source == "lxcat_phelps":
        if xsec_text is None:
            raise ValueError("lxcat_phelpsには断面積CSV本文が必要です。")
        funcs = load_lxcat_ion_xsec_text(xsec_text)
        return funcs["backscat"], funcs["isotropic"]
    if gas.cross_section_source == "approximation":
        sigma_cx = sigma_cx_approx_factory(gas.cross_section_scale)
        ratio = max(gas.elastic_to_cx_ratio, 0.0)

        def sigma_el(E_eV):
            return ratio * sigma_cx(E_eV)

        return sigma_cx, sigma_el
    raise ValueError("cross_section_sourceはlxcat_phelps/approximation")
