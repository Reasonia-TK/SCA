"""Compare a read-only IAEDF-Sim source snapshot with the bundled model."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from memory_hole.iaedf_core.schemas import Config1D, Config2D
from memory_hole.upstream import ASSETS, clean_json


def compare(original, migrated, path="output"):
    ignored = {"tpmc_seconds", "elapsed_s", "basis_seconds"}
    if isinstance(original, dict):
        for key, value in original.items():
            if key not in ignored:
                compare(value, migrated[key], f"{path}.{key}")
    elif isinstance(original, list):
        assert len(original) == len(migrated), path
        for i, (a, b) in enumerate(zip(original, migrated, strict=True)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(original, np.ndarray):
        np.testing.assert_array_equal(original, migrated, err_msg=path)
    elif isinstance(original, (float, np.floating)):
        assert original == migrated or (np.isnan(original) and np.isnan(migrated)), path
    elif hasattr(original, "__dataclass_fields__"):
        for name in original.__dataclass_fields__:
            compare(getattr(original, name), getattr(migrated, name), f"{path}.{name}")
    else:
        assert original == migrated, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="IAEDF-Sim project directory (read-only)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    backend = args.source.resolve() / "backend"
    if not (backend / "bkmcore" / "schemas.py").is_file():
        parser.error("IAEDF-Sim backend/bkmcore が見つかりません。")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(backend))
    source_schema = importlib.import_module("bkmcore.schemas")
    xsec = (ASSETS / "xsec_ar_ion_phelps_lxcat.csv").read_text(encoding="utf-8")
    csv = (ASSETS / "tailored_waveform_5harmonic.csv").read_text(encoding="utf-8-sig")
    results = []
    for model, waveform, magnetic, space_charge in [
        ("1d", "sinusoid", False, False),
        ("1d", "csv", True, False),
        ("2d", "sinusoid", False, False),
        ("2d", "csv", True, True),
    ]:
        config = Config1D() if model == "1d" else Config2D()
        config.tpmc.n_particles = 1000
        config.gas.pressures_mTorr = [0, 10]
        if magnetic:
            config.magnetic.bz_T = 0.01
        if model == "2d":
            config.field2d.nx, config.field2d.ny = 65, 40
            config.space_charge.enabled = space_charge
            config.space_charge.outer_iterations = 2
            config.space_charge.deposition_particles = 800
        if waveform == "csv":
            waves = [config.waveform] if model == "1d" else [config.wafer_waveform, config.ring_waveform]
            for wave in waves:
                wave.mode, wave.csv_text = "csv", csv
        config = type(config).model_validate(config.model_dump())
        runner_name = f"run_{model}"
        old_runner = getattr(importlib.import_module(f"bkmcore.model{model}.runner"), runner_name)
        new_runner = getattr(
            importlib.import_module(f"memory_hole.iaedf_core.model{model}.runner"), runner_name
        )
        source_class = getattr(source_schema, "Config1D" if model == "1d" else "Config2D")
        a = old_runner(source_class.model_validate(config.model_dump()), xsec_text=xsec)
        b = new_runner(config, xsec_text=xsec)
        compare(a, b)
        entry = dict(
            model=model,
            waveform=waveform,
            magnetic=magnetic,
            space_charge=space_charge,
            pressure_cases=2,
            original_outputs="exactly_equal",
            validation=b["validation"],
        )
        results.append(entry)
        print(f"PASS: {model}, waveform={waveform}, B={magnetic}, space_charge={space_charge}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(clean_json(results), ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
