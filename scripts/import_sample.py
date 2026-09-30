"""Import the supplied pair without modifying the source files."""

import uuid
from pathlib import Path

import numpy as np

from memory_hole.config import CaseConfig
from memory_hole.engine import atomic_json
from memory_hole.geometry import derive
from memory_hole.iaedf import case_from_pair


def main():
    root = Path(__file__).resolve().parents[1]
    payload = (root / "sample-20mTorr.npz").read_bytes()
    source = (root / "sample-20mTorr.json").read_bytes()
    c, data, meta = case_from_pair(
        payload, source, "sample-20mTorr.npz", CaseConfig(), projection_policy="zero_out_of_plane"
    )
    sid = uuid.uuid4().hex
    target = root / "data" / "distributions" / sid
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target.with_suffix(".npz"), **{k: v if v is not None else np.empty(0) for k, v in data.items()}
    )
    atomic_json(target.with_suffix(".json"), meta)
    target.with_suffix(".source.json").write_bytes(source)
    c.ions[0].distribution_id = sid
    c.numerics.phase_bins = 16
    c.numerics.samples_per_species = 2000
    c.numerics.backend = "gpu"
    atomic_json(root / "data" / "preset.json", c.model_dump())
    examples = root / "examples"
    examples.mkdir(exist_ok=True)
    atomic_json(
        examples / "sample_10mtorr_case.json",
        {"config": c.model_dump(), "derived": derive(c), "input_metadata": meta},
    )
    print(
        f"pressure={meta['pressure_mtorr']} mTorr, samples={meta['sample_count']}, reached={meta['reached_fraction']}"
    )
    print(f"wafer={c.waveform.dc_v} + {c.waveform.amplitude_v} sin(phase) V")
    print(f"ion flux estimate={c.ions[0].flux_m2_s:.6e} m^-2 s^-1")
    for warning in meta["warnings"]:
        print(warning)


if __name__ == "__main__":
    main()
