"""Replay a validation case and inspect unresolved particles at larger limits."""

import json
from pathlib import Path

import numpy as np

from memory_hole.config import CaseConfig
from memory_hole.engine import atomic_json, run_job
from memory_hole.gpu import GPUTracer


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/qa/numerical_validation_20261001"
    source = output / "jobs/steps_16_seed_161803/config.json"
    c = CaseConfig.model_validate_json(source.read_text(encoding="utf-8"))
    c.name = "未解決粒子の原因調査 · seed 161803"
    directory = output / "unresolved_diagnostic"
    directory.mkdir(exist_ok=True)
    if (directory / "checkpoint.h5").exists():
        raise RuntimeError("Diagnostic already exists; inspect unresolved_diagnostic.json")
    atomic_json(directory / "config.json", c.model_dump())
    records = []
    original = GPUTracer.__call__

    def inspect(tracer, *args):
        result = original(tracer, *args)
        missing = result[2] == 3
        if np.any(missing):
            record = {
                "count": int(missing.sum()),
                "starts": args[2][missing].tolist(),
                "initial_positions_nm": (args[0][missing] * 1e9).tolist(),
                "flight_time_s": result[5][missing].tolist(),
                "end_positions_nm": (result[0][missing] * 1e9).tolist(),
                "retry": [],
            }
            retry_args = list(args)
            for index in range(5):
                retry_args[index] = args[index][missing]
            retry_args[15] = 0
            for limit in (5000, 10000, 20000):
                retry_args[11] = limit
                retry = original(tracer, *retry_args)
                record["retry"].append(
                    {
                        "max_steps": limit,
                        "status": retry[2].tolist(),
                        "patch": retry[3].tolist(),
                        "flight_time_s": retry[5].tolist(),
                    }
                )
            records.append(record)
        return result

    GPUTracer.__call__ = inspect
    try:
        run_job(str(directory), str(root / "data/distributions"))
    finally:
        GPUTracer.__call__ = original
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise RuntimeError(status["stage"])
    atomic_json(output / "unresolved_diagnostic.json", {"config": c.model_dump(), "records": records})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
