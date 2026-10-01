"""Run local saturation/limit examples through the real scheduler."""

import json
import time
from pathlib import Path

import httpx

from memory_hole.config import CaseConfig, SaturationCriteria

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8765"


def main():
    c = CaseConfig(mode="self_consistent")
    c.numerics.run_until = "saturation"
    c.numerics.samples_per_species = 40
    c.numerics.phase_bins = 1
    c.numerics.save_every_steps = 3
    c.initial_sigma_c_m2 = 1e-4
    c.leakage_tau_s = 5e-7
    c.ions[0].flux_m2_s = 0
    c.electron.flux_m2_s = 0
    c.numerics.saturation = SaturationCriteria(
        max_time_s=20e-6,
        window_s=1e-6,
        steps_per_window=2,
        min_windows=2,
        consecutive_windows=2,
        charge_relative_tolerance=0,
        charge_absolute_tolerance_c_m2=1e-6,
        voltage_tolerance_v=0.1,
    )
    c.name = "飽和判定検証 · 漏れによる減衰 · CPU"
    limited = c.model_copy(deep=True)
    limited.name = "飽和判定検証 · 最大時間で終了 · CPU"
    limited.numerics.saturation.max_time_s = 1.5e-6
    preset = ROOT / "data/preset.json"
    cap = (
        CaseConfig.model_validate_json(preset.read_text(encoding="utf-8"))
        if preset.exists()
        else c.model_copy(deep=True)
    )
    cap.mode = "self_consistent"
    cap.numerics.run_until = "saturation"
    cap.numerics.saturation = SaturationCriteria(max_updates=3)
    cap.numerics.samples_per_species = 100
    cap.numerics.representative_trajectories = 12
    cap.name = "飽和判定検証 · 入力ケースの更新上限"
    reports = {}
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        cap.numerics.backend = "gpu" if client.get("/api/system").json()["gpu"]["available"] else "cpu"
        for config, reason in [(c, "saturated"), (limited, "time_limit"), (cap, "update_limit")]:
            response = client.post("/api/jobs", json=config.model_dump())
            response.raise_for_status()
            job_id = response.json()["id"]
            deadline = time.monotonic() + 180
            while True:
                status = client.get(f"/api/jobs/{job_id}").json()
                if status["status"] in {"completed", "failed", "paused"}:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Job {job_id} is still active")
                time.sleep(0.25)
            assert status["status"] == "completed", status
            assert status["termination_reason"] == reason, status
            result = client.get(f"/api/jobs/{job_id}/results").json()
            assert result["termination_reason"] == reason
            assert result["saturation"]["saturated"] == (reason == "saturated")
            reports[reason] = {
                "job_id": job_id,
                "name": config.name,
                "time_s": status["time_s"],
                "steps": status["steps"],
                "backend": config.numerics.backend,
                "saturation": result["saturation"],
            }
            print(f"{reason}: {job_id}, time={status['time_s']:.6e} s, updates={status['steps']}", flush=True)
    qa = ROOT / "data/qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "saturation_api_check.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
