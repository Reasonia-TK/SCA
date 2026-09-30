"""Check stop/resume through the actual process scheduler, not just the engine."""

import json
import time
from pathlib import Path

import httpx

from memory_hole.engine import atomic_json


def wait_status(client, jid, desired, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{jid}").json()
        if status["status"] == "failed":
            raise RuntimeError(status["stage"])
        if status["status"] in desired:
            return status
        time.sleep(0.1)
    raise TimeoutError(jid)


def main():
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=30) as client:
        c = client.get("/api/defaults").json()["config"]
        c["name"] = "停止再開の検証 · CPU"
        c["mode"] = "self_consistent"
        c["numerics"]["backend"] = "cpu"
        c["numerics"]["charging_steps"] = 8
        c["numerics"]["duration_s"] = 1e-4
        response = client.post("/api/jobs", json=c)
        response.raise_for_status()
        jid = response.json()["id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = client.get(f"/api/jobs/{jid}").json()
            if status["status"] in {"completed", "failed"}:
                raise RuntimeError("Job ended before pause test: " + status["stage"])
            if status.get("time_s", 0) > 0:
                break
            time.sleep(0.05)
        client.post(f"/api/jobs/{jid}/pause", json={}).raise_for_status()
        wait_status(client, jid, {"paused"})
        saved = client.get(f"/api/jobs/{jid}/results").json()
        assert 0 < saved["time_s"] < c["numerics"]["duration_s"]
        for _ in range(20):
            response = client.post(f"/api/jobs/{jid}/resume", json={})
            if response.is_success:
                break
            time.sleep(0.1)
        response.raise_for_status()
        wait_status(client, jid, {"completed"})
        resumed = client.get(f"/api/jobs/{jid}/results").json()
        c["name"] = "連続実行の検証 · CPU"
        response = client.post("/api/jobs", json=c)
        response.raise_for_status()
        control = response.json()["id"]
        wait_status(client, control, {"completed"})
        continuous = client.get(f"/api/jobs/{control}/results").json()
        assert resumed["ledger"] == continuous["ledger"]
        assert resumed["statistics"] == continuous["statistics"]
        assert resumed["surface"]["sigma_c_m2"] == continuous["surface"]["sigma_c_m2"]
        summary = {
            "resumed_job": jid,
            "control_job": control,
            "paused_at_s": saved["time_s"],
            "final_time_s": resumed["time_s"],
            "sigma": "exact match",
            "ledger": "exact match",
            "statistics": "exact match",
        }
        atomic_json(Path("data/qa/resume_api_check.json"), summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
