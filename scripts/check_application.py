"""Run the supplied inlet through the local API on CPU/GPU and charged mode."""

import json
import time
from pathlib import Path

import httpx
import numpy as np

from memory_hole.engine import atomic_json


def main():
    client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=30)
    baseline = client.get("/api/defaults").json()["config"]
    ids = []
    for backend, mode in [("gpu", "uncharged"), ("cpu", "uncharged"), ("gpu", "self_consistent")]:
        c = json.loads(json.dumps(baseline))
        c["name"] = f"提供IAEDF 10 mTorr · {backend.upper()} · {mode}"
        c["mode"] = mode
        c["numerics"]["backend"] = backend
        if mode == "self_consistent":
            c["numerics"]["duration_s"] = 1e-4
            c["numerics"]["charging_steps"] = 4
        response = client.post("/api/jobs", json=c)
        response.raise_for_status()
        ids.append(response.json()["id"])
    previous = {}
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        all_done = True
        for jid in ids:
            r = client.get("/api/jobs/" + jid).json()
            status = r["status"]
            key = (status, r.get("step"), r.get("time_s"))
            if previous.get(jid) != key:
                print(jid, status, r.get("step"), r.get("time_s"), flush=True)
                previous[jid] = key
            if status == "failed":
                raise RuntimeError(r["stage"])
            if status != "completed":
                all_done = False
        if all_done:
            break
        time.sleep(1)
    else:
        raise TimeoutError("Application jobs did not complete in 300 seconds")
    results = [client.get(f"/api/jobs/{jid}/results").json() for jid in ids]
    comparison = {}
    for species in results[0]["statistics"]:
        gpu, cpu = (r["statistics"][species] for r in results[:2])
        difference = abs(gpu["bottom_arrival"]["rate"] - cpu["bottom_arrival"]["rate"])
        assert difference < 1e-12
        comparison[species] = {
            "bottom_arrival_difference": difference,
            "rate": gpu["bottom_arrival"]["rate"],
            "interval": [gpu["bottom_arrival"]["low"], gpu["bottom_arrival"]["high"]],
        }
    max_residual = max(h["charge_residual"] for r in results for h in r["history"])
    assert max_residual < 1e-6
    for jid in ids:
        for kind in ("config", "csv", "hdf5", "checkpoint"):
            response = client.get(f"/api/jobs/{jid}/export/{kind}")
            response.raise_for_status()
            assert len(response.content) > 0
    for state in results[2]["saved_times"]:
        response = client.get(f"/api/jobs/{ids[2]}/results", params={"step": state["step"]})
        response.raise_for_status()
        assert response.json()["time_s"] == state["time_s"]
    charged = results[2]
    assert np.max(abs(np.array(charged["surface"]["sigma_c_m2"]))) > 0
    summary = {
        "job_ids": ids,
        "cpu_gpu": comparison,
        "max_charge_residual": max_residual,
        "charged_steps": len(charged["history"]),
        "charged_time_s": charged["time_s"],
        "maximum_sigma_c_m2": float(np.max(abs(np.array(charged["surface"]["sigma_c_m2"])))),
        "all_exports": "passed",
        "time_selection": "passed",
    }
    output = Path("data/qa/application_check.json")
    output.parent.mkdir(exist_ok=True)
    atomic_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
