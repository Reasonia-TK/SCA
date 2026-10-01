"""IAEDF job service; shares SCA's single-worker budget and storage retry policy."""

from __future__ import annotations

import logging
import multiprocessing
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .geometry import derive
from .storage import atomic_json, read_json, validate_data_directory
from .upstream import (
    ApplyRequest,
    UpstreamRequest,
    connected_case,
    defaults,
    distribution_plot,
    run_upstream,
    validated_config,
)

logger = logging.getLogger(__name__)


class UpstreamService:
    def __init__(self, data_root, distributions, read_status):
        self.root = Path(data_root) / "iaedf_jobs"
        self.distributions = Path(distributions)
        self.read_status = read_status
        self.workers = {}
        self.startup_jobs = set()
        self.warned = set()
        self.router = APIRouter(prefix="/api/iaedf", tags=["IAEDF"])
        self.routes()

    def directory(self, jid):
        directory = self.root / jid
        if len(jid) != 32 or not jid.isalnum() or not directory.is_dir():
            raise HTTPException(404, "IAEDFジョブが見つかりません。")
        return directory

    def completed(self, jid):
        directory = self.directory(jid)
        if self.read_status(directory)["status"] != "completed":
            raise HTTPException(409, "完了したIAEDFジョブを選んでください。")
        return directory

    def start(self):
        validate_data_directory(self.root)
        self.startup_jobs = {
            p.name for p in self.root.iterdir() if p.is_dir() and len(p.name) == 32 and p.name.isalnum()
        }

    def warn(self, jid, error):
        if jid not in self.warned:
            logger.warning("IAEDFジョブ %s の状態を保存できません。再試行します: %s", jid, error)
            self.warned.add(jid)

    def active(self):
        return any(p.is_alive() for p in self.workers.values())

    def tick(self, can_launch=True):
        for jid in list(self.startup_jobs):
            directory = self.directory(jid)
            status = self.read_status(directory)
            if status["status"] == "unavailable":
                continue
            try:
                if status["status"] in {"running", "cancelling"}:
                    atomic_json(
                        directory / "status.json",
                        dict(status="failed", stage="前回のIAEDF計算が中断しました。再計算できます。"),
                    )
            except OSError as error:
                self.warn(jid, error)
                continue
            self.startup_jobs.discard(jid)
            self.warned.discard(jid)
        for jid, proc in list(self.workers.items()):
            if proc.is_alive():
                continue
            proc.join()
            directory = self.directory(jid)
            status = self.read_status(directory)
            if status["status"] == "unavailable":
                continue
            try:
                if status["status"] in {"queued", "running", "cancelling"}:
                    cancelled = (directory / "cancel.request").exists()
                    atomic_json(
                        directory / "status.json",
                        dict(
                            status="cancelled" if cancelled else "failed",
                            stage="計算を中止しました"
                            if cancelled
                            else "ワーカーが終了しました。error.logを確認してください。",
                        ),
                    )
            except OSError as error:
                self.warn(jid, error)
                continue
            del self.workers[jid]
            self.warned.discard(jid)
        if can_launch and not self.active():
            for directory in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime):
                if (
                    directory.is_dir()
                    and len(directory.name) == 32
                    and directory.name.isalnum()
                    and directory.name not in self.workers
                    and directory.name not in self.startup_jobs
                    and self.read_status(directory)["status"] == "queued"
                ):
                    proc = multiprocessing.get_context("spawn").Process(
                        target=run_upstream, args=(str(directory),)
                    )
                    proc.start()
                    self.workers[directory.name] = proc
                    break

    def shutdown(self):
        for jid, proc in list(self.workers.items()):
            directory = self.directory(jid)
            (directory / "cancel.request").touch()
            proc.join(2)
            if proc.is_alive():
                proc.terminate()
                proc.join(3)
            status = self.read_status(directory)
            if status["status"] in {"queued", "running", "cancelling"}:
                atomic_json(
                    directory / "status.json",
                    dict(status="cancelled", stage="サーバー終了でIAEDF計算を中止しました。再計算できます。"),
                )
        self.workers.clear()

    def routes(self):
        router = self.router

        @router.get("/defaults")
        def get_defaults():
            return defaults()

        @router.post("/validate")
        def validate(request: UpstreamRequest):
            request.config = validated_config(request.model, request.config).model_dump()
            return request.model_dump()

        @router.post("/jobs")
        def submit(request: UpstreamRequest):
            config = validated_config(request.model, request.config)
            request.config = config.model_dump()
            jid = uuid.uuid4().hex
            directory = self.root / jid
            directory.mkdir(parents=True)
            atomic_json(directory / "request.json", request.model_dump())
            atomic_json(
                directory / "status.json",
                dict(status="queued", stage="IAEDF待機中", created_at=time.time(), progress=0),
            )
            return {"id": jid}

        @router.get("/jobs")
        def jobs():
            rows = []
            if not self.root.exists():
                return rows
            for directory in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if directory.is_dir() and (directory / "request.json").exists():
                    request = read_json(directory / "request.json")
                    rows.append(
                        dict(
                            id=directory.name,
                            name=request["name"],
                            model=request["model"],
                            **self.read_status(directory),
                        )
                    )
            return rows

        @router.get("/jobs/{jid}")
        def job(jid: str):
            directory = self.directory(jid)
            data = dict(id=jid, request=read_json(directory / "request.json"), **self.read_status(directory))
            if data["status"] == "completed":
                data["summary"] = read_json(directory / "summary.json")
                data["plots"] = read_json(directory / "plots.json")
            return data

        @router.post("/jobs/{jid}/cancel")
        def cancel(jid: str):
            directory = self.directory(jid)
            status = self.read_status(directory)
            if status["status"] not in {"queued", "running", "cancelling"}:
                raise HTTPException(409, "待機中または実行中のIAEDFのみ中止できます。")
            (directory / "cancel.request").touch()
            proc = self.workers.get(jid)
            if proc is not None:
                proc.terminate()
                proc.join(3)
            atomic_json(
                directory / "status.json", dict(status, status="cancelled", stage="計算を中止しました")
            )
            return {"ok": True}

        @router.get("/jobs/{jid}/distribution")
        def plot(
            jid: str,
            pressure_index: int = Query(0, ge=0),
            collector_min_m: float | None = Query(None, allow_inf_nan=False),
            collector_max_m: float | None = Query(None, allow_inf_nan=False),
        ):
            return distribution_plot(self.completed(jid), pressure_index, collector_min_m, collector_max_m)

        @router.post("/jobs/{jid}/apply")
        def apply(jid: str, request: ApplyRequest):
            directory = self.completed(jid)
            config, data, metadata = connected_case(directory, request)
            derive(config)
            sid = uuid.uuid4().hex
            root = self.distributions / sid
            self.distributions.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root.with_suffix(".npz"), **data)
            atomic_json(root.with_suffix(".source.json"), read_json(directory / "config.json"))
            atomic_json(root.with_suffix(".json"), metadata)
            config.ions[0].distribution_id = sid
            return {"config": config.model_dump(), "id": sid, "metadata": metadata}

        @router.get("/jobs/{jid}/export/{kind}")
        def export(jid: str, kind: str):
            directory = self.directory(jid)
            if kind not in {"raw", "config", "summary", "error", "request"}:
                raise HTTPException(404, "出力形式が不正です。")
            filename = {
                "raw": "raw.npz",
                "config": "config.json",
                "summary": "summary.json",
                "error": "error.log",
                "request": "request.json",
            }[kind]
            if not (directory / filename).exists():
                raise HTTPException(404, "出力ファイルがまだありません。")
            return FileResponse(directory / filename, filename=f"IAEDF-{jid}-{filename}")
