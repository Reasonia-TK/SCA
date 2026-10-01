from __future__ import annotations

import asyncio
import csv
import io
import itertools
import json
import logging
import multiprocessing
import os
import time
import uuid
import zipfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .config import CaseConfig
from .engine import SPACE_CHARGE_WARNING, atomic_json, gpu_info, run_job
from .geometry import derive
from .inlet import parse_distribution
from .section import saved_section
from .storage import permission_message, read_json, validate_data_directory

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MEMORY_HOLE_DATA", str(ROOT / "data"))).resolve()
JOBS = DATA / "jobs"
DISTRIBUTIONS = DATA / "distributions"
WORKERS = {}
STARTUP_JOBS = set()
logger = logging.getLogger(__name__)


def job_dir(job_id):
    if len(job_id) != 32 or not job_id.isalnum() or not (JOBS / job_id).is_dir():
        raise HTTPException(404, "ジョブが見つかりません。")
    return JOBS / job_id


def read_status(directory):
    try:
        return read_json(directory / "status.json")
    except PermissionError as error:
        return {"status": "unavailable", "stage": permission_message(error, directory / "status.json")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "status": "unavailable",
            "stage": "status.jsonが存在しないか、内容を読み取れません。保存先の状態を確認してください。",
        }


def scheduler_tick(warned):
    for jid in list(STARTUP_JOBS):
        directory = JOBS / jid
        status = read_status(directory)
        if status["status"] == "unavailable":
            if jid not in warned:
                logger.warning("ジョブ %s の起動時の状態を確認できません: %s", jid, status["stage"])
                warned.add(jid)
            continue
        if status["status"] == "running":
            try:
                atomic_json(
                    directory / "status.json",
                    dict(status="paused", stage="前回終了時のチェックポイントから再開可能"),
                )
            except OSError as error:
                if jid not in warned:
                    logger.warning("ジョブ %s の停止状態を保存できません: %s", jid, error)
                    warned.add(jid)
                continue
        STARTUP_JOBS.discard(jid)
        warned.discard(jid)
    for jid, proc in list(WORKERS.items()):
        if proc.is_alive():
            continue
        proc.join()
        directory = JOBS / jid
        try:
            status = read_status(directory)
            if status["status"] == "unavailable":
                if jid not in warned:
                    logger.warning("ジョブ %s の終了状態を確認できません: %s", jid, status["stage"])
                    warned.add(jid)
                continue
            if status["status"] in {"running", "queued"}:
                atomic_json(
                    directory / "status.json",
                    dict(
                        status="failed",
                        stage="ワーカープロセスが終了しました。error.logまたはserver.error.logを確認してください。"
                        "保存済みのチェックポイントから再開できます。",
                    ),
                )
        except OSError as error:
            if jid not in warned:
                logger.warning("ジョブ %s の終了状態を保存できません: %s", jid, error)
                warned.add(jid)
            continue
        del WORKERS[jid]
        warned.discard(jid)
    # A dead worker waiting for storage recovery must not block other queued jobs.
    if not any(proc.is_alive() for proc in WORKERS.values()):
        for directory in sorted(JOBS.iterdir(), key=lambda p: p.stat().st_mtime):
            if (
                directory.is_dir()
                and directory.name not in WORKERS
                and directory.name not in STARTUP_JOBS
                and read_status(directory)["status"] == "queued"
            ):
                proc = multiprocessing.get_context("spawn").Process(
                    target=run_job, args=(str(directory), str(DISTRIBUTIONS))
                )
                proc.start()
                WORKERS[directory.name] = proc
                break


async def scheduler():
    warned = set()
    last_error = None
    while True:
        try:
            scheduler_tick(warned)
            last_error = None
        except OSError as error:
            if str(error) != last_error:
                logger.warning("保存先にアクセスできません。再試行します: %s", error)
                last_error = str(error)
        await asyncio.sleep(0.3)


@asynccontextmanager
async def lifespan(app):
    for directory in (DATA, JOBS, DISTRIBUTIONS):
        validate_data_directory(directory)
    STARTUP_JOBS.clear()
    STARTUP_JOBS.update(directory.name for directory in JOBS.iterdir() if directory.is_dir())
    task = asyncio.create_task(scheduler())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    for jid in WORKERS:
        try:
            (JOBS / jid / "pause.request").touch()
        except OSError as error:
            logger.warning("ジョブ %s の停止要求を保存できません: %s", jid, error)
    for proc in WORKERS.values():
        await asyncio.to_thread(proc.join, 5)
    WORKERS.clear()


app = FastAPI(title="Memory Hole Simulator", lifespan=lifespan)


@app.middleware("http")
async def local_origin_only(request: Request, call_next):
    host = request.headers.get("host", "").split(":")[0]
    if host not in {"localhost", "127.0.0.1", "testserver"}:
        return JSONResponse(status_code=403, content={"detail": "ローカルホストからアクセスしてください。"})
    origin = request.headers.get("origin")
    if (
        request.method not in {"GET", "HEAD", "OPTIONS"}
        and origin
        and origin
        not in {
            "http://localhost:8765",
            "http://127.0.0.1:8765",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        }
    ):
        return JSONResponse(status_code=403, content={"detail": "異なるサイトからの操作は受け付けません。"})
    return await call_next(request)


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PermissionError)
async def permission_error(request, exc):
    return JSONResponse(status_code=403, content={"detail": permission_message(exc, DATA)})


@app.get("/api/defaults")
def defaults():
    preset = DATA / "preset.json"
    c = (
        CaseConfig.model_validate_json(preset.read_text(encoding="utf-8"))
        if preset.exists()
        else CaseConfig()
    )
    return {"config": c.model_dump(), "derived": derive(c)}


@app.get("/api/system")
def system():
    return {
        "gpu": gpu_info(),
        "active_workers": sum(proc.is_alive() for proc in WORKERS.values()),
        "app_version": "0.1.0",
        "data_dir": str(DATA),
    }


@app.post("/api/geometry")
def geometry(config: CaseConfig):
    return derive(config)


def create_job(config):
    derive(config)
    from .inlet import verify_metadata

    for species in [*config.ions, config.electron]:
        if species.distribution_id:
            sid = species.distribution_id
            if len(sid) != 32 or not sid.isalnum() or not (DISTRIBUTIONS / (sid + ".json")).exists():
                raise ValueError("入口分布が見つかりません。再取込してください。")
            verify_metadata(config, json.loads((DISTRIBUTIONS / (sid + ".json")).read_text(encoding="utf-8")))
    jid = uuid.uuid4().hex
    directory = JOBS / jid
    directory.mkdir()
    atomic_json(directory / "config.json", config.model_dump())
    atomic_json(
        directory / "status.json", dict(status="queued", stage="待機中", created_at=time.time(), progress=0)
    )
    return jid


@app.post("/api/jobs")
def submit(config: CaseConfig):
    return {"id": create_job(config)}


@app.get("/api/jobs")
def jobs():
    result = []
    for directory in sorted(JOBS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if directory.is_dir() and (directory / "config.json").exists():
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            result.append(
                {
                    "id": directory.name,
                    "name": config["name"],
                    "mode": config["mode"],
                    "backend": config["numerics"]["backend"],
                    "run_until": config["numerics"].get("run_until", "time"),
                    "maximum_time_s": config["numerics"].get("saturation", {}).get("max_time_s", 1e-3),
                    **read_status(directory),
                }
            )
    return result


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    directory = job_dir(job_id)
    return {
        "id": job_id,
        "config": json.loads((directory / "config.json").read_text(encoding="utf-8")),
        **read_status(directory),
    }


@app.post("/api/jobs/{job_id}/pause")
def pause(job_id: str):
    directory = job_dir(job_id)
    status = read_status(directory)["status"]
    if status == "queued" and job_id not in WORKERS:
        atomic_json(directory / "status.json", dict(status="paused", stage="開始前に停止"))
    elif status == "running" or job_id in WORKERS:
        (directory / "pause.request").touch()
    else:
        raise HTTPException(409, "実行中または待機中のジョブのみ停止できます。")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/resume")
def resume(job_id: str):
    directory = job_dir(job_id)
    if read_status(directory)["status"] not in {"paused", "failed"}:
        raise HTTPException(409, "停止または失敗したジョブのみ再開できます。")
    if job_id in WORKERS:
        raise HTTPException(409, "ワーカー終了処理中です。少し待って再開してください。")
    (directory / "pause.request").unlink(missing_ok=True)
    atomic_json(directory / "status.json", dict(status="queued", stage="チェックポイントから再開待ち"))
    return {"ok": True}


@app.get("/api/jobs/{job_id}/results")
def result(job_id: str, step: int | None = None):
    path = (
        job_dir(job_id) / "preview.json"
        if step is None
        else job_dir(job_id) / "snapshots" / f"{step:05}.json"
    )
    if not path.exists():
        raise HTTPException(404, "完了した帯電更新の結果がまだありません。")
    data = json.loads(path.read_text(encoding="utf-8"))
    # 保存記録は保持し、旧結果の表示にも現在の検証方針を適用する。
    data["metadata"]["warnings"] = [
        SPACE_CHARGE_WARNING if warning == "空間電荷を省略。5%感度検証とCOMSOL比較は未実施。" else warning
        for warning in data["metadata"]["warnings"]
    ]
    latest = json.loads((job_dir(job_id) / "preview.json").read_text(encoding="utf-8"))
    data["saved_times"] = [
        {"step": h["step"], "time_s": h["time_s"]}
        for h in latest["history"]
        if (job_dir(job_id) / "snapshots" / f"{h['step']:05}.json").exists()
    ]
    return data


@app.get("/api/jobs/{job_id}/export/{kind}")
def export(job_id: str, kind: str):
    directory = job_dir(job_id)
    if kind in {"config", "hdf5", "checkpoint"}:
        filename = {"config": "config.json", "hdf5": "results.h5", "checkpoint": "checkpoint.h5"}[kind]
        path = directory / filename
        if not path.exists():
            raise HTTPException(404, "出力ファイルがまだありません。")
        return FileResponse(path, filename=f"{job_id}_{filename}")
    if kind == "csv":
        preview = result(job_id)
        names = preview["metadata"]["collision_labels"]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["x_nm", "y_nm", "z_nm", "area_m2", "sigma_c_m2", *[n + "_impact_weight" for n in names]]
        )
        surf = preview["surface"]
        for i, point in enumerate(surf["centers_nm"]):
            writer.writerow(
                [*point, surf["area_m2"][i], surf["sigma_c_m2"][i], *[v[i] for v in surf["impact_weight"]]]
            )
        return Response(
            "\ufeff" + buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{job_id}_surface.csv"'},
        )
    raise HTTPException(404, "出力形式が不正です。")


@app.get("/api/jobs/{job_id}/slice")
def section(
    job_id: str,
    plane: Literal["xy", "xz", "yz", "vertical"] = "xz",
    position_nm: float = Query(0, allow_inf_nan=False),
    angle_deg: float = Query(0, ge=-180, le=180, allow_inf_nan=False),
    resolution: int = Query(241, ge=65, le=401),
    step: int | None = Query(None, ge=0),
):
    try:
        return saved_section(
            job_dir(job_id),
            plane=plane,
            position_nm=position_nm,
            angle_deg=angle_deg,
            resolution=resolution,
            step=step,
        )
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error


@app.post("/api/distributions")
async def import_distribution(
    file: UploadFile = File(...),
    config_json: str = Form(...),
    mass_amu: float = Form(...),
    pressure_case: str = Form("p0"),
    projection_policy: str = Form("reject"),
    collector_min_m: float | None = Form(None),
    collector_max_m: float | None = Form(None),
    source_config: UploadFile | None = File(None),
):
    config = CaseConfig.model_validate_json(config_json)
    payload = await file.read(128 * 1024 * 1024 + 1)
    if len(payload) > 128 * 1024 * 1024:
        raise ValueError(
            "入力ファイルは128MB以下にしてください。大きなデータはコレクタ単位に分割してください。"
        )
    if (file.filename or "").lower().endswith(".npz"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 512 * 1024 * 1024:
                raise ValueError("展開後のNPZが512MBを超えています。")
    metadata = None
    if collector_min_m is not None and collector_max_m is not None:
        if collector_min_m >= collector_max_m:
            raise ValueError("コレクタ範囲の上下限が不正です。")
        metadata = {"collector_x_min_m": collector_min_m, "collector_x_max_m": collector_max_m}
    data, meta = parse_distribution(
        payload, file.filename or "", config, mass_amu, pressure_case, projection_policy, metadata
    )
    jid = uuid.uuid4().hex
    root = DISTRIBUTIONS / jid
    np.savez_compressed(
        root.with_suffix(".npz"), **{k: v if v is not None else np.empty(0) for k, v in data.items()}
    )
    if source_config:
        source = await source_config.read(1024 * 1024 + 1)
        if len(source) > 1024 * 1024:
            raise ValueError("上流設定ファイルが大きすぎます。")
        json.loads(source)
        import hashlib

        meta["source_config_sha256"] = hashlib.sha256(source).hexdigest()
        root.with_suffix(".source.json").write_bytes(source)
    atomic_json(root.with_suffix(".json"), meta)
    return {"id": jid, "metadata": meta}


@app.post("/api/iaedf-case")
async def import_iaedf_pair(
    file: UploadFile = File(...),
    source_config: UploadFile = File(...),
    config_json: str = Form(...),
    pressure_case: str = Form("p0"),
    projection_policy: str = Form("reject"),
):
    from .iaedf import case_from_pair

    payload = await file.read(128 * 1024 * 1024 + 1)
    source = await source_config.read(1024 * 1024 + 1)
    if len(payload) > 128 * 1024 * 1024 or len(source) > 1024 * 1024:
        raise ValueError("入力ファイルのサイズ上限を超えています。")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(info.file_size for info in archive.infolist()) > 512 * 1024 * 1024:
            raise ValueError("展開後のNPZが512MBを超えています。")
    config = CaseConfig.model_validate_json(config_json)
    c, data, meta = case_from_pair(
        payload, source, file.filename or "raw.npz", config, pressure_case, projection_policy
    )
    derive(c)
    sid = uuid.uuid4().hex
    root = DISTRIBUTIONS / sid
    np.savez_compressed(
        root.with_suffix(".npz"), **{k: v if v is not None else np.empty(0) for k, v in data.items()}
    )
    root.with_suffix(".source.json").write_bytes(source)
    atomic_json(root.with_suffix(".json"), meta)
    c.ions[0].distribution_id = sid
    return {"config": c.model_dump(), "id": sid, "metadata": meta}


class SweepDimension(BaseModel):
    path: str
    values: list[float] = Field(min_length=1, max_length=20)


class SweepRequest(BaseModel):
    config: CaseConfig
    dimensions: list[SweepDimension] = Field(min_length=1, max_length=3)


@app.post("/api/sweeps")
def sweep(request: SweepRequest):
    if math_product([len(d.values) for d in request.dimensions]) > 64:
        raise ValueError("1回の掃引は64ケース以内にしてください。")
    configs = []
    for values in itertools.product(*(dimension.values for dimension in request.dimensions)):
        data = request.config.model_dump()
        for dim, value in zip(request.dimensions, values, strict=True):
            if not dim.path.startswith("geometry.") or not dim.path.endswith(("_nm", "_deg", "_count")):
                raise ValueError("掃引には形状寸法・角度・個数を指定してください。")
            parts = dim.path.split(".")
            target = data
            try:
                for part in parts[:-1]:
                    target = target[int(part)] if isinstance(target, list) else target[part]
                if parts[-1] not in target:
                    raise ValueError("未知の寸法です。")
                target[parts[-1]] = value
            except (KeyError, IndexError, TypeError):
                raise ValueError("掃引パラメーターのパスが不正です。") from None
        data["name"] = (
            request.config.name
            + " · "
            + ", ".join(
                f"{d.path.split('.')[-1]}={v:g}" for d, v in zip(request.dimensions, values, strict=True)
            )
        )
        try:
            c = CaseConfig.model_validate(data)
        except ValidationError as error:
            raise ValueError(str(error)) from error
        derive(c)
        configs.append(c)
    return {"ids": [create_job(c) for c in configs]}


def math_product(values):
    import math

    return math.prod(values)


static = ROOT / "frontend" / "dist"
if static.exists():
    app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
