import ctypes
import errno
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from memory_hole import api, engine, storage
from memory_hole.config import CaseConfig


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing regression")
def test_atomic_json_recovers_after_windows_reader_releases_file(tmp_path):
    path = tmp_path / "status.json"
    storage.atomic_json(path, {"status": "running"})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    # Read/write sharing without delete sharing: replacing the open file is denied.
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    release = threading.Timer(0.2, lambda: kernel.CloseHandle(handle))
    release.start()
    try:
        storage.atomic_json(path, {"status": "completed"})
        assert storage.read_json(path) == {"status": "completed"}
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        release.join()


def test_concurrent_json_writers_publish_complete_output(tmp_path):
    path = tmp_path / "status.json"
    barrier = threading.Barrier(8)

    def update(index):
        barrier.wait(timeout=5)
        storage.atomic_json(path, {"writer": index, "message": "日本語" * 1000})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(update, range(8)))
    result = storage.read_json(path)
    assert result["writer"] in range(8)
    assert result["message"] == "日本語" * 1000
    assert list(tmp_path.iterdir()) == [path]


def test_permanent_replace_denial_keeps_old_status_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    storage.atomic_json(path, {"status": "running"})
    attempts = []

    def denied(source, target):
        attempts.append(source)
        raise PermissionError(errno.EACCES, "Access denied", str(target))

    monkeypatch.setattr(storage.os, "replace", denied)
    with pytest.raises(PermissionError):
        storage.atomic_json(path, {"status": "completed"}, timeout_s=0.04)
    assert len(attempts) >= 2
    assert storage.read_json(path) == {"status": "running"}
    assert list(tmp_path.iterdir()) == [path]


def test_non_permission_errors_are_not_retried(tmp_path, monkeypatch):
    attempts = []

    def full(source, target):
        attempts.append(source)
        raise OSError(errno.ENOSPC, "Disk full", str(target))

    monkeypatch.setattr(storage.os, "replace", full)
    with pytest.raises(OSError) as caught:
        storage.atomic_json(tmp_path / "status.json", {})
    assert caught.value.errno == errno.ENOSPC
    assert len(attempts) == 1
    assert not list(tmp_path.iterdir())


def test_read_json_recovers_from_temporary_denial(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    storage.atomic_json(path, {"status": "running"})
    original = type(path).read_text
    attempts = []

    def read(target, *args, **kwargs):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError(errno.EACCES, "Access denied", str(target))
        return original(target, *args, **kwargs)

    monkeypatch.setattr(type(path), "read_text", read)
    assert storage.read_json(path) == {"status": "running"}
    assert len(attempts) == 3


@pytest.fixture
def isolated_api(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(api, "DATA", tmp_path)
    monkeypatch.setattr(api, "JOBS", jobs)
    monkeypatch.setattr(api, "DISTRIBUTIONS", tmp_path / "distributions")
    monkeypatch.setattr(api, "WORKERS", {})
    monkeypatch.setattr(api, "STARTUP_JOBS", set())
    return jobs


def test_unreadable_status_is_visible_and_cannot_be_resumed(isolated_api, monkeypatch):
    jid = "a" * 32
    directory = isolated_api / jid
    directory.mkdir()
    storage.atomic_json(directory / "config.json", CaseConfig().model_dump())

    def denied(path):
        raise PermissionError(errno.EACCES, "Access denied", str(path))

    monkeypatch.setattr(api, "read_json", denied)
    client = TestClient(api.app)
    response = client.get("/api/jobs")
    assert response.status_code == 200
    status = response.json()[0]
    assert status["status"] == "unavailable"
    assert "status.json" in status["stage"] and "-DataDir" in status["stage"]
    assert client.post(f"/api/jobs/{jid}/resume").status_code == 409
    assert api.WORKERS == {}


@pytest.mark.parametrize("contents", [None, "invalid json"])
def test_missing_or_invalid_status_is_never_assumed_queued(tmp_path, contents):
    if contents is not None:
        (tmp_path / "status.json").write_text(contents, encoding="utf-8")
    assert api.read_status(tmp_path)["status"] == "unavailable"


def test_scheduler_recovers_finalization_without_blocking_another_job(isolated_api, monkeypatch, caplog):
    jid, other = "a" * 32, "b" * 32
    for name, status in [(jid, "running"), (other, "queued")]:
        directory = isolated_api / name
        directory.mkdir()
        storage.atomic_json(directory / "status.json", {"status": status})
    dead = SimpleNamespace(is_alive=lambda: False, join=lambda: None)
    active = SimpleNamespace(is_alive=lambda: True, start=lambda: None)
    api.WORKERS[jid] = dead
    launches = []

    def process(**kwargs):
        launches.append(kwargs)
        return active

    monkeypatch.setattr(api.multiprocessing, "get_context", lambda _: SimpleNamespace(Process=process))
    original = api.atomic_json

    def denied(path, value):
        raise PermissionError(errno.EACCES, "Access denied", str(path))

    monkeypatch.setattr(api, "atomic_json", denied)
    warned = set()
    api.scheduler_tick(warned)
    api.scheduler_tick(warned)
    assert api.WORKERS == {jid: dead, other: active}
    assert len(launches) == 1
    assert launches[0]["args"][0] == str(isolated_api / other)
    assert len(caplog.records) == 1
    monkeypatch.setattr(api, "atomic_json", original)
    api.scheduler_tick(warned)
    assert api.WORKERS == {other: active}
    assert api.read_status(isolated_api / jid)["status"] == "failed"
    assert warned == set()


def test_scheduler_does_not_relaunch_worker_with_unreadable_status(isolated_api, monkeypatch):
    jid = "a" * 32
    directory = isolated_api / jid
    directory.mkdir()
    storage.atomic_json(directory / "status.json", {"status": "queued"})
    dead = SimpleNamespace(is_alive=lambda: False, join=lambda: None)
    api.WORKERS[jid] = dead
    original = api.read_json

    def denied(path):
        raise PermissionError(errno.EACCES, "Access denied", str(path))

    monkeypatch.setattr(api, "read_json", denied)
    api.scheduler_tick(set())
    assert api.WORKERS == {jid: dead}
    monkeypatch.setattr(api, "read_json", original)
    # Once readable again, stale queued state from a failed worker becomes failed.
    api.scheduler_tick(set())
    assert api.WORKERS == {}
    assert api.read_status(directory)["status"] == "failed"


def test_startup_recovers_prior_running_job_after_lock_is_released(isolated_api, monkeypatch):
    jid = "a" * 32
    directory = isolated_api / jid
    directory.mkdir()
    storage.atomic_json(directory / "status.json", {"status": "running"})
    api.STARTUP_JOBS.add(jid)
    original = api.read_json

    def denied(path):
        raise PermissionError(errno.EACCES, "Access denied", str(path))

    monkeypatch.setattr(api, "read_json", denied)
    warned = set()
    api.scheduler_tick(warned)
    assert api.STARTUP_JOBS == {jid}
    assert api.WORKERS == {}
    monkeypatch.setattr(api, "read_json", original)
    api.scheduler_tick(warned)
    assert api.STARTUP_JOBS == set()
    assert api.WORKERS == {}
    assert api.read_status(directory)["status"] == "paused"


def test_permission_error_response_explains_save_path(isolated_api, monkeypatch):
    def denied(config):
        raise PermissionError(errno.EACCES, "Access denied", str(isolated_api))

    monkeypatch.setattr(api, "create_job", denied)
    response = TestClient(api.app).post("/api/jobs", json=CaseConfig().model_dump())
    assert response.status_code == 403
    assert str(isolated_api) in response.json()["detail"]
    assert "-DataDir" in response.json()["detail"]


def test_worker_keeps_original_error_when_status_cannot_be_written(tmp_path, monkeypatch, caplog):
    config = CaseConfig()
    config.numerics.backend = "cpu"
    storage.atomic_json(tmp_path / "config.json", config.model_dump())
    storage.atomic_json(tmp_path / "status.json", {"status": "queued"})

    def denied(path, value):
        raise PermissionError(errno.EACCES, "Access denied", str(path))

    monkeypatch.setattr(engine, "atomic_json", denied)
    with pytest.raises(PermissionError):
        engine.run_job(str(tmp_path), str(tmp_path / "distributions"))
    details = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "PermissionError" in details and "status.json" in details
    assert api.read_status(tmp_path)["status"] == "queued"
    assert "status.json" in caplog.text


def test_data_directory_probe_leaves_existing_files_intact(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    storage.validate_data_directory(tmp_path)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.iterdir()) == [marker]


def test_data_directory_probe_reports_denied_path(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, "Access denied", str(tmp_path))

    monkeypatch.setattr(storage.tempfile, "mkstemp", denied)
    monkeypatch.setattr(storage, "retry_permission", lambda operation, **kwargs: operation())
    with pytest.raises(PermissionError) as caught:
        storage.validate_data_directory(tmp_path)
    assert str(tmp_path) in str(caught.value)
    assert "書込権限" in str(caught.value) and "-DataDir" in str(caught.value)
