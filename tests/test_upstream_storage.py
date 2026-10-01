import errno
from types import SimpleNamespace

from memory_hole.storage import atomic_json, read_json
from memory_hole.upstream_api import UpstreamService


def test_denied_final_status_does_not_block_other_iaedf_jobs(tmp_path, monkeypatch, caplog):
    service = UpstreamService(tmp_path, tmp_path / "distributions", lambda p: read_json(p / "status.json"))
    service.start()
    dead_id, other_id = "c" * 32, "d" * 32
    for jid, status in [(dead_id, "running"), (other_id, "queued")]:
        (service.root / jid).mkdir()
        atomic_json(service.root / jid / "status.json", {"status": status})
    dead = SimpleNamespace(is_alive=lambda: False, join=lambda: None)
    live = SimpleNamespace(is_alive=lambda: True, start=lambda: None)
    service.workers[dead_id] = dead
    monkeypatch.setattr(
        "memory_hole.upstream_api.multiprocessing.get_context",
        lambda _: SimpleNamespace(Process=lambda **kwargs: live),
    )

    def denied(path, value):
        raise PermissionError(errno.EACCES, "Access denied", str(path))

    monkeypatch.setattr("memory_hole.upstream_api.atomic_json", denied)
    service.tick()
    service.tick()
    assert service.workers == {dead_id: dead, other_id: live}
    assert len(caplog.records) == 1
    monkeypatch.setattr("memory_hole.upstream_api.atomic_json", atomic_json)
    service.tick()
    assert service.workers == {other_id: live}
    assert read_json(service.root / dead_id / "status.json")["status"] == "failed"
