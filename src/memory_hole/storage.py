"""Bounded retries for Windows file sharing conflicts and atomic local output."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def retry_permission(operation, *, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    delay = 0.01
    while True:
        try:
            return operation()
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.2)


def replace_file(source: Path, target: Path, *, timeout_s=2.0):
    retry_permission(lambda: os.replace(source, target), timeout_s=timeout_s)


def atomic_json(path: Path, value, *, timeout_s=2.0):
    # Serialize before opening a file; invalid values must leave the old output intact.
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
    fd, name = retry_permission(
        lambda: tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent),
        timeout_s=timeout_s,
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(payload)
        replace_file(temporary, path, timeout_s=timeout_s)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A cleanup failure must not hide the original storage error.
            pass


def read_json(path: Path, *, timeout_s=0.15):
    return retry_permission(lambda: json.loads(path.read_text(encoding="utf-8")), timeout_s=timeout_s)


def permission_message(error: PermissionError, fallback: Path):
    path = error.filename2 or error.filename or fallback
    return (
        f"ファイルへのアクセスが拒否されました: {path}。"
        "一時ロックへの再試行後もアクセスできません。"
        "このファイルと保存先の書込権限・他のアプリによる使用を確認してください。"
        "保存先を変更する場合はサーバーを終了し、.\\launch.ps1 -DataDir "
        '"$env:LOCALAPPDATA\\MemoryHoleLab" で起動してください。'
    )


def validate_data_directory(directory: Path):
    """Check creation, reading and replacement without touching existing job output."""
    probe = None
    try:
        retry_permission(lambda: directory.mkdir(parents=True, exist_ok=True))
        fd, name = retry_permission(
            lambda: tempfile.mkstemp(prefix=".memory-hole-check-", suffix=".json", dir=directory)
        )
        os.close(fd)
        probe = Path(name)
        atomic_json(probe, {"check": 1})
        read_json(probe)
        atomic_json(probe, {"check": 2})
        retry_permission(probe.unlink)
        probe = None
    except PermissionError as error:
        raise PermissionError(permission_message(error, directory)) from error
    finally:
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-path", required=True, type=Path)
    args = parser.parse_args()
    try:
        validate_data_directory(args.check_path.resolve())
    except OSError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
