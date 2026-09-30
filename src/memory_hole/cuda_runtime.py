"""Process-local DLL registration for CUDA component wheels on Windows.

PATH is inherited by spawned workers; AddDllDirectory registrations are not.
NVRTC loads its builtins dynamically, so register and retain their directory
in each process and preload the matching wheel library by its absolute path.
"""

from __future__ import annotations

import ctypes
import os
import sys
import sysconfig
from pathlib import Path

_HANDLES: list = []
_READY = False


def prepare_cuda_runtime():
    global _READY
    if _READY or sys.platform != "win32":
        return
    root = Path(sysconfig.get_path("purelib")) / "nvidia" / "cu13" / "bin"
    if root.exists():
        for library in root.glob("**/nvrtc-builtins*.dll"):
            folder = str(library.parent.resolve())
            _HANDLES.append(os.add_dll_directory(folder))
            parts = os.environ.get("PATH", "").split(os.pathsep)
            if folder not in parts:
                os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            _HANDLES.append(ctypes.WinDLL(str(library.resolve())))
    _READY = True
