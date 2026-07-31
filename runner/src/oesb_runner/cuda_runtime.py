"""Best-effort loader for a pip-installed cuBLAS runtime, on Linux only.

ctranslate2 (faster-whisper's backend) dlopen's `libcublas.so.12` lazily when
`--backend cuda` is used — it declares no pip dependency on it (confirmed by
inspecting the actual PyPI wheel: no bundled .so, no NEEDED entry, only a
bare `libcublas.so.12` string referenced internally for a runtime dlopen)
and expects it to already be resolvable, typically via a full system CUDA
Toolkit install. A fresh Ubuntu box with an NVIDIA driver but no toolkit (or
one pinned to a different CUDA major version) has no cuBLAS on the loader's
search path at all, and `WhisperModel(device="cuda")` fails deep inside
ctranslate2 the first time it actually tries to use the GPU.

NVIDIA also publishes cuBLAS as a standalone pip wheel, `nvidia-cublas-cu12`
— the same mechanism PyTorch's cu12x wheels rely on to avoid requiring the
multi-GB Toolkit installer. This module lets goesb-runner use that wheel
instead: install it once (`_offer_install` in cli.py), then preload it every
time `--backend cuda` is used so ctranslate2's own later dlopen finds it
already resident under the matching SONAME.
"""
from __future__ import annotations

import ctypes
import importlib.metadata
import platform
from pathlib import Path

CUBLAS_PACKAGE = "nvidia-cublas-cu12"
_CUBLAS_SONAME = "libcublas.so.12"


def cuda_libs_supported() -> bool:
    """False everywhere except Linux — this whole pip-wheel-preload trick
    is Linux `dlopen`/SONAME-specific; macOS has no CUDA path at all, and
    Windows' `cublas64_12.dll` search-path story is different enough
    (needs its own verification on real hardware) that it's out of scope
    here."""
    return platform.system() == "Linux"


def cublas_loadable() -> bool:
    """Best-effort: can this process load cuBLAS right now, from wherever
    the OS loader already looks (a system CUDA Toolkit install, a conda
    env, or an already-preloaded pip wheel)? False on any failure,
    including on platforms this module doesn't cover at all."""
    if not cuda_libs_supported():
        return False
    try:
        ctypes.CDLL(_CUBLAS_SONAME)
        return True
    except OSError:
        return False


def _pip_cublas_lib_path() -> Path | None:
    """Where the `nvidia-cublas-cu12` wheel actually put libcublas.so.12,
    if that wheel is installed — its package layout ships the library
    under `nvidia/cublas/lib/`, which is never on the loader's default
    search path just from being pip-installed."""
    try:
        dist = importlib.metadata.distribution(CUBLAS_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return None
    for f in dist.files or []:
        if f.name == _CUBLAS_SONAME:
            return Path(dist.locate_file(f))
    return None


def preload_installed_cublas() -> bool:
    """If `nvidia-cublas-cu12` is pip-installed, load it explicitly by
    its absolute path (`RTLD_GLOBAL`) so ctranslate2's own later
    `dlopen("libcublas.so.12")` sees a library with that exact SONAME
    already resident in the process and reuses it — standard ELF dlopen
    behavior, the same preload trick PyTorch's `cu12x` wheels use, not a
    path/env-var hack that could silently miss.

    Safe to call unconditionally and repeatedly: a no-op returning True
    if cuBLAS is already loadable some other way (system Toolkit, conda),
    and a no-op returning False if the pip wheel isn't installed either
    (cpu-only installs, or a platform this module doesn't cover) — the
    caller falls back to its existing "ask goesb doctor" error path in
    that case."""
    if cublas_loadable():
        return True
    lib_path = _pip_cublas_lib_path()
    if lib_path is None:
        return False
    try:
        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return cublas_loadable()
