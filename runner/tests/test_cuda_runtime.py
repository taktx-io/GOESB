import importlib.metadata
from pathlib import Path

from oesb_runner import cuda_runtime


def test_cuda_libs_supported_only_on_linux(monkeypatch):
    monkeypatch.setattr(cuda_runtime.platform, "system", lambda: "Linux")
    assert cuda_runtime.cuda_libs_supported() is True

    for other in ("Darwin", "Windows"):
        monkeypatch.setattr(cuda_runtime.platform, "system", lambda other=other: other)
        assert cuda_runtime.cuda_libs_supported() is False


def test_cublas_loadable_false_on_unsupported_platform(monkeypatch):
    """No ctypes probe should even be attempted off Linux — cheap, and
    avoids any platform-specific dlopen quirks on macOS/Windows test
    runners."""
    monkeypatch.setattr(cuda_runtime, "cuda_libs_supported", lambda: False)

    def _unexpected(*a, **k):
        raise AssertionError("should not probe ctypes on an unsupported platform")

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", _unexpected)

    assert cuda_runtime.cublas_loadable() is False


def test_cublas_loadable_true_when_ctypes_loads_it(monkeypatch):
    monkeypatch.setattr(cuda_runtime, "cuda_libs_supported", lambda: True)
    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", lambda name: object())

    assert cuda_runtime.cublas_loadable() is True


def test_cublas_loadable_false_when_ctypes_raises(monkeypatch):
    monkeypatch.setattr(cuda_runtime, "cuda_libs_supported", lambda: True)

    def _raise(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", _raise)

    assert cuda_runtime.cublas_loadable() is False


def test_preload_installed_cublas_short_circuits_when_already_loadable(monkeypatch):
    """If cuBLAS is already loadable some other way (system CUDA Toolkit,
    conda env), there's nothing to preload — must never touch
    importlib.metadata at all."""
    monkeypatch.setattr(cuda_runtime, "cublas_loadable", lambda: True)

    def _unexpected(*a, **k):
        raise AssertionError("should not check for the pip wheel when already loadable")

    monkeypatch.setattr(cuda_runtime.importlib.metadata, "distribution", _unexpected)

    assert cuda_runtime.preload_installed_cublas() is True


def test_preload_installed_cublas_false_when_pip_wheel_not_installed(monkeypatch):
    monkeypatch.setattr(cuda_runtime, "cublas_loadable", lambda: False)
    monkeypatch.setattr(
        cuda_runtime.importlib.metadata, "distribution",
        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)),
    )

    assert cuda_runtime.preload_installed_cublas() is False


class _FakeDistFile:
    def __init__(self, name):
        self.name = name


class _FakeDist:
    def __init__(self, files):
        self.files = files

    def locate_file(self, f):
        return f"/fake/site-packages/nvidia/cublas/lib/{f.name}"


def test_preload_installed_cublas_finds_and_loads_the_pip_wheels_library(monkeypatch):
    """The realistic fresh-Ubuntu-box path: nvidia-cublas-cu12 is
    pip-installed but not on the loader's default search path yet (never
    is, just from being pip-installed) -- preload must find its exact
    libcublas.so.12 file and load it by absolute path with RTLD_GLOBAL,
    then report success once the follow-up loadability check passes."""
    calls = {"loadable_check_count": 0}

    def fake_cublas_loadable():
        calls["loadable_check_count"] += 1
        # False the first time (nothing loaded yet), True after the
        # preload's own CDLL call below has "loaded" it.
        return calls["loadable_check_count"] > 1

    monkeypatch.setattr(cuda_runtime, "cublas_loadable", fake_cublas_loadable)
    monkeypatch.setattr(
        cuda_runtime.importlib.metadata, "distribution",
        lambda name: _FakeDist(files=[
            _FakeDistFile("some_other_file.txt"),
            _FakeDistFile(cuda_runtime._CUBLAS_SONAME),
        ]),
    )

    loaded = {}

    def fake_cdll(path, mode=None):
        loaded["path"] = path
        loaded["mode"] = mode
        return object()

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", fake_cdll)

    assert cuda_runtime.preload_installed_cublas() is True
    # cuda_runtime.py wraps locate_file()'s result in Path(...), which
    # normalizes separators for whatever OS pytest itself runs on (this
    # module's actual behavior is Linux-only, but the test suite still
    # collects and runs this file on every CI platform) -- build the
    # expectation the same way instead of hardcoding forward slashes.
    expected = Path(f"/fake/site-packages/nvidia/cublas/lib/{cuda_runtime._CUBLAS_SONAME}")
    assert loaded["path"] == str(expected)
    assert loaded["mode"] == cuda_runtime.ctypes.RTLD_GLOBAL


def test_preload_installed_cublas_returns_false_when_wheel_installed_but_file_missing(monkeypatch):
    """Defensive: an installed nvidia-cublas-cu12 whose file layout doesn't
    match what this module expects (e.g. a future release renaming the
    library) must fail closed, not raise."""
    monkeypatch.setattr(cuda_runtime, "cublas_loadable", lambda: False)
    monkeypatch.setattr(
        cuda_runtime.importlib.metadata, "distribution",
        lambda name: _FakeDist(files=[_FakeDistFile("some_other_file.txt")]),
    )

    assert cuda_runtime.preload_installed_cublas() is False


def test_preload_installed_cublas_returns_false_when_cdll_raises(monkeypatch):
    monkeypatch.setattr(cuda_runtime, "cublas_loadable", lambda: False)
    monkeypatch.setattr(
        cuda_runtime.importlib.metadata, "distribution",
        lambda name: _FakeDist(files=[_FakeDistFile(cuda_runtime._CUBLAS_SONAME)]),
    )

    def _raise(path, mode=None):
        raise OSError("wrong ELF class or corrupt file")

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", _raise)

    assert cuda_runtime.preload_installed_cublas() is False
