import io
import json
import os
import sys
import tarfile
import types
import urllib.error
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

# Captured at import time, before the autouse `_no_network_version_check`
# fixture (conftest.py) monkeypatches `cli_module._warn_if_runner_outdated`
# to a no-op for every other test -- this reference still points at the
# real function regardless, since monkeypatch only rebinds the module
# attribute, not this already-held object.
from oesb_runner.cli import (
    _warn_if_runner_outdated as _real_warn_if_runner_outdated,
)
from oesb_runner.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "goesb-runner" in result.stdout


def test_env_command_prints_json():
    result = runner.invoke(app, ["env"])
    assert result.exit_code == 0
    assert "schema_version" in result.stdout


def test_validate_valid_profile():
    path = REPO_ROOT / "profiles" / "whisper-medium-en-batch" / "profile.yaml"
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_validate_valid_pack():
    path = REPO_ROOT / "packs" / "librispeech-en" / "pack.yaml"
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_validate_invalid_file_exits_nonzero(tmp_path):
    bad = tmp_path / "profile.yaml"
    bad.write_text("id: not-enough-fields\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1


def test_list_profiles_offline_lists_local_profiles():
    result = runner.invoke(
        app, ["list-profiles", "--offline", "--profiles-dir", str(REPO_ROOT / "profiles")]
    )
    assert result.exit_code == 0
    assert "whisper-medium-en-batch" in result.stdout


def test_list_packs_offline_lists_local_packs():
    result = runner.invoke(
        app, ["list-packs", "--offline", "--packs-dir", str(REPO_ROOT / "packs")]
    )
    assert result.exit_code == 0
    assert "librispeech-en" in result.stdout
    assert "open" in result.stdout


def test_list_profiles_offline_no_local_dir_fails(tmp_path):
    result = runner.invoke(
        app, ["list-profiles", "--offline", "--profiles-dir", str(tmp_path / "nope")]
    )
    assert result.exit_code == 1


def test_run_prints_fetch_instructions_for_a_pack_with_no_manifest_yet(tmp_path, monkeypatch):
    # A pack that declares fetch_instructions but no auto-fetchable
    # source.type, and has no manifest.jsonl on disk at all (e.g. a
    # not-yet-completed contribution) — `run` must fail cleanly with those
    # instructions, not crash trying to read a manifest.jsonl that was
    # never written (regression: this used to raise an uncaught
    # FileNotFoundError). Not what this test is about, so stub out the
    # engine-install check — CI machines don't have any engine extra
    # installed, and this test's profile just needs to validate, not run.
    from oesb_runner import cli as cli_module
    from oesb_runner.hashing import canonical_asset_sha256

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)

    pack = {
        "id": "incomplete-pack",
        "version": "1.0.0",
        "profile_id": "whisper-medium-en-batch",
        "visibility": "open",
        "license": "CC0-1.0",
        "audio": {"source": {"fetch_instructions": "Visit https://example.invalid/dataset and follow its steps."}},
        "metadata": {"language": "en-US", "recording_environment": "quiet", "speech_style": "read"},
    }
    pack["sha256"] = canonical_asset_sha256(pack)
    packs_dir = tmp_path / "packs" / "incomplete-pack"
    packs_dir.mkdir(parents=True)
    (packs_dir / "pack.yaml").write_text(yaml.safe_dump(pack, sort_keys=False))

    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "incomplete-pack",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
    ])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "fetch it manually" in result.output
    assert "example.invalid/dataset" in result.output


def test_run_hints_at_upgrade_for_a_pack_source_type_this_runner_predates(tmp_path, monkeypatch):
    # A pack whose audio.source.type this runner's bundled schema doesn't
    # know about (simulates an old install fetching a pack.yaml built for a
    # newer platform, e.g. a future provider added after this runner
    # shipped) — should point at `pip install --upgrade goesb-runner`, not
    # dump a raw jsonschema enum-mismatch message.
    from oesb_runner import cli as cli_module
    from oesb_runner.hashing import canonical_asset_sha256

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)

    pack = {
        "id": "future-pack",
        "version": "1.0.0",
        "profile_id": "whisper-medium-en-batch",
        "visibility": "open",
        "license": "CC0-1.0",
        "audio": {"source": {"type": "some_future_provider", "params": {}}},
        "metadata": {"language": "en-US", "recording_environment": "quiet", "speech_style": "read"},
    }
    pack["sha256"] = canonical_asset_sha256(pack)
    packs_dir = tmp_path / "packs" / "future-pack"
    packs_dir.mkdir(parents=True)
    (packs_dir / "pack.yaml").write_text(yaml.safe_dump(pack, sort_keys=False))

    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "future-pack",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
    ])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "pip install --upgrade goesb-runner" in result.output
    assert "some_future_provider" in result.output


def test_run_refuses_a_pack_that_declares_a_newer_min_runner_version(tmp_path, monkeypatch):
    # A pack explicitly declaring it needs a newer runner than what's
    # installed (e.g. it relies on a manifest.jsonl field like audio_sha256
    # that only a newer load_pack checks) — must refuse with a clear
    # upgrade message before even reaching schema validation, not silently
    # run without whatever guarantee the older runner doesn't know to check.
    from oesb_runner import cli as cli_module
    from oesb_runner.hashing import canonical_asset_sha256

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)
    monkeypatch.setattr(cli_module, "__version__", "0.4.1")

    pack = {
        "id": "newer-pack",
        "version": "1.0.0",
        "profile_id": "whisper-medium-en-batch",
        "visibility": "open",
        "license": "CC0-1.0",
        "min_runner_version": "0.5.0",
        "metadata": {"language": "en-US", "recording_environment": "quiet", "speech_style": "read"},
    }
    pack["sha256"] = canonical_asset_sha256(pack)
    packs_dir = tmp_path / "packs" / "newer-pack"
    packs_dir.mkdir(parents=True)
    (packs_dir / "pack.yaml").write_text(yaml.safe_dump(pack, sort_keys=False))

    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "newer-pack",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
    ])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "pip install --upgrade goesb-runner" in result.output
    assert "0.5.0" in result.output
    assert "0.4.1" in result.output


def test_run_refuses_a_pack_whose_language_does_not_match_the_profile(tmp_path, monkeypatch):
    """ADR-0011: eligibility is decided by language, not by a pack pinning
    one exact profile_id — a pack for a different language must still be
    refused, hard, before anything runs."""
    from oesb_runner import cli as cli_module
    from oesb_runner.hashing import canonical_asset_sha256

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)

    pack = {
        "id": "wrong-language-pack",
        "version": "1.0.0",
        "visibility": "open",
        "license": "CC0-1.0",
        "audio": {"source": {"fetch_instructions": "n/a"}},
        "metadata": {"language": "de-DE", "recording_environment": "quiet", "speech_style": "read"},
    }
    pack["sha256"] = canonical_asset_sha256(pack)
    packs_dir = tmp_path / "packs" / "wrong-language-pack"
    packs_dir.mkdir(parents=True)
    (packs_dir / "pack.yaml").write_text(yaml.safe_dump(pack, sort_keys=False))

    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "wrong-language-pack",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
    ])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "wrong-language-pack" in result.output
    assert "'de-DE'" in result.output
    assert "'en-US'" in result.output


def test_run_refuses_a_pack_when_profile_has_no_language_declared(tmp_path, monkeypatch):
    """A profile with no `language` at all (schema-legal — language is
    optional) can't be verified eligible for any pack — refuse rather than
    silently letting it through."""
    from oesb_runner import cli as cli_module
    from oesb_runner.hashing import canonical_asset_sha256

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)

    profile = {
        "id": "no-language-profile", "version": "1.0.0", "benchmark_type": "batch",
        "runtime": {"name": "faster-whisper"},
        "model": {"name": "whisper-medium"},
        "scoring": {"primary_metric": "wer"},
        "metrics": ["wer"],
    }
    profiles_dir = tmp_path / "profiles" / "no-language-profile"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))

    pack = {
        "id": "some-pack", "version": "1.0.0", "visibility": "open", "license": "CC0-1.0",
        "audio": {"source": {"fetch_instructions": "n/a"}},
        "metadata": {"language": "en-US", "recording_environment": "quiet", "speech_style": "read"},
    }
    pack["sha256"] = canonical_asset_sha256(pack)
    packs_dir = tmp_path / "packs" / "some-pack"
    packs_dir.mkdir(parents=True)
    (packs_dir / "pack.yaml").write_text(yaml.safe_dump(pack, sort_keys=False))

    result = runner.invoke(app, [
        "run", "no-language-profile", "some-pack",
        "--profiles-dir", str(tmp_path / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
    ])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "no-language-profile" in result.output
    assert "no `language` declared" in result.output


def test_bare_invocation_shows_help_instead_of_hanging():
    # CliRunner's stdin isn't a tty, same as any piped/scripted invocation —
    # exercises the non-interactive fallback path, not the wizard itself.
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


# --- ADR-0008: `goesb doctor` reports, never runs anything ---


def _assume_all_engines_installed(monkeypatch, cli_module):
    """`doctor` reports differently for an engine it can't even import
    ("not installed — supports [...] once installed"). CI's runner matrix
    installs only `./runner[dev]` (no faster-whisper/vosk/whisper-cpp
    extras — see ci.yml), so the doctor scenarios below, which are about
    backend *readiness* given an installed engine, must not depend on
    whatever happens to actually be pip-installed in the environment the
    test runs in."""
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())


def test_doctor_reports_no_gpu_and_exits_cleanly(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "GPU: none detected" in result.output
    assert "cuda unavailable (no NVIDIA GPU detected)" in result.output
    assert "vosk (batch): cpu ready" in result.output


def test_doctor_reports_gpu_present_but_cublas_missing_with_a_next_step(monkeypatch):
    """Acceptance criterion 4: GPU present but cuBLAS not found -> reports it
    plainly with an actionable next step, exits cleanly (0), never a
    partial/crashed state. Pinned to a platform where cuda_runtime's own
    pip-wheel fix doesn't apply (see the Linux-specific test below), so this
    exercises the fallback "install the full Toolkit" wording."""
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    fake_gpu = {"model": "NVIDIA RTX 3060", "driver": "550.54.14", "vram": "12288 MiB"}
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: fake_gpu)
    monkeypatch.setattr(cli_module, "_cuda_device_count", lambda: 0)
    monkeypatch.setattr(cli_module.cuda_runtime, "cuda_libs_supported", lambda: False)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "NVIDIA RTX 3060" in result.output
    assert "driver 550.54.14" in result.output
    assert "cuBLAS is likely missing" in result.output
    assert "developer.nvidia.com/cudnn" in result.output  # the actionable next step
    assert "--backend cpu" in result.output  # the escape hatch


def test_doctor_reports_gpu_present_but_cublas_missing_on_linux_suggests_pip_fix(monkeypatch):
    """On a platform cuda_runtime.py actually covers (Linux), doctor's
    message should point at the lightweight pip-wheel fix goesb run now
    offers automatically, not just the full Toolkit installer."""
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    fake_gpu = {"model": "NVIDIA RTX 3060", "driver": "550.54.14", "vram": "12288 MiB"}
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: fake_gpu)
    monkeypatch.setattr(cli_module, "_cuda_device_count", lambda: 0)
    monkeypatch.setattr(cli_module.cuda_runtime, "cuda_libs_supported", lambda: True)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert 'pip install "goesb-runner[cuda]"' in result.output
    assert "--backend cpu" in result.output  # the escape hatch


def test_doctor_reports_gpu_ready_when_cuda_devices_visible(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    fake_gpu = {"model": "NVIDIA RTX 4090", "driver": "550.54.14", "vram": "24576 MiB"}
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: fake_gpu)
    monkeypatch.setattr(cli_module, "_cuda_device_count", lambda: 1)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "cuda ready (1 device(s) visible to ctranslate2)" in result.output


def test_doctor_reports_not_installed_engines_with_an_install_hint(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert 'not installed — supports' in result.output
    assert 'pip install "goesb-runner[faster-whisper]"' in result.output
    assert 'pip install "goesb-runner[vosk]"' in result.output


def test_doctor_never_touches_disk_beyond_reading(monkeypatch, tmp_path):
    """Detection informs the human, it never changes the experiment (ADR-0008
    §2) — confirm doctor writes nothing, regardless of what it detects."""
    from oesb_runner import cli as cli_module

    fake_gpu = {"model": "NVIDIA RTX 3060", "driver": "550.54.14", "vram": "12288 MiB"}
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: fake_gpu)
    monkeypatch.setattr(cli_module, "_cuda_device_count", lambda: 0)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere


def test_doctor_survives_an_unexpected_probe_failure(monkeypatch):
    """A report command must never itself crash or leave a partial/garbled
    state — even if a probe raises something entirely unanticipated."""
    from oesb_runner import cli as cli_module

    def _boom(unavailable):
        raise RuntimeError("simulated probe failure")

    monkeypatch.setattr(cli_module, "_capture_gpu", _boom)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "bug" in result.output.lower()


def _fake_pywhispercpp(monkeypatch, model_cls) -> None:
    """Stand in for a real `import pywhispercpp.model.Model` — doctor's
    whisper-cpp branch imports this lazily, so faking the module in
    sys.modules is the only way to control what Model.system_info()
    reports without pywhispercpp genuinely being installed."""
    fake_pkg = types.ModuleType("pywhispercpp")
    fake_model_module = types.ModuleType("pywhispercpp.model")
    fake_model_module.Model = model_cls
    monkeypatch.setitem(sys.modules, "pywhispercpp", fake_pkg)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", fake_model_module)


def test_doctor_engine_line_whisper_cpp_reports_real_cuda_support(monkeypatch):
    from oesb_runner import cli as cli_module

    class _FakeModel:
        @staticmethod
        def system_info():
            # Realistic ggml dynamic-backend-registry format (confirmed
            # against upstream source) — CUDA is its own "CUDA : ..."
            # section, never a flat "CUDA = 1" pair.
            return "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : ARCHS = 89 | "

    _fake_pywhispercpp(monkeypatch, _FakeModel)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())

    fake_gpu = {"model": "NVIDIA RTX 4090", "driver": "550.54.14", "vram": "24576 MiB"}
    line = cli_module._doctor_engine_line("whisper-cpp", "batch", fake_gpu)

    assert "cuda ready (NVIDIA RTX 4090 detected)" in line
    assert "metal unavailable" in line  # not compiled into this fake build


def test_doctor_engine_line_whisper_cpp_reports_no_cuda_support(monkeypatch):
    """This is the real gap the fix closes: previously this line always
    said "can't be checked without running a real transcription" — now it
    gives a definitive answer from whisper.cpp's own build info."""
    from oesb_runner import cli as cli_module

    class _FakeModel:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | "

    _fake_pywhispercpp(monkeypatch, _FakeModel)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())

    line = cli_module._doctor_engine_line("whisper-cpp", "batch", None)

    assert "cuda unavailable (not compiled into this build)" in line
    assert "metal ready" in line
    assert "can't be checked without running a real transcription" not in line


def test_doctor_engine_line_whisper_cpp_cuda_ready_but_no_nvidia_gpu_warns(monkeypatch):
    from oesb_runner import cli as cli_module

    class _FakeModel:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : ARCHS = 89 | "

    _fake_pywhispercpp(monkeypatch, _FakeModel)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())

    line = cli_module._doctor_engine_line("whisper-cpp", "batch", None)

    assert "cuda compiled in, but no NVIDIA GPU detected" in line
    assert "unlikely to work" in line


def test_doctor_engine_line_whisper_cpp_reports_real_metal_support(monkeypatch):
    """Metal doesn't need nvidia-smi/gpu at all — unlike cuda, it's
    self-contained: no NVIDIA-GPU cross-check applies to it."""
    from oesb_runner import cli as cli_module

    class _FakeModel:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | "

    _fake_pywhispercpp(monkeypatch, _FakeModel)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())

    line = cli_module._doctor_engine_line("whisper-cpp", "batch", None)

    assert "metal ready" in line


def test_doctor_engine_line_whisper_cpp_handles_find_spec_import_mismatch(monkeypatch):
    """find_spec (used for the earlier `installed` check) and a real import
    can disagree — e.g. a partial install, or (the actual case this
    guards against) a test/CI environment where `pywhispercpp` isn't
    genuinely installed but something upstream reports it as available.
    Must report that clearly, not let it surface as doctor's generic
    "an internal probe failed" catch-all."""
    from oesb_runner import cli as cli_module

    # Both entries must be forced to None — pywhispercpp is genuinely
    # installed in this dev/CI environment for other tests, so
    # "pywhispercpp.model" is independently cached in sys.modules the
    # moment any test imports it; nulling only the top-level package
    # leaves `from pywhispercpp.model import Model` free to succeed via
    # that cached submodule regardless.
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", None)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())

    line = cli_module._doctor_engine_line("whisper-cpp", "batch", None)

    assert "gpu readiness unknown" in line
    assert "broken or partial install" in line


# --- _ready_backends: wizard-facing "actually usable now" narrowing ---


def test_ready_backends_not_installed_returns_empty(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: None)

    assert cli_module._ready_backends("faster-whisper", "batch", None) == frozenset()


def test_ready_backends_cpu_only_engine_returns_cpu(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())

    assert cli_module._ready_backends("vosk", "batch", None) == frozenset({"cpu"})


def test_ready_backends_whisper_cpp_cuda_needs_both_build_and_gpu(monkeypatch):
    from oesb_runner import cli as cli_module

    class _FakeModel:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : ARCHS = 89 | "

    _fake_pywhispercpp(monkeypatch, _FakeModel)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())

    fake_gpu = {"model": "NVIDIA RTX 4090", "driver": "550.54.14", "vram": "24576 MiB"}
    assert cli_module._ready_backends("whisper-cpp", "batch", fake_gpu) == frozenset({"cpu", "cuda"})
    # Compiled in, but no NVIDIA GPU detected — same caveat doctor prints,
    # excluded here rather than offered as a choice certain to fail.
    assert cli_module._ready_backends("whisper-cpp", "batch", None) == frozenset({"cpu"})


def test_ready_backends_whisper_cpp_metal_needs_no_gpu_probe(monkeypatch):
    from oesb_runner import cli as cli_module

    class _FakeModel:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | "

    _fake_pywhispercpp(monkeypatch, _FakeModel)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())

    assert cli_module._ready_backends("whisper-cpp", "batch", None) == frozenset({"cpu", "metal"})


def test_ready_backends_faster_whisper_needs_gpu_and_cuda_devices(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli_module, "_cuda_device_count", lambda: 1)

    fake_gpu = {"model": "NVIDIA RTX 4090", "driver": "550.54.14", "vram": "24576 MiB"}
    assert cli_module._ready_backends("faster-whisper", "batch", fake_gpu) == frozenset({"cpu", "cuda"})
    assert cli_module._ready_backends("faster-whisper", "batch", None) == frozenset({"cpu"})


def test_ready_backends_faster_whisper_zero_cuda_devices_is_cpu_only(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli_module, "_cuda_device_count", lambda: 0)

    fake_gpu = {"model": "NVIDIA RTX 4090", "driver": "550.54.14", "vram": "24576 MiB"}
    assert cli_module._ready_backends("faster-whisper", "batch", fake_gpu) == frozenset({"cpu"})


def test_ready_backends_parakeet_reports_cuda_and_metal_independently(monkeypatch):
    """Unlike whisper-cpp/faster-whisper, parakeet's cuda readiness never
    needs the NVIDIA-only `gpu` probe -- torch.cuda.is_available() is
    already the real answer -- and metal is a wholly separate axis."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli_module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(cli_module, "_torch_mps_available", lambda: True)

    assert cli_module._ready_backends("parakeet", "batch", None) == frozenset({"cpu", "cuda", "metal"})


def test_ready_backends_parakeet_cpu_only_when_neither_backend_available(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(cli_module, "_torch_mps_available", lambda: False)

    assert cli_module._ready_backends("parakeet", "batch", None) == frozenset({"cpu"})


def test_ready_backends_parakeet_metal_only(monkeypatch):
    """The real shape on a Mac with no NVIDIA GPU at all."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(cli_module, "_torch_mps_available", lambda: True)

    assert cli_module._ready_backends("parakeet", "batch", None) == frozenset({"cpu", "metal"})


def test_doctor_engine_line_parakeet_reports_metal_even_with_no_nvidia_gpu(monkeypatch):
    """Regression test: this branch used to sit after a generic `if gpu is
    None: return ...cuda unavailable...` early-exit shared with
    faster-whisper -- correct for faster-whisper (cuda-only), but it
    silently swallowed parakeet's own metal readiness on every Mac (no
    NVIDIA GPU there at all). Moved above that early-exit; this is the
    exact case it was hiding."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(cli_module, "_torch_mps_available", lambda: True)

    line = cli_module._doctor_engine_line("parakeet", "batch", None)

    assert "cuda unavailable (no NVIDIA GPU detected)" in line
    assert "metal ready (torch.backends.mps.is_available() is True)" in line


def test_doctor_engine_line_parakeet_reports_real_cuda_with_gpu_detected(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli_module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(cli_module, "_torch_mps_available", lambda: False)

    fake_gpu = {"model": "NVIDIA RTX 4090", "driver": "550.54.14", "vram": "24576 MiB"}
    line = cli_module._doctor_engine_line("parakeet", "batch", fake_gpu)

    assert "cuda ready (torch.cuda.is_available() is True, NVIDIA RTX 4090 detected)" in line
    assert "metal unavailable" in line


def test_doctor_engine_line_parakeet_gpu_present_but_torch_reports_cpu_only(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli_module, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(cli_module, "_torch_mps_available", lambda: False)

    fake_gpu = {"model": "NVIDIA RTX 4090", "driver": "550.54.14", "vram": "24576 MiB"}
    line = cli_module._doctor_engine_line("parakeet", "batch", fake_gpu)

    assert "NVIDIA RTX 4090 detected but torch.cuda.is_available() is False" in line
    assert "CPU-only torch build" in line


def test_torch_mps_available_none_when_torch_not_installed(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setitem(sys.modules, "torch", None)

    assert cli_module._torch_mps_available() is None


# --- _wizard_pick_backends ---


def test_wizard_pick_backends_skips_prompt_when_only_cpu_ready(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ready_backends", lambda *a, **k: frozenset({"cpu"}))
    prompted = []
    monkeypatch.setattr(
        cli_module.questionary, "select",
        lambda *a, **k: prompted.append(1) or _FakeAsk("cpu"),
    )

    result = cli_module._wizard_pick_backends({"p1": "vosk"}, None)

    assert result == {}
    assert not prompted


def test_wizard_pick_backends_offers_choice_and_omits_cpu_selection(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ready_backends", lambda *a, **k: frozenset({"cpu", "cuda"}))
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: _FakeAsk("cuda"))

    result = cli_module._wizard_pick_backends({"p1": "faster-whisper"}, None)

    assert result == {"faster-whisper": "cuda"}


def test_wizard_pick_backends_picking_cpu_stays_omitted(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ready_backends", lambda *a, **k: frozenset({"cpu", "cuda"}))
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: _FakeAsk("cpu"))

    result = cli_module._wizard_pick_backends({"p1": "faster-whisper"}, None)

    assert result == {}


def test_wizard_pick_backends_aborts_on_none(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ready_backends", lambda *a, **k: frozenset({"cpu", "cuda"}))
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: _FakeAsk(None))

    assert cli_module._wizard_pick_backends({"p1": "faster-whisper"}, None) is None


def test_wizard_pick_backends_one_prompt_per_distinct_engine(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ready_backends", lambda *a, **k: frozenset({"cpu", "cuda"}))
    engines_prompted = []

    def _fake_select(question, *a, **k):
        engines_prompted.append(question)
        return _FakeAsk("cuda")

    monkeypatch.setattr(cli_module.questionary, "select", _fake_select)

    result = cli_module._wizard_pick_backends(
        {"p1": "faster-whisper", "p2": "faster-whisper", "p3": "vosk"}, None
    )

    assert len(engines_prompted) == 2  # once for faster-whisper, once for vosk
    assert result == {"faster-whisper": "cuda", "vosk": "cuda"}


def test_wizard_pick_backends_aborts_after_a_successful_whisper_cpp_cuda_rebuild(monkeypatch):
    """A rebuilt pywhispercpp extension can't take effect until a real
    process restart (see _offer_whisper_cpp_cuda_build's own docstring) --
    the wizard must abort here the same way a declined backend prompt
    does, never silently continue with the stale cpu-only build."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ready_backends", lambda *a, **k: frozenset({"cpu"}))
    monkeypatch.setattr(cli_module, "_offer_whisper_cpp_cuda_build", lambda: True)

    def _unexpected(*a, **k):
        raise AssertionError("must not prompt for a backend after aborting to rebuild")

    monkeypatch.setattr(cli_module.questionary, "select", _unexpected)

    assert cli_module._wizard_pick_backends({"p1": "whisper-cpp"}, None) is None


def test_wizard_pick_backends_continues_normally_when_no_rebuild_happens(monkeypatch):
    """Declining the rebuild offer (or it being a no-op on this
    platform/hardware) must fall through to the ordinary cpu-only-skip
    behavior, not abort the wizard."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ready_backends", lambda *a, **k: frozenset({"cpu"}))
    monkeypatch.setattr(cli_module, "_offer_whisper_cpp_cuda_build", lambda: False)
    prompted = []
    monkeypatch.setattr(
        cli_module.questionary, "select",
        lambda *a, **k: prompted.append(1) or _FakeAsk("cpu"),
    )

    result = cli_module._wizard_pick_backends({"p1": "whisper-cpp"}, None)

    assert result == {}
    assert not prompted  # cpu-only, no rebuild -- same as any other cpu-only engine


def test_wizard_run_threads_chosen_backend_into_reexec_args(monkeypatch):
    """End-to-end proof the wizard's backend choice actually reaches the
    re-exec'd `run` invocation, not just an internal dict."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "language": "en-US"}],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-en-batch"])
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "custom"))
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(
        cli_module, "_load_profile_for_wizard",
        lambda *a, **k: {"runtime": {"name": "faster-whisper"}},
    )
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_wizard_pick_backends", lambda *a, **k: {"faster-whisper": "cuda"})
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))

    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))

    cli_module._wizard_run()

    assert reexec_calls == [
        [
            "run", "whisper-medium-en-batch", "pack-a", "--repeats", "1", "--hardware", "custom",
            "--backend", "cuda",
        ],
    ]


def test_wizard_run_asks_backend_before_hardware(monkeypatch):
    """The hardware picker needs the chosen backend to preselect the right
    catalog entry (see _guess_hardware_label_for_backends) -- prove the
    wizard actually asks in that order, not just that both get asked."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "language": "en-US"}],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-en-batch"])
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(
        cli_module, "_load_profile_for_wizard",
        lambda *a, **k: {"runtime": {"name": "faster-whisper"}},
    )
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)

    call_order = []

    def fake_pick_backends(*a, **k):
        call_order.append("backend")
        return {"faster-whisper": "cuda"}

    def fake_pick_hardware(backends, gpu):
        call_order.append("hardware")
        return dict.fromkeys(backends, "custom")

    monkeypatch.setattr(cli_module, "_wizard_pick_backends", fake_pick_backends)
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", fake_pick_hardware)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_reexec", lambda args: None)

    cli_module._wizard_run()

    assert call_order == ["backend", "hardware"]


def test_wizard_run_passes_gpu_and_distinct_backends_into_hardware_picker(monkeypatch):
    """The hardware picker must actually receive what the backend step
    decided (as a deduplicated list of backend values), not just run
    after it."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "language": "en-US"}],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-en-batch"])
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(
        cli_module, "_load_profile_for_wizard",
        lambda *a, **k: {"runtime": {"name": "faster-whisper"}},
    )
    fake_gpu = {"model": "NVIDIA RTX A4000", "vram": "16384 MiB", "driver": "550.54"}
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: fake_gpu)
    monkeypatch.setattr(cli_module, "_wizard_pick_backends", lambda *a, **k: {"faster-whisper": "cuda"})

    captured = {}

    def fake_pick_by_backend(backends, gpu):
        captured["backends"] = backends
        captured["gpu"] = gpu
        return dict.fromkeys(backends, "custom")

    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", fake_pick_by_backend)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_reexec", lambda args: None)

    cli_module._wizard_run()

    assert captured["gpu"] == fake_gpu
    assert captured["backends"] == ["cuda"]


def test_wizard_run_asks_hardware_separately_per_distinct_backend(monkeypatch):
    """The actual gap this exists to fix: a batch mixing cuda for one
    engine and cpu for another must ask (and apply) hardware separately
    per backend, not one shared --hardware for the whole batch."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [
            {"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"},
            {"id": "vosk-small-en-batch", "language": "en-US", "benchmark_type": "batch"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "language": "en-US"}],
    )
    monkeypatch.setattr(
        cli_module, "_ask_matrix",
        lambda matrix: ["whisper-medium-en-batch", "vosk-small-en-batch"],
    )
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)

    def fake_load_profile(profile_id, *a, **k):
        runtime = "faster-whisper" if profile_id == "whisper-medium-en-batch" else "vosk"
        return {"runtime": {"name": runtime}}

    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", fake_load_profile)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    # faster-whisper gets cuda, vosk is never prompted (cpu-only engine) --
    # matches _wizard_pick_backends' own real contract (cpu entries omitted).
    monkeypatch.setattr(cli_module, "_wizard_pick_backends", lambda *a, **k: {"faster-whisper": "cuda"})

    captured_backends = {}

    def fake_pick_by_backend(backends, gpu):
        captured_backends["backends"] = backends
        return {"cpu": "intel-xeon-e3-1240-v6", "cuda": "nvidia-rtx-a4000"}

    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", fake_pick_by_backend)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))

    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))

    cli_module._wizard_run()

    assert captured_backends["backends"] == ["cpu", "cuda"]
    assert reexec_calls == [
        [
            "run", "whisper-medium-en-batch", "pack-a", "--repeats", "1",
            "--hardware", "nvidia-rtx-a4000", "--backend", "cuda",
        ],
        ["run", "vosk-small-en-batch", "pack-a", "--repeats", "1", "--hardware", "intel-xeon-e3-1240-v6"],
    ]


def test_detect_non_nvidia_gpu_darwin_reports_metal(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Darwin")

    result = cli_module._detect_non_nvidia_gpu()

    assert result is not None
    assert "Metal" in result


def test_detect_non_nvidia_gpu_linux_parses_lspci(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    lspci_output = (
        '00:02.0 "VGA compatible controller" "Advanced Micro Devices, Inc. [AMD/ATI]" '
        '"Radeon RX 7900 XTX" -r00 "Advanced Micro Devices, Inc. [AMD/ATI]" "Radeon RX 7900 XTX"\n'
    )
    monkeypatch.setattr(cli_module, "_run", lambda cmd: lspci_output if cmd[0] == "lspci" else None)

    result = cli_module._detect_non_nvidia_gpu()

    assert result is not None
    assert "Radeon RX 7900 XTX" in result


def test_detect_non_nvidia_gpu_linux_no_gpu_returns_none(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module, "_run", lambda cmd: None)

    assert cli_module._detect_non_nvidia_gpu() is None


def test_doctor_shows_non_nvidia_gpu_when_nvidia_absent(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_detect_non_nvidia_gpu", lambda: "Apple GPU (Metal-capable; this doesn't identify the exact model)")
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])
    # whisper-cpp's own branch does a real pywhispercpp import — stub it out
    # so this test exercises only the top-level GPU line, not that path.
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "GPU: none detected via nvidia-smi" in result.output
    assert "Non-NVIDIA GPU detected: Apple GPU" in result.output


def test_doctor_omits_non_nvidia_line_when_nothing_detected(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_detect_non_nvidia_gpu", lambda: None)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Non-NVIDIA GPU detected" not in result.output


_CPU_HARDWARE_ROWS = [
    {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
    {"id": "amd-epyc-7203", "display_name": "AMD EPYC 7203", "vendor": "AMD", "category": "cpu"},
]


def test_guess_hardware_id_returns_the_matched_row_id(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )

    assert cli_module._guess_hardware_id(_CPU_HARDWARE_ROWS) == "intel-xeon-e3-1240-v6"


def test_guess_hardware_label_still_returns_the_full_label(monkeypatch):
    """Regression check after extracting _guess_hardware_id out of this
    function — the wizard picker's own preselection behavior must be
    unchanged."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )

    assert cli_module._guess_hardware_label(_CPU_HARDWARE_ROWS) == "Intel Xeon E3-1240 v6 (Intel)"


_GPU_HARDWARE_ROWS = [
    {"id": "nvidia-rtx-a4000", "display_name": "NVIDIA RTX A4000", "vendor": "NVIDIA", "category": "gpu"},
    {"id": "nvidia-rtx-a6000", "display_name": "NVIDIA RTX A6000", "vendor": "NVIDIA", "category": "gpu"},
]
_MIXED_HARDWARE_ROWS = _CPU_HARDWARE_ROWS + _GPU_HARDWARE_ROWS


def test_guess_gpu_hardware_id_returns_the_matched_row_id(monkeypatch):
    from oesb_runner import cli as cli_module

    assert (
        cli_module._guess_gpu_hardware_id(_GPU_HARDWARE_ROWS, {"model": "NVIDIA RTX A4000"})
        == "nvidia-rtx-a4000"
    )


def test_guess_gpu_hardware_id_returns_none_on_no_match(monkeypatch):
    from oesb_runner import cli as cli_module

    assert cli_module._guess_gpu_hardware_id(_GPU_HARDWARE_ROWS, {"model": "Some Unknown Card"}) is None


def test_guess_gpu_hardware_id_returns_none_when_model_missing(monkeypatch):
    from oesb_runner import cli as cli_module

    assert cli_module._guess_gpu_hardware_id(_GPU_HARDWARE_ROWS, {}) is None


def test_guess_hardware_label_for_backends_prefers_gpu_when_backend_chosen(monkeypatch):
    from oesb_runner import cli as cli_module

    label = cli_module._guess_hardware_label_for_backends(
        _MIXED_HARDWARE_ROWS, {"model": "NVIDIA RTX A4000"}, {"faster-whisper": "cuda"}
    )

    assert label == "NVIDIA RTX A4000 (NVIDIA)"


def test_guess_hardware_label_for_backends_no_fallback_to_cpu_on_gpu_miss(monkeypatch):
    """A GPU backend was chosen but nothing in the catalog matches -- must
    NOT fall back to suggesting the CPU entry (that's the exact wrong-
    preselection bug this whole change fixes)."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )

    label = cli_module._guess_hardware_label_for_backends(
        _MIXED_HARDWARE_ROWS, {"model": "Some Unknown Card"}, {"faster-whisper": "cuda"}
    )

    assert label is None


def test_guess_hardware_label_for_backends_falls_back_to_cpu_when_cpu_only(monkeypatch):
    """No GPU backend chosen (empty dict, _wizard_pick_backends' cpu-only
    shape) -- today's exact CPU-guess behavior, unchanged."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )

    label = cli_module._guess_hardware_label_for_backends(_MIXED_HARDWARE_ROWS, None, {})

    assert label == "Intel Xeon E3-1240 v6 (Intel)"


def test_hardware_result_gaps_reports_uncovered_official_profiles(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)

    def fake_get_json(url, timeout):
        if "/profiles" in url:
            return {"profiles": [
                {"id": "whisper-tiny-en-batch", "runtime": "faster-whisper"},
                {"id": "whisper-medium-en-batch", "runtime": "faster-whisper"},
                {"id": "vosk-small-en-batch", "runtime": "vosk"},
                {"id": "some-other-runtime-profile", "runtime": "an-engine-not-installed"},
            ]}
        assert "/leaderboards" in url and "hardware=intel-xeon-e3-1240-v6" in url
        return {"results": [{"profile_id": "whisper-tiny-en-batch"}]}

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)

    gaps = cli_module._hardware_result_gaps("intel-xeon-e3-1240-v6", "http://api.example")

    # Covered profile excluded, uninstalled-runtime profile excluded, the
    # two genuinely-uncovered ones for installed engines remain.
    assert gaps == ["vosk-small-en-batch", "whisper-medium-en-batch"]


def test_hardware_result_gaps_empty_when_everything_covered(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)

    def fake_get_json(url, timeout):
        if "/profiles" in url:
            return {"profiles": [{"id": "whisper-tiny-en-batch", "runtime": "faster-whisper"}]}
        return {"results": [{"profile_id": "whisper-tiny-en-batch"}]}

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)

    assert cli_module._hardware_result_gaps("intel-xeon-e3-1240-v6", "http://api.example") == []


def test_hardware_result_gaps_returns_none_on_network_failure(monkeypatch):
    """None, not an empty list — the caller needs to tell "checked, found
    nothing missing" apart from "couldn't check at all"."""
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)

    def fake_get_json(url, timeout):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)

    assert cli_module._hardware_result_gaps("intel-xeon-e3-1240-v6", "http://api.example") is None


def test_hardware_result_gaps_empty_when_no_engines_installed(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: None)

    def _fail_if_called(url, timeout):
        raise AssertionError("should not hit the network with no installed engines to check")

    monkeypatch.setattr(cli_module, "_get_json", _fail_if_called)

    assert cli_module._hardware_result_gaps("intel-xeon-e3-1240-v6", "http://api.example") == []


def test_doctor_reports_hardware_result_gaps(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: _CPU_HARDWARE_ROWS)
    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )
    monkeypatch.setattr(
        cli_module, "_hardware_result_gaps",
        lambda hardware_id, api_url: ["whisper-medium-en-batch", "vosk-small-en-batch"],
    )
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "2 official profile(s)" in result.output
    assert "intel-xeon-e3-1240-v6" in result.output
    assert "whisper-medium-en-batch" in result.output
    assert "vosk-small-en-batch" in result.output


def test_doctor_reports_full_coverage_when_no_gaps(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: _CPU_HARDWARE_ROWS)
    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )
    monkeypatch.setattr(cli_module, "_hardware_result_gaps", lambda hardware_id, api_url: [])
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "already has a public result" in result.output


def test_doctor_reports_when_hardware_not_detected(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: _CPU_HARDWARE_ROWS)
    monkeypatch.setattr(cli_module, "_capture_cpu", lambda unavailable: {"model": None})
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "couldn't confidently match this CPU" in result.output


def test_doctor_reports_when_result_coverage_check_cant_reach_network(monkeypatch):
    from oesb_runner import cli as cli_module

    _assume_all_engines_installed(monkeypatch, cli_module)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: _CPU_HARDWARE_ROWS)
    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )
    monkeypatch.setattr(cli_module, "_hardware_result_gaps", lambda hardware_id, api_url: None)
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "couldn't reach" in result.output


def test_wizard_list_profiles_reexecs_the_subcommand(monkeypatch):
    from oesb_runner import cli as cli_module

    calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: calls.append(args))

    # First select() call picks the action; loop must then see "Exit" or it spins forever.
    responses = iter(["List available profiles", "Exit"])
    monkeypatch.setattr(
        cli_module.questionary, "select", lambda *a, **k: _FakeAsk(next(responses))
    )

    cli_module._run_wizard()

    assert calls == [["list-profiles"]]


def test_wizard_dispatch_calls_wizard_run(monkeypatch):
    from oesb_runner import cli as cli_module

    called = []
    monkeypatch.setattr(cli_module, "_wizard_run", lambda: called.append(True))

    responses = iter(["Run benchmark(s)", "Exit"])
    monkeypatch.setattr(
        cli_module.questionary, "select", lambda *a, **k: _FakeAsk(next(responses))
    )

    cli_module._run_wizard()

    assert called == [True]


def test_wizard_dispatch_calls_wizard_run_streaming(monkeypatch):
    from oesb_runner import cli as cli_module

    called = []
    monkeypatch.setattr(cli_module, "_wizard_run_streaming", lambda: called.append(True))

    responses = iter(["Run streaming benchmark(s)", "Exit"])
    monkeypatch.setattr(
        cli_module.questionary, "select", lambda *a, **k: _FakeAsk(next(responses))
    )

    cli_module._run_wizard()

    assert called == [True]


def test_wizard_run_streaming_calls_wizard_run_matrix_with_streaming(monkeypatch):
    from oesb_runner import cli as cli_module

    calls = []
    monkeypatch.setattr(cli_module, "_wizard_run_matrix", lambda benchmark_type: calls.append(benchmark_type))

    cli_module._wizard_run_streaming()

    assert calls == ["streaming"]


def test_wizard_run_calls_wizard_run_matrix_with_batch(monkeypatch):
    from oesb_runner import cli as cli_module

    calls = []
    monkeypatch.setattr(cli_module, "_wizard_run_matrix", lambda benchmark_type: calls.append(benchmark_type))

    cli_module._wizard_run()

    assert calls == ["batch"]


def test_wizard_run_builds_combos_and_continues_past_failures(monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [
            {"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"},
            {"id": "whisper-medium-fr-batch", "language": "fr-FR", "benchmark_type": "batch"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "pack-a", "visibility": "open", "language": "en-US"},
            {"id": "pack-b", "visibility": "open", "language": "fr-FR"},
            {"id": "unrelated-pack", "visibility": "open", "language": "de-DE"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_ask_matrix",
        lambda matrix: ["whisper-medium-en-batch", "whisper-medium-fr-batch"],
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "intel-xeon-e3-1240-v6"))
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )

    reexec_calls = []

    def fake_reexec(args):
        reexec_calls.append(args)
        if args[1] == "whisper-medium-en-batch":
            raise typer.Exit(code=1)

    monkeypatch.setattr(cli_module, "_reexec", fake_reexec)

    cli_module._wizard_run()

    assert reexec_calls == [
        ["run", "whisper-medium-en-batch", "pack-a", "--repeats", "1", "--hardware", "intel-xeon-e3-1240-v6"],
        ["run", "whisper-medium-fr-batch", "pack-b", "--repeats", "1", "--hardware", "intel-xeon-e3-1240-v6"],
    ]
    out = capsys.readouterr().out
    assert "✗ whisper-medium-en-batch  x  pack-a" in out
    assert "✓ whisper-medium-fr-batch  x  pack-b" in out


def test_wizard_run_expands_combos_when_multiple_packs_chosen_for_one_profile(monkeypatch):
    """A cell whose profile matches >1 pack and where the user checks more
    than one of them runs all of them, not just the first match."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-nl-batch", "language": "nl-NL", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "fleurs-nl", "visibility": "open", "language": "nl-NL"},
            {"id": "common-voice-nl-elderly", "visibility": "open", "language": "nl-NL"},
        ],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-nl-batch"])
    monkeypatch.setattr(
        cli_module, "_choose_packs_for_language",
        lambda profile_id, matching_packs, *a, **k: ["fleurs-nl", "common-voice-nl-elderly"],
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "custom"))
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )
    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))

    cli_module._wizard_run()

    assert reexec_calls == [
        ["run", "whisper-medium-nl-batch", "fleurs-nl", "--repeats", "1", "--hardware", "custom"],
        ["run", "whisper-medium-nl-batch", "common-voice-nl-elderly", "--repeats", "1", "--hardware", "custom"],
    ]


def test_wizard_run_drops_cell_silently_when_pack_choice_empty(monkeypatch):
    """Declining every pack for an ambiguous cell drops only that cell —
    other cells in the same batch still run."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [
            {"id": "whisper-medium-nl-batch", "language": "nl-NL", "benchmark_type": "batch"},
            {"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "fleurs-nl", "visibility": "open", "language": "nl-NL"},
            {"id": "common-voice-nl-elderly", "visibility": "open", "language": "nl-NL"},
            {"id": "pack-en", "visibility": "open", "language": "en-US"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_ask_matrix",
        lambda matrix: ["whisper-medium-nl-batch", "whisper-medium-en-batch"],
    )
    monkeypatch.setattr(
        cli_module, "_choose_packs_for_language",
        lambda language, matching_packs, *a, **k: (
            [] if language == "nl-NL" else [matching_packs[0]["id"]]
        ),
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "custom"))
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )
    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))

    cli_module._wizard_run()

    assert reexec_calls == [
        ["run", "whisper-medium-en-batch", "pack-en", "--repeats", "1", "--hardware", "custom"],
    ]


def test_wizard_run_aborts_when_pack_choice_cancelled(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-nl-batch", "language": "nl-NL", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "fleurs-nl", "visibility": "open", "language": "nl-NL"},
            {"id": "common-voice-nl-elderly", "visibility": "open", "language": "nl-NL"},
        ],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-nl-batch"])
    monkeypatch.setattr(cli_module, "_choose_packs_for_language", lambda profile_id, matching_packs, *a, **k: None)
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "custom"))
    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))

    cli_module._wizard_run()

    assert reexec_calls == []


def test_wizard_run_asks_pack_choice_once_per_language_not_once_per_profile(monkeypatch):
    """The actual gap this exists to fix: a matrix selection spanning two
    engines that share one language (e.g. whisper + vosk, both nl-NL)
    must prompt for which pack(s) to run exactly once for that language
    and reuse the answer for every profile sharing it -- not repeat an
    identical prompt once per profile/engine."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [
            {"id": "whisper-medium-nl-batch", "language": "nl-NL", "benchmark_type": "batch"},
            {"id": "vosk-small-nl-batch", "language": "nl-NL", "benchmark_type": "batch"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "fleurs-nl", "visibility": "open", "language": "nl-NL"},
            {"id": "common-voice-nl-elderly", "visibility": "open", "language": "nl-NL"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_ask_matrix",
        lambda matrix: ["whisper-medium-nl-batch", "vosk-small-nl-batch"],
    )

    calls = []

    def fake_choose_packs(language, matching_packs, *a, **k):
        calls.append(language)
        return ["fleurs-nl"]

    monkeypatch.setattr(cli_module, "_choose_packs_for_language", fake_choose_packs)
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "custom"))
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )
    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))

    cli_module._wizard_run()

    assert calls == ["nl-NL"]  # asked once, not once per profile
    assert reexec_calls == [
        ["run", "whisper-medium-nl-batch", "fleurs-nl", "--repeats", "1", "--hardware", "custom"],
        ["run", "vosk-small-nl-batch", "fleurs-nl", "--repeats", "1", "--hardware", "custom"],
    ]


def test_wizard_run_declines_confirmation_runs_nothing(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "language": "en-US"}],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-en-batch"])
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("2"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "custom"))
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {}) for pid, pack in combos],
    )

    calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: calls.append(args))

    cli_module._wizard_run()

    assert calls == []


# --- ADR-0012: _wizard_run_concurrency ---


def test_wizard_run_concurrency_no_profiles_found_echoes_and_returns(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )

    def _unexpected(*a, **k):
        raise AssertionError("must never prompt when no concurrency profiles exist")

    monkeypatch.setattr(cli_module.questionary, "checkbox", _unexpected)

    cli_module._wizard_run_concurrency()  # must not raise


def test_wizard_run_concurrency_skips_the_language_pack_matrix_entirely(monkeypatch):
    """The whole point of ADR-0012: no _ask_matrix, no _choose_packs_for_language
    -- a pack is picked automatically, content doesn't matter."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-concurrency", "benchmark_type": "concurrency"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "fleurs-nl", "visibility": "open", "language": "nl-NL"},
            {"id": "librispeech-en", "visibility": "open", "language": "en-US"},
        ],
    )
    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk(["whisper-medium-concurrency"]))

    def _unexpected(*a, **k):
        raise AssertionError("must never touch the language/pack matrix machinery")

    monkeypatch.setattr(cli_module, "_ask_matrix", _unexpected)
    monkeypatch.setattr(cli_module, "_choose_packs_for_language", _unexpected)

    captured_combos = {}

    def _fake_confirm_and_run(combos):
        captured_combos["combos"] = combos

    monkeypatch.setattr(cli_module, "_wizard_confirm_and_run", _fake_confirm_and_run)

    cli_module._wizard_run_concurrency()

    # librispeech-en preferred over fleurs-nl -- a deterministic pick, not
    # asked of the user (content is irrelevant to this benchmark type).
    assert captured_combos["combos"] == [("whisper-medium-concurrency", "librispeech-en")]


def test_wizard_run_concurrency_falls_back_to_first_open_pack_alphabetically(monkeypatch):
    """When librispeech-en isn't available, still picks deterministically
    rather than prompting."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-concurrency", "benchmark_type": "concurrency"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "zzz-pack", "visibility": "open", "language": "de-DE"},
            {"id": "aaa-pack", "visibility": "open", "language": "fr-FR"},
        ],
    )
    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk(["whisper-medium-concurrency"]))

    captured_combos = {}
    monkeypatch.setattr(cli_module, "_wizard_confirm_and_run", lambda combos: captured_combos.update(combos=combos))

    cli_module._wizard_run_concurrency()

    assert captured_combos["combos"] == [("whisper-medium-concurrency", "aaa-pack")]


def test_wizard_run_concurrency_no_open_packs_echoes_and_returns(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-concurrency", "benchmark_type": "concurrency"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "gated-pack", "visibility": "gated", "language": "en-US"}],
    )
    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk(["whisper-medium-concurrency"]))

    def _unexpected(combos):
        raise AssertionError("must not proceed with no open packs available")

    monkeypatch.setattr(cli_module, "_wizard_confirm_and_run", _unexpected)

    cli_module._wizard_run_concurrency()  # must not raise


def test_wizard_run_concurrency_multiple_profiles_selected(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [
            {"id": "whisper-medium-concurrency", "benchmark_type": "concurrency"},
            {"id": "whisper-large-v3-concurrency", "benchmark_type": "concurrency"},
            {"id": "whisper-medium-en-batch", "benchmark_type": "batch", "language": "en-US"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "librispeech-en", "visibility": "open", "language": "en-US"}],
    )

    offered_choices = []

    def _fake_checkbox(message, choices):
        offered_choices.extend(c.value for c in choices)
        return _FakeAsk([c.value for c in choices])

    monkeypatch.setattr(cli_module.questionary, "checkbox", _fake_checkbox)

    captured_combos = {}
    monkeypatch.setattr(cli_module, "_wizard_confirm_and_run", lambda combos: captured_combos.update(combos=combos))

    cli_module._wizard_run_concurrency()

    # The batch profile is never offered as a choice -- only concurrency ones.
    assert offered_choices == ["whisper-medium-concurrency", "whisper-large-v3-concurrency"]
    assert captured_combos["combos"] == [
        ("whisper-medium-concurrency", "librispeech-en"),
        ("whisper-large-v3-concurrency", "librispeech-en"),
    ]


# --- ADR-0009: wizard per-engine parameter step ---
# Real, already-migrated profiles under REPO_ROOT/profiles are used
# directly (no synthetic fixtures, no network) — whisper-medium-en-batch
# and whisper-tiny-en-batch both declare beam_size (allowed [1,2,4,5,8],
# default 5) and vad (default true); vosk-small-en-batch declares nothing.


def test_wizard_engine_parameters_enter_through_matches_todays_behavior(monkeypatch):
    """Regression guard (hard constraint): a full Enter-through must
    produce the exact same expanded combos/overrides as before this
    feature existed — one entry per combo, empty overrides, so `_reexec`
    gets no extra --param flags at all."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk(""))

    combos = [("whisper-medium-en-batch", "pack-a"), ("vosk-small-en-batch", "pack-b")]
    expanded = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )

    assert expanded == [
        ("whisper-medium-en-batch", "pack-a", {}),
        ("vosk-small-en-batch", "pack-b", {}),
    ]


def _fake_text_by_param(**responses: str):
    """Build a questionary.text stub that answers by which parameter name
    appears in the prompt text, defaulting to "" (Enter) for anything not
    named — robust to however many parameters a profile's `overridable`
    block declares, rather than depending on prompt order/count."""
    def _fake_text(question, *a, **k):
        for param_name, value in responses.items():
            if f"] {param_name} " in question:
                return _FakeAsk(value)
        return _FakeAsk("")
    return _fake_text


def test_wizard_engine_parameters_single_value_overrides_all_cells_of_engine(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", _fake_text_by_param(beam_size="8"))

    combos = [("whisper-medium-en-batch", "pack-a"), ("whisper-tiny-en-batch", "pack-b")]
    expanded = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )

    assert expanded == [
        ("whisper-medium-en-batch", "pack-a", {"beam_size": "8"}),
        ("whisper-tiny-en-batch", "pack-b", {"beam_size": "8"}),
    ]


def test_wizard_engine_parameters_comma_list_sweeps_cells_x_values(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", _fake_text_by_param(beam_size="1,4,8"))

    combos = [("whisper-medium-en-batch", "pack-a")]
    expanded = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )

    assert expanded == [
        ("whisper-medium-en-batch", "pack-a", {"beam_size": "1"}),
        ("whisper-medium-en-batch", "pack-a", {"beam_size": "4"}),
        ("whisper-medium-en-batch", "pack-a", {"beam_size": "8"}),
    ]


def test_wizard_engine_parameters_cross_product_of_multiple_swept_params(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module.questionary, "text",
        _fake_text_by_param(beam_size="1,8", vad="true,false"),
    )

    combos = [("whisper-medium-en-batch", "pack-a")]
    expanded = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )

    assert len(expanded) == 4
    assert {
        (o.get("beam_size"), o.get("vad")) for _, _, o in expanded
    } == {("1", "true"), ("1", "false"), ("8", "true"), ("8", "false")}


def test_wizard_engine_parameters_never_leaks_onto_engine_without_it(monkeypatch):
    """Mixed whisper+vosk selection: whisper gets prompted (once per its
    overridable parameters, all scoped to faster-whisper), vosk — no
    overridable parameters at all — is never asked anything and never
    receives an override."""
    from oesb_runner import cli as cli_module

    prompts = []

    def _fake_text(question, *a, **k):
        prompts.append(question)
        return _FakeAsk("8" if "] beam_size " in question else "")

    monkeypatch.setattr(cli_module.questionary, "text", _fake_text)

    combos = [("whisper-medium-en-batch", "pack-a"), ("vosk-small-en-batch", "pack-b")]
    expanded = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )

    assert prompts  # at least one prompt happened
    assert all(p.startswith("[faster-whisper]") for p in prompts)
    assert ("vosk-small-en-batch", "pack-b", {}) in expanded
    assert ("whisper-medium-en-batch", "pack-a", {"beam_size": "8"}) in expanded


def test_wizard_engine_parameters_rejects_out_of_domain_sweep_value(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", _fake_text_by_param(beam_size="3"))

    combos = [("whisper-medium-en-batch", "pack-a")]
    with pytest.raises(typer.Exit):
        cli_module._wizard_engine_parameters(
            combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
        )


def test_wizard_engine_parameters_aborts_on_none(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk(None))

    combos = [("whisper-medium-en-batch", "pack-a")]
    result = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )
    assert result is None


# --- ADR-0012 addendum: the concurrency prompt's blank-Enter sweep nudge ---


def test_suggested_concurrency_sweep_clamped_to_profile_range():
    from oesb_runner import cli as cli_module

    unclamped = {"overridable": {"concurrency": {"range": {"min": 1, "max": 64}}}}
    assert cli_module._suggested_concurrency_sweep(unclamped) == "1,4,8,16"

    clamped = {"overridable": {"concurrency": {"range": {"min": 1, "max": 8}}}}
    assert cli_module._suggested_concurrency_sweep(clamped) == "1,4,8"

    no_range = {"overridable": {"concurrency": {}}}
    assert cli_module._suggested_concurrency_sweep(no_range) == "1,4,8,16"


def test_wizard_engine_parameters_blank_concurrency_defaults_to_auto(monkeypatch):
    """The actual gap this exists to fix: plain Enter-through on every
    other parameter means "run once, at the default" -- concurrency=1
    alone shows nothing about load behavior. Blank input here must resolve
    to the "auto" sentinel (expanded into real levels later, by
    `_run_concurrency_auto_sweep`, once throughput is actually measured)
    rather than silently doing the single boring run or guessing a static
    sweep that ignores this hardware's real ceiling."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk(""))

    combos = [("whisper-medium-concurrency", "pack-a")]
    expanded = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )

    assert expanded == [("whisper-medium-concurrency", "pack-a", {"concurrency": "auto"})]


def test_wizard_engine_parameters_explicit_auto_also_resolves_to_the_sentinel(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", _fake_text_by_param(concurrency="auto"))

    combos = [("whisper-medium-concurrency", "pack-a")]
    expanded = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )

    assert expanded == [("whisper-medium-concurrency", "pack-a", {"concurrency": "auto"})]


def test_wizard_engine_parameters_concurrency_prompt_explains_auto_and_shows_an_example(monkeypatch):
    from oesb_runner import cli as cli_module

    prompts = []

    def _fake_text(question, *a, **k):
        prompts.append(question)
        return _FakeAsk("")

    monkeypatch.setattr(cli_module.questionary, "text", _fake_text)

    combos = [("whisper-medium-concurrency", "pack-a")]
    cli_module._wizard_engine_parameters(combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL)

    assert any("auto-detect" in p and "1,4,8,16" in p for p in prompts)


def test_wizard_engine_parameters_explicit_concurrency_value_overrides_auto(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", _fake_text_by_param(concurrency="2"))

    combos = [("whisper-medium-concurrency", "pack-a")]
    expanded = cli_module._wizard_engine_parameters(
        combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL
    )

    assert expanded == [("whisper-medium-concurrency", "pack-a", {"concurrency": "2"})]


# --- _run_concurrency_auto_sweep: doubling until throughput plateaus ---


def test_next_concurrency_level_doubles_then_steps_by_four():
    from oesb_runner import cli as cli_module

    assert cli_module._next_concurrency_level(1) == 2
    assert cli_module._next_concurrency_level(2) == 4
    assert cli_module._next_concurrency_level(4) == 8
    assert cli_module._next_concurrency_level(8) == 12
    assert cli_module._next_concurrency_level(12) == 16
    assert cli_module._next_concurrency_level(20) == 24


def test_run_concurrency_auto_sweep_stops_after_two_consecutive_low_gain_levels(monkeypatch):
    from oesb_runner import cli as cli_module

    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))
    monkeypatch.setattr(cli_module, "_concurrency_ceiling", lambda *a, **k: 64)
    # 1 -> 10, 2 -> 20 (+100%, keep climbing), 4 -> 22 (+10%, strike 1),
    # 8 -> 23 (+4.5%, strike 2 -- two low-gain levels in a row, stop)
    throughputs = iter([10.0, 20.0, 22.0, 23.0])
    monkeypatch.setattr(cli_module, "_latest_result_throughput", lambda *a, **k: next(throughputs))

    outcomes = cli_module._run_concurrency_auto_sweep(
        "whispercpp-medium-concurrency", "pack-a", "2", "cpu", "some-cpu", {}
    )

    assert [o[2]["concurrency"] for o in outcomes] == ["1", "2", "4", "8"]
    assert all(o[3] for o in outcomes)
    assert len(reexec_calls) == 4
    assert "concurrency=1" in reexec_calls[0]
    assert "concurrency=8" in reexec_calls[3]


def test_run_concurrency_auto_sweep_a_single_low_gain_dip_does_not_stop_it(monkeypatch):
    """A single noisy low reading must not be mistaken for a real plateau
    -- only two low-gain levels IN A ROW should stop the sweep. A level
    that recovers back above the gain floor resets the streak, and the
    step size switches from doubling to +4 once level >= 8."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_reexec", lambda args: None)
    monkeypatch.setattr(cli_module, "_concurrency_ceiling", lambda *a, **k: 64)
    # 1->10, 2->20 (+100%), 4->21 (+5%, strike 1), 8->50 (+138%, recovers --
    # resets the streak), 12->51 (+2%, strike 1 again), 16->52 (+2%, strike 2 -- stop)
    throughputs = iter([10.0, 20.0, 21.0, 50.0, 51.0, 52.0])
    monkeypatch.setattr(cli_module, "_latest_result_throughput", lambda *a, **k: next(throughputs))

    outcomes = cli_module._run_concurrency_auto_sweep(
        "whispercpp-medium-concurrency", "pack-a", "2", "cpu", "some-cpu", {}
    )

    assert [o[2]["concurrency"] for o in outcomes] == ["1", "2", "4", "8", "12", "16"]


def test_run_concurrency_auto_sweep_stops_at_the_profile_ceiling(monkeypatch):
    """Throughput that never plateaus on its own (unrealistic, but the
    ceiling must still be a hard stop -- ADR-0012's whisper-cpp per-instance
    memory cost is exactly why this ceiling exists)."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_reexec", lambda args: None)
    monkeypatch.setattr(cli_module, "_concurrency_ceiling", lambda *a, **k: 4)
    throughputs = iter([10.0, 100.0, 1000.0])
    monkeypatch.setattr(cli_module, "_latest_result_throughput", lambda *a, **k: next(throughputs))

    outcomes = cli_module._run_concurrency_auto_sweep(
        "whispercpp-medium-concurrency", "pack-a", "2", "cpu", "some-cpu", {}
    )

    assert [o[2]["concurrency"] for o in outcomes] == ["1", "2", "4"]


def test_run_concurrency_auto_sweep_stops_on_a_failed_level(monkeypatch):
    from oesb_runner import cli as cli_module

    def _fail_at_two(args):
        if "concurrency=2" in args:
            raise typer.Exit(code=1)

    monkeypatch.setattr(cli_module, "_reexec", _fail_at_two)
    monkeypatch.setattr(cli_module, "_concurrency_ceiling", lambda *a, **k: 64)
    monkeypatch.setattr(cli_module, "_latest_result_throughput", lambda *a, **k: 10.0)

    outcomes = cli_module._run_concurrency_auto_sweep(
        "whispercpp-medium-concurrency", "pack-a", "2", "cpu", "some-cpu", {}
    )

    assert [(o[2]["concurrency"], o[3]) for o in outcomes] == [("1", True), ("2", False)]


def test_run_concurrency_auto_sweep_preserves_other_overrides_at_every_level(monkeypatch):
    from oesb_runner import cli as cli_module

    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))
    monkeypatch.setattr(cli_module, "_concurrency_ceiling", lambda *a, **k: 2)
    monkeypatch.setattr(cli_module, "_latest_result_throughput", lambda *a, **k: 10.0)

    outcomes = cli_module._run_concurrency_auto_sweep(
        "whispercpp-medium-concurrency", "pack-a", "2", "cpu", "some-cpu", {"threads": "8"}
    )

    assert all(o[2]["threads"] == "8" for o in outcomes)
    assert all("threads=8" in c for c in reexec_calls)


def test_run_concurrency_auto_sweep_passes_backend_and_hardware_through(monkeypatch):
    from oesb_runner import cli as cli_module

    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))
    monkeypatch.setattr(cli_module, "_concurrency_ceiling", lambda *a, **k: 1)
    monkeypatch.setattr(cli_module, "_latest_result_throughput", lambda *a, **k: 10.0)

    cli_module._run_concurrency_auto_sweep(
        "whispercpp-medium-concurrency", "pack-a", "3", "cuda", "nvidia-rtx-4090", {}
    )

    assert reexec_calls == [
        [
            "run", "whispercpp-medium-concurrency", "pack-a", "--repeats", "3",
            "--hardware", "nvidia-rtx-4090", "--backend", "cuda", "--param", "concurrency=1",
        ]
    ]


def test_latest_result_throughput_reads_the_newest_matching_file(tmp_path):
    from oesb_runner import cli as cli_module

    older = tmp_path / "whispercpp-medium-concurrency__pack-a__20260101T000000Z.json"
    older.write_text(json.dumps({"metrics": {"throughput": {"value": 1.0}}}))
    newer = tmp_path / "whispercpp-medium-concurrency__pack-a__20260101T000100Z.json"
    newer.write_text(json.dumps({"metrics": {"throughput": {"value": 2.0}}}))
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    result = cli_module._latest_result_throughput(str(tmp_path), "whispercpp-medium-concurrency", "pack-a")

    assert result == 2.0


def test_latest_result_throughput_none_when_no_result_written(tmp_path):
    from oesb_runner import cli as cli_module

    assert cli_module._latest_result_throughput(str(tmp_path), "whispercpp-medium-concurrency", "pack-a") is None


# --- _wizard_confirm_and_run routes concurrency=auto to the sweep helper ---


def _stub_confirm_and_run_collaborators(monkeypatch, cli_module, engine="whisper-cpp"):
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", lambda *a, **k: {"runtime": {"name": engine}})
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module, "_wizard_pick_backends", lambda runtime_by_profile, gpu: dict.fromkeys(runtime_by_profile.values(), "cpu"))
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "apple-m1-pro"))
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("2"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))


def test_wizard_confirm_and_run_routes_auto_concurrency_to_the_sweep_helper(monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    _stub_confirm_and_run_collaborators(monkeypatch, cli_module)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {"concurrency": "auto"}) for pid, pack in combos],
    )
    sweep_calls = []
    monkeypatch.setattr(
        cli_module,
        "_run_concurrency_auto_sweep",
        lambda *a, **k: sweep_calls.append(a)
        or [("whispercpp-medium-concurrency", "pack-a", {"concurrency": "1"}, True)],
    )
    monkeypatch.setattr(
        cli_module, "_reexec", lambda args: pytest.fail("auto-sweep combos must not fall through to a plain _reexec")
    )

    cli_module._wizard_confirm_and_run([("whispercpp-medium-concurrency", "pack-a")])

    out = capsys.readouterr().out
    assert "auto-sweeps will run more" in out
    assert len(sweep_calls) == 1
    assert sweep_calls[0][:2] == ("whispercpp-medium-concurrency", "pack-a")


def test_wizard_confirm_and_run_manual_sweep_skips_the_auto_note_and_helper(monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    _stub_confirm_and_run_collaborators(monkeypatch, cli_module)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [(pid, pack, {"concurrency": "4"}) for pid, pack in combos],
    )
    monkeypatch.setattr(
        cli_module, "_run_concurrency_auto_sweep",
        lambda *a, **k: pytest.fail("a manual sweep value must not trigger the auto-sweep helper"),
    )
    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))

    cli_module._wizard_confirm_and_run([("whispercpp-medium-concurrency", "pack-a")])

    out = capsys.readouterr().out
    assert "auto-sweeps will run more" not in out
    assert len(reexec_calls) == 1
    assert "concurrency=4" in reexec_calls[0]


def test_wizard_run_confirmation_states_total_including_repeats(monkeypatch, capsys):
    """Fixes the pre-existing undercount: --repeats 2 must be reflected in
    the stated total, not just the combo count."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "language": "en-US"}],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-en-batch"])
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "intel-xeon-e3-1240-v6"))
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [
            (pid, pack, {"beam_size": "1"}) for pid, pack in combos
        ] + [(pid, pack, {"beam_size": "8"}) for pid, pack in combos],
    )
    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("3"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_reexec", lambda args: None)

    cli_module._wizard_run()

    out = capsys.readouterr().out
    assert "About to run 2 benchmark(s) (6 runs incl. 3 repeats)" in out


def test_wizard_run_warns_above_twenty_expanded_combos(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "language": "en-US"}],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-en-batch"])
    monkeypatch.setattr(cli_module, "_pick_hardware_ids_by_backend", lambda backends, gpu: dict.fromkeys(backends, "intel-xeon-e3-1240-v6"))
    monkeypatch.setattr(cli_module, "_preflight_pack_credentials", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_preflight_engines", lambda combos, *a, **k: combos)
    monkeypatch.setattr(cli_module, "_load_profile_for_wizard", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)
    monkeypatch.setattr(
        cli_module, "_wizard_engine_parameters",
        lambda combos, *a, **k: [
            (pid, pack, {"beam_size": str(i)}) for pid, pack in combos for i in range(25)
        ],
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))

    confirm_prompts = []

    def _fake_confirm(question, *a, **k):
        confirm_prompts.append(question)
        return _FakeAsk(False)  # decline the soft-warning confirm

    monkeypatch.setattr(cli_module.questionary, "confirm", _fake_confirm)
    reexec_calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: reexec_calls.append(args))

    cli_module._wizard_run()

    assert any("25 combos" in p for p in confirm_prompts)
    assert reexec_calls == []  # declined the warning, never proceeded to run anything


_MDC_CREDENTIAL = {
    "env_var": "MDC_API_KEY",
    "signup_url": "https://mozilladatacollective.com",
    "instructions": "Get an API key from the MDC dashboard, then run `pip install datacollective`.",
}


def _write_pack_for_credential_test(packs_dir, pack_id, credential=None):
    pack_dir = packs_dir / pack_id
    pack_dir.mkdir(parents=True)
    source = {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}}
    if credential:
        source["credential"] = credential
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump({"id": pack_id, "audio": {"source": source}}))


def test_preflight_pack_credentials_prompts_once_for_shared_env_var(tmp_path, monkeypatch):
    """ADR-0010 acceptance #3: two packs declaring the same env_var are
    asked about exactly once, not once per pack."""
    from oesb_runner import cli as cli_module

    store: dict[str, str] = {}
    monkeypatch.setattr(cli_module.credentials, "load_credential", lambda env_var, **kw: store.get(env_var))
    monkeypatch.setattr(
        cli_module.credentials, "save_credential",
        lambda env_var, value, **kw: store.__setitem__(env_var, value),
    )
    monkeypatch.delenv("MDC_API_KEY", raising=False)

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "pack-a", credential=_MDC_CREDENTIAL)
    _write_pack_for_credential_test(packs_dir, "pack-b", credential=_MDC_CREDENTIAL)
    combos = [("profile-a", "pack-a"), ("profile-b", "pack-b")]

    prompts = []

    def _fake_password(message):
        prompts.append(message)
        return _FakeAsk("the-secret-key")

    monkeypatch.setattr(cli_module.questionary, "password", _fake_password)

    kept = cli_module._preflight_pack_credentials(combos, str(packs_dir), cli_module.DEFAULT_API_URL)

    assert kept == combos
    assert len(prompts) == 1
    assert "MDC_API_KEY" in prompts[0]
    assert store["MDC_API_KEY"] == "the-secret-key"
    assert os.environ["MDC_API_KEY"] == "the-secret-key"  # _reexec's subprocess inherits this


def test_preflight_pack_credentials_shows_instructions_and_signup_url(tmp_path, monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.credentials, "load_credential", lambda env_var, **kw: None)
    monkeypatch.setattr(cli_module.credentials, "save_credential", lambda env_var, value, **kw: None)
    monkeypatch.delenv("MDC_API_KEY", raising=False)

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "pack-a", credential=_MDC_CREDENTIAL)
    monkeypatch.setattr(cli_module.questionary, "password", lambda message: _FakeAsk("the-secret-key"))

    cli_module._preflight_pack_credentials([("profile-a", "pack-a")], str(packs_dir), cli_module.DEFAULT_API_URL)

    err = capsys.readouterr().err
    assert _MDC_CREDENTIAL["instructions"] in err
    assert _MDC_CREDENTIAL["signup_url"] in err


def test_preflight_pack_credentials_declining_drops_only_affected_combos(tmp_path, monkeypatch, capsys):
    """ADR-0010 acceptance #4: an empty answer drops only the combos
    needing that credential — a batch that also has ungated packs still
    runs those."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.credentials, "load_credential", lambda env_var, **kw: None)
    monkeypatch.setattr(cli_module.credentials, "save_credential", lambda env_var, value, **kw: None)
    monkeypatch.delenv("MDC_API_KEY", raising=False)

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "gated-pack", credential=_MDC_CREDENTIAL)
    _write_pack_for_credential_test(packs_dir, "ungated-pack", credential=None)
    combos = [("profile-a", "gated-pack"), ("profile-b", "ungated-pack")]

    monkeypatch.setattr(cli_module.questionary, "password", lambda message: _FakeAsk(""))

    kept = cli_module._preflight_pack_credentials(combos, str(packs_dir), cli_module.DEFAULT_API_URL)

    assert kept == [("profile-b", "ungated-pack")]
    assert "MDC_API_KEY" not in os.environ
    assert "profile-a  x  gated-pack — skipping" in capsys.readouterr().err


def test_preflight_pack_credentials_skips_prompt_when_already_resolvable(tmp_path, monkeypatch):
    """ADR-0010 acceptance #5: a credential already resolvable (env var or
    the local store) is never prompted for.

    Real report, second bug found chasing the same "Missing API key"
    failure: a credential resolved from the on-disk store (as opposed to
    an env var the user already exported themselves) must still end up in
    os.environ — `_reexec`'s subprocess and audio_sources.
    fetch_common_voice_audio (whose own docstring assumes the credential
    is "already in os.environ by the time it runs") both read it directly,
    never the store. Skipping the prompt without exporting meant a
    credential saved on run N was silently unusable on every run after."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.credentials, "load_credential", lambda env_var, **kw: "already-saved-key")
    monkeypatch.delenv("MDC_API_KEY", raising=False)

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "pack-a", credential=_MDC_CREDENTIAL)
    combos = [("profile-a", "pack-a")]

    def _fail_if_called(message):
        raise AssertionError("should not prompt — credential already resolvable")

    monkeypatch.setattr(cli_module.questionary, "password", _fail_if_called)

    kept = cli_module._preflight_pack_credentials(combos, str(packs_dir), cli_module.DEFAULT_API_URL)

    assert kept == combos
    assert os.environ["MDC_API_KEY"] == "already-saved-key"


def test_preflight_pack_credentials_reprompts_on_blank_stored_value(tmp_path, monkeypatch):
    """Real report: a batch of Common Voice combos never prompted for
    MDC_API_KEY at all — auto-fetch kept failing downstream with "Missing
    API key" instead. Root cause: this function checked `is not None`, so
    a blank string sitting in the on-disk credential store looked
    "already resolved" and skipped the prompt forever, even though the
    value is unusable. Must treat a blank stored value the same as no
    value — prompt again."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.credentials, "load_credential", lambda env_var, **kw: "")
    # save_credential must be mocked too, not just load_credential — the
    # prompt-answered path below calls the real one, which would otherwise
    # write a live "the-real-key" entry into this machine's actual
    # ~/.goesb/credentials.json (confirmed: this is exactly what an
    # earlier, unmocked version of this same test did on the dev machine
    # that wrote it, and that leftover real file was itself then mistaken
    # for reproducing the bug it was written to catch).
    monkeypatch.setattr(cli_module.credentials, "save_credential", lambda *a, **k: None)
    monkeypatch.delenv("MDC_API_KEY", raising=False)

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "pack-a", credential=_MDC_CREDENTIAL)
    combos = [("profile-a", "pack-a")]

    monkeypatch.setattr(cli_module.questionary, "password", lambda message: _FakeAsk("the-real-key"))

    kept = cli_module._preflight_pack_credentials(combos, str(packs_dir), cli_module.DEFAULT_API_URL)

    assert kept == combos
    assert os.environ["MDC_API_KEY"] == "the-real-key"


def test_preflight_pack_credentials_returns_none_on_abort(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.credentials, "load_credential", lambda env_var, **kw: None)
    monkeypatch.delenv("MDC_API_KEY", raising=False)

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "pack-a", credential=_MDC_CREDENTIAL)
    monkeypatch.setattr(cli_module.questionary, "password", lambda message: _FakeAsk(None))

    kept = cli_module._preflight_pack_credentials([("profile-a", "pack-a")], str(packs_dir), cli_module.DEFAULT_API_URL)

    assert kept is None


def test_preflight_pack_credentials_no_gated_packs_never_touches_credentials_module(tmp_path, monkeypatch):
    """Acceptance #1: a batch with no gated packs behaves byte-identically
    to today — the credential store is never even consulted."""
    from oesb_runner import cli as cli_module

    def _fail_if_called(*a, **kw):
        raise AssertionError("should not be called — no pack declares a credential")

    monkeypatch.setattr(cli_module.credentials, "load_credential", _fail_if_called)
    monkeypatch.setattr(cli_module.questionary, "password", _fail_if_called)

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "pack-a", credential=None)
    combos = [("profile-a", "pack-a")]

    kept = cli_module._preflight_pack_credentials(combos, str(packs_dir), cli_module.DEFAULT_API_URL)

    assert kept == combos


def test_choose_packs_for_language_single_match_never_prompts(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    def _fail_if_called(*a, **kw):
        raise AssertionError("should not prompt — only one pack matches")

    monkeypatch.setattr(cli_module.questionary, "checkbox", _fail_if_called)

    chosen = cli_module._choose_packs_for_language(
        "profile-a", [{"id": "pack-a", "visibility": "open", "profile_id": "profile-a"}],
        "irrelevant-packs-dir", cli_module.DEFAULT_API_URL,
    )

    assert chosen == ["pack-a"]


def test_choose_packs_for_language_multiple_matches_prompts_with_default_checked(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "gated-pack", credential=_MDC_CREDENTIAL)
    _write_pack_for_credential_test(packs_dir, "open-pack", credential=None)
    matching_packs = [
        {"id": "open-pack", "visibility": "open", "profile_id": "profile-a"},
        {"id": "gated-pack", "visibility": "open", "profile_id": "profile-a"},
    ]

    captured = {}

    def _fake_checkbox(message, choices):
        captured["message"] = message
        captured["choices"] = choices
        return _FakeAsk(["open-pack", "gated-pack"])

    monkeypatch.setattr(cli_module.questionary, "checkbox", _fake_checkbox)

    chosen = cli_module._choose_packs_for_language(
        "profile-a", matching_packs, str(packs_dir), cli_module.DEFAULT_API_URL
    )

    assert chosen == ["open-pack", "gated-pack"]
    assert "profile-a" in captured["message"]
    by_value = {c.value: c for c in captured["choices"]}
    assert by_value["open-pack"].checked is True  # the ungated pack is the pre-checked default
    assert by_value["gated-pack"].checked is False
    assert "open" in by_value["open-pack"].title
    assert "gated" in by_value["gated-pack"].title


def test_choose_packs_for_language_defaults_to_ungated_even_when_listed_second(tmp_path, monkeypatch):
    """Regression test: matching_packs order comes from _pack_rows' local-dir
    listing, which is alphabetical and has nothing to do with which pack is
    the sensible default. A gated pack sorting before the ungated one (e.g.
    'common-voice-nl-elderly' before 'fleurs-nl') must not end
    up pre-checked ahead of it — caught live: the wizard pre-checked the
    gated pack for whisper-medium-nl-batch because it happened to sort
    first."""
    from oesb_runner import cli as cli_module

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "gated-pack", credential=_MDC_CREDENTIAL)
    _write_pack_for_credential_test(packs_dir, "open-pack", credential=None)
    matching_packs = [
        {"id": "gated-pack", "visibility": "open", "profile_id": "profile-a"},  # sorts/lists first
        {"id": "open-pack", "visibility": "open", "profile_id": "profile-a"},
    ]

    captured = {}

    def _fake_checkbox(message, choices):
        captured["choices"] = choices
        return _FakeAsk(["open-pack"])

    monkeypatch.setattr(cli_module.questionary, "checkbox", _fake_checkbox)

    cli_module._choose_packs_for_language("profile-a", matching_packs, str(packs_dir), cli_module.DEFAULT_API_URL)

    by_value = {c.value: c for c in captured["choices"]}
    assert by_value["open-pack"].checked is True
    assert by_value["gated-pack"].checked is False


def test_choose_packs_for_language_defaults_to_first_when_all_gated(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "gated-pack-a", credential=_MDC_CREDENTIAL)
    _write_pack_for_credential_test(packs_dir, "gated-pack-b", credential=_MDC_CREDENTIAL)
    matching_packs = [
        {"id": "gated-pack-a", "visibility": "open", "profile_id": "profile-a"},
        {"id": "gated-pack-b", "visibility": "open", "profile_id": "profile-a"},
    ]

    captured = {}

    def _fake_checkbox(message, choices):
        captured["choices"] = choices
        return _FakeAsk(["gated-pack-a"])

    monkeypatch.setattr(cli_module.questionary, "checkbox", _fake_checkbox)

    cli_module._choose_packs_for_language("profile-a", matching_packs, str(packs_dir), cli_module.DEFAULT_API_URL)

    by_value = {c.value: c for c in captured["choices"]}
    assert by_value["gated-pack-a"].checked is True
    assert by_value["gated-pack-b"].checked is False


def test_choose_packs_for_language_returns_none_on_abort(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "pack-a", credential=None)
    _write_pack_for_credential_test(packs_dir, "pack-b", credential=None)
    matching_packs = [
        {"id": "pack-a", "visibility": "open", "profile_id": "profile-a"},
        {"id": "pack-b", "visibility": "open", "profile_id": "profile-a"},
    ]
    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda message, choices: _FakeAsk(None))

    chosen = cli_module._choose_packs_for_language(
        "profile-a", matching_packs, str(packs_dir), cli_module.DEFAULT_API_URL
    )

    assert chosen is None


def test_choose_packs_for_language_empty_selection_returns_empty_list(tmp_path, monkeypatch):
    """Unchecking everything is a deliberate decline, not an abort — the
    caller drops this cell silently rather than bailing the whole wizard."""
    from oesb_runner import cli as cli_module

    packs_dir = tmp_path / "packs"
    _write_pack_for_credential_test(packs_dir, "pack-a", credential=None)
    _write_pack_for_credential_test(packs_dir, "pack-b", credential=None)
    matching_packs = [
        {"id": "pack-a", "visibility": "open", "profile_id": "profile-a"},
        {"id": "pack-b", "visibility": "open", "profile_id": "profile-a"},
    ]
    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda message, choices: _FakeAsk([]))

    chosen = cli_module._choose_packs_for_language(
        "profile-a", matching_packs, str(packs_dir), cli_module.DEFAULT_API_URL
    )

    assert chosen == []


def test_preflight_engines_installs_each_distinct_engine_once(monkeypatch):
    from oesb_runner import cli as cli_module

    combos = [
        ("whisper-tiny-en-batch", "pack-a"),  # faster-whisper
        ("whisper-medium-en-batch", "pack-b"),  # faster-whisper too — same engine
        ("vosk-small-en-batch", "pack-c"),  # vosk
    ]
    installed = []
    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda name: installed.append(name))

    kept = cli_module._preflight_engines(combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL)

    assert sorted(installed) == ["faster-whisper", "vosk"]  # deduped, not once per combo
    assert kept == combos


def test_preflight_engines_drops_only_combos_needing_a_declined_engine(monkeypatch):
    from oesb_runner import cli as cli_module

    combos = [
        ("whisper-tiny-en-batch", "pack-a"),  # faster-whisper
        ("vosk-small-en-batch", "pack-b"),  # vosk
    ]

    def fake_ensure(name):
        if name == "vosk":
            raise typer.Exit(code=1)

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", fake_ensure)

    kept = cli_module._preflight_engines(combos, str(REPO_ROOT / "profiles"), cli_module.DEFAULT_API_URL)

    assert kept == [("whisper-tiny-en-batch", "pack-a")]


def _stub_wizard_submit_prompts(monkeypatch, comment="", identity=None):
    """Common stub for the comment/callsign prompts `_wizard_submit` now
    asks between picking files and actually submitting -- covers every
    test below that doesn't care about those prompts specifically.
    `resolve_identity` is stubbed directly rather than faking
    `load_identity`/a second `questionary.text` call underneath it: it's
    already covered by its own dedicated tests (see `resolve_identity`'s
    own test block), so re-exercising its internals here would just be
    redundant coverage of a different function."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk(comment))
    monkeypatch.setattr(cli_module, "resolve_identity", lambda callsign, anonymous: identity)


def test_wizard_submit_no_results_prints_message(monkeypatch, tmp_path, capsys):
    from oesb_runner import cli as cli_module

    monkeypatch.chdir(tmp_path)

    cli_module._wizard_submit()

    assert "no result files found" in capsys.readouterr().err


def test_wizard_submit_multiple_and_deletes_confirmed(monkeypatch, tmp_path):
    from oesb_runner import cli as cli_module

    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "runs" / "results"
    results_dir.mkdir(parents=True)
    file_a, file_b = results_dir / "a.json", results_dir / "b.json"
    file_a.write_text("{}")
    file_b.write_text("{}")

    monkeypatch.setattr(
        cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk([str(file_a), str(file_b)])
    )
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    _stub_wizard_submit_prompts(monkeypatch)
    submit_calls = []

    def fake_submit_paths(paths, api_url, *, comment=None, identity=None):
        submit_calls.append((paths, api_url, comment, identity))
        return [(p, True, f"Submitted: {p}") for p in paths]

    monkeypatch.setattr(cli_module, "_submit_paths", fake_submit_paths)

    cli_module._wizard_submit()

    # one shared batch call for every chosen file, not one call per file
    assert submit_calls == [([str(file_a), str(file_b)], cli_module.DEFAULT_API_URL, None, None)]
    assert not file_a.exists()
    assert not file_b.exists()


def test_wizard_submit_declined_delete_keeps_files(monkeypatch, tmp_path):
    from oesb_runner import cli as cli_module

    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "runs" / "results"
    results_dir.mkdir(parents=True)
    file_a = results_dir / "a.json"
    file_a.write_text("{}")

    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk([str(file_a)]))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))
    _stub_wizard_submit_prompts(monkeypatch)
    monkeypatch.setattr(
        cli_module, "_submit_paths",
        lambda paths, api_url, **kw: [(p, True, "ok") for p in paths],
    )

    cli_module._wizard_submit()

    assert file_a.exists()


def test_wizard_submit_only_deletes_successfully_submitted_files(monkeypatch, tmp_path):
    from oesb_runner import cli as cli_module

    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "runs" / "results"
    results_dir.mkdir(parents=True)
    file_a, file_b = results_dir / "a.json", results_dir / "b.json"
    file_a.write_text("{}")
    file_b.write_text("{}")

    monkeypatch.setattr(
        cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk([str(file_a), str(file_b)])
    )
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    _stub_wizard_submit_prompts(monkeypatch)

    def fake_submit_paths(paths, api_url, **kw):
        return [
            (p, False, "submission rejected: ...") if p == str(file_b) else (p, True, "Submitted: ...")
            for p in paths
        ]

    monkeypatch.setattr(cli_module, "_submit_paths", fake_submit_paths)

    cli_module._wizard_submit()

    assert not file_a.exists()  # submitted successfully -> deleted
    assert file_b.exists()  # submission failed -> kept, never offered for deletion


def test_wizard_submit_no_selection_does_nothing(monkeypatch, tmp_path):
    from oesb_runner import cli as cli_module

    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "runs" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "a.json").write_text("{}")

    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk([]))
    calls = []
    monkeypatch.setattr(cli_module, "_submit_paths", lambda paths, api_url, **kw: calls.append(paths))

    cli_module._wizard_submit()

    assert calls == []


def test_wizard_submit_passes_comment_and_identity_through(monkeypatch, tmp_path):
    from oesb_runner import cli as cli_module
    from oesb_runner.identity import Identity

    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "runs" / "results"
    results_dir.mkdir(parents=True)
    file_a = results_dir / "a.json"
    file_a.write_text("{}")

    fake_identity = Identity("scout", "deadbeef")
    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk([str(file_a)]))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))
    _stub_wizard_submit_prompts(monkeypatch, comment="  a real note  ", identity=fake_identity)

    captured = {}

    def fake_submit_paths(paths, api_url, *, comment=None, identity=None):
        captured["comment"] = comment
        captured["identity"] = identity
        return [(p, True, "ok") for p in paths]

    monkeypatch.setattr(cli_module, "_submit_paths", fake_submit_paths)

    cli_module._wizard_submit()

    # Stripped, not the raw whitespace-padded entry.
    assert captured["comment"] == "a real note"
    assert captured["identity"] is fake_identity


def test_wizard_submit_blank_comment_becomes_none(monkeypatch, tmp_path):
    from oesb_runner import cli as cli_module

    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "runs" / "results"
    results_dir.mkdir(parents=True)
    file_a = results_dir / "a.json"
    file_a.write_text("{}")

    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk([str(file_a)]))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))
    _stub_wizard_submit_prompts(monkeypatch, comment="   ")

    captured = {}
    monkeypatch.setattr(
        cli_module, "_submit_paths",
        lambda paths, api_url, *, comment=None, identity=None: (
            captured.__setitem__("comment", comment) or [(p, True, "ok") for p in paths]
        ),
    )

    cli_module._wizard_submit()

    assert captured["comment"] is None


def test_wizard_submit_aborts_on_cancelled_comment_prompt(monkeypatch, tmp_path):
    """`.ask()` returning None is the Ctrl-C/abort convention used
    throughout this module -- must not proceed to submit anything."""
    from oesb_runner import cli as cli_module

    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "runs" / "results"
    results_dir.mkdir(parents=True)
    file_a = results_dir / "a.json"
    file_a.write_text("{}")

    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk([str(file_a)]))
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk(None))

    def _unexpected(*a, **k):
        raise AssertionError("must not submit after the comment prompt was cancelled")

    monkeypatch.setattr(cli_module, "_submit_paths", _unexpected)
    monkeypatch.setattr(cli_module, "resolve_identity", _unexpected)

    cli_module._wizard_submit()

    assert file_a.exists()


def _write_fake_result(path, **overrides):
    from oesb_runner.hashing import canonical_asset_sha256

    result = {"schema_version": "0.2", "runner": {"version": "0.0.1"}, "metrics": {}}
    result.update(overrides)
    result["payload_sha256"] = canonical_asset_sha256(result, exclude=("payload_sha256", "signature"))
    path.write_text(json.dumps(result))
    return result


def _write_full_result(path, **overrides):
    """A fully schema-valid result document, unlike `_write_fake_result`'s
    deliberately-minimal stub -- needed for tests exercising the
    comment/identity mutation path in `_submit_paths`, since that path
    re-validates the mutated document against the real schema."""
    from oesb_runner.hashing import canonical_asset_sha256

    example = json.loads(
        (REPO_ROOT / "schemas" / "examples" / "benchmark-result.example.json").read_text()
    )
    example.update(overrides)
    example["payload_sha256"] = canonical_asset_sha256(example, exclude=("payload_sha256", "signature"))
    path.write_text(json.dumps(example))
    return example


def test_submit_paths_attaches_comment_and_identity(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module
    from oesb_runner.identity import Identity

    file_a = tmp_path / "a.json"
    original = _write_full_result(file_a)
    captured = {}

    def fake_get_json(url, timeout):
        return {"min_runner_version": "0.0.1"}

    def fake_post_json(url, payload, timeout):
        if url.endswith("/runner-tokens"):
            return {"token_id": "tok-1"}
        captured["sent"] = payload["results"][0]
        return {"results": [{"accepted": True, "id": payload["results"][0]["payload_sha256"]}]}

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)
    monkeypatch.setattr(cli_module, "_post_json", fake_post_json)

    outcomes = cli_module._submit_paths(
        [str(file_a)], "http://api.example",
        comment="great little board", identity=Identity("anon", "a1b2c3d4"),
    )

    assert outcomes[0][1] is True
    sent = captured["sent"]
    assert sent["comment"] == "great little board"
    assert sent["submitted_by"] == {"callsign": "anon", "discriminator": "a1b2c3d4"}
    # payload_sha256 must reflect the mutated content, not the original file's
    assert sent["payload_sha256"] != original["payload_sha256"]

    # the on-disk file `run` wrote is untouched -- comment/identity are
    # attached to the in-memory network copy only
    on_disk = json.loads(file_a.read_text())
    assert on_disk["payload_sha256"] == original["payload_sha256"]
    assert "comment" not in on_disk
    assert "submitted_by" not in on_disk


def test_submit_paths_without_comment_or_identity_leaves_payload_unchanged(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    file_a = tmp_path / "a.json"
    original = _write_full_result(file_a)

    def fake_get_json(url, timeout):
        return {"min_runner_version": "0.0.1"}

    def fake_post_json(url, payload, timeout):
        if url.endswith("/runner-tokens"):
            return {"token_id": "tok-1"}
        sent = payload["results"][0]
        assert sent["payload_sha256"] == original["payload_sha256"]
        assert "comment" not in sent
        assert "submitted_by" not in sent
        return {"results": [{"accepted": True, "id": sent["payload_sha256"]}]}

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)
    monkeypatch.setattr(cli_module, "_post_json", fake_post_json)

    outcomes = cli_module._submit_paths([str(file_a)], "http://api.example")

    assert outcomes[0][1] is True


def _unexpected(*args, **kwargs):
    raise AssertionError("should not have been called")


def test_warn_if_runner_outdated_skips_entirely_when_offline(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_get_json", _unexpected)

    _real_warn_if_runner_outdated("http://api.example", offline=True)  # must not raise


def test_warn_if_runner_outdated_silent_when_api_unreachable(monkeypatch):
    """`run` has never required network access -- an unreachable API must
    degrade to a silent no-op, not block or fail the run."""
    from oesb_runner import cli as cli_module

    def fake_get_json(url, timeout):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)

    _real_warn_if_runner_outdated("http://api.example", offline=False)  # must not raise


def test_warn_if_runner_outdated_passes_when_current(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_get_json", lambda url, timeout: {"min_runner_version": "0.0.1"})

    _real_warn_if_runner_outdated("http://api.example", offline=False)  # must not raise


def test_warn_if_runner_outdated_exits_when_outdated_and_reachable(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_get_json", lambda url, timeout: {"min_runner_version": "999.0.0"})

    with pytest.raises(typer.Exit):
        _real_warn_if_runner_outdated("http://api.example", offline=False)


def test_warn_if_runner_outdated_skips_when_env_var_set(monkeypatch):
    """The env var _wizard_run sets before re-execing each combo -- must
    short-circuit before ever attempting the network call."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_get_json", _unexpected)
    monkeypatch.setenv(cli_module._SKIP_OUTDATED_CHECK_ENV_VAR, "1")

    _real_warn_if_runner_outdated("http://api.example", offline=False)  # must not raise


def test_wizard_run_checks_outdated_once_not_once_per_reexec(monkeypatch):
    """A wizard batch re-execs `goesb run` as a fresh subprocess per combo
    (_reexec) -- the version check must happen once here, up front, not
    once per subprocess. Proven two ways: the check itself is called
    exactly once, and the skip env var is set afterward so any _reexec'd
    child inherits a no-op."""
    from oesb_runner import cli as cli_module

    monkeypatch.delenv(cli_module._SKIP_OUTDATED_CHECK_ENV_VAR, raising=False)
    check_calls = []
    monkeypatch.setattr(cli_module, "_warn_if_runner_outdated", lambda *a, **k: check_calls.append(a))
    monkeypatch.setattr(cli_module, "_profile_rows", lambda *a, **k: [])

    cli_module._wizard_run()

    assert check_calls == [(cli_module.DEFAULT_API_URL,)]
    assert os.environ.get(cli_module._SKIP_OUTDATED_CHECK_ENV_VAR) == "1"


def test_run_command_exits_before_any_profile_lookup_when_outdated(tmp_path, monkeypatch):
    """Integration-level: `run` itself must consult the check, and must do
    so before touching profiles/packs -- not just that the helper function
    works in isolation."""
    from oesb_runner import cli as cli_module

    def fake_check(*a, **k):
        raise typer.Exit(code=1)

    monkeypatch.setattr(cli_module, "_warn_if_runner_outdated", fake_check)
    # profiles-dir doesn't exist, so a correctly-ordered `run` would fall
    # through to fetching the profile over the network -- prove the check
    # actually stops it first, not just that exit_code happens to be 1.
    monkeypatch.setattr(cli_module, "fetch_profile", _unexpected)

    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "some-pack",
        "--profiles-dir", str(tmp_path / "profiles"), "--packs-dir", str(tmp_path / "packs"),
    ])

    assert result.exit_code == 1


def test_resolve_identity_anonymous_flag_skips_lookup(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "load_identity", _unexpected)

    assert cli_module.resolve_identity(None, True) is None


def test_resolve_identity_callsign_flag_matching_saved_reuses_without_prompt(monkeypatch):
    from oesb_runner import cli as cli_module
    from oesb_runner.identity import Identity

    saved = Identity("anon", "a1b2c3d4")
    monkeypatch.setattr(cli_module, "load_identity", lambda: saved)
    monkeypatch.setattr(cli_module.questionary, "password", _unexpected)

    assert cli_module.resolve_identity("anon", False) == saved


def test_resolve_identity_callsign_flag_new_value_interactive_prompts_and_saves(monkeypatch):
    from oesb_runner import cli as cli_module
    from oesb_runner.identity import Identity

    monkeypatch.setattr(cli_module, "load_identity", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.questionary, "password", lambda *a, **k: _FakeAsk("s3cr3t"))
    monkeypatch.setattr(cli_module, "compute_discriminator", lambda callsign, secret: "deadbeef")
    save_calls = []
    monkeypatch.setattr(cli_module, "save_identity", lambda identity: save_calls.append(identity))

    result = cli_module.resolve_identity("newname", False)

    assert result == Identity("newname", "deadbeef")
    assert save_calls == [Identity("newname", "deadbeef")]


def test_resolve_identity_callsign_flag_new_value_noninteractive_errors(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "load_identity", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(typer.Exit):
        cli_module.resolve_identity("newname", False)


def test_resolve_identity_no_flag_noninteractive_reuses_saved_silently(monkeypatch):
    from oesb_runner import cli as cli_module
    from oesb_runner.identity import Identity

    saved = Identity("anon", "a1b2c3d4")
    monkeypatch.setattr(cli_module, "load_identity", lambda: saved)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli_module.questionary, "text", _unexpected)

    assert cli_module.resolve_identity(None, False) == saved


def test_resolve_identity_no_flag_noninteractive_nothing_saved_returns_none(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "load_identity", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    assert cli_module.resolve_identity(None, False) is None


def test_resolve_identity_no_flag_interactive_default_prefilled_with_saved(monkeypatch):
    from oesb_runner import cli as cli_module
    from oesb_runner.identity import Identity

    saved = Identity("anon", "a1b2c3d4")
    monkeypatch.setattr(cli_module, "load_identity", lambda: saved)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    captured_default = {}

    def fake_text(prompt, default=""):
        captured_default["default"] = default
        return _FakeAsk("anon")  # user hits Enter -> default echoed back

    monkeypatch.setattr(cli_module.questionary, "text", fake_text)
    monkeypatch.setattr(cli_module.questionary, "password", _unexpected)

    result = cli_module.resolve_identity(None, False)

    assert captured_default["default"] == "anon"
    assert result == saved


def test_resolve_identity_no_flag_interactive_cleared_field_is_anonymous_once(monkeypatch):
    from oesb_runner import cli as cli_module
    from oesb_runner.identity import Identity

    saved = Identity("anon", "a1b2c3d4")
    monkeypatch.setattr(cli_module, "load_identity", lambda: saved)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk(""))
    save_calls = []
    monkeypatch.setattr(cli_module, "save_identity", lambda identity: save_calls.append(identity))

    result = cli_module.resolve_identity(None, False)

    assert result is None
    assert save_calls == []  # saved identity on disk is not touched


def test_resolve_identity_no_flag_interactive_new_callsign_prompts_secret(monkeypatch):
    from oesb_runner import cli as cli_module
    from oesb_runner.identity import Identity

    monkeypatch.setattr(cli_module, "load_identity", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("newname"))
    monkeypatch.setattr(cli_module.questionary, "password", lambda *a, **k: _FakeAsk("s3cr3t"))
    monkeypatch.setattr(cli_module, "compute_discriminator", lambda callsign, secret: "deadbeef")
    save_calls = []
    monkeypatch.setattr(cli_module, "save_identity", lambda identity: save_calls.append(identity))

    result = cli_module.resolve_identity(None, False)

    assert result == Identity("newname", "deadbeef")
    assert save_calls == [Identity("newname", "deadbeef")]


def test_submit_paths_batches_all_files_under_one_token(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    file_a, file_b = tmp_path / "a.json", tmp_path / "b.json"
    _write_fake_result(file_a, repeats=1)
    _write_fake_result(file_b, repeats=2)

    calls = []

    def fake_get_json(url, timeout):
        calls.append(("GET", url))
        return {"min_runner_version": "0.0.1"}

    def fake_post_json(url, payload, timeout):
        calls.append(("POST", url, payload))
        if url.endswith("/runner-tokens"):
            return {"token_id": "tok-1"}
        assert url.endswith("/benchmark/batch")
        assert payload["token_id"] == "tok-1"
        assert len(payload["results"]) == 2
        return {"results": [{"accepted": True, "id": r["payload_sha256"]} for r in payload["results"]]}

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)
    monkeypatch.setattr(cli_module, "_post_json", fake_post_json)

    outcomes = cli_module._submit_paths([str(file_a), str(file_b)], "http://api.example")

    assert [(p, accepted) for p, accepted, _ in outcomes] == [(str(file_a), True), (str(file_b), True)]
    # exactly one health check, one token request, one batch POST -- not
    # one round-trip per file
    assert [c[0:2] for c in calls] == [
        ("GET", "http://api.example/health"),
        ("POST", "http://api.example/runner-tokens"),
        ("POST", "http://api.example/benchmark/batch"),
    ]


def test_submit_paths_locally_invalid_file_never_reaches_network(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps({"payload_sha256": "not-the-real-hash", "metrics": {}}))
    good = tmp_path / "good.json"
    _write_fake_result(good)

    posted = []

    def fake_get_json(url, timeout):
        return {"min_runner_version": "0.0.1"}

    def fake_post_json(url, payload, timeout):
        posted.append(url)
        if url.endswith("/runner-tokens"):
            return {"token_id": "tok-1"}
        assert len(payload["results"]) == 1  # tampered file never joins the batch
        return {"results": [{"accepted": True, "id": payload["results"][0]["payload_sha256"]}]}

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)
    monkeypatch.setattr(cli_module, "_post_json", fake_post_json)

    outcomes = cli_module._submit_paths([str(tampered), str(good)], "http://api.example")

    outcome_by_path = {p: (accepted, msg) for p, accepted, msg in outcomes}
    assert outcome_by_path[str(tampered)][0] is False
    assert "does not match its own payload_sha256" in outcome_by_path[str(tampered)][1]
    assert outcome_by_path[str(good)][0] is True


def test_submit_paths_reports_per_item_rejection_without_failing_others(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    file_a, file_b = tmp_path / "a.json", tmp_path / "b.json"
    _write_fake_result(file_a)
    _write_fake_result(file_b)

    def fake_get_json(url, timeout):
        return {"min_runner_version": "0.0.1"}

    def fake_post_json(url, payload, timeout):
        if url.endswith("/runner-tokens"):
            return {"token_id": "tok-1"}
        return {"results": [
            {"accepted": True, "id": payload["results"][0]["payload_sha256"]},
            {"accepted": False, "detail": {"reason": "not_an_official_profile"}},
        ]}

    monkeypatch.setattr(cli_module, "_get_json", fake_get_json)
    monkeypatch.setattr(cli_module, "_post_json", fake_post_json)

    outcomes = cli_module._submit_paths([str(file_a), str(file_b)], "http://api.example")

    assert outcomes[0][1] is True
    assert outcomes[1][1] is False
    assert "not_an_official_profile" in outcomes[1][2]


def test_submit_command_exits_nonzero_if_any_file_rejected(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    file_a, file_b = tmp_path / "a.json", tmp_path / "b.json"
    _write_fake_result(file_a)
    _write_fake_result(file_b)

    monkeypatch.setattr(cli_module, "resolve_identity", lambda callsign, anonymous: None)
    monkeypatch.setattr(
        cli_module,
        "_submit_paths",
        lambda paths, api_url, **kw: [(paths[0], True, "Submitted: ok"), (paths[1], False, "submission rejected: nope")],
    )

    result = runner.invoke(app, ["submit", str(file_a), str(file_b)])

    assert result.exit_code == 1
    assert "Submitted: ok" in result.output
    assert "submission rejected: nope" in result.output


def test_submit_command_single_file_still_works(tmp_path, monkeypatch):
    from oesb_runner import cli as cli_module

    file_a = tmp_path / "a.json"
    _write_fake_result(file_a)

    monkeypatch.setattr(cli_module, "resolve_identity", lambda callsign, anonymous: None)
    monkeypatch.setattr(
        cli_module, "_submit_paths", lambda paths, api_url, **kw: [(paths[0], True, "Submitted: ok")]
    )

    result = runner.invoke(app, ["submit", str(file_a)])

    assert result.exit_code == 0
    assert "Submitted: ok" in result.output


def _sample_matrix_profiles():
    return [
        {"id": "whisper-tiny-en-batch", "language": "en-US", "benchmark_type": "batch"},
        {"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"},
        {"id": "vosk-small-en-batch", "language": "en-US", "benchmark_type": "batch"},
        {"id": "whisper-tiny-fr-batch", "language": "fr-FR", "benchmark_type": "batch"},
        # Sparse language: only one of the columns the other languages have.
        {"id": "whisper-medium-nl-batch", "language": "nl-NL", "benchmark_type": "batch"},
        # Not part of the batch grid at all — must be ignored, not crash.
        {"id": "whisper-medium-en-streaming", "language": "en-US", "benchmark_type": "streaming"},
        {"id": "vosk-small-en-streaming", "language": "en-US", "benchmark_type": "streaming"},
    ]


def test_build_matrix_shapes_languages_columns_and_cells():
    from oesb_runner import cli as cli_module

    matrix = cli_module._build_matrix(_sample_matrix_profiles())

    assert matrix.languages == ["en-US", "fr-FR", "nl-NL"]
    assert ("whisper", "tiny") in matrix.columns
    assert ("vosk", "small") in matrix.columns
    assert matrix.cells == {
        ("en-US", "whisper-tiny"): "whisper-tiny-en-batch",
        ("en-US", "whisper-medium"): "whisper-medium-en-batch",
        ("en-US", "vosk-small"): "vosk-small-en-batch",
        ("fr-FR", "whisper-tiny"): "whisper-tiny-fr-batch",
        ("nl-NL", "whisper-medium"): "whisper-medium-nl-batch",
    }


def test_build_matrix_picks_up_large_v3_turbo():
    # Real regression: _MATRIX_SIZES/_MATRIX_ID_RE is a fixed size list, not
    # derived from what profiles actually exist -- adding whisper*-large-v3-
    # turbo-*-batch profiles without also adding "large-v3-turbo" here made
    # _build_matrix silently drop every one of them (id_re.match() returns
    # None, `continue`s past them), with no error anywhere. The size
    # alternation must list "large-v3-turbo" for this to pass.
    from oesb_runner import cli as cli_module

    profiles = _sample_matrix_profiles() + [
        {"id": "whisper-large-v3-turbo-en-batch", "language": "en-US", "benchmark_type": "batch"},
        {"id": "whispercpp-large-v3-turbo-en-batch", "language": "en-US", "benchmark_type": "batch"},
    ]
    matrix = cli_module._build_matrix(profiles)

    assert ("whisper", "large-v3-turbo") in matrix.columns
    assert ("whispercpp", "large-v3-turbo") in matrix.columns
    assert matrix.cells[("en-US", "whisper-large-v3-turbo")] == "whisper-large-v3-turbo-en-batch"
    assert matrix.cells[("en-US", "whispercpp-large-v3-turbo")] == "whispercpp-large-v3-turbo-en-batch"


def test_build_matrix_streaming_picks_up_every_streaming_profile():
    """No profile gets singled out and hidden here, even one that measures
    badly on the machine building this test -- GOESB is hardware-generic
    (docs/00-vision.md): "slow on this one CPU" isn't a property of the
    profile, it's a property of THIS run's hardware. See _build_matrix's
    own docstring for the one time this policy was tried and reverted."""
    from oesb_runner import cli as cli_module

    matrix = cli_module._build_matrix(_sample_matrix_profiles(), "streaming")

    assert matrix.languages == ["en-US"]
    assert set(matrix.columns) == {("whisper", "medium"), ("vosk", "small")}
    assert matrix.cells == {
        ("en-US", "whisper-medium"): "whisper-medium-en-streaming",
        ("en-US", "vosk-small"): "vosk-small-en-streaming",
    }


def test_streaming_matches_batchs_real_language_and_engine_coverage():
    """Regression lock, not synthetic data: streaming should cover the
    same languages/engine-size combos batch does (both score WER, so
    language is a real accuracy axis for streaming the same way it is
    for batch -- unlike concurrency, which is language-agnostic since it
    only measures throughput physics). Reads the actual profiles/
    directory, not a fixture -- fails the moment someone adds a new batch
    combo without its streaming counterpart, or vice versa."""
    from oesb_runner import cli as cli_module

    profiles = [
        yaml.safe_load(p.read_text())
        for p in (REPO_ROOT / "profiles").glob("*/profile.yaml")
    ]

    batch = cli_module._build_matrix(profiles, "batch")
    streaming = cli_module._build_matrix(profiles, "streaming")

    assert streaming.languages == batch.languages
    assert set(streaming.columns) == set(batch.columns)
    assert set(streaming.cells) == set(batch.cells)  # same (language, column) keys covered


def test_toggle_selection_column_header_selects_and_clears_the_whole_column():
    from oesb_runner import cli as cli_module

    matrix = cli_module._build_matrix(_sample_matrix_profiles())
    en_row, nl_row = 1, 3  # languages sorted: en-US, fr-FR, nl-NL
    medium_col = matrix.columns.index(("whisper", "medium")) + 1

    selected = cli_module._toggle_selection(set(), matrix, 0, medium_col)
    # Only en-US and nl-NL have a whisper-medium cell; fr-FR doesn't.
    assert selected == {(en_row, medium_col), (nl_row, medium_col)}

    cleared = cli_module._toggle_selection(selected, matrix, 0, medium_col)
    assert cleared == set()


def test_toggle_selection_row_header_selects_and_clears_the_whole_row():
    from oesb_runner import cli as cli_module

    matrix = cli_module._build_matrix(_sample_matrix_profiles())
    en_row = matrix.languages.index("en-US") + 1
    n_cols = len(matrix.columns)

    selected = cli_module._toggle_selection(set(), matrix, en_row, 0)
    assert selected == {
        (en_row, c) for c in range(1, n_cols + 1)
        if cli_module._matrix_cell_exists(matrix, en_row, c)
    }
    assert len(selected) == 3  # tiny, medium, vosk-small

    cleared = cli_module._toggle_selection(selected, matrix, en_row, 0)
    assert cleared == set()


def test_toggle_selection_body_cell_and_missing_cell():
    from oesb_runner import cli as cli_module

    matrix = cli_module._build_matrix(_sample_matrix_profiles())
    nl_row = matrix.languages.index("nl-NL") + 1
    tiny_col = matrix.columns.index(("whisper", "tiny")) + 1  # nl-NL has no tiny profile

    # Missing cell: no-op.
    assert cli_module._toggle_selection(set(), matrix, nl_row, tiny_col) == set()

    # Real cell: toggles on, then off.
    medium_col = matrix.columns.index(("whisper", "medium")) + 1
    on = cli_module._toggle_selection(set(), matrix, nl_row, medium_col)
    assert on == {(nl_row, medium_col)}
    off = cli_module._toggle_selection(on, matrix, nl_row, medium_col)
    assert off == set()


def test_toggle_selection_corner_is_a_noop():
    from oesb_runner import cli as cli_module

    matrix = cli_module._build_matrix(_sample_matrix_profiles())
    assert cli_module._toggle_selection(set(), matrix, 0, 0) == set()


def test_selection_to_profile_ids_maps_cells_back_to_ids():
    from oesb_runner import cli as cli_module

    matrix = cli_module._build_matrix(_sample_matrix_profiles())
    en_row = matrix.languages.index("en-US") + 1
    fr_row = matrix.languages.index("fr-FR") + 1
    tiny_col = matrix.columns.index(("whisper", "tiny")) + 1
    vosk_col = matrix.columns.index(("vosk", "small")) + 1

    selected = {(en_row, tiny_col), (en_row, vosk_col), (fr_row, tiny_col)}
    assert cli_module._selection_to_profile_ids(selected, matrix) == sorted([
        "whisper-tiny-en-batch", "vosk-small-en-batch", "whisper-tiny-fr-batch",
    ])


def test_list_hardware_offline_lists_local_hardware():
    result = runner.invoke(
        app, ["list-hardware", "--offline", "--hardware-dir", str(REPO_ROOT / "hardware")]
    )
    assert result.exit_code == 0
    assert "intel-xeon-e3-1240-v6" in result.stdout
    assert "Intel" in result.stdout


def test_pick_hardware_id_resolves_label_back_to_id(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_hardware_rows",
        lambda *a, **k: [
            {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
            {"id": "custom", "display_name": "Other / not yet in the catalog", "vendor": "Other", "category": "other"},
        ],
    )
    monkeypatch.setattr(
        cli_module.questionary, "autocomplete",
        lambda *a, **k: _FakeAsk("Intel Xeon E3-1240 v6 (Intel)"),
    )

    assert cli_module._pick_hardware_id("http://api", "hardware", offline=False) == "intel-xeon-e3-1240-v6"


def test_pick_hardware_id_preselects_gpu_when_backend_chosen(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_hardware_rows",
        lambda *a, **k: [
            {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
            {"id": "nvidia-rtx-a4000", "display_name": "NVIDIA RTX A4000", "vendor": "NVIDIA", "category": "gpu"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )
    captured_default = {}

    def fake_autocomplete(prompt, choices, default, **kwargs):
        captured_default["default"] = default
        return _FakeAsk("NVIDIA RTX A4000 (NVIDIA)")

    monkeypatch.setattr(cli_module.questionary, "autocomplete", fake_autocomplete)

    result = cli_module._pick_hardware_id(
        "http://api", "hardware", offline=False,
        gpu={"model": "NVIDIA RTX A4000"}, backend_by_runtime={"faster-whisper": "cuda"},
    )

    # The CPU model would also match here -- proves the GPU guess actually
    # won, not just that a match happened to be found.
    assert captured_default["default"] == "NVIDIA RTX A4000 (NVIDIA)"
    assert result == "nvidia-rtx-a4000"


def test_pick_hardware_id_explicit_other_falls_back_to_custom_silently(monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_hardware_rows",
        lambda *a, **k: [
            {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
        ],
    )
    monkeypatch.setattr(
        cli_module.questionary, "autocomplete",
        lambda *a, **k: _FakeAsk("Other / not yet in the catalog"),
    )

    assert cli_module._pick_hardware_id("http://api", "hardware", offline=False) == "custom"
    # Deliberately picking "Other" is not a mistake -- no warning needed.
    assert "doesn't match a catalog entry" not in capsys.readouterr().err


def test_pick_hardware_id_typo_falls_back_to_custom_with_warning(monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_hardware_rows",
        lambda *a, **k: [
            {"id": "intel-n150", "display_name": "Intel N150", "vendor": "Intel", "category": "cpu"},
        ],
    )
    monkeypatch.setattr(
        cli_module.questionary, "autocomplete",
        lambda *a, **k: _FakeAsk("intel-n150"),  # typed the id/slug, not the shown label
    )

    result = cli_module._pick_hardware_id("http://api", "hardware", offline=False)

    assert result == "custom"
    assert "'intel-n150' doesn't match a catalog entry" in capsys.readouterr().err


def test_pick_hardware_id_empty_catalog_falls_back_to_custom(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_hardware_rows", lambda *a, **k: [])

    assert cli_module._pick_hardware_id("http://api", "hardware", offline=False) == "custom"


def test_pick_hardware_id_cancelled_returns_none(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_hardware_rows",
        lambda *a, **k: [
            {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
        ],
    )
    monkeypatch.setattr(cli_module.questionary, "autocomplete", lambda *a, **k: _FakeAsk(None))

    assert cli_module._pick_hardware_id("http://api", "hardware", offline=False) is None


def test_pick_hardware_ids_by_backend_asks_once_for_a_single_backend(monkeypatch):
    from oesb_runner import cli as cli_module

    calls = []

    def fake_pick(api_url, hardware_dir, offline, *, gpu=None, backend_by_runtime=None, question=None):
        calls.append((backend_by_runtime, question))
        return "intel-xeon-e3-1240-v6"

    monkeypatch.setattr(cli_module, "_pick_hardware_id", fake_pick)

    result = cli_module._pick_hardware_ids_by_backend(["cpu"], gpu=None)

    assert result == {"cpu": "intel-xeon-e3-1240-v6"}
    assert len(calls) == 1
    backend_by_runtime, question = calls[0]
    assert backend_by_runtime == {}  # cpu -> no GPU signal, same as today's plain CPU guess
    assert "CPU" in question
    assert "'cpu'" in question


def test_pick_hardware_ids_by_backend_asks_separately_per_distinct_backend(monkeypatch):
    """The actual gap: a batch mixing cuda for one engine and cpu for
    another ran on physically different hardware for each -- prove each
    backend gets its own prompt with its own noun/question, not one
    shared prompt reused for both."""
    from oesb_runner import cli as cli_module

    calls = []

    def fake_pick(api_url, hardware_dir, offline, *, gpu=None, backend_by_runtime=None, question=None):
        calls.append((backend_by_runtime, question))
        return {} if backend_by_runtime == {} else "custom"

    monkeypatch.setattr(cli_module, "_pick_hardware_id", lambda *a, **k: fake_pick(*a, **k))
    fake_gpu = {"model": "NVIDIA RTX A4000"}

    result = cli_module._pick_hardware_ids_by_backend(["cuda", "cpu"], gpu=fake_gpu)

    assert len(calls) == 2  # sorted -- cpu asked before cuda
    cpu_backend_by_runtime, cpu_question = calls[0]
    cuda_backend_by_runtime, cuda_question = calls[1]
    assert cpu_backend_by_runtime == {}
    assert "CPU" in cpu_question and "'cpu'" in cpu_question
    assert cuda_backend_by_runtime == {"_": "cuda"}
    assert "GPU" in cuda_question and "'cuda'" in cuda_question
    assert set(result.keys()) == {"cpu", "cuda"}


def test_pick_hardware_ids_by_backend_aborts_on_any_cancelled_prompt(monkeypatch):
    """Same bail-the-whole-run convention as every other wizard preflight
    step -- one Ctrl-C anywhere in the loop must propagate as None, not
    a partial dict."""
    from oesb_runner import cli as cli_module

    def fake_pick(api_url, hardware_dir, offline, *, gpu=None, backend_by_runtime=None, question=None):
        return None if backend_by_runtime == {"_": "cuda"} else "custom"

    monkeypatch.setattr(cli_module, "_pick_hardware_id", fake_pick)

    assert cli_module._pick_hardware_ids_by_backend(["cpu", "cuda"], gpu=None) is None


def test_normalize_cpu_model_strips_register_marks_and_clock_speed():
    from oesb_runner import cli as cli_module

    assert cli_module._normalize_cpu_model(
        "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"
    ) == "Intel Xeon E3-1240 v6"
    assert cli_module._normalize_cpu_model(
        "Intel(R) Core(TM) i7-9700T CPU @ 2.00GHz"
    ) == "Intel Core i7-9700T"


def test_guess_hardware_label_matches_normalized_probe_against_catalog(monkeypatch):
    """Real report: a user had to type-to-search the full catalog by hand
    for every run, described as very error prone. The exact scenario that
    prompted this — an Intel Xeon E3-1240 v6 box — should preselect."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )
    rows = [
        {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
        {"id": "amd-epyc-7203", "display_name": "AMD EPYC 7203", "vendor": "AMD", "category": "cpu"},
    ]

    assert cli_module._guess_hardware_label(rows) == "Intel Xeon E3-1240 v6 (Intel)"


def test_guess_hardware_label_ignores_gpu_rows(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )
    rows = [
        {"id": "nvidia-rtx-4090", "display_name": "Intel Xeon E3-1240 v6", "vendor": "NVIDIA", "category": "gpu"},
    ]

    assert cli_module._guess_hardware_label(rows) is None


def test_guess_hardware_label_no_match_for_unrelated_or_virtualized_cpu(monkeypatch):
    """Under virtualization the probed string is unrecoverable (e.g. "QEMU
    Virtual CPU version 2.5+") — must not guess something wrong; no default
    is the safe outcome, same as no match at all."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "QEMU Virtual CPU version 2.5+"},
    )
    rows = [
        {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
    ]

    assert cli_module._guess_hardware_label(rows) is None


def test_guess_hardware_label_no_probe_available_returns_none(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_capture_cpu", lambda unavailable: {"model": None})
    rows = [
        {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
    ]

    assert cli_module._guess_hardware_label(rows) is None


def test_pick_hardware_id_preselects_detected_hardware_and_announces_it(monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_hardware_rows",
        lambda *a, **k: [
            {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_capture_cpu",
        lambda unavailable: {"model": "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz"},
    )
    autocomplete_kwargs = {}

    def _fake_autocomplete(message, **kwargs):
        autocomplete_kwargs.update(kwargs)
        return _FakeAsk("Intel Xeon E3-1240 v6 (Intel)")

    monkeypatch.setattr(cli_module.questionary, "autocomplete", _fake_autocomplete)

    result = cli_module._pick_hardware_id("http://api", "hardware", offline=False)

    assert result == "intel-xeon-e3-1240-v6"
    assert autocomplete_kwargs["default"] == "Intel Xeon E3-1240 v6 (Intel)"
    assert "Detected: Intel Xeon E3-1240 v6 (Intel)" in capsys.readouterr().err


def test_pick_hardware_id_no_detection_leaves_prompt_blank(monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_hardware_rows",
        lambda *a, **k: [
            {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6", "vendor": "Intel", "category": "cpu"},
        ],
    )
    monkeypatch.setattr(cli_module, "_capture_cpu", lambda unavailable: {"model": None})
    autocomplete_kwargs = {}

    def _fake_autocomplete(message, **kwargs):
        autocomplete_kwargs.update(kwargs)
        return _FakeAsk("Intel Xeon E3-1240 v6 (Intel)")

    monkeypatch.setattr(cli_module.questionary, "autocomplete", _fake_autocomplete)

    cli_module._pick_hardware_id("http://api", "hardware", offline=False)

    assert autocomplete_kwargs["default"] == ""
    assert "Detected:" not in capsys.readouterr().err


class _FakeAsk:
    """Stands in for whatever questionary.select/.text(...) returns — real
    code only ever calls .ask() on it."""

    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


def test_ensure_engine_installed_noop_if_already_importable(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: object())
    calls = []
    monkeypatch.setattr(cli_module.subprocess, "run", lambda *a, **k: calls.append(a))

    cli_module._ensure_engine_installed("vosk")

    assert calls == []


def test_ensure_engine_installed_non_tty_refuses_without_prompting(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: False)
    asked = []
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: asked.append(a) or _FakeAsk(True))

    with pytest.raises(typer.Exit):
        cli_module._ensure_engine_installed("vosk")

    assert asked == []  # never even prompted


def test_ensure_engine_installed_declines_prompt_exits(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))
    calls = []
    monkeypatch.setattr(cli_module.subprocess, "run", lambda *a, **k: calls.append(a))

    with pytest.raises(typer.Exit):
        cli_module._ensure_engine_installed("vosk")

    assert calls == []  # declined -> never installs


def test_ensure_engine_installed_confirms_and_installs(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))

    calls = []

    class _FakeResult:
        returncode = 0

    def fake_run(args, **kwargs):
        calls.append(args)
        return _FakeResult()

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    cli_module._ensure_engine_installed("vosk")

    assert calls == [[
        cli_module.sys.executable, "-m", "pip", "install",
        f"goesb-runner[vosk]=={cli_module.__version__}",
    ]]


def test_install_package_returns_on_first_pip_success(monkeypatch):
    from oesb_runner import cli as cli_module

    calls = []

    class _Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append(args)
        return _Result(0)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    result = cli_module._install_package("goesb-runner[vosk]==0.2.1")

    assert result.returncode == 0
    assert calls == [[cli_module.sys.executable, "-m", "pip", "install", "goesb-runner[vosk]==0.2.1"]]


def test_install_package_bootstraps_pip_via_ensurepip_then_retries(monkeypatch):
    from oesb_runner import cli as cli_module

    calls = []

    class _Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["-m", "ensurepip"]:
            return _Result(0)
        # first pip attempt fails (no pip module), retry after ensurepip succeeds
        return _Result(0 if len(calls) > 1 else 1)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    result = cli_module._install_package("goesb-runner[vosk]==0.2.1")

    assert result.returncode == 0
    assert calls == [
        [cli_module.sys.executable, "-m", "pip", "install", "goesb-runner[vosk]==0.2.1"],
        [cli_module.sys.executable, "-m", "ensurepip", "--upgrade"],
        [cli_module.sys.executable, "-m", "pip", "install", "goesb-runner[vosk]==0.2.1"],
    ]


def test_install_package_falls_back_to_uv_when_ensurepip_fails(monkeypatch):
    from oesb_runner import cli as cli_module

    calls = []

    class _Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0] == "/usr/local/bin/uv":
            return _Result(0)
        return _Result(1)  # both pip and ensurepip fail

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: "/usr/local/bin/uv" if name == "uv" else None)

    result = cli_module._install_package("goesb-runner[vosk]==0.2.1")

    assert result.returncode == 0
    assert calls[-1] == [
        "/usr/local/bin/uv", "pip", "install", "--python", cli_module.sys.executable,
        "goesb-runner[vosk]==0.2.1",
    ]


def test_install_package_surfaces_original_pip_failure_when_no_uv_available(monkeypatch):
    from oesb_runner import cli as cli_module

    class _Result:
        def __init__(self, returncode):
            self.returncode = returncode

    monkeypatch.setattr(cli_module.subprocess, "run", lambda args, **kwargs: _Result(1))
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: None)

    result = cli_module._install_package("goesb-runner[vosk]==0.2.1")

    assert result.returncode == 1


def test_ensure_engine_installed_frozen_binary_refuses(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli_module.sys, "frozen", True, raising=False)

    with pytest.raises(typer.Exit):
        cli_module._ensure_engine_installed("vosk")


def test_ensure_cuda_runtime_ready_noop_for_cpu_backend(monkeypatch):
    from oesb_runner import cli as cli_module

    def _unexpected(*a, **k):
        raise AssertionError("must never probe cuda_runtime for --backend cpu")

    monkeypatch.setattr(cli_module.cuda_runtime, "cuda_libs_supported", _unexpected)

    cli_module._ensure_cuda_runtime_ready("faster-whisper", "cpu")  # must not raise


def test_ensure_cuda_runtime_ready_noop_for_non_faster_whisper_engine(monkeypatch):
    """whisper-cpp's GPU path is a compiled-binary property, not fixable by
    installing a cuBLAS wheel -- must never offer this install for it."""
    from oesb_runner import cli as cli_module

    def _unexpected(*a, **k):
        raise AssertionError("must never probe cuda_runtime for a non-faster-whisper engine")

    monkeypatch.setattr(cli_module.cuda_runtime, "cuda_libs_supported", _unexpected)

    cli_module._ensure_cuda_runtime_ready("whisper-cpp", "cuda")  # must not raise


def test_ensure_cuda_runtime_ready_noop_on_unsupported_platform(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.cuda_runtime, "cuda_libs_supported", lambda: False)

    def _unexpected(*a, **k):
        raise AssertionError("must not check loadability on a platform cuda_runtime doesn't cover")

    monkeypatch.setattr(cli_module.cuda_runtime, "preload_installed_cublas", _unexpected)
    monkeypatch.setattr(cli_module, "_offer_install", _unexpected)

    cli_module._ensure_cuda_runtime_ready("faster-whisper", "cuda")  # must not raise


def test_ensure_cuda_runtime_ready_noop_when_already_loadable(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.cuda_runtime, "cuda_libs_supported", lambda: True)
    monkeypatch.setattr(cli_module.cuda_runtime, "preload_installed_cublas", lambda: True)

    def _unexpected(*a, **k):
        raise AssertionError("must not offer to install what's already loadable")

    monkeypatch.setattr(cli_module, "_offer_install", _unexpected)

    cli_module._ensure_cuda_runtime_ready("faster-whisper", "cuda")  # must not raise


def test_ensure_cuda_runtime_ready_does_not_reprompt_a_pip_installed_wheel(monkeypatch):
    """Regression test for a real report: this used to check via the
    narrower `cublas_loadable()` alone, which never sees a pip-installed
    `nvidia-cublas-cu12` wheel (its library lives under
    `nvidia/cublas/lib/`, never on the OS loader's default search path) --
    so it kept re-offering the install on every single run even right
    after the user had already installed it. `preload_installed_cublas()`
    is the function that actually checks the pip-wheel location too."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.cuda_runtime, "cuda_libs_supported", lambda: True)
    # A bare loadability check would say False here (nothing preloaded
    # yet); preload_installed_cublas() finds the pip wheel, loads it, and
    # correctly reports True -- must be trusted over the narrower check.
    monkeypatch.setattr(cli_module.cuda_runtime, "preload_installed_cublas", lambda: True)

    def _unexpected(*a, **k):
        raise AssertionError("must not re-offer installing an already pip-installed cuBLAS wheel")

    monkeypatch.setattr(cli_module, "_offer_install", _unexpected)

    cli_module._ensure_cuda_runtime_ready("faster-whisper", "cuda")  # must not raise


def test_ensure_cuda_runtime_ready_offers_the_cublas_wheel_when_missing(monkeypatch):
    """The actual gap this exists to fix: a fresh Ubuntu box with an
    NVIDIA driver but no cuBLAS on the loader's search path (system
    Toolkit or pip wheel) gets offered the pip-installable fix right
    here, before it ever hits the crash inside ctranslate2."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.cuda_runtime, "cuda_libs_supported", lambda: True)
    monkeypatch.setattr(cli_module.cuda_runtime, "preload_installed_cublas", lambda: False)

    offered = []
    monkeypatch.setattr(cli_module, "_offer_install", lambda package: offered.append(package) or True)

    cli_module._ensure_cuda_runtime_ready("faster-whisper", "cuda")

    assert offered == [cli_module.cuda_runtime.CUBLAS_PACKAGE]


# --- _whisper_cpp_cuda_build_ready / _offer_whisper_cpp_cuda_build ---


def test_whisper_cpp_cuda_build_ready_false_when_not_importable(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setitem(sys.modules, "pywhispercpp", None)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", None)

    assert cli_module._whisper_cpp_cuda_build_ready() is False


def test_whisper_cpp_cuda_build_ready_delegates_to_adapters_cuda_available(monkeypatch):
    """Needs the real `pywhispercpp` import to succeed (the function under
    test only reaches the patched `cuda_available` call after that import
    lands) -- CI's `[dev]`-only install never has it, same as every other
    whisper-cpp-extra test in this repo."""
    pytest.importorskip("pywhispercpp", reason="requires `pip install goesb-runner[whisper-cpp]`")
    from oesb_runner import cli as cli_module
    from oesb_runner.adapters import whisper_cpp as whisper_cpp_adapter

    monkeypatch.setattr(whisper_cpp_adapter, "cuda_available", lambda model_cls: True)

    assert cli_module._whisper_cpp_cuda_build_ready() is True


def test_offer_whisper_cpp_cuda_build_noop_on_non_linux(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Darwin")

    def _unexpected(*a, **k):
        raise AssertionError("must not probe gpu/nvcc on a platform this doesn't cover")

    monkeypatch.setattr(cli_module, "_capture_gpu", _unexpected)

    assert cli_module._offer_whisper_cpp_cuda_build() is False


def test_offer_whisper_cpp_cuda_build_noop_when_not_a_tty(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: False)

    assert cli_module._offer_whisper_cpp_cuda_build() is False


def test_offer_whisper_cpp_cuda_build_noop_without_a_gpu(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: None)

    def _unexpected(*a, **k):
        raise AssertionError("must not offer a build with no GPU to build for")

    monkeypatch.setattr(cli_module.questionary, "confirm", _unexpected)

    assert cli_module._offer_whisper_cpp_cuda_build() is False


def test_offer_whisper_cpp_cuda_build_noop_without_nvcc(monkeypatch):
    """A real NVIDIA GPU (driver) is not the same as a real CUDA Toolkit
    (nvcc) -- the source build needs the latter, and a fresh GPU box very
    commonly has only the former."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: {"model": "H100"})
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: None)

    def _unexpected(*a, **k):
        raise AssertionError("must not offer a build without a CUDA Toolkit to build with")

    monkeypatch.setattr(cli_module.questionary, "confirm", _unexpected)

    assert cli_module._offer_whisper_cpp_cuda_build() is False


def _stub_questionary_confirm(monkeypatch, answer: bool):
    class _FakeConfirm:
        def ask(self):
            return answer

    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeConfirm())


def test_offer_whisper_cpp_cuda_build_returns_false_when_declined(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: {"model": "H100"})
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    _stub_questionary_confirm(monkeypatch, False)

    def _unexpected(*a, **k):
        raise AssertionError("must not run pip when the user declined")

    monkeypatch.setattr(cli_module.subprocess, "run", _unexpected)

    assert cli_module._offer_whisper_cpp_cuda_build() is False


def test_offer_whisper_cpp_cuda_build_runs_pip_with_ggml_cuda_env(monkeypatch):
    """The actual gap this exists to fix: rebuild pywhispercpp from source
    with CUDA baked in, via the exact command upstream documents
    (GGML_CUDA=1 pip install git+...), so this doesn't have to be run by
    hand on every fresh GPU box."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: {"model": "H100"})
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    _stub_questionary_confirm(monkeypatch, True)

    captured = {}

    def _fake_run(cmd, env=None, check=False):
        captured["cmd"] = cmd
        captured["env"] = env
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_module.subprocess, "run", _fake_run)

    assert cli_module._offer_whisper_cpp_cuda_build() is True
    assert cli_module._PYWHISPERCPP_CUDA_SOURCE in captured["cmd"]
    assert captured["env"]["GGML_CUDA"] == "1"


def test_offer_whisper_cpp_cuda_build_returns_false_when_pip_fails(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_capture_gpu", lambda unavailable: {"model": "H100"})
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    _stub_questionary_confirm(monkeypatch, True)
    monkeypatch.setattr(
        cli_module.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1)
    )

    assert cli_module._offer_whisper_cpp_cuda_build() is False


# --- regression: _load_profile_for_wizard/_load_pack_for_wizard must not
# re-fetch the same id over and over within one process ---


def test_load_profile_for_wizard_caches_within_one_process(monkeypatch, tmp_path):
    """Regression test for a real report ("packs and profiles ... each
    time it seems to reinstall"): `fetch_profile` hits the network
    unconditionally on every call (a deliberate, separate fix for a real
    stale-cache bug -- see remote.py's own docstring), which broke this
    function's own "second call within one wizard run is a cheap cached
    read" contract, since a single wizard run calls this several times
    for the same profile (e.g. by both `_preflight_engines` and
    `_wizard_engine_parameters`). Memoized here instead, at the right
    layer: still one fresh fetch per profile per `goesb` process (each
    `_reexec`'d run gets its own fresh fetch), just not one fetch per
    call within that process's own preflight sequence."""
    from oesb_runner import cli as cli_module

    cli_module._load_profile_for_wizard.cache_clear()
    fetch_calls = []
    monkeypatch.setattr(
        cli_module, "fetch_profile",
        lambda profile_id, api_url: fetch_calls.append(profile_id) or {"id": profile_id},
    )

    first = cli_module._load_profile_for_wizard("whisper-medium-en-batch", str(tmp_path), "https://example.test/api")
    second = cli_module._load_profile_for_wizard("whisper-medium-en-batch", str(tmp_path), "https://example.test/api")

    assert first == second == {"id": "whisper-medium-en-batch"}
    assert fetch_calls == ["whisper-medium-en-batch"]  # not fetched twice
    cli_module._load_profile_for_wizard.cache_clear()


def test_load_profile_for_wizard_reads_local_file_without_ever_fetching(monkeypatch, tmp_path):
    from oesb_runner import cli as cli_module

    cli_module._load_profile_for_wizard.cache_clear()

    def _unexpected(*a, **k):
        raise AssertionError("must not fetch over the network when a local profile.yaml exists")

    monkeypatch.setattr(cli_module, "fetch_profile", _unexpected)

    profile_dir = tmp_path / "whisper-medium-en-batch"
    profile_dir.mkdir()
    (profile_dir / "profile.yaml").write_text("id: whisper-medium-en-batch\n")

    result = cli_module._load_profile_for_wizard("whisper-medium-en-batch", str(tmp_path), "https://example.test/api")

    assert result == {"id": "whisper-medium-en-batch"}
    cli_module._load_profile_for_wizard.cache_clear()


def test_load_pack_for_wizard_caches_within_one_process(monkeypatch, tmp_path):
    from oesb_runner import cli as cli_module

    cli_module._load_pack_for_wizard.cache_clear()
    fetch_calls = []
    fake_dir = tmp_path / "fetched-pack"
    fake_dir.mkdir()
    (fake_dir / "pack.yaml").write_text("id: some-pack\n")
    monkeypatch.setattr(
        cli_module, "fetch_pack",
        lambda pack_id, api_url: fetch_calls.append(pack_id) or fake_dir,
    )

    first = cli_module._load_pack_for_wizard("some-pack", str(tmp_path / "does-not-exist"), "https://example.test/api")
    second = cli_module._load_pack_for_wizard("some-pack", str(tmp_path / "does-not-exist"), "https://example.test/api")

    assert first == second == {"id": "some-pack"}
    assert fetch_calls == ["some-pack"]  # not fetched twice
    cli_module._load_pack_for_wizard.cache_clear()


class _FakeAudioResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_pack(pack_dir: Path, source: dict) -> dict:
    """A minimal, self-contained pack fixture with real (not skipped)
    hashes — load_pack() checks pack.yaml's sha256 and manifest.jsonl's
    manifest_sha256 against actual content, so a fixture with fake hashes
    would fail before ever reaching the audio-resolution logic under
    test."""
    from oesb_runner.hashing import canonical_asset_sha256, sha256_file

    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pack_dir / "manifest.jsonl"
    manifest_path.write_text(json.dumps({
        "utterance_id": "u1", "relative_path": "wanted.wav",
        "reference_text": "hello", "duration_s": 1.0,
    }) + "\n")

    pack_yaml = {
        "id": "fake-pack", "version": "1.0.0", "profile_id": "fake-profile",
        "visibility": "open", "license": "CC0-1.0",
        "metadata": {"language": "en-US", "recording_environment": "quiet", "speech_style": "read"},
        "audio": {"manifest_sha256": sha256_file(manifest_path), "source": source},
    }
    pack_yaml["sha256"] = canonical_asset_sha256(pack_yaml)
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(pack_yaml))
    return pack_yaml


def test_run_command_loads_audio_from_wherever_it_was_actually_fetched(tmp_path, monkeypatch):
    """End-to-end regression test through the real `run` command's actual
    control flow — not just its pieces in isolation. A real bug shipped in
    0.2.4: audio was auto-fetched into the shared cache directory, but
    load_pack() was called with a different (stale) directory, so every
    run of an auto-fetchable pack failed with PackAudioMissingError
    despite the fetch itself succeeding and reporting success. Unit tests
    of the fetch logic and of load_pack() both passed in isolation — a
    prior version of this very test called `_resolve_pack_audio()` and
    `load_pack()` separately and also passed despite the bug, because it
    re-did the correct wiring itself instead of exercising `run()`'s own
    call site. This invokes the real `run` subcommand end to end and only
    stops (via a recognizable sentinel) right after load_pack succeeds —
    proving the exact call site that broke actually works, without
    needing a real ML engine installed."""
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    class _StoppedRightAfterLoadPack(Exception):
        pass

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)
    monkeypatch.setattr(cli_module, "_ensure_cuda_runtime_ready", lambda runtime_name, backend: None)
    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")

    def _fake_get_adapter(runtime_name, benchmark_type="batch"):
        raise _StoppedRightAfterLoadPack()

    monkeypatch.setattr(cli_module, "get_adapter", _fake_get_adapter)

    source = {"type": "fleurs", "params": {"language": "xx_xx", "split": "dev"}}
    packs_dir = tmp_path / "packs"
    pack_dir = packs_dir / "fake-pack"
    _fake_pack(pack_dir, source)
    # _fake_pack() sets profile_id "fake-profile" — swap in a real,
    # already-committed profile id so `run` has an actual profile.yaml to load.
    pack_yaml_path = pack_dir / "pack.yaml"
    pack_yaml = yaml.safe_load(pack_yaml_path.read_text())
    pack_yaml["profile_id"] = "whisper-medium-en-batch"
    del pack_yaml["sha256"]
    from oesb_runner.hashing import canonical_asset_sha256
    pack_yaml["sha256"] = canonical_asset_sha256(pack_yaml)
    pack_yaml_path.write_text(yaml.safe_dump(pack_yaml))

    archive_buf = io.BytesIO()
    with tarfile.open(fileobj=archive_buf, mode="w:gz") as tar:
        content = b"fake audio bytes"
        info = tarfile.TarInfo(name="xx_xx/audio/dev/wanted.wav")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    archive_bytes = archive_buf.getvalue()
    monkeypatch.setattr(
        audio_sources.urllib.request, "urlopen",
        lambda url, **kw: _FakeAudioResponse(archive_bytes),
    )

    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "fake-pack",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(packs_dir),
    ])

    assert "PackAudioMissingError" not in result.output
    assert "audio file(s) missing" not in result.output
    assert isinstance(result.exception, _StoppedRightAfterLoadPack), (
        f"expected to reach get_adapter(), got: {result.output}"
    )


def test_resolve_pack_audio_prefers_existing_pack_local_audio(tmp_path, monkeypatch):
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")

    source = {"type": "fleurs", "params": {"language": "xx_xx", "split": "dev"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)
    (pack_dir / "audio").mkdir()
    (pack_dir / "audio" / "wanted.wav").write_bytes(b"already here")

    def _fail_if_called(url, **kw):
        raise AssertionError("should not fetch — pack already has local audio")

    monkeypatch.setattr(audio_sources.urllib.request, "urlopen", _fail_if_called)

    resolved_audio_dir = cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert resolved_audio_dir == pack_dir / "audio"


def test_resolve_pack_audio_retries_after_stale_empty_shared_cache_dir(tmp_path, monkeypatch):
    """A prior auto-fetch interrupted mid-stream (network blip, Ctrl-C) can
    leave shared_audio_dir() existing but empty (or partial). Since that
    same path is reused by every sibling pack pointing at the same source,
    treating bare directory existence as "already fetched" would poison
    every one of them permanently — this is the real bug seen in
    production: a batch run's first pack left an empty cache dir behind,
    and every subsequent sibling pack failed with PackAudioMissingError
    without even attempting a fetch."""
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")

    source = {"type": "fleurs", "params": {"language": "xx_xx", "split": "dev"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    # Simulate the poisoned state: the shared cache dir already exists
    # (created by mkdir at the start of a prior, interrupted _stream_extract)
    # but has none of the wanted files in it.
    stale_dir = audio_sources.shared_audio_dir(source)
    stale_dir.mkdir(parents=True)

    archive_buf = io.BytesIO()
    with tarfile.open(fileobj=archive_buf, mode="w:gz") as tar:
        content = b"fake audio bytes"
        info = tarfile.TarInfo(name="wanted.wav")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    archive_bytes = archive_buf.getvalue()
    monkeypatch.setattr(
        audio_sources.urllib.request, "urlopen",
        lambda url, **kw: _FakeAudioResponse(archive_bytes),
    )

    resolved_audio_dir = cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert resolved_audio_dir == stale_dir
    assert (stale_dir / "wanted.wav").read_bytes() == content


def test_resolve_pack_audio_skips_fetch_when_shared_cache_already_complete(tmp_path, monkeypatch):
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")

    source = {"type": "fleurs", "params": {"language": "xx_xx", "split": "dev"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    complete_dir = audio_sources.shared_audio_dir(source)
    complete_dir.mkdir(parents=True)
    (complete_dir / "wanted.wav").write_bytes(b"already fetched")

    def _fail_if_called(url, **kw):
        raise AssertionError("should not re-fetch — shared cache is already complete")

    monkeypatch.setattr(audio_sources.urllib.request, "urlopen", _fail_if_called)

    resolved_audio_dir = cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert resolved_audio_dir == complete_dir


def test_resolve_pack_audio_reports_clean_message_on_gated_fetch_auth_error(tmp_path, monkeypatch, capsys):
    """ADR-0010: a rejected/expired/revoked gated-source credential must
    surface as a clear stderr message and a normal typer.Exit(1), never an
    uncaught traceback."""
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    source = {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    def _fail(*a, **k):
        raise cli_module.GatedFetchAuthError("Mozilla Data Collective rejected the 'abc123' request: Access denied.")

    monkeypatch.setattr(cli_module, "auto_fetch_audio", _fail)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert exc_info.value.exit_code == 1
    err = capsys.readouterr().err
    assert "credential rejected" in err
    assert "Access denied" in err


def test_resolve_pack_audio_reports_clean_message_on_generic_fetch_failure(tmp_path, monkeypatch, capsys):
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    source = {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    def _fail(*a, **k):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(cli_module, "auto_fetch_audio", _fail)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert exc_info.value.exit_code == 1
    assert "Auto-fetch failed: network exploded" in capsys.readouterr().err


def test_resolve_pack_audio_missing_dependency_non_tty_suggests_command_no_prompt(tmp_path, monkeypatch, capsys):
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    source = {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: False)
    asked = []
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: asked.append(a) or _FakeAsk(True))

    def _fail(*a, **k):
        raise audio_sources.MissingDependencyError("datacollective")

    monkeypatch.setattr(cli_module, "auto_fetch_audio", _fail)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert exc_info.value.exit_code == 1
    assert asked == []  # non-interactive — never prompted
    err = capsys.readouterr().err
    assert "datacollective is not installed" in err
    assert "pip install datacollective" in err


def test_resolve_pack_audio_missing_dependency_declines_prompt_suggests_command(tmp_path, monkeypatch, capsys):
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    source = {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))

    def _fail(*a, **k):
        raise audio_sources.MissingDependencyError("datacollective")

    monkeypatch.setattr(cli_module, "auto_fetch_audio", _fail)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert exc_info.value.exit_code == 1
    assert "run `pip install datacollective` and retry" in capsys.readouterr().err


def test_resolve_pack_audio_missing_dependency_pipx_install_suggests_inject(tmp_path, monkeypatch, capsys):
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    source = {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))
    monkeypatch.setattr(cli_module, "_is_pipx_install", lambda: True)

    def _fail(*a, **k):
        raise audio_sources.MissingDependencyError("datacollective")

    monkeypatch.setattr(cli_module, "auto_fetch_audio", _fail)

    with pytest.raises(typer.Exit):
        cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert "pipx inject goesb-runner datacollective" in capsys.readouterr().err


def test_resolve_pack_audio_missing_dependency_confirms_installs_and_retries(tmp_path, monkeypatch, capsys):
    """Confirming the install prompt must retry the fetch exactly once —
    without a retry, the wizard's per-combo `_reexec` subprocesses would
    each hit the same missing-dependency prompt again instead of the
    install actually unblocking the run that triggered it."""
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    source = {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))

    class _FakeResult:
        returncode = 0

    install_calls = []
    monkeypatch.setattr(cli_module.subprocess, "run", lambda *a, **k: install_calls.append(a) or _FakeResult())

    fetch_calls = []

    def _fetch(source, wanted_names, audio_dir):
        fetch_calls.append(len(fetch_calls))
        if len(fetch_calls) == 1:
            raise audio_sources.MissingDependencyError("datacollective")
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "wanted.wav").write_bytes(b"fetched after install")
        return {"wanted.wav"}

    monkeypatch.setattr(cli_module, "auto_fetch_audio", _fetch)

    resolved_audio_dir = cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert len(fetch_calls) == 2  # failed once, retried once after install
    assert (resolved_audio_dir / "wanted.wav").read_bytes() == b"fetched after install"
    assert len(install_calls) == 1
    assert "Installing datacollective" in capsys.readouterr().err


def test_resolve_pack_audio_missing_dependency_only_retries_once(tmp_path, monkeypatch, capsys):
    """A second MissingDependencyError after an already-attempted install
    must not loop forever or prompt again — fail with the suggested command
    instead, same as a decline."""
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    source = {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    confirm_calls = []
    monkeypatch.setattr(
        cli_module.questionary, "confirm",
        lambda *a, **k: confirm_calls.append(a) or _FakeAsk(True),
    )

    class _FakeResult:
        returncode = 0

    monkeypatch.setattr(cli_module.subprocess, "run", lambda *a, **k: _FakeResult())

    def _always_fail(*a, **k):
        raise audio_sources.MissingDependencyError("datacollective")

    monkeypatch.setattr(cli_module, "auto_fetch_audio", _always_fail)

    with pytest.raises(typer.Exit) as exc_info:
        cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, False)

    assert exc_info.value.exit_code == 1
    assert len(confirm_calls) == 1  # only ever prompted once
    assert "run `pip install datacollective` and retry" in capsys.readouterr().err


def test_suggest_install_command_plain_venv():
    from oesb_runner import cli as cli_module

    assert cli_module._suggest_install_command("datacollective") == "pip install datacollective"


def test_suggest_install_command_pipx(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_is_pipx_install", lambda: True)
    assert cli_module._suggest_install_command("datacollective") == "pipx inject goesb-runner datacollective"


def test_suggest_install_command_frozen_binary(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.sys, "frozen", True, raising=False)
    try:
        assert "pip install goesb-runner" in cli_module._suggest_install_command("datacollective")
    finally:
        monkeypatch.delattr(cli_module.sys, "frozen", raising=False)


def test_is_pipx_install_detects_pipx_venv_path(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module.sys, "executable", "/home/eric/.local/pipx/venvs/goesb-runner/bin/python3",
    )
    assert cli_module._is_pipx_install() is True


def test_is_pipx_install_false_for_plain_venv(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.sys, "executable", "/home/eric/.venvs/goesb/bin/python3")
    assert cli_module._is_pipx_install() is False


def test_coerce_param_value_bool_parses_common_spellings():
    from oesb_runner import cli as cli_module

    assert cli_module._coerce_param_value("true", True) is True
    assert cli_module._coerce_param_value("false", True) is False
    assert cli_module._coerce_param_value("1", False) is True
    assert cli_module._coerce_param_value("no", True) is False


def test_coerce_param_value_bool_rejects_garbage():
    from oesb_runner import cli as cli_module

    with pytest.raises(ValueError, match="true/false"):
        cli_module._coerce_param_value("maybe", True)


def test_coerce_param_value_int_and_float():
    from oesb_runner import cli as cli_module

    assert cli_module._coerce_param_value("8", 5) == 8
    assert cli_module._coerce_param_value("0.5", 0.0) == 0.5


# --- _sample_during: GPU sampling ---


def test_sample_during_collects_gpu_samples_when_available(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.gpu_pct, "sample_gpu_pct", lambda: 42.0)

    _result, _samples, _temp, _rapl, gpu_samples = cli_module._sample_during(
        lambda: None, interval_s=0.01, gpu_interval_s=0.01
    )

    assert gpu_samples and all(v == 42.0 for v in gpu_samples)


def test_sample_during_gpu_samples_empty_without_nvidia_smi(monkeypatch):
    """No NVIDIA GPU/nvidia-smi -- same "absent, not fabricated zero"
    convention as temperature/RAPL."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.gpu_pct, "sample_gpu_pct", lambda: None)

    _result, _samples, _temp, _rapl, gpu_samples = cli_module._sample_during(
        lambda: None, interval_s=0.01, gpu_interval_s=0.01
    )

    assert gpu_samples == []


def test_sample_during_samples_gpu_less_often_than_cpu(monkeypatch):
    """GPU sampling spawns a subprocess per call (nvidia-smi) -- must not
    fire on every fast CPU/RAM tick, only its own coarser cadence."""
    import time

    from oesb_runner import cli as cli_module

    calls = []
    monkeypatch.setattr(cli_module.gpu_pct, "sample_gpu_pct", lambda: calls.append(1) or 10.0)

    cli_module._sample_during(lambda: time.sleep(0.05), interval_s=0.01, gpu_interval_s=0.05)

    # ~5 CPU ticks (0.05s / 0.01s) but gpu_interval_s is 5x coarser -- far
    # fewer GPU samples than CPU ticks, not one per tick.
    assert 1 <= len(calls) <= 3


def test_coerce_param_value_int_rejects_non_numeric():
    from oesb_runner import cli as cli_module

    with pytest.raises(ValueError, match="integer"):
        cli_module._coerce_param_value("eight", 5)


def test_check_param_domain_allowed_list():
    from oesb_runner import cli as cli_module

    cli_module._check_param_domain("beam_size", 8, {"allowed": [1, 2, 4, 5, 8]})
    with pytest.raises(ValueError, match="not in allowed values"):
        cli_module._check_param_domain("beam_size", 3, {"allowed": [1, 2, 4, 5, 8]})


def test_check_param_domain_range():
    from oesb_runner import cli as cli_module

    cli_module._check_param_domain("threads", 4, {"range": {"min": 1, "max": 8}})
    with pytest.raises(ValueError, match="outside range"):
        cli_module._check_param_domain("threads", 16, {"range": {"min": 1, "max": 8}})


_SAMPLE_OVERRIDABLE_PROFILE = {
    "id": "sample-profile",
    "model": {"name": "whisper-medium", "beam_size": 5, "vad": True},
    "configuration": {"threads": 4},
    "overridable": {"beam_size": {"allowed": [1, 2, 4, 5, 8]}, "vad": {}},
}


def test_resolve_one_param_override_within_domain():
    from oesb_runner import cli as cli_module

    resolved = cli_module._resolve_one_param(_SAMPLE_OVERRIDABLE_PROFILE, "beam_size", "8")
    assert resolved == {"value": 8, "default": 5}


def test_resolve_one_param_no_override_returns_default():
    from oesb_runner import cli as cli_module

    resolved = cli_module._resolve_one_param(_SAMPLE_OVERRIDABLE_PROFILE, "beam_size", None)
    assert resolved == {"value": 5, "default": 5}


def test_resolve_one_param_rejects_undeclared_key():
    from oesb_runner import cli as cli_module

    with pytest.raises(ValueError, match="not declared overridable"):
        cli_module._resolve_one_param(_SAMPLE_OVERRIDABLE_PROFILE, "temperature", "0.5")


def test_resolve_one_param_rejects_out_of_domain_value():
    from oesb_runner import cli as cli_module

    with pytest.raises(ValueError, match="not in allowed values"):
        cli_module._resolve_one_param(_SAMPLE_OVERRIDABLE_PROFILE, "beam_size", "3")


def test_resolve_parameters_returns_every_declared_key_overridden_or_not():
    from oesb_runner import cli as cli_module

    resolved = cli_module._resolve_parameters(_SAMPLE_OVERRIDABLE_PROFILE, {"beam_size": "8"})

    assert resolved == {
        "beam_size": {"value": 8, "default": 5},
        "vad": {"value": True, "default": True},
    }


def test_resolve_parameters_empty_for_profile_without_overridable():
    from oesb_runner import cli as cli_module

    profile = {"id": "no-overridable", "model": {"name": "x"}}
    assert cli_module._resolve_parameters(profile, {}) == {}


def test_resolve_parameters_rejects_unknown_param():
    from oesb_runner import cli as cli_module

    with pytest.raises(ValueError, match="not declared overridable"):
        cli_module._resolve_parameters(_SAMPLE_OVERRIDABLE_PROFILE, {"quantization": "int4"})


# --- ADR-0009 §2 "no silent knobs": profile <-> adapter cross-validation ---


def test_validate_overridable_against_adapter_accepts_correctly_declared_profile():
    """Acceptance criterion 1 in spirit: a correctly-authored profile (the
    real, migrated whisper-medium-en-batch) never trips this check."""
    from oesb_runner import cli as cli_module

    profile = yaml.safe_load(
        (REPO_ROOT / "profiles" / "whisper-medium-en-batch" / "profile.yaml").read_text()
    )
    cli_module._validate_overridable_against_adapter(profile)  # must not raise


def test_validate_overridable_against_adapter_rejects_unapplied_parameter():
    """Acceptance criterion 7 part 1: a profile declaring a parameter its
    adapter doesn't apply fails with a clear error naming both the
    parameter and the adapter."""
    from oesb_runner import cli as cli_module

    profile = {
        "id": "misdeclared-whispercpp-profile",
        "runtime": {"name": "whisper-cpp"},
        "benchmark_type": "batch",
        "model": {"name": "whisper-base", "beam_size": 5},
        "overridable": {"beam_size": {"allowed": [1, 5, 8]}},
    }
    with pytest.raises(ValueError, match="beam_size") as exc_info:
        cli_module._validate_overridable_against_adapter(profile)
    assert "whisper-cpp" in str(exc_info.value)


def test_validate_overridable_against_adapter_noop_when_nothing_declared():
    from oesb_runner import cli as cli_module

    profile = {"id": "x", "runtime": {"name": "vosk"}, "benchmark_type": "batch", "model": {"name": "m"}}
    cli_module._validate_overridable_against_adapter(profile)  # must not raise


def test_run_command_param_against_whispercpp_fails_before_model_load(tmp_path, monkeypatch):
    """Acceptance criterion 7 part 2: `goesb run <whispercpp-profile> <pack>
    --param beam_size=8` fails before model load, for the correct reason
    (beam_size isn't declared overridable on whisper-cpp profiles at all,
    since the adapter never applies it) — not just any failure."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)

    result = runner.invoke(app, [
        "run", "whispercpp-base-en-batch", "some-pack-that-does-not-exist",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
        "--param", "beam_size=8",
    ])

    assert result.exit_code == 1
    assert "not declared overridable" in result.output
    assert "Loaded" not in result.output  # never reached pack/audio resolution


def test_run_command_exits_cleanly_after_a_successful_whisper_cpp_cuda_rebuild(tmp_path, monkeypatch):
    """A successful rebuild can't take effect in this same process (see
    _offer_whisper_cpp_cuda_build's own docstring) -- `run` must stop and
    ask for a re-run rather than continue on with the stale cpu-only
    build it already checked."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)
    monkeypatch.setattr(cli_module, "_ensure_cuda_runtime_ready", lambda runtime_name, backend: None)
    monkeypatch.setattr(cli_module, "_whisper_cpp_cuda_build_ready", lambda: False)
    monkeypatch.setattr(cli_module, "_offer_whisper_cpp_cuda_build", lambda: True)

    result = runner.invoke(app, [
        "run", "whispercpp-base-en-batch", "some-pack-that-does-not-exist",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
        "--backend", "cuda",
    ])

    assert result.exit_code == 0
    assert "Loaded" not in result.output  # never reached pack/audio resolution


def test_run_command_continues_normally_when_whisper_cpp_cuda_already_built(tmp_path, monkeypatch):
    """No rebuild offered/needed when the build already has CUDA -- must
    not early-exit in that case."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)
    monkeypatch.setattr(cli_module, "_ensure_cuda_runtime_ready", lambda runtime_name, backend: None)
    monkeypatch.setattr(cli_module, "_whisper_cpp_cuda_build_ready", lambda: True)

    def _unexpected():
        raise AssertionError("must not offer a rebuild when CUDA is already built")

    monkeypatch.setattr(cli_module, "_offer_whisper_cpp_cuda_build", _unexpected)

    result = runner.invoke(app, [
        "run", "whispercpp-base-en-batch", "some-pack-that-does-not-exist",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
        "--backend", "cuda",
    ])

    # Proceeds past the rebuild check to real pack resolution, which then
    # fails for the unrelated reason that this pack doesn't exist --
    # proves it got past the rebuild gate, not that the whole run succeeded.
    assert result.exit_code == 1
    assert "not found" in result.output


def test_parse_param_overrides_splits_key_value_pairs():
    from oesb_runner import cli as cli_module

    assert cli_module._parse_param_overrides(["beam_size=8", "vad=false"]) == {
        "beam_size": "8", "vad": "false",
    }


def test_parse_param_overrides_rejects_missing_equals():
    from oesb_runner import cli as cli_module

    with pytest.raises(ValueError, match="KEY=VALUE"):
        cli_module._parse_param_overrides(["beam_size8"])


def test_run_with_param_override_records_parameters_and_verifies(tmp_path, monkeypatch):
    """Acceptance criterion 3: `goesb run p pack --param beam_size=8` ->
    signed result contains parameters.beam_size = {value: 8, default: 5}
    and verifies. Uses the real, already-migrated whisper-medium-en-batch
    profile (beam_size default 5, allowed [1,2,4,5,8]) rather than a
    synthetic one, so this exercises the actual shipped profile data."""
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module
    from oesb_runner.adapters import Transcription
    from oesb_runner.signing import verify_result_document

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)
    monkeypatch.setattr(cli_module, "_ensure_cuda_runtime_ready", lambda runtime_name, backend: None)
    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")

    captured_kwargs = {}

    def _fake_get_adapter(runtime_name, benchmark_type="batch"):
        def _fake_run_batch(model_name, utterances, **kwargs):
            captured_kwargs.update(kwargs)
            return [
                Transcription(
                    utterance_id=u.utterance_id, hypothesis_text=u.reference_text, processing_time_s=0.01
                )
                for u in utterances
            ]
        return _fake_run_batch

    monkeypatch.setattr(cli_module, "get_adapter", _fake_get_adapter)

    source = {"type": "fleurs", "params": {"language": "xx_xx", "split": "dev"}}
    packs_dir = tmp_path / "packs"
    pack_dir = packs_dir / "fake-pack"
    _fake_pack(pack_dir, source)
    pack_yaml_path = pack_dir / "pack.yaml"
    pack_yaml = yaml.safe_load(pack_yaml_path.read_text())
    pack_yaml["profile_id"] = "whisper-medium-en-batch"
    del pack_yaml["sha256"]
    from oesb_runner.hashing import canonical_asset_sha256
    pack_yaml["sha256"] = canonical_asset_sha256(pack_yaml)
    pack_yaml_path.write_text(yaml.safe_dump(pack_yaml))

    archive_buf = io.BytesIO()
    with tarfile.open(fileobj=archive_buf, mode="w:gz") as tar:
        content = b"fake audio bytes"
        info = tarfile.TarInfo(name="xx_xx/audio/dev/wanted.wav")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    archive_bytes = archive_buf.getvalue()
    monkeypatch.setattr(
        audio_sources.urllib.request, "urlopen",
        lambda url, **kw: _FakeAudioResponse(archive_bytes),
    )

    results_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "fake-pack",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(packs_dir),
        "--results-dir", str(results_dir),
        "--repeats", "1",
        "--param", "beam_size=8",
    ])

    assert result.exit_code == 0, result.output
    assert captured_kwargs["beam_size"] == 8  # the adapter actually received the override

    [result_path] = list(results_dir.glob("*.json"))
    written = json.loads(result_path.read_text())
    assert written["parameters"]["beam_size"] == {"value": 8, "default": 5}
    assert written["parameters"]["vad"] == {"value": True, "default": True}
    assert written["schema_version"] == "0.4"
    assert verify_result_document(written)


def test_run_with_out_of_domain_param_fails_before_pack_resolution(tmp_path, monkeypatch):
    """A bad --param value must fail before run 1 — before pack/audio
    resolution or model load, not partway through."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)

    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "some-pack-that-does-not-exist",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
        "--param", "beam_size=3",
    ])

    assert result.exit_code == 1
    assert "not in allowed values" in result.output
    assert "Loaded" not in result.output  # never reached pack/audio resolution


def _run_with_backend(tmp_path, monkeypatch, backend_args):
    """Shared setup for the --backend passthrough tests below — same fake
    pack/adapter/audio-archive scaffolding as
    test_run_with_param_override_records_parameters_and_verifies, just
    varying --backend instead of --param. Returns (captured_kwargs, written
    result dict)."""
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module
    from oesb_runner.adapters import Transcription

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)
    monkeypatch.setattr(cli_module, "_ensure_cuda_runtime_ready", lambda runtime_name, backend: None)
    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")

    captured_kwargs = {}

    def _fake_get_adapter(runtime_name, benchmark_type="batch"):
        def _fake_run_batch(model_name, utterances, **kwargs):
            captured_kwargs.update(kwargs)
            return [
                Transcription(
                    utterance_id=u.utterance_id, hypothesis_text=u.reference_text, processing_time_s=0.01
                )
                for u in utterances
            ]
        return _fake_run_batch

    monkeypatch.setattr(cli_module, "get_adapter", _fake_get_adapter)

    source = {"type": "fleurs", "params": {"language": "xx_xx", "split": "dev"}}
    packs_dir = tmp_path / "packs"
    pack_dir = packs_dir / "fake-pack"
    _fake_pack(pack_dir, source)
    pack_yaml_path = pack_dir / "pack.yaml"
    pack_yaml = yaml.safe_load(pack_yaml_path.read_text())
    pack_yaml["profile_id"] = "whisper-medium-en-batch"
    del pack_yaml["sha256"]
    from oesb_runner.hashing import canonical_asset_sha256
    pack_yaml["sha256"] = canonical_asset_sha256(pack_yaml)
    pack_yaml_path.write_text(yaml.safe_dump(pack_yaml))

    archive_buf = io.BytesIO()
    with tarfile.open(fileobj=archive_buf, mode="w:gz") as tar:
        content = b"fake audio bytes"
        info = tarfile.TarInfo(name="xx_xx/audio/dev/wanted.wav")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    archive_bytes = archive_buf.getvalue()
    monkeypatch.setattr(
        audio_sources.urllib.request, "urlopen",
        lambda url, **kw: _FakeAudioResponse(archive_bytes),
    )

    results_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "fake-pack",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(packs_dir),
        "--results-dir", str(results_dir),
        "--repeats", "1",
        *backend_args,
    ])
    assert result.exit_code == 0, result.output

    [result_path] = list(results_dir.glob("*.json"))
    written = json.loads(result_path.read_text())
    return captured_kwargs, written


def test_run_defaults_to_cpu_backend_and_records_it(tmp_path, monkeypatch):
    """Acceptance criterion 3: no --backend flag -> runs on cpu, never left
    to the underlying library's own auto-selection, and that choice is
    recorded on the signed result."""
    from oesb_runner.signing import verify_result_document

    captured_kwargs, written = _run_with_backend(tmp_path, monkeypatch, [])

    assert captured_kwargs["backend"] == "cpu"
    assert written["runtime"]["backend"] == "cpu"
    assert verify_result_document(written)


def test_run_with_explicit_backend_flag_passes_through_and_records_it(tmp_path, monkeypatch):
    from oesb_runner.signing import verify_result_document

    captured_kwargs, written = _run_with_backend(tmp_path, monkeypatch, ["--backend", "cuda"])

    assert captured_kwargs["backend"] == "cuda"
    assert written["runtime"]["backend"] == "cuda"
    assert verify_result_document(written)


def test_run_rejects_backend_unsupported_by_this_runtime_before_engine_install(tmp_path, monkeypatch):
    """Acceptance criterion 5's general case: an adapter that never declared
    a backend (vosk is genuinely cpu-only) must refuse --backend cuda
    immediately — before the engine-install prompt, before pack resolution
    — never a silent fallback to cpu."""
    from oesb_runner import cli as cli_module

    install_calls = []
    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: install_calls.append(runtime_name))

    result = runner.invoke(app, [
        "run", "vosk-small-fr-batch", "some-pack-that-does-not-exist",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(tmp_path / "packs"),
        "--backend", "cuda",
    ])

    assert result.exit_code == 1
    assert "not supported by 'vosk'" in result.output
    assert "cpu" in result.output  # names what it does support
    assert install_calls == []  # never reached engine-install
    assert "Loaded" not in result.output  # never reached pack/audio resolution


# --- ADR-0012: `concurrency` benchmark_type ---


def _run_concurrency(tmp_path, monkeypatch, calls_by_repeat=None):
    """Shared setup mirroring `_run_with_backend` -- real
    `whisper-medium-concurrency` profile (no `language`), a fake pack whose
    content is irrelevant (ADR-0012: audio content doesn't matter for this
    benchmark type), and a fake adapter returning `ConcurrentCall`s instead
    of `Transcription`s. Returns (captured_kwargs, written result dict)."""
    from oesb_runner import audio_sources
    from oesb_runner import cli as cli_module
    from oesb_runner.adapters import ConcurrentCall

    monkeypatch.setattr(cli_module, "_ensure_engine_installed", lambda runtime_name: None)
    monkeypatch.setattr(cli_module, "_ensure_cuda_runtime_ready", lambda runtime_name, backend: None)
    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")

    default_calls = [
        ConcurrentCall(processing_time_s=0.5, audio_duration_s=1.0),
        ConcurrentCall(processing_time_s=1.0, audio_duration_s=1.0),
    ]
    captured_kwargs = {}

    def _fake_get_adapter(runtime_name, benchmark_type="batch"):
        def _fake_run_concurrency(model_name, utterances, **kwargs):
            captured_kwargs.update(kwargs)
            return calls_by_repeat if calls_by_repeat is not None else default_calls
        return _fake_run_concurrency

    monkeypatch.setattr(cli_module, "get_adapter", _fake_get_adapter)

    source = {"type": "fleurs", "params": {"language": "xx_xx", "split": "dev"}}
    packs_dir = tmp_path / "packs"
    pack_dir = packs_dir / "fake-pack"
    _fake_pack(pack_dir, source)
    pack_yaml_path = pack_dir / "pack.yaml"
    pack_yaml = yaml.safe_load(pack_yaml_path.read_text())
    pack_yaml["profile_id"] = "whisper-medium-concurrency"
    del pack_yaml["sha256"]
    from oesb_runner.hashing import canonical_asset_sha256
    pack_yaml["sha256"] = canonical_asset_sha256(pack_yaml)
    pack_yaml_path.write_text(yaml.safe_dump(pack_yaml))

    archive_buf = io.BytesIO()
    with tarfile.open(fileobj=archive_buf, mode="w:gz") as tar:
        content = b"fake audio bytes"
        info = tarfile.TarInfo(name="xx_xx/audio/dev/wanted.wav")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    archive_bytes = archive_buf.getvalue()
    monkeypatch.setattr(
        audio_sources.urllib.request, "urlopen",
        lambda url, **kw: _FakeAudioResponse(archive_bytes),
    )

    results_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "run", "whisper-medium-concurrency", "fake-pack",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(packs_dir),
        "--results-dir", str(results_dir),
        "--repeats", "1",
    ])
    assert result.exit_code == 0, result.output

    [result_path] = list(results_dir.glob("*.json"))
    written = json.loads(result_path.read_text())
    return captured_kwargs, written


def test_run_allows_a_language_null_pack_when_benchmark_type_is_concurrency(tmp_path, monkeypatch):
    """The ADR-0011 exception this whole feature depends on: a concurrency
    profile declares no `language` at all, and must NOT hit "cannot verify
    pack is eligible" the way a language-less batch/streaming profile
    would (see test_run_refuses_a_pack_when_profile_has_no_language_declared)."""
    from oesb_runner.signing import verify_result_document

    _captured_kwargs, written = _run_concurrency(tmp_path, monkeypatch)

    assert verify_result_document(written)
    assert written["profile"]["id"] == "whisper-medium-concurrency"


def test_run_concurrency_pools_real_time_factor_with_spread(tmp_path, monkeypatch):
    """real_time_factor is the per-call distribution for concurrency, not a
    single corpus-aggregate scalar like batch/streaming -- p50/p95 must
    always be attached, same as the existing streaming latency metrics."""
    from oesb_runner.adapters import ConcurrentCall

    calls = [
        ConcurrentCall(processing_time_s=0.5, audio_duration_s=1.0),  # rtf 0.5
        ConcurrentCall(processing_time_s=1.5, audio_duration_s=1.0),  # rtf 1.5
    ]
    _captured_kwargs, written = _run_concurrency(tmp_path, monkeypatch, calls_by_repeat=calls)

    rtf_metric = written["metrics"]["real_time_factor"]
    assert rtf_metric["value"] == pytest.approx(1.0)  # mean of 0.5 and 1.5
    assert "spread" in rtf_metric
    assert rtf_metric["spread"]["p50"] == pytest.approx(1.0)


def test_run_concurrency_computes_throughput(tmp_path, monkeypatch):
    from oesb_runner.adapters import ConcurrentCall

    calls = [
        ConcurrentCall(processing_time_s=0.5, audio_duration_s=10.0),
        ConcurrentCall(processing_time_s=0.5, audio_duration_s=10.0),
    ]
    _captured_kwargs, written = _run_concurrency(tmp_path, monkeypatch, calls_by_repeat=calls)

    # 20s of audio total / configuration.duration_s (30, the profile default) wall-clock
    assert written["metrics"]["throughput"]["value"] == pytest.approx(20.0 / 30.0)


def test_run_concurrency_never_scores_accuracy(tmp_path, monkeypatch):
    """No WER/CER -- this benchmark type doesn't touch reference_text at
    all, and the profile itself never declares these metric ids."""
    _captured_kwargs, written = _run_concurrency(tmp_path, monkeypatch)

    assert "wer" not in written["metrics"]
    assert "cer" not in written["metrics"]


def test_run_concurrency_passes_concurrency_param_to_adapter(tmp_path, monkeypatch):
    captured_kwargs, _written = _run_concurrency(tmp_path, monkeypatch)

    assert captured_kwargs["concurrency"] == 1  # profile default, no --param override
    assert captured_kwargs["duration_s"] == 30


def test_resolve_pack_audio_offline_with_nothing_local_exits(tmp_path):
    from oesb_runner import cli as cli_module

    source = {"type": "fleurs", "params": {"language": "xx_xx", "split": "dev"}}
    pack_dir = tmp_path / "packs" / "fake-pack"
    pack_yaml = _fake_pack(pack_dir, source)

    with pytest.raises(typer.Exit):
        cli_module._resolve_pack_audio(pack_dir, pack_yaml, None, True)
