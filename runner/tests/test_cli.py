from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

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
    path = REPO_ROOT / "packs" / "librispeech-en-batch" / "pack.yaml"
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
    assert "librispeech-en-batch" in result.stdout
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


def test_bare_invocation_shows_help_instead_of_hanging():
    # CliRunner's stdin isn't a tty, same as any piped/scripted invocation —
    # exercises the non-interactive fallback path, not the wizard itself.
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


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


def test_wizard_run_builds_expected_run_args(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "librispeech-en-batch", "visibility": "open", "profile_id": "whisper-medium-en-batch"},
            {"id": "unrelated-pack", "visibility": "open", "profile_id": "some-other-profile"},
        ],
    )

    text_responses = iter(["tiny", "1"])  # model override, then repeats
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk(next(text_responses)))

    select_responses = iter(["whisper-medium-en-batch", "librispeech-en-batch"])

    def fake_select(_prompt, choices):
        # Choice objects carry .value; a plain string choice is its own value.
        wanted = next(select_responses)
        for c in choices:
            value = getattr(c, "value", c)
            if value == wanted:
                return _FakeAsk(wanted)
        raise AssertionError(f"{wanted!r} not offered: {choices}")

    monkeypatch.setattr(cli_module.questionary, "select", fake_select)

    calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: calls.append(args))

    cli_module._wizard_run()

    assert calls == [[
        "run", "whisper-medium-en-batch", "librispeech-en-batch",
        "--repeats", "1", "--model-override", "tiny",
    ]]


def test_wizard_batch_dispatch_calls_wizard_run_batch(monkeypatch):
    from oesb_runner import cli as cli_module

    called = []
    monkeypatch.setattr(cli_module, "_wizard_run_batch", lambda: called.append(True))

    responses = iter(["Run multiple benchmarks (batch)", "Exit"])
    monkeypatch.setattr(
        cli_module.questionary, "select", lambda *a, **k: _FakeAsk(next(responses))
    )

    cli_module._run_wizard()

    assert called == [True]


def test_wizard_run_batch_builds_combos_and_continues_past_failures(monkeypatch, capsys):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [
            {"id": "profile-a", "language": "en-US", "benchmark_type": "batch"},
            {"id": "profile-b", "language": "fr-FR", "benchmark_type": "batch"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [
            {"id": "pack-a", "visibility": "open", "profile_id": "profile-a"},
            {"id": "pack-b", "visibility": "open", "profile_id": "profile-b"},
            {"id": "unrelated-pack", "visibility": "open", "profile_id": "some-other-profile"},
        ],
    )
    monkeypatch.setattr(
        cli_module.questionary, "checkbox",
        lambda *a, **k: _FakeAsk(["profile-a", "profile-b"]),
    )

    select_responses = iter(["pack-a", "pack-b"])

    def fake_select(_prompt, choices):
        wanted = next(select_responses)
        for c in choices:
            value = getattr(c, "value", c)
            if value == wanted:
                return _FakeAsk(wanted)
        raise AssertionError(f"{wanted!r} not offered: {choices}")

    monkeypatch.setattr(cli_module.questionary, "select", fake_select)
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))

    reexec_calls = []

    def fake_reexec(args):
        reexec_calls.append(args)
        if args[1] == "profile-a":
            raise typer.Exit(code=1)

    monkeypatch.setattr(cli_module, "_reexec", fake_reexec)

    cli_module._wizard_run_batch()

    assert reexec_calls == [
        ["run", "profile-a", "pack-a", "--repeats", "1"],
        ["run", "profile-b", "pack-b", "--repeats", "1"],
    ]
    out = capsys.readouterr().out
    assert "✗ profile-a  x  pack-a" in out
    assert "✓ profile-b  x  pack-b" in out


def test_wizard_run_batch_declines_confirmation_runs_nothing(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "profile-a", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "profile_id": "profile-a"}],
    )
    monkeypatch.setattr(cli_module.questionary, "checkbox", lambda *a, **k: _FakeAsk(["profile-a"]))
    monkeypatch.setattr(cli_module.questionary, "select", lambda *a, **k: _FakeAsk("pack-a"))
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("2"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))

    calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: calls.append(args))

    cli_module._wizard_run_batch()

    assert calls == []


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


def test_ensure_engine_installed_frozen_binary_refuses(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli_module.sys, "frozen", True, raising=False)

    with pytest.raises(typer.Exit):
        cli_module._ensure_engine_installed("vosk")
