from pathlib import Path

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
    path = REPO_ROOT / "packs" / "example-librispeech-en-batch" / "pack.yaml"
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
    assert "example-librispeech-en-batch" in result.stdout
    assert "open" in result.stdout


def test_list_profiles_offline_no_local_dir_fails(tmp_path):
    result = runner.invoke(
        app, ["list-profiles", "--offline", "--profiles-dir", str(tmp_path / "nope")]
    )
    assert result.exit_code == 1


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
            {"id": "example-librispeech-en-batch", "visibility": "open", "profile_id": "whisper-medium-en-batch"},
            {"id": "unrelated-pack", "visibility": "open", "profile_id": "some-other-profile"},
        ],
    )

    text_responses = iter(["tiny", "1"])  # model override, then repeats
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk(next(text_responses)))

    select_responses = iter(["whisper-medium-en-batch", "example-librispeech-en-batch"])

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
        "run", "whisper-medium-en-batch", "example-librispeech-en-batch",
        "--repeats", "1", "--model-override", "tiny",
    ]]


class _FakeAsk:
    """Stands in for whatever questionary.select/.text(...) returns — real
    code only ever calls .ask() on it."""

    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value
