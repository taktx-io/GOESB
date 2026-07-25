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
            {"id": "pack-a", "visibility": "open", "profile_id": "whisper-medium-en-batch"},
            {"id": "pack-b", "visibility": "open", "profile_id": "whisper-medium-fr-batch"},
            {"id": "unrelated-pack", "visibility": "open", "profile_id": "some-other-profile"},
        ],
    )
    monkeypatch.setattr(
        cli_module, "_ask_matrix",
        lambda matrix: ["whisper-medium-en-batch", "whisper-medium-fr-batch"],
    )
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_pick_hardware_id", lambda *a, **k: "intel-xeon-e3-1240-v6")

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


def test_wizard_run_declines_confirmation_runs_nothing(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "profile_id": "whisper-medium-en-batch"}],
    )
    monkeypatch.setattr(cli_module, "_ask_matrix", lambda matrix: ["whisper-medium-en-batch"])
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("2"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(False))
    monkeypatch.setattr(cli_module, "_pick_hardware_id", lambda *a, **k: "custom")

    calls = []
    monkeypatch.setattr(cli_module, "_reexec", lambda args: calls.append(args))

    cli_module._wizard_run()

    assert calls == []


def test_wizard_run_hardware_back_re_shows_the_matrix(monkeypatch):
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(
        cli_module, "_profile_rows",
        lambda *a, **k: [{"id": "whisper-medium-en-batch", "language": "en-US", "benchmark_type": "batch"}],
    )
    monkeypatch.setattr(
        cli_module, "_pack_rows",
        lambda *a, **k: [{"id": "pack-a", "visibility": "open", "profile_id": "whisper-medium-en-batch"}],
    )
    matrix_calls = []
    monkeypatch.setattr(
        cli_module, "_ask_matrix",
        lambda matrix: (matrix_calls.append(1), ["whisper-medium-en-batch"])[1],
    )
    hardware_responses = iter([cli_module._WIZARD_BACK, "intel-xeon-e3-1240-v6"])
    monkeypatch.setattr(cli_module, "_pick_hardware_id", lambda *a, **k: next(hardware_responses))
    monkeypatch.setattr(cli_module.questionary, "text", lambda *a, **k: _FakeAsk("1"))
    monkeypatch.setattr(cli_module.questionary, "confirm", lambda *a, **k: _FakeAsk(True))
    monkeypatch.setattr(cli_module, "_reexec", lambda args: None)

    cli_module._wizard_run()

    assert len(matrix_calls) == 2  # matrix re-shown once after the "back" pick


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


def test_pick_hardware_id_unmatched_answer_falls_back_to_custom(monkeypatch):
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
