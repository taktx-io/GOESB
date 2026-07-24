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
