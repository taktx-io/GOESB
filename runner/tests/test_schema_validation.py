import yaml

from oesb_runner.schema_validation import _find_repo_schemas_dir, validate_against

REPO_ROOT = _find_repo_schemas_dir().parent


def test_finds_valid_profile():
    data = yaml.safe_load(
        (REPO_ROOT / "profiles" / "whisper-medium-en-batch" / "profile.yaml").read_text()
    )
    assert validate_against(data, "benchmark-profile.schema.json") == []


def test_finds_valid_pack():
    data = yaml.safe_load(
        (REPO_ROOT / "packs" / "example-librispeech-en-batch" / "pack.yaml").read_text()
    )
    assert validate_against(data, "benchmark-pack.schema.json") == []


def test_reports_errors_for_invalid_profile():
    errors = validate_against({"id": "x"}, "benchmark-profile.schema.json")
    assert errors  # missing required fields
