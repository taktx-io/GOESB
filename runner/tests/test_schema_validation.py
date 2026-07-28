from pathlib import Path

import yaml

from oesb_runner.schema_validation import (
    unmet_min_runner_version,
    unrecognized_pack_source_type,
    validate_against,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_finds_valid_profile():
    data = yaml.safe_load(
        (REPO_ROOT / "profiles" / "whisper-medium-en-batch" / "profile.yaml").read_text()
    )
    assert validate_against(data, "benchmark-profile.schema.json") == []


def test_finds_valid_pack():
    data = yaml.safe_load(
        (REPO_ROOT / "packs" / "librispeech-en-batch" / "pack.yaml").read_text()
    )
    assert validate_against(data, "benchmark-pack.schema.json") == []


def test_reports_errors_for_invalid_profile():
    errors = validate_against({"id": "x"}, "benchmark-profile.schema.json")
    assert errors  # missing required fields


def test_migrated_profile_declares_overridable_beam_size_and_vad():
    """The real, already-migrated (ADR-0009) profile — asserts the schema
    accepts overridable exactly as generate_bulk_assets.py/
    add_overridable_params.py actually emit it, not just a hand-crafted
    fixture."""
    data = yaml.safe_load(
        (REPO_ROOT / "profiles" / "whisper-medium-en-batch" / "profile.yaml").read_text()
    )
    assert data["overridable"]["beam_size"]["allowed"] == [1, 2, 4, 5, 8]
    assert data["overridable"]["vad"] == {}
    assert data["overridable"]["quantization"]["allowed"] == ["int8", "float32"]
    assert data["overridable"]["threads"]["range"] == {"min": 1, "max": 16}


def test_whispercpp_profile_only_declares_threads():
    """whisper-cpp's adapter accepts beam_size/vad/quantization purely for
    call-shape parity with the other batch adapters and never applies them
    (pywhispercpp's flat params don't wire beam_size; there's no VAD;
    quantization is a ggml model-file choice) — "no silent knobs" (ADR-0009
    §2) means declaring them overridable there is exactly the mistake
    generate_bulk_assets.py's overridable_block_for() exists to prevent."""
    data = yaml.safe_load(
        (REPO_ROOT / "profiles" / "whispercpp-base-en-batch" / "profile.yaml").read_text()
    )
    assert set(data["overridable"]) == {"threads"}
    assert "beam_size" in data["model"]  # set, but not overridable — adapter ignores it
    assert "vad" not in data["model"]  # only faster-whisper's model block sets vad at all
    assert "vad" not in data["overridable"]
    assert "vad" not in data["model"]


def test_overridable_rejects_unknown_domain_shape():
    profile = {
        "id": "x", "version": "1.0.0", "benchmark_type": "batch",
        "runtime": {"name": "faster-whisper"}, "model": {"name": "m", "beam_size": 5},
        "scoring": {"primary_metric": "wer"}, "metrics": ["wer"],
        "overridable": {"beam_size": {"not_a_real_domain_key": [1, 2]}},
    }
    errors = validate_against(profile, "benchmark-profile.schema.json")
    assert errors


def test_result_schema_version_is_0_3():
    example = yaml.safe_load(
        (REPO_ROOT / "schemas" / "examples" / "benchmark-result.example.json").read_text()
    )
    assert example["schema_version"] == "0.3"
    assert validate_against(example, "benchmark-result.schema.json") == []


def test_result_schema_requires_runtime_backend():
    example = yaml.safe_load(
        (REPO_ROOT / "schemas" / "examples" / "benchmark-result.example.json").read_text()
    )
    assert example["runtime"]["backend"] == "cpu"
    del example["runtime"]["backend"]
    errors = validate_against(example, "benchmark-result.schema.json")
    assert errors


def test_result_schema_accepts_parameters_field():
    example = yaml.safe_load(
        (REPO_ROOT / "schemas" / "examples" / "benchmark-result.example.json").read_text()
    )
    example["parameters"] = {"beam_size": {"value": 8, "default": 5}}
    assert validate_against(example, "benchmark-result.schema.json") == []


def test_unrecognized_pack_source_type_flags_a_type_this_runner_does_not_know():
    """Simulates an old runner's bundled schema fetching a pack.yaml built
    for a newer platform (e.g. a hypothetical future provider this runner
    predates) — should name the offending type, not silently pass or
    surface only the raw jsonschema enum-mismatch message."""
    pack = {"audio": {"source": {"type": "some_future_provider"}}}
    assert unrecognized_pack_source_type(pack) == "some_future_provider"


def test_unrecognized_pack_source_type_accepts_known_type():
    pack = {"audio": {"source": {"type": "mozilla_data_collective"}}}
    assert unrecognized_pack_source_type(pack) is None


def test_unrecognized_pack_source_type_handles_missing_source():
    assert unrecognized_pack_source_type({"id": "x"}) is None
    assert unrecognized_pack_source_type({"audio": {}}) is None


def test_unmet_min_runner_version_flags_an_installed_version_that_is_too_old():
    pack = {"min_runner_version": "0.5.0"}
    assert unmet_min_runner_version(pack, "0.4.1") == "0.5.0"


def test_unmet_min_runner_version_accepts_a_satisfying_installed_version():
    pack = {"min_runner_version": "0.5.0"}
    assert unmet_min_runner_version(pack, "0.5.0") is None
    assert unmet_min_runner_version(pack, "1.0.0") is None


def test_unmet_min_runner_version_handles_absent_field():
    assert unmet_min_runner_version({"id": "x"}, "0.0.1") is None


def test_result_schema_rejects_parameters_entry_missing_default():
    example = yaml.safe_load(
        (REPO_ROOT / "schemas" / "examples" / "benchmark-result.example.json").read_text()
    )
    example["parameters"] = {"beam_size": {"value": 8}}  # missing required "default"
    errors = validate_against(example, "benchmark-result.schema.json")
    assert errors
