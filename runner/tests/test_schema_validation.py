from pathlib import Path

import pytest
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
        (REPO_ROOT / "packs" / "librispeech-en" / "pack.yaml").read_text()
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
    generate_bulk_assets.py's overridable_block_for() exists to prevent.

    `vad`/`context_reset` are still declared in `model` (real, fixed
    values a reader can see), just never in `overridable` — a fixed
    declared value and an override-eligible knob are different claims;
    the adapter genuinely ignoring `vad` only rules out the second one."""
    data = yaml.safe_load(
        (REPO_ROOT / "profiles" / "whispercpp-base-en-batch" / "profile.yaml").read_text()
    )
    assert set(data["overridable"]) == {"threads"}
    assert "beam_size" in data["model"]  # set, but not overridable — adapter ignores it
    assert data["model"]["vad"] is False  # explicit, not silently absent (real feedback fix)
    assert data["model"]["context_reset"] == "per_utterance"
    assert "vad" not in data["overridable"]
    assert "context_reset" not in data["overridable"]


def test_overridable_rejects_unknown_domain_shape():
    profile = {
        "id": "x", "version": "1.0.0", "benchmark_type": "batch",
        "runtime": {"name": "faster-whisper"}, "model": {"name": "m", "beam_size": 5},
        "scoring": {"primary_metric": "wer"}, "metrics": ["wer"],
        "overridable": {"beam_size": {"not_a_real_domain_key": [1, 2]}},
    }
    errors = validate_against(profile, "benchmark-profile.schema.json")
    assert errors


def test_result_schema_version_is_0_4():
    example = yaml.safe_load(
        (REPO_ROOT / "schemas" / "examples" / "benchmark-result.example.json").read_text()
    )
    assert example["schema_version"] == "0.4"
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


# --- nemotron profiles + the streaming_latency_ms configuration key (ADR-0013) ---

NEMOTRON_PROFILE_IDS = [
    f"nemotron-3-5-{language}-{benchmark_type}"
    for benchmark_type in ("batch", "streaming")
    for language in ("en", "nl", "de", "fr", "es", "pt")
] + ["nemotron-3-5-concurrency"]


@pytest.mark.parametrize("profile_id", NEMOTRON_PROFILE_IDS)
def test_every_nemotron_profile_validates(profile_id):
    data = yaml.safe_load((REPO_ROOT / "profiles" / profile_id / "profile.yaml").read_text())
    assert validate_against(data, "benchmark-profile.schema.json") == []
    assert data["id"] == profile_id
    assert data["runtime"]["name"] == "nemotron"
    assert data["model"]["name"] == "nemotron-3.5-asr-streaming-0.6b"


@pytest.mark.parametrize(
    "profile_id", [p for p in NEMOTRON_PROFILE_IDS if p.endswith("-streaming")]
)
def test_nemotron_streaming_profiles_declare_streaming_latency_ms_and_never_chunk_ms(profile_id):
    """ADR-0013 §3 / ADR-0009 §2: these are different physical quantities
    (encoder right-attention context vs bounded re-decode window), and this
    adapter applies only the first. Declaring `chunk_ms` here — in
    `configuration` or `overridable` — would sign a result asserting a value
    that had no effect.

    The `allowed` enum is exactly the four modes this checkpoint's own
    processor reports via `supported_streaming_latencies_ms`, read off the
    real checkpoint. NVIDIA's model card lists a fifth (160 ms) that this
    checkpoint does not have; it must not appear here."""
    data = yaml.safe_load((REPO_ROOT / "profiles" / profile_id / "profile.yaml").read_text())

    assert data["configuration"]["streaming_latency_ms"] == 320
    assert "chunk_ms" not in data["configuration"]
    assert "chunk_ms" not in data["overridable"]
    assert data["overridable"]["streaming_latency_ms"]["allowed"] == [80, 320, 560, 1120]
    assert data["configuration"]["streaming_latency_ms"] in data["overridable"]["streaming_latency_ms"]["allowed"]
    assert data["model"]["context_reset"] == "per_utterance"


def test_schema_declares_streaming_latency_ms_in_configuration():
    """The `configuration` block isn't `additionalProperties: false`, so this
    is documentation rather than a gate — declared anyway, per that file's own
    convention, so a reader meeting `streaming_latency_ms` in a profile finds
    it defined instead of inferring it from adapter code."""
    import json

    schema = json.loads(
        (REPO_ROOT / "runner" / "src" / "oesb_runner" / "schemas" / "benchmark-profile.schema.json").read_text()
    )
    entry = schema["properties"]["configuration"]["properties"]["streaming_latency_ms"]
    assert entry["type"] == "integer"
    assert entry["minimum"] == 1
    assert "chunk_ms" in entry["description"]  # says what it is NOT, not just what it is


def test_nemotron_concurrency_profile_has_no_language_or_accuracy_scoring():
    """ADR-0012's corrected shape (one profile, not one per language), reused
    verbatim — plus a `concurrency` ceiling set from this engine's own
    measured per-instance VRAM rather than copied from another engine."""
    data = yaml.safe_load(
        (REPO_ROOT / "profiles" / "nemotron-3-5-concurrency" / "profile.yaml").read_text()
    )
    assert "language" not in data
    assert "normalization" not in data
    assert not {"wer", "cer"} & set(data["metrics"])
    assert data["overridable"]["concurrency"]["range"]["max"] == 8
