"""M5 exit criterion (docs/03-roadmap.md): streaming results validate, verify,
and report tail latency (p50/p95, never mean alone)."""
import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from oesb_runner.cli import app
from oesb_runner.schema_validation import validate_against
from oesb_runner.signing import verify_result_document

faster_whisper = pytest.importorskip(
    "faster_whisper", reason="requires `pip install goesb-runner[faster-whisper]`"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_AUDIO_DIR = REPO_ROOT / "packs" / "librispeech-en" / "audio"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not BATCH_AUDIO_DIR.exists(),
        reason="requires fetched audio: run scripts/fetch_librispeech_subset.py first",
    ),
]

runner = CliRunner()


def test_streaming_run_produces_valid_signed_result_with_latency_percentiles(tmp_path):
    results_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "run", "whisper-medium-en-streaming", "librispeech-en-streaming",
        "--repeats", "1",
        "--model-override", "tiny",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(REPO_ROOT / "packs"),
        "--audio-dir", str(BATCH_AUDIO_DIR),
        "--results-dir", str(results_dir),
        "--models-root", str(tmp_path / "models"),
    ])
    assert result.exit_code == 0, result.stdout

    written = list(results_dir.glob("*.json"))
    assert len(written) == 1
    doc = json.loads(written[0].read_text())

    assert validate_against(doc, "benchmark-result.schema.json") == []
    assert verify_result_document(doc) is True

    for metric_id in (
        "wer", "real_time_factor", "cpu_pct", "ram_mb",
        "update_frequency", "partial_stability", "streaming_responsiveness",
        "first_partial_latency", "first_final_latency", "end_of_speech_latency",
    ):
        assert metric_id in doc["metrics"]

    # Latency metrics must always report p50/p95, never mean alone (docs/specs/metrics.md).
    for metric_id in ("first_partial_latency", "first_final_latency", "end_of_speech_latency"):
        block = doc["metrics"][metric_id]
        assert block["unit"] == "ms"
        assert "spread" in block
        assert "p50" in block["spread"]
        assert "p95" in block["spread"]
        # Cross-metric sanity, not just "has the right shape": a real
        # correctness bug (e.g. a negative-latency arithmetic error, or
        # first_partial computed after first_final) would pass every
        # assertion above this point despite being obviously wrong.
        assert block["value"] >= 0
        assert block["spread"]["p50"] >= 0
        assert block["spread"]["p95"] >= 0

    # A partial hypothesis can never be emitted after the first committed
    # (final) one — first_partial_latency must not exceed
    # first_final_latency for the same run. Compared at the aggregate
    # (p50) level pooled across this pack's utterances, same convention
    # partial_stability/update_frequency already use.
    assert (
        doc["metrics"]["first_partial_latency"]["value"]
        <= doc["metrics"]["first_final_latency"]["value"]
    )


# --- nemotron (ADR-0013): the genuinely incremental, GPU-only engine ---
#
# Guarded lazily (importlib + an in-test probe) rather than with a
# module-level `pytest.importorskip("torch")`: this file's own
# faster-whisper importorskip already gates the whole module, and adding a
# second one here would silently skip the pre-existing faster-whisper test
# above on any environment that has faster-whisper but not torch. A skip
# that takes unrelated coverage with it is worse than no guard.

NEMOTRON_PACK_AUDIO_DIR = REPO_ROOT / "packs" / "fleurs-nl" / "audio"


def _nemotron_gpu_backend() -> str | None:
    """"cuda" / "metal" if this machine has one, else None."""
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "metal"
    return None


nemotron_gpu = pytest.mark.skipif(
    _nemotron_gpu_backend() is None,
    reason="nemotron is GPU-only (ADR-0013 §4): requires CUDA or Apple Silicon MPS",
)
nemotron_pack = pytest.mark.skipif(
    not NEMOTRON_PACK_AUDIO_DIR.exists(),
    reason="requires fetched audio: run scripts/fetch_fleurs_subset.py --language nl_nl first",
)


@nemotron_gpu
@nemotron_pack
def test_nemotron_streaming_run_signs_streaming_latency_ms_not_chunk_ms(tmp_path):
    """ADR-0013 §3 end to end: the signed result must record the encoder
    right-attention-context mode that actually ran, and must not carry a
    `chunk_ms` the adapter never applied. The two are different physical
    quantities, so a result asserting the wrong one is worse than no result.

    Also pins the two metric consequences of a genuinely incremental engine
    at the document level (docs/specs/metrics.md): `partial_stability` is
    exactly 1.0, and `first_partial_latency` equals `first_final_latency`.
    """
    results_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "run", "nemotron-3-5-nl-streaming", "fleurs-nl",
        "--repeats", "1",
        "--backend", _nemotron_gpu_backend(),
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(REPO_ROOT / "packs"),
        "--results-dir", str(results_dir),
        "--models-root", str(tmp_path / "models"),
    ])
    assert result.exit_code == 0, result.stdout

    doc = json.loads(next(iter(results_dir.glob("*.json"))).read_text())

    assert validate_against(doc, "benchmark-result.schema.json") == []
    assert verify_result_document(doc) is True

    assert doc["parameters"]["streaming_latency_ms"]["value"] == 320
    assert "chunk_ms" not in doc["parameters"]
    assert doc["runtime"]["name"] == "nemotron"
    assert doc["runtime"]["backend"] == _nemotron_gpu_backend()

    assert doc["metrics"]["partial_stability"]["value"] == 1.0
    assert (
        doc["metrics"]["first_partial_latency"]["value"]
        == doc["metrics"]["first_final_latency"]["value"]
    )


@nemotron_pack
def test_nemotron_streaming_run_rejects_cpu_backend_instead_of_running_slowly():
    """ADR-0008/ADR-0013 §4's guarantee at the CLI layer: a CPU-only
    contributor gets a hard error naming the supported backends, not a slow
    success that lands real signed results NVIDIA never claims support for.
    Deliberately not gated on GPU availability — the refusal must happen on
    any machine, before any weights load."""
    result = runner.invoke(app, [
        "run", "nemotron-3-5-nl-streaming", "fleurs-nl",
        "--repeats", "1",
        "--backend", "cpu",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(REPO_ROOT / "packs"),
    ])

    assert result.exit_code == 1


@nemotron_pack
def test_nemotron_streaming_run_rejects_an_unsupported_latency_override():
    """160 ms is on NVIDIA's model card but is not one of the four modes
    this checkpoint reports. The profile's `allowed` enum must refuse it at
    the --param gate, before anything loads — never snapped to 320."""
    result = runner.invoke(app, [
        "run", "nemotron-3-5-nl-streaming", "fleurs-nl",
        "--repeats", "1",
        "--param", "streaming_latency_ms=160",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(REPO_ROOT / "packs"),
    ])

    assert result.exit_code == 1
