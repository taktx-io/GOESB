"""M5 exit criterion (docs/03-roadmap.md): streaming results validate, verify,
and report tail latency (p50/p95, never mean alone)."""
import json

import pytest
from typer.testing import CliRunner

from oesb_runner.cli import app
from oesb_runner.schema_validation import _find_repo_schemas_dir, validate_against
from oesb_runner.signing import verify_result_document

faster_whisper = pytest.importorskip(
    "faster_whisper", reason="requires `pip install oesb-runner[faster-whisper]`"
)

REPO_ROOT = _find_repo_schemas_dir().parent
BATCH_AUDIO_DIR = REPO_ROOT / "packs" / "example-librispeech-en-batch" / "audio"

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
        "run", "whisper-medium-en-streaming", "example-librispeech-en-streaming",
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
