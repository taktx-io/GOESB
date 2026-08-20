"""M1 exit criterion (docs/03-roadmap.md): two runs on the same machine agree
within tolerance, and the result validates + its hashes verify."""
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
PACK_DIR = REPO_ROOT / "packs" / "librispeech-en"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (PACK_DIR / "audio").exists(),
        reason="requires fetched audio: run scripts/fetch_librispeech_subset.py first",
    ),
]

runner = CliRunner()


def test_run_produces_valid_signed_reproducible_result(tmp_path):
    results_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "run", "whisper-medium-en-batch", "librispeech-en",
        "--repeats", "2",
        "--model-override", "tiny",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(REPO_ROOT / "packs"),
        "--results-dir", str(results_dir),
        "--models-root", str(tmp_path / "models"),
        "--hardware", "intel-xeon-e3-1240-v6",
    ])
    assert result.exit_code == 0, result.stdout

    written = list(results_dir.glob("*.json"))
    assert len(written) == 1
    doc = json.loads(written[0].read_text())

    # Validates against its own schema.
    assert validate_against(doc, "benchmark-result.schema.json") == []

    # Hashes verify: signature covers exactly the content it claims to.
    assert verify_result_document(doc) is True

    # User-asserted hardware id makes it into the signed document as-is.
    assert doc["hardware_id"] == "intel-xeon-e3-1240-v6"

    # Reproducibility: primary metric (wer) is identical across both
    # corpus-level repeats for this deterministic (beam_size,
    # temperature=0.0) config — "two runs agree within tolerance" lives in
    # per_repeat, not spread. spread for wer/cer is pooled per-recording
    # (docs/specs/metrics.md "Reporting"), not per-repeat — a per-repeat
    # spread would be degenerate (always zero for a deterministic decoder,
    # exactly the case this test exercises) and hide the real per-recording
    # distribution checked below.
    assert doc["repeats"] == 2
    wer_block = doc["metrics"]["wer"]
    assert wer_block["per_repeat"][0] == pytest.approx(wer_block["per_repeat"][1])
    assert wer_block["value"] < 0.25  # tiny model on clean read speech, sanity bound
    assert "spread" in wer_block
    assert set(wer_block["spread"]) == {"std", "min", "max", "p50", "p95"}

    # Every metric M1 implements (of the profile's required set) is present.
    # energy_wh is profile-required but not yet implemented — a known M1/M2 gap.
    for metric_id in ("wer", "cer", "real_time_factor", "cpu_pct", "ram_mb"):
        assert metric_id in doc["metrics"]

    # Per-utterance recognition log: one JSONL line per utterance per
    # repeat, next to but separate from the result document.
    utterances_written = list(results_dir.glob("*.utterances.jsonl"))
    assert len(utterances_written) == 1
    lines = [json.loads(line) for line in utterances_written[0].read_text().splitlines()]
    manifest_lines = (PACK_DIR / "manifest.jsonl").read_text().splitlines()
    utterance_count = sum(1 for line in manifest_lines if line.strip())
    assert len(lines) == utterance_count * doc["repeats"]
    assert {entry["repeat"] for entry in lines} == {1, 2}
    for entry in lines:
        assert entry.keys() == {"repeat", "utterance_id", "reference_text", "hypothesis_text"}
        assert entry["reference_text"]  # every LibriSpeech utterance has a non-empty transcript


# --- nemotron batch (ADR-0013): GPU-only, one multilingual checkpoint ---
#
# Guarded lazily rather than with a module-level `pytest.importorskip
# ("torch")` — a second module-level skip here would take the pre-existing
# faster-whisper test above with it on any environment that has
# faster-whisper but not torch.

NEMOTRON_PACK_AUDIO_DIR = REPO_ROOT / "packs" / "fleurs-nl" / "audio"


def _nemotron_gpu_backend() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "metal"
    return None


@pytest.mark.skipif(
    _nemotron_gpu_backend() is None,
    reason="nemotron is GPU-only (ADR-0013 §4): requires CUDA or Apple Silicon MPS",
)
@pytest.mark.skipif(
    not NEMOTRON_PACK_AUDIO_DIR.exists(),
    reason="requires fetched audio: run scripts/fetch_fleurs_subset.py --language nl_nl first",
)
def test_nemotron_batch_run_produces_a_valid_signed_gpu_result(tmp_path):
    """A `nemotron` result must ingest exactly like any other: `runtime.name`
    and `runtime.backend` are free strings owned by the adapter registry, and
    nothing downstream carries a hardcoded engine list. Proven here at the
    document layer rather than asserted in the ADR."""
    results_dir = tmp_path / "results"
    backend = _nemotron_gpu_backend()
    result = runner.invoke(app, [
        "run", "nemotron-3-5-nl-batch", "fleurs-nl",
        "--repeats", "1",
        "--backend", backend,
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(REPO_ROOT / "packs"),
        "--results-dir", str(results_dir),
        "--models-root", str(tmp_path / "models"),
    ])
    assert result.exit_code == 0, result.stdout

    doc = json.loads(next(iter(results_dir.glob("*.json"))).read_text())

    assert validate_against(doc, "benchmark-result.schema.json") == []
    assert verify_result_document(doc) is True
    assert doc["runtime"] == {**doc["runtime"], "name": "nemotron", "backend": backend}
    assert doc["model"]["name"] == "nemotron-3.5-asr-streaming-0.6b"
    # Real Dutch out of a real multilingual checkpoint — measured 0.131 WER
    # on this pack; a loose bound, not a pinned accuracy number.
    assert doc["metrics"]["wer"]["value"] < 0.5
