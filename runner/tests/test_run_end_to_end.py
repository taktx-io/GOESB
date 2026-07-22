"""M1 exit criterion (docs/03-roadmap.md): two runs on the same machine agree
within tolerance, and the result validates + its hashes verify."""
import json

import pytest
from typer.testing import CliRunner

from oesb_runner.cli import app
from oesb_runner.schema_validation import _find_repo_schemas_dir, validate_against
from oesb_runner.signing import verify_result_document

faster_whisper = pytest.importorskip(
    "faster_whisper", reason="requires `pip install goesb-runner[faster-whisper]`"
)

REPO_ROOT = _find_repo_schemas_dir().parent
PACK_DIR = REPO_ROOT / "packs" / "example-librispeech-en-batch"

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
        "run", "whisper-medium-en-batch", "example-librispeech-en-batch",
        "--repeats", "2",
        "--model-override", "tiny",
        "--profiles-dir", str(REPO_ROOT / "profiles"),
        "--packs-dir", str(REPO_ROOT / "packs"),
        "--results-dir", str(results_dir),
        "--models-root", str(tmp_path / "models"),
    ])
    assert result.exit_code == 0, result.stdout

    written = list(results_dir.glob("*.json"))
    assert len(written) == 1
    doc = json.loads(written[0].read_text())

    # Validates against its own schema.
    assert validate_against(doc, "benchmark-result.schema.json") == []

    # Hashes verify: signature covers exactly the content it claims to.
    assert verify_result_document(doc) is True

    # Reproducibility: primary metric (wer) has zero spread across 2 repeats
    # for this deterministic (beam_size, temperature=0.0) config — the
    # concrete form of "two runs agree within tolerance" for this profile.
    assert doc["repeats"] == 2
    wer_block = doc["metrics"]["wer"]
    assert "spread" in wer_block
    assert wer_block["spread"]["std"] == pytest.approx(0.0)
    assert wer_block["value"] < 0.25  # tiny model on clean read speech, sanity bound

    # Every metric M1 implements (of the profile's required set) is present.
    # energy_wh is profile-required but not yet implemented — a known M1/M2 gap.
    for metric_id in ("wer", "cer", "real_time_factor", "cpu_pct", "ram_mb"):
        assert metric_id in doc["metrics"]
