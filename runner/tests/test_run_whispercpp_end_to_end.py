"""M2 exit criterion (docs/03-roadmap.md): a third runtime adapter proves
the plugin interface — swaps in without any core-code change."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from oesb_runner.cli import app
from oesb_runner.schema_validation import validate_against
from oesb_runner.signing import verify_result_document

pywhispercpp = pytest.importorskip(
    "pywhispercpp", reason="requires `pip install goesb-runner[whisper-cpp]`"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_AUDIO_DIR = REPO_ROOT / "packs" / "example-librispeech-en-batch" / "audio"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not BATCH_AUDIO_DIR.exists(),
        reason="requires fetched audio: run scripts/fetch_librispeech_subset.py first",
    ),
]

runner = CliRunner()


def test_whispercpp_run_produces_valid_signed_result(tmp_path):
    results_dir = tmp_path / "results"
    result = runner.invoke(app, [
        "run", "whispercpp-base-en-batch", "example-librispeech-en-whispercpp-batch",
        "--repeats", "1",
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
    assert doc["runtime"]["name"] == "whisper-cpp"

    for metric_id in ("wer", "cer", "real_time_factor", "cpu_pct", "ram_mb"):
        assert metric_id in doc["metrics"]
