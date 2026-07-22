from pathlib import Path

import pytest

from oesb_runner.normalization import normalize
from oesb_runner.pack import load_pack

pywhispercpp = pytest.importorskip(
    "pywhispercpp", reason="requires `pip install goesb-runner[whisper-cpp]`"
)

from oesb_runner.adapters.whisper_cpp import run_batch  # noqa: E402
from oesb_runner.metrics import rtf, wer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "packs" / "example-librispeech-en-batch"

pytestmark = pytest.mark.skipif(
    not (PACK_DIR / "audio").exists(),
    reason="requires fetched audio: run scripts/fetch_librispeech_subset.py first",
)


@pytest.mark.slow
def test_run_batch_transcribes_real_audio_within_wer_tolerance(tmp_path):
    """End-to-end proof: pack -> adapter -> normalization -> metrics.

    Proves the third runtime adapter interface (docs/03-roadmap.md M2:
    adapters swap without core changes) — same shape as
    test_adapter_faster_whisper.py's batch test, different runtime.
    """
    pack = load_pack(PACK_DIR)
    transcriptions = run_batch("base.en", pack.utterances, download_root=tmp_path / "models")
    by_id = {t.utterance_id: t for t in transcriptions}

    pairs = []
    for utterance in pack.utterances:
        hypothesis = by_id[utterance.utterance_id].hypothesis_text
        pairs.append((
            normalize("goesb-en-v1", utterance.reference_text),
            normalize("goesb-en-v1", hypothesis),
        ))

    result_wer = wer.compute(pairs)
    total_processing_s = sum(t.processing_time_s for t in transcriptions)
    result_rtf = rtf.compute(total_processing_s, pack.total_duration_s)

    # whisper.cpp base.en on clean read speech: loose bound, just proving the wiring.
    assert result_wer < 0.25
    assert result_rtf < 1.0  # faster than realtime even on CPU
