from pathlib import Path

import pytest

from oesb_runner.normalization import normalize
from oesb_runner.pack import load_pack

# NOTE: test_resolve_model_id lives in test_faster_whisper_model_id.py, not
# here — pytest.importorskip below fails the whole module's *collection* if
# faster-whisper isn't installed, which would wrongly skip a pure-string-logic
# test that has nothing to do with the actual package being present.
faster_whisper = pytest.importorskip(
    "faster_whisper", reason="requires `pip install goesb-runner[faster-whisper]`"
)

from oesb_runner.adapters.faster_whisper import run_batch, run_streaming
from oesb_runner.metrics import rtf, wer

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "packs" / "example-librispeech-en-batch"

pytestmark = pytest.mark.skipif(
    not (PACK_DIR / "audio").exists(),
    reason="requires fetched audio: run scripts/fetch_librispeech_subset.py first",
)


@pytest.mark.slow
def test_run_batch_transcribes_real_audio_within_wer_tolerance():
    """End-to-end proof: pack -> adapter -> normalization -> metrics.

    Uses `tiny` (not the profile's official `whisper-medium`) to keep this
    test fast; it validates the *pipeline*, not the official profile's
    accuracy bar.
    """
    pack = load_pack(PACK_DIR)
    transcriptions = run_batch("tiny", pack.utterances, beam_size=5, temperature=0.0)
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

    # whisper-tiny on clean read speech: loose bound, just proving the wiring.
    assert result_wer < 0.25
    assert result_rtf < 1.0  # faster than realtime even on CPU with tiny


@pytest.mark.slow
def test_run_streaming_transcribes_real_audio_in_chunks():
    """End-to-end proof: pack -> chunked adapter -> per-utterance StreamTrace.

    Uses `tiny` (not the profile's official `whisper-medium`) to keep this
    test fast; validates the chunked-feed pipeline and trace shape.
    """
    pack = load_pack(PACK_DIR)
    traces = run_streaming("tiny", pack.utterances, chunk_ms=1000, beam_size=5, temperature=0.0)

    assert len(traces) == len(pack.utterances)
    by_id = {t.utterance_id: t for t in traces}
    for utterance in pack.utterances:
        trace = by_id[utterance.utterance_id]
        assert trace.updates, f"{utterance.utterance_id} produced no chunk updates"
        assert trace.final_text == trace.updates[-1].text
        assert trace.audio_duration_s == pytest.approx(utterance.duration_s, rel=0.05)
        # Every chunk boundary is monotonically increasing and ends at the
        # utterance's own duration on the final chunk.
        chunk_ends = [u.chunk_end_s for u in trace.updates]
        assert chunk_ends == sorted(chunk_ends)
        assert chunk_ends[-1] == pytest.approx(trace.audio_duration_s)

    pairs = []
    for utterance in pack.utterances:
        hypothesis = by_id[utterance.utterance_id].final_text
        pairs.append((
            normalize("goesb-en-v1", utterance.reference_text),
            normalize("goesb-en-v1", hypothesis),
        ))
    assert wer.compute(pairs) < 0.25
