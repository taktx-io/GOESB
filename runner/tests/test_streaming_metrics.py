import pytest

from oesb_runner.metrics import (
    end_of_speech_latency,
    first_final_latency,
    first_partial_latency,
    partial_stability,
    streaming_responsiveness,
    update_frequency,
)
from oesb_runner.streaming import PartialUpdate, StreamTrace


def _trace(utterance_id, audio_duration_s, updates, processing_time_s=0.1):
    return StreamTrace(
        utterance_id=utterance_id,
        audio_duration_s=audio_duration_s,
        processing_time_s=processing_time_s,
        updates=updates,
        final_text=updates[-1].text if updates else "",
    )


def test_first_partial_latency_is_first_nonempty_update():
    trace = _trace("u1", 2.0, [
        PartialUpdate(chunk_end_s=1.0, emit_time_s=1.05, text="", committed_word_count=0),
        PartialUpdate(chunk_end_s=2.0, emit_time_s=2.03, text="hello world", committed_word_count=2),
    ])
    assert first_partial_latency.compute([trace]) == pytest.approx([2030.0])


def test_first_partial_latency_rejects_utterance_with_no_hypothesis():
    trace = _trace("u1", 1.0, [
        PartialUpdate(chunk_end_s=1.0, emit_time_s=1.02, text="", committed_word_count=0),
    ])
    with pytest.raises(ValueError):
        first_partial_latency.compute([trace])


def test_first_final_latency_waits_for_committed_words():
    trace = _trace("u1", 3.0, [
        PartialUpdate(chunk_end_s=1.0, emit_time_s=1.05, text="hello", committed_word_count=0),
        PartialUpdate(chunk_end_s=2.0, emit_time_s=2.04, text="hello there", committed_word_count=1),
        PartialUpdate(chunk_end_s=3.0, emit_time_s=3.02, text="hello there friend", committed_word_count=3),
    ])
    assert first_final_latency.compute([trace]) == pytest.approx([2040.0])


def test_end_of_speech_latency_is_final_decode_delay():
    trace = _trace("u1", 2.0, [
        PartialUpdate(chunk_end_s=1.0, emit_time_s=1.05, text="hello", committed_word_count=0),
        PartialUpdate(chunk_end_s=2.0, emit_time_s=2.03, text="hello world", committed_word_count=2),
    ])
    assert end_of_speech_latency.compute([trace]) == pytest.approx([30.0])


def test_update_frequency_counts_only_text_changing_updates():
    trace = _trace("u1", 2.0, [
        PartialUpdate(chunk_end_s=1.0, emit_time_s=1.0, text="hello", committed_word_count=0),
        PartialUpdate(chunk_end_s=2.0, emit_time_s=2.0, text="hello", committed_word_count=0),  # unchanged
        PartialUpdate(chunk_end_s=3.0, emit_time_s=3.0, text="hello world", committed_word_count=2),
    ])
    # 2 distinct texts over 2.0s of audio
    assert update_frequency.compute([trace]) == pytest.approx(1.0)


def test_update_frequency_rejects_zero_duration_corpus():
    with pytest.raises(ValueError):
        update_frequency.compute([])


def test_partial_stability_perfect_when_partials_only_extend():
    trace = _trace("u1", 2.0, [
        PartialUpdate(chunk_end_s=1.0, emit_time_s=1.0, text="hello", committed_word_count=0),
        PartialUpdate(chunk_end_s=2.0, emit_time_s=2.0, text="hello world", committed_word_count=2),
    ])
    assert partial_stability.compute([trace]) == pytest.approx(1.0)


def test_partial_stability_penalizes_rewrites():
    trace = _trace("u1", 2.0, [
        PartialUpdate(chunk_end_s=1.0, emit_time_s=1.0, text="hello word", committed_word_count=0),
        # second word rewritten: "word" -> "world"
        PartialUpdate(chunk_end_s=2.0, emit_time_s=2.0, text="hello world", committed_word_count=2),
    ])
    # step 1: prev=[] -> 0 rewritten, 2 emitted
    # step 2: prev=["hello","word"], new=["hello","world"] -> lcp=1, rewritten=1, emitted=2
    # total: rewritten=1, emitted=4 -> stability = 1 - 1/4 = 0.75
    assert partial_stability.compute([trace]) == pytest.approx(0.75)


def test_partial_stability_rejects_traces_with_no_updates():
    trace = _trace("u1", 1.0, [])
    with pytest.raises(ValueError):
        partial_stability.compute([trace])


def test_streaming_responsiveness_formula():
    value = streaming_responsiveness.compute(
        update_frequency_hz=2.0, partial_stability=0.5, first_partial_latency_p50_ms=500.0
    )
    assert value == pytest.approx((2.0 * 0.5) / 0.5)


def test_streaming_responsiveness_rejects_nonpositive_latency():
    with pytest.raises(ValueError):
        streaming_responsiveness.compute(
            update_frequency_hz=1.0, partial_stability=1.0, first_partial_latency_p50_ms=0.0
        )
