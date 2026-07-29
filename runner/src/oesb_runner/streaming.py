"""Streaming trace types shared between streaming runtime adapters and the
streaming metric plugins (docs/specs/metrics.md "Realtime (streaming)").

A streaming adapter feeds audio in chunks and, after each chunk, gets back
whatever hypothesis the runtime has so far. `StreamTrace` records that
incremental timeline on a *virtual* real-time clock, zeroed at detected
speech onset (not clip-buffer position 0 — see `audio_duration_s` below
for why): chunk `k`'s audio is deemed to "arrive" at `chunk_end_s` (its
position relative to that speech-onset zero point), and its hypothesis
becomes available `decode wall-clock time` later. This lets latency be
measured against simulated real-time playback without actually sleeping
the run out for real, while `processing_time_s` separately tracks true
wall-clock compute (for RTF), which is not the same number as the virtual
emit time (that also includes the simulated chunk arrival gaps).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartialUpdate:
    chunk_end_s: float
    emit_time_s: float
    text: str
    committed_word_count: int
    """Words considered locked in via local agreement (docs/specs/metrics.md
    `first_final_latency`: "non-revisable") — a prefix that has agreed
    across two consecutive hypotheses, monotonic (never decreases once
    reached), plus everything once the utterance's audio is exhausted.
    Confirmed by direct measurement why this can't be a cruder rule like
    "every segment but the last": VAD re-segments as more audio is
    appended, so a segment not being the chunk's last one doesn't mean
    it's actually stable — real audio showed "committed" text revised in
    14 of 30 chunks under that simpler rule."""


@dataclass(frozen=True)
class StreamTrace:
    utterance_id: str
    audio_duration_s: float
    """Despite the name, this is the duration of *detected speech* within
    the clip (VAD-segment offset minus onset), not the raw clip length —
    confirmed by direct measurement that this pack's audio carries
    ~500-600ms of leading/trailing silence on every file, and both
    consumers of this field (`end_of_speech_latency`'s "true end of
    speech", `update_frequency`'s "during continuous speech") want
    speech-relative duration, not clip-relative. Falls back to the full
    clip length only if VAD never detects any speech in the utterance at
    all. Distinct from `pack.total_duration_s` (used for RTF), which
    correctly stays clip-length — RTF is compute-time per second of real
    audio input, silence included, since a live deployment still has to
    stream that silence through."""
    processing_time_s: float
    updates: list[PartialUpdate]
    final_text: str
