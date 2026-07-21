"""Streaming trace types shared between streaming runtime adapters and the
streaming metric plugins (docs/specs/metrics.md "Realtime (streaming)").

A streaming adapter feeds audio in chunks and, after each chunk, gets back
whatever hypothesis the runtime has so far. `StreamTrace` records that
incremental timeline on a *virtual* real-time clock: chunk `k`'s audio is
deemed to "arrive" at `chunk_end_s` (its position in the utterance's own
timeline), and its hypothesis becomes available `decode wall-clock time`
later. This lets latency be measured against simulated real-time playback
without actually sleeping the run out for real, while `processing_time_s`
separately tracks true wall-clock compute (for RTF), which is not the same
number as the virtual emit time (that also includes the simulated chunk
arrival gaps).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartialUpdate:
    chunk_end_s: float
    emit_time_s: float
    text: str
    committed_word_count: int
    """Words belonging to segments the runtime will not revise further —
    every segment but the last while more audio is still to come, all
    segments once the utterance's audio is exhausted."""


@dataclass(frozen=True)
class StreamTrace:
    utterance_id: str
    audio_duration_s: float
    processing_time_s: float
    updates: list[PartialUpdate]
    final_text: str
