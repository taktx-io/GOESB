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

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .pack import Utterance


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


# How many of a window's most-recent agreeing words to hold back from
# committing even once they agree across two consecutive decodes — the
# same reason whisper_streaming's own LocalAgreement-2 policy never
# commits its freshest words: a decode's last word or two is the part
# most likely to be revised once more audio (more context) lands, since
# it's closest to wherever the decode's own attention window currently
# ends. 2 is a conservative starting point, not a measured optimum.
_COMMIT_SAFETY_MARGIN = 2

# Real report, confirmed by directly measuring real audio (faster-whisper,
# the first engine this ran against): trimming the window flush to the
# last committed word's own `end` timestamp silently dropped single words
# at the seam on 5+ of 15 real LibriSpeech utterances ("...OF ART MISTER
# QUILTER WRITES..." -> "...of Mr. Quilter writes...", "ART" gone) --
# Whisper-family word end timestamps aren't precise enough to cut flush
# against, and VAD (re-run fresh on every window) reads a hard mid-phoneme
# cut as leading silence and skips real speech at the new window's own
# start. This cushion, subtracted from the first still-uncommitted word's
# own START (a real word boundary, not an imprecise end-time) before
# trimming, gives the next window's VAD/encoder genuine leading acoustic
# context instead of a splice. Not tuned against real audio beyond
# confirming it stops the word-dropping observed above -- not a measured
# optimum.
_TRIM_CUSHION_S = 0.3


def _normalize_word_for_overlap_match(word: str) -> str:
    """Real report: comparing raw words for the committed/window overlap
    check below missed real duplicates on actual audio ("and we are We
    are glad", "finish in art Art is") -- Whisper-family engines
    capitalize a window's own first word as if it were a fresh sentence
    start, even when that word was already committed mid-sentence,
    lowercase, from the previous window. Trailing punctuation (a comma
    landing on one decode's version of a word but not the other's)
    caused the same kind of miss. Comparison-only -- the caller's own
    `committed_words` keeps whatever casing/punctuation it was first
    committed with."""
    return word.strip(".,!?;:\"'").lower()


def run_windowed_local_agreement_streaming(
    utterance: Utterance,
    samples: Any,
    *,
    sample_rate: int,
    chunk_ms: int,
    decode_window: Callable[[Any], list[dict]],
) -> StreamTrace:
    """Shared bounded-window, local-agreement streaming loop for
    Whisper-family engines that have no incremental decoder state to
    resume and must re-decode to produce a streaming trace at all
    (faster-whisper, whisper.cpp — unlike e.g. vosk's KaldiRecognizer,
    which genuinely resumes decoder state across `AcceptWaveform` calls
    and needs none of this; see its own `run_streaming`). This is the
    shared "how" so one real, hard-won correctness fix only has to exist
    once for every such engine, not get independently rediscovered (or
    worse, diverge) in a second copy — see `_TRIM_CUSHION_S` and
    `_normalize_word_for_overlap_match`'s own docstrings for the two real
    bugs (word-dropping, then duplication) found and fixed via 4 rounds
    of real-audio validation against faster-whisper, the first engine
    this ran against.

    Re-decodes only the audio *since the last commit*, not the whole
    clip so far every chunk — bounding per-chunk decode cost instead of
    letting it grow with clip length (real report: an earlier
    whole-buffer version measured RTF 3.19x on Apple M1 Pro, and its
    naive `chunk_end_s + decode_wall_s` latency math understated real
    latency once behind realtime, since it assumed each chunk starts
    decoding the instant its audio "arrives," ignoring backlog from
    slower earlier chunks).

    `decode_window(samples_slice)` is the only engine-specific piece —
    it must decode exactly the slice it's given and return that window's
    words as `[{"word": str, "start": float, "end": float}]`, timestamps
    relative to position 0 of the slice (not the whole clip) — this
    function handles converting those to absolute clip-relative time,
    the local-agreement commit/trim/dedup state machine, and assembling
    the final `StreamTrace` itself.
    """
    total_samples = len(samples)
    chunk_samples = max(1, int(chunk_ms / 1000 * sample_rate))
    clip_duration_s = total_samples / sample_rate

    updates: list[PartialUpdate] = []
    processing_time_s = 0.0
    # Real report, confirmed by directly measuring real audio: LibriSpeech-
    # sourced clips carry ~500-600ms of leading and trailing silence,
    # consistently, on every file. Zeroing the virtual clock at position 0
    # of the raw buffer (rather than at detected speech onset) means every
    # latency number below would include that dead air, when
    # docs/specs/metrics.md defines these relative to real speech, not
    # clip boundaries. onset locks in at first detection (the earliest
    # chunk that produces any word); offset keeps updating to the latest
    # chunk's last-word end, so by the final chunk it reflects where
    # speech actually stopped.
    speech_onset_s: float | None = None
    speech_offset_s = clip_duration_s  # fallback if no words ever land

    # Bounded-window state — see this function's own docstring.
    window_start_sample = 0
    committed_words: list[str] = []  # locked in from PRIOR windows, never revised
    previous_window_words: list[dict] = []  # this window's previous decode, for local agreement

    end = 0
    while end < total_samples:
        end = min(end + chunk_samples, total_samples)
        is_last_chunk = end >= total_samples
        chunk_end_s = end / sample_rate
        window_offset_s = window_start_sample / sample_rate

        start = time.perf_counter()
        raw_words = decode_window(samples[window_start_sample:end])
        decode_wall_s = time.perf_counter() - start
        processing_time_s += decode_wall_s

        # Absolute clip-relative time, not window-relative — decode_window
        # hands back timestamps from position 0 of whatever slice it was
        # given, which is `window_offset_s` into the real clip once the
        # window has trimmed forward.
        window_words = [
            {"word": w["word"], "start": w["start"] + window_offset_s, "end": w["end"] + window_offset_s}
            for w in raw_words
        ]

        # `_TRIM_CUSHION_S` deliberately keeps a little already-committed
        # audio in the new window (real report: cutting flush against it
        # dropped words instead — see that constant's own docstring) —
        # which means this decode can re-transcribe the tail end of
        # `committed_words` a second time. Strip however many of
        # `window_words`' own leading entries match the tail of
        # `committed_words` (case/punctuation-insensitive — see
        # `_normalize_word_for_overlap_match`), in order, before treating
        # the rest as this window's real (new, uncommitted) content —
        # otherwise that re-transcribed overlap gets appended AGAIN,
        # duplicating words in the final text ("and we are We are glad",
        # confirmed on real audio without this check).
        if committed_words:
            overlap = 0
            committed_tail_norm = [_normalize_word_for_overlap_match(w) for w in committed_words]
            for k in range(min(len(window_words), len(committed_words)), 0, -1):
                window_head_norm = [_normalize_word_for_overlap_match(w["word"]) for w in window_words[:k]]
                if window_head_norm == committed_tail_norm[-k:]:
                    overlap = k
                    break
            window_words = window_words[overlap:]

        if window_words:
            if speech_onset_s is None:
                speech_onset_s = window_words[0]["start"]
            # Clamp to this chunk's own buffer end — Whisper-family
            # predicted timestamps can slightly overshoot the real audio
            # fed in (a known model quirk, confirmed by direct
            # measurement).
            speech_offset_s = min(window_words[-1]["end"], chunk_end_s)

        # Local agreement WITHIN this window: a word only counts as
        # agreeing once it matches, by position, across two consecutive
        # decodes of the SAME (untrimmed) window. Real report that shaped
        # this rule: a cruder "every word but the last" rule had
        # committed text revised in 14 of 30 chunks on real audio, since
        # VAD/segmentation isn't stable just because a word isn't the
        # decode's last one.
        agreeing = 0
        for a, b in zip(previous_window_words, window_words):
            if a["word"] != b["word"]:
                break
            agreeing += 1
        newly_stable = len(window_words) if is_last_chunk else max(0, agreeing - _COMMIT_SAFETY_MARGIN)
        previous_window_words = window_words

        if newly_stable > 0:
            newly_committed = window_words[:newly_stable]
            committed_words.extend(w["word"] for w in newly_committed)
            # Trim: advance the window, bounding decode cost (the whole
            # point of this function) — but NOT flush against the last
            # committed word's own `end` timestamp; cut `_TRIM_CUSHION_S`
            # before the first still-uncommitted word's own START instead
            # (see that constant's own docstring for the real-audio
            # word-dropping this fixes). Monotonic: never moves backward
            # past where the window already is.
            next_word_start_s = (
                window_words[newly_stable]["start"] if newly_stable < len(window_words)
                else newly_committed[-1]["end"]
            )
            cushioned_start_s = max(0.0, next_word_start_s - _TRIM_CUSHION_S)
            window_start_sample = max(
                window_start_sample,
                min(total_samples, round(cushioned_start_s * sample_rate)),
            )
            # The window just moved out from under `previous_window_words`
            # (its offsets no longer correspond to any future decode's
            # window) — reset so the next decode's agreement check starts
            # fresh against the new window, same as the very first
            # chunk's own bootstrap (empty previous_window_words).
            previous_window_words = []

        tail_words = [w["word"] for w in window_words[newly_stable:]]
        text = " ".join(committed_words + tail_words).strip()

        updates.append(PartialUpdate(
            chunk_end_s=chunk_end_s,
            emit_time_s=chunk_end_s + decode_wall_s,
            text=text,
            committed_word_count=len(committed_words),
        ))

    onset = speech_onset_s if speech_onset_s is not None else 0.0
    speech_duration_s = max(0.0, speech_offset_s - onset)
    updates = [
        PartialUpdate(
            chunk_end_s=u.chunk_end_s - onset,
            emit_time_s=u.emit_time_s - onset,
            text=u.text,
            committed_word_count=u.committed_word_count,
        )
        for u in updates
    ]

    return StreamTrace(
        utterance_id=utterance.utterance_id,
        audio_duration_s=speech_duration_s,
        processing_time_s=processing_time_s,
        updates=updates,
        final_text=updates[-1].text if updates else "",
    )
