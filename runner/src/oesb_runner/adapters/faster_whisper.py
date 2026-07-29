"""`faster-whisper` batch runtime adapter (docs/02-architecture.md §4).

Optional dependency (`pip install goesb-runner[faster-whisper]`) — the actual
`faster_whisper` package is only imported inside `run_batch`, so importing
`oesb_runner.adapters` never requires it, matching the normalization plugin
pattern.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..pack import Utterance
from ..streaming import PartialUpdate, StreamTrace
from . import Transcription, log_progress, register


def _resolve_model_id(model_name: str) -> str:
    """Translate GOESB's runtime-agnostic model name (profiles say
    'whisper-medium') into the identifier faster-whisper's own API expects
    ('medium') — this translation belongs in the adapter, not the profile,
    so profiles stay independent of any one runtime's naming convention."""
    prefix = "whisper-"
    return model_name.removeprefix(prefix)


# ADR-0008: ctranslate2's own default (device="auto") silently tries CUDA
# whenever a GPU looks present — on Linux, cuBLAS/cuDNN often come along for
# free via pip wheel dependencies, so this mostly worked by accident; on
# Windows there's no equivalent auto-bundling, so "GPU present, CUDA
# libraries not actually installed" failed deep inside model load, not at
# install time. `backend` must always be passed explicitly as `device=`
# instead of ever leaving it to ctranslate2's own default.
_DEVICE_BY_BACKEND = {"cpu": "cpu", "cuda": "cuda"}


def _load_model(model_name: str, backend: str, quantization: str, threads: int, download_root):
    """Shared `WhisperModel(...)` construction for both run_batch and
    run_streaming. `--backend cuda` on a CTranslate2 build without CUDA
    support raises a raw ValueError deep inside ctranslate2 — caught and
    re-raised as a clear, actionable RuntimeError (ADR-0008: fails
    immediately, before any model weights load, never a silent CPU
    fallback) rather than surfacing a bare third-party stack trace as the
    only explanation."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "faster-whisper is not installed; run "
            "`pip install goesb-runner[faster-whisper]`"
        ) from exc

    try:
        return WhisperModel(
            _resolve_model_id(model_name),
            device=_DEVICE_BY_BACKEND[backend],
            compute_type=quantization,
            cpu_threads=threads,
            download_root=str(download_root) if download_root is not None else None,
        )
    except ValueError as exc:
        if backend == "cuda" and "CUDA" in str(exc):
            raise RuntimeError(
                f"--backend cuda failed: {exc}. Run `goesb doctor` to check what's "
                "missing, or use --backend cpu."
            ) from exc
        raise


@register(
    "faster-whisper", benchmark_type="batch",
    applied_parameters=frozenset({"quantization", "beam_size", "temperature", "vad", "threads"}),
    backends=frozenset(_DEVICE_BY_BACKEND),
)
def run_batch(
    model_name: str,
    utterances: list[Utterance],
    *,
    quantization: str = "int8",
    beam_size: int = 5,
    temperature: float = 0.0,
    vad: bool = True,
    threads: int = 4,
    download_root: str | Path | None = None,
    language: str | None = None,
    backend: str = "cpu",
) -> list[Transcription]:
    """Transcribe every utterance once, batch-style, and time each call.

    Model load time is deliberately excluded from per-utterance timing (it is
    a one-off cost, not part of what RTF measures) but the loaded model is
    reused across all utterances, matching how a real deployment would run.

    `download_root`, when given, pins exactly where the model snapshot is
    cached — the caller (the CLI) hashes that directory as `model.sha256`, so
    it must actually be where the weights land, not faster-whisper's default
    shared HF cache.

    `language` (2-letter code, e.g. "es"; `None` = faster-whisper's own
    auto-detect) should be set from the profile whenever it's known —
    auto-detection is not required to fail, but it's strictly less reliable
    than telling the decoder the language up front.

    `backend` (ADR-0008) is passed straight through as `device=` — the CLI
    has already validated it against this adapter's declared supported set
    before calling in, so an unsupported value never reaches here.
    """
    model = _load_model(model_name, backend, quantization, threads, download_root)

    results: list[Transcription] = []
    for i, utterance in enumerate(utterances, start=1):
        start = time.perf_counter()
        segments, _info = model.transcribe(
            str(utterance.audio_path),
            beam_size=beam_size,
            temperature=temperature,
            vad_filter=vad,
            language=language,
        )
        hypothesis_text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - start
        log_progress(i, len(utterances), utterance.utterance_id, elapsed)
        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results


@register(
    "faster-whisper", benchmark_type="streaming",
    applied_parameters=frozenset(
        {"quantization", "beam_size", "temperature", "vad", "threads", "chunk_ms"}
    ),
    backends=frozenset(_DEVICE_BY_BACKEND),
)
def run_streaming(
    model_name: str,
    utterances: list[Utterance],
    *,
    chunk_ms: int = 1000,
    quantization: str = "int8",
    beam_size: int = 5,
    temperature: float = 0.0,
    vad: bool = True,
    threads: int = 4,
    download_root: str | Path | None = None,
    language: str | None = None,
    backend: str = "cpu",
) -> list[StreamTrace]:
    """Feed each utterance to faster-whisper in `chunk_ms` chunks, re-decoding
    the growing buffer after every chunk (faster-whisper has no incremental
    decoder state to resume, so "streaming" here means repeated whole-buffer
    re-transcription — the same "local agreement" pattern used by e.g.
    whisper_streaming). One `StreamTrace` per utterance records the resulting
    hypothesis timeline for the streaming metric plugins to score.
    """
    try:
        from faster_whisper.audio import decode_audio
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "faster-whisper is not installed; run "
            "`pip install goesb-runner[faster-whisper]`"
        ) from exc

    sample_rate = 16000
    chunk_samples = max(1, int(chunk_ms / 1000 * sample_rate))

    model = _load_model(model_name, backend, quantization, threads, download_root)

    traces: list[StreamTrace] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_audio(str(utterance.audio_path), sampling_rate=sample_rate)
        total_samples = len(samples)
        clip_duration_s = total_samples / sample_rate

        updates: list[PartialUpdate] = []
        processing_time_s = 0.0
        # Real report, confirmed by directly measuring this pack's actual
        # audio: LibriSpeech-sourced clips carry ~500-600ms of leading and
        # trailing silence, consistently, on every file. Zeroing the
        # virtual clock at position 0 of the raw buffer (the previous
        # behavior) meant every latency number below included that dead
        # air, when docs/specs/metrics.md defines these relative to real
        # speech, not clip boundaries. speech_onset_s/speech_offset_s come
        # from faster-whisper's own VAD segment timestamps — no new
        # dependency, no separate detection pass, since we already decode
        # every chunk anyway. onset locks in at first detection (the
        # earliest chunk that produces any segment); offset keeps updating
        # to the latest chunk's last-segment end, so by the final
        # (whole-buffer) chunk it reflects where speech actually stopped.
        speech_onset_s: float | None = None
        speech_offset_s = clip_duration_s  # fallback if VAD never detects any speech at all
        # Local-agreement commit tracking (the "local agreement" strategy
        # the module docstring above already names as the design this
        # mirrors, e.g. whisper_streaming's LocalAgreement-2): a prefix of
        # words is only "committed" once it agrees with the immediately
        # preceding hypothesis, and stays committed even if a later,
        # fresh-VAD-segmented decode revises its own text — monotonic,
        # never decreases. Real report, confirmed by direct measurement
        # on real audio: the previous "every segment but the last" rule
        # had "committed" text revised in 14 of 30 chunks on one real
        # utterance (VAD re-segments as more audio is appended — it isn't
        # stable just because a segment isn't the last one in a given
        # chunk's output), directly contradicting this metric's own
        # "non-revisable" definition.
        committed_word_count = 0
        previous_words: list[str] = []
        end = 0
        while end < total_samples:
            end = min(end + chunk_samples, total_samples)
            is_last_chunk = end >= total_samples
            chunk_end_s = end / sample_rate

            start = time.perf_counter()
            segments, _info = model.transcribe(
                samples[:end],
                beam_size=beam_size,
                temperature=temperature,
                vad_filter=vad,
                language=language,
            )
            segments = list(segments)
            decode_wall_s = time.perf_counter() - start
            processing_time_s += decode_wall_s

            if segments:
                if speech_onset_s is None:
                    speech_onset_s = segments[0].start
                # Clamp to this chunk's own buffer end: Whisper's predicted
                # segment timestamps can slightly overshoot the actual audio
                # fed in (a known model quirk, confirmed by direct
                # measurement — segments[-1].end reported ~40ms past the
                # final chunk's own buffer length on real audio). Without
                # this, speech_offset_s could end up *after* the last
                # chunk's own chunk_end_s, which is nonsensical: "end of
                # speech" can't be later than "end of the whole clip".
                speech_offset_s = min(segments[-1].end, chunk_end_s)

            text = " ".join(segment.text.strip() for segment in segments).strip()
            words = text.split()
            if is_last_chunk:
                committed_word_count = len(words)
            else:
                common_prefix = 0
                for a, b in zip(previous_words, words):
                    if a != b:
                        break
                    common_prefix += 1
                committed_word_count = max(committed_word_count, common_prefix)
            previous_words = words

            updates.append(PartialUpdate(
                chunk_end_s=chunk_end_s,
                emit_time_s=chunk_end_s + decode_wall_s,
                text=text,
                committed_word_count=committed_word_count,
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

        log_progress(i, len(utterances), utterance.utterance_id, processing_time_s)
        traces.append(StreamTrace(
            utterance_id=utterance.utterance_id,
            audio_duration_s=speech_duration_s,
            processing_time_s=processing_time_s,
            updates=updates,
            final_text=updates[-1].text if updates else "",
        ))
    return traces
