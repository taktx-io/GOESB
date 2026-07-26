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


@register(
    "faster-whisper", benchmark_type="batch",
    applied_parameters=frozenset({"quantization", "beam_size", "temperature", "vad", "threads"}),
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
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "faster-whisper is not installed; run "
            "`pip install goesb-runner[faster-whisper]`"
        ) from exc

    model = WhisperModel(
        _resolve_model_id(model_name),
        compute_type=quantization,
        cpu_threads=threads,
        download_root=str(download_root) if download_root is not None else None,
    )

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
) -> list[StreamTrace]:
    """Feed each utterance to faster-whisper in `chunk_ms` chunks, re-decoding
    the growing buffer after every chunk (faster-whisper has no incremental
    decoder state to resume, so "streaming" here means repeated whole-buffer
    re-transcription — the same "local agreement" pattern used by e.g.
    whisper_streaming). One `StreamTrace` per utterance records the resulting
    hypothesis timeline for the streaming metric plugins to score.
    """
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "faster-whisper is not installed; run "
            "`pip install goesb-runner[faster-whisper]`"
        ) from exc

    sample_rate = 16000
    chunk_samples = max(1, int(chunk_ms / 1000 * sample_rate))

    model = WhisperModel(
        _resolve_model_id(model_name),
        compute_type=quantization,
        cpu_threads=threads,
        download_root=str(download_root) if download_root is not None else None,
    )

    traces: list[StreamTrace] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_audio(str(utterance.audio_path), sampling_rate=sample_rate)
        total_samples = len(samples)
        audio_duration_s = total_samples / sample_rate

        updates: list[PartialUpdate] = []
        processing_time_s = 0.0
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

            text = " ".join(segment.text.strip() for segment in segments).strip()
            if is_last_chunk:
                committed_word_count = len(text.split())
            else:
                committed_text = " ".join(s.text.strip() for s in segments[:-1]).strip()
                committed_word_count = len(committed_text.split())

            updates.append(PartialUpdate(
                chunk_end_s=chunk_end_s,
                emit_time_s=chunk_end_s + decode_wall_s,
                text=text,
                committed_word_count=committed_word_count,
            ))

        log_progress(i, len(utterances), utterance.utterance_id, processing_time_s)
        traces.append(StreamTrace(
            utterance_id=utterance.utterance_id,
            audio_duration_s=audio_duration_s,
            processing_time_s=processing_time_s,
            updates=updates,
            final_text=updates[-1].text if updates else "",
        ))
    return traces
