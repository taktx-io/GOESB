"""`faster-whisper` batch runtime adapter (docs/02-architecture.md §4).

Optional dependency (`pip install oesb-runner[faster-whisper]`) — the actual
`faster_whisper` package is only imported inside `run_batch`, so importing
`oesb_runner.adapters` never requires it, matching the normalization plugin
pattern.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ..pack import Utterance
from . import register


@dataclass(frozen=True)
class Transcription:
    utterance_id: str
    hypothesis_text: str
    processing_time_s: float


def _resolve_model_id(model_name: str) -> str:
    """Translate OESB's runtime-agnostic model name (profiles say
    'whisper-medium') into the identifier faster-whisper's own API expects
    ('medium') — this translation belongs in the adapter, not the profile,
    so profiles stay independent of any one runtime's naming convention."""
    prefix = "whisper-"
    return model_name[len(prefix):] if model_name.startswith(prefix) else model_name


@register("faster-whisper")
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
) -> list[Transcription]:
    """Transcribe every utterance once, batch-style, and time each call.

    Model load time is deliberately excluded from per-utterance timing (it is
    a one-off cost, not part of what RTF measures) but the loaded model is
    reused across all utterances, matching how a real deployment would run.

    `download_root`, when given, pins exactly where the model snapshot is
    cached — the caller (the CLI) hashes that directory as `model.sha256`, so
    it must actually be where the weights land, not faster-whisper's default
    shared HF cache.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "faster-whisper is not installed; run "
            "`pip install oesb-runner[faster-whisper]`"
        ) from exc

    model = WhisperModel(
        _resolve_model_id(model_name),
        compute_type=quantization,
        cpu_threads=threads,
        download_root=str(download_root) if download_root is not None else None,
    )

    results: list[Transcription] = []
    for utterance in utterances:
        start = time.perf_counter()
        segments, _info = model.transcribe(
            str(utterance.audio_path),
            beam_size=beam_size,
            temperature=temperature,
            vad_filter=vad,
        )
        hypothesis_text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - start
        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results
