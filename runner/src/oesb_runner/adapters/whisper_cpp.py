"""`whisper.cpp` (via `pywhispercpp`) batch runtime adapter
(docs/02-architecture.md §4).

Optional dependency (`pip install oesb-runner[whisper-cpp]`) —
`pywhispercpp`/`soundfile` are only imported inside `run_batch`, matching the
lazy-import pattern used by every other adapter.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..audio import decode_pcm
from ..pack import Utterance
from . import Transcription, register


def _resolve_model_id(model_name: str) -> str:
    """Translate OESB's runtime-agnostic model name ('whisper-base.en') into
    the ggml model id pywhispercpp/whisper.cpp expects ('base.en') — same
    translation role as faster_whisper._resolve_model_id; each adapter
    carries its own runtime's naming convention rather than the profile."""
    prefix = "whisper-"
    return model_name[len(prefix):] if model_name.startswith(prefix) else model_name


@register("whisper-cpp", benchmark_type="batch")
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

    `quantization`/`beam_size`/`vad` are accepted for call-shape parity with
    the other batch adapters (docs/03-roadmap.md M2 exit criterion: adapters
    swap without core changes) but unused here — whisper.cpp's ggml models
    are pre-quantized by model-file choice rather than a runtime flag,
    `beam_size` lives under a nested `beam_search` param pywhispercpp's flat
    `**params` doesn't set (defaults to whisper.cpp's own greedy strategy),
    and this batch adapter doesn't chunk audio (that's M5's streaming
    concern).
    """
    try:
        from pywhispercpp.model import Model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "pywhispercpp is not installed; run "
            "`pip install oesb-runner[whisper-cpp]`"
        ) from exc

    model = Model(
        _resolve_model_id(model_name),
        models_dir=str(download_root) if download_root is not None else None,
        n_threads=threads,
        temperature=temperature,
        print_realtime=False,
        print_progress=False,
    )

    results: list[Transcription] = []
    for utterance in utterances:
        samples = decode_pcm(utterance.audio_path, dtype="float32")
        start = time.perf_counter()
        segments = model.transcribe(samples)
        hypothesis_text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - start
        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results
