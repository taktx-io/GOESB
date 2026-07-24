"""`whisper.cpp` (via `pywhispercpp`) batch runtime adapter
(docs/02-architecture.md §4).

Optional dependency (`pip install goesb-runner[whisper-cpp]`) —
`pywhispercpp`/`soundfile` are only imported inside `run_batch`, matching the
lazy-import pattern used by every other adapter.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..audio import decode_pcm
from ..pack import Utterance
from . import Transcription, log_progress, register


def _resolve_model_id(model_name: str) -> str:
    """Translate GOESB's runtime-agnostic model name ('whisper-base.en') into
    the ggml model id pywhispercpp/whisper.cpp expects ('base.en') — same
    translation role as faster_whisper._resolve_model_id; each adapter
    carries its own runtime's naming convention rather than the profile."""
    prefix = "whisper-"
    return model_name.removeprefix(prefix)


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
    language: str | None = None,
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

    `language` (2-letter code, e.g. "es") matters more here than it might
    look: whisper.cpp's own default (`whisper_full_default_params()`) is a
    hardcoded `language="en"`, not auto-detect — leaving it unset on
    non-English audio doesn't just skip a nice-to-have, it makes the model
    decode conditioned on the wrong language and produce fluent-sounding but
    wrong (translation-flavored hallucination) English text instead of a
    real transcription. `None` (profile has no declared language) falls
    back to real auto-detection instead of silently keeping that "en"
    default — except for an English-only (`.en`) model, which has no other
    language to detect and produces near-random noise (confirmed: "detected"
    unrelated languages at ~1% confidence) if asked to try.
    """
    try:
        from pywhispercpp.model import Model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "pywhispercpp is not installed; run "
            "`pip install goesb-runner[whisper-cpp]`"
        ) from exc

    resolved_model_id = _resolve_model_id(model_name)
    if language:
        language_params = {"language": language}
    elif resolved_model_id.endswith(".en"):
        language_params = {"language": "en"}
    else:
        language_params = {"detect_language": True}
    model = Model(
        resolved_model_id,
        models_dir=str(download_root) if download_root is not None else None,
        n_threads=threads,
        temperature=temperature,
        print_realtime=False,
        print_progress=False,
        **language_params,
    )

    results: list[Transcription] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, dtype="float32")
        start = time.perf_counter()
        segments = model.transcribe(samples)
        hypothesis_text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - start
        log_progress(i, len(utterances), utterance.utterance_id, elapsed)
        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results
