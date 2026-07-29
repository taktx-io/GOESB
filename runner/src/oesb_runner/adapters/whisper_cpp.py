"""`whisper.cpp` (via `pywhispercpp`) batch runtime adapter
(docs/02-architecture.md §4).

Optional dependency (`pip install goesb-runner[whisper-cpp]`) —
`pywhispercpp`/`soundfile` are only imported inside `run_batch`, matching the
lazy-import pattern used by every other adapter.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..audio import decode_pcm
from ..pack import Utterance
from . import Transcription, log_progress, register

# whisper.cpp's own build-info string (whisper_print_system_info(), bound
# as Model.system_info() — a static method, no model file needed) is the
# only reliable way to know whether this exact compiled pywhispercpp
# binary actually has CUDA support. Confirmed by direct inspection: e.g.
# "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | CPU :
# NEON = 1 | ..." on a Metal-only Mac build — a CUDA build's string
# includes "CUDA = 1" the same way.
_CUDA_SYSTEM_INFO_RE = re.compile(r"\bCUDA\s*=\s*1\b", re.IGNORECASE)


def cuda_available(model_cls: Any) -> bool:
    """True iff `model_cls` (pywhispercpp's `Model`) reports real CUDA
    support in its own compiled-in system_info string. Exposed at module
    level (not adapter-private) so `cli._doctor_engine_line` can give the
    same real answer `goesb doctor` currently can't for this engine,
    instead of "can't be checked without running a real transcription"."""
    return _CUDA_SYSTEM_INFO_RE.search(model_cls.system_info() or "") is not None


def _resolve_model_id(model_name: str) -> str:
    """Translate GOESB's runtime-agnostic model name ('whisper-base.en') into
    the ggml model id pywhispercpp/whisper.cpp expects ('base.en') — same
    translation role as faster_whisper._resolve_model_id; each adapter
    carries its own runtime's naming convention rather than the profile."""
    prefix = "whisper-"
    return model_name.removeprefix(prefix)


# ADR-0008: leaving context_params unset lets pywhispercpp fall back to
# whisper.cpp's own compiled-in default (whisper_context_default_params()),
# which silently uses whatever `use_gpu` value the binary was built with —
# same "library decides, not us" failure mode ADR-0008 rejects for
# faster-whisper. `use_gpu` is whisper.cpp's own accelerator toggle: it's
# not vendor-specific the way faster-whisper's `device=` is (it means
# "whichever GPU backend this binary was compiled with" — CUDA, Metal, or
# Vulkan), so on its own "cuda" here would just mean "ask whisper.cpp to
# use its compiled-in GPU backend, whatever that is" — a no-op rather than
# a hard error on a Metal-only or CPU-only build. `run_batch` below checks
# `cuda_available()` before ever setting this, specifically for the
# "cuda" case, so that gap doesn't reach here anymore.
_USE_GPU_BY_BACKEND = {"cpu": False, "cuda": True}


@register(
    "whisper-cpp", benchmark_type="batch",
    applied_parameters=frozenset({"threads", "temperature"}),
    backends=frozenset(_USE_GPU_BY_BACKEND),
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

    `backend` (ADR-0008) sets `use_gpu` explicitly via `context_params` —
    the CLI has already validated it against this adapter's declared
    supported set before calling in.
    """
    try:
        from pywhispercpp.model import Model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "pywhispercpp is not installed; run "
            "`pip install goesb-runner[whisper-cpp]`"
        ) from exc

    if backend == "cuda":
        info = Model.system_info()
        if _CUDA_SYSTEM_INFO_RE.search(info or "") is None:
            raise RuntimeError(
                f"--backend cuda failed: this pywhispercpp build has no CUDA support "
                f"(system_info: {info!r}). Run `goesb doctor` to check what's "
                "available, or use --backend cpu."
            )

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
        context_params={"use_gpu": _USE_GPU_BY_BACKEND[backend]},
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
