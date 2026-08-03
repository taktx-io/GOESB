"""`faster-whisper` batch runtime adapter (docs/02-architecture.md §4).

Optional dependency (`pip install goesb-runner[faster-whisper]`) — the actual
`faster_whisper` package is only imported inside `run_batch`, so importing
`oesb_runner.adapters` never requires it, matching the normalization plugin
pattern.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import cuda_runtime
from ..pack import Utterance
from ..streaming import StreamTrace, run_windowed_local_agreement_streaming
from . import ConcurrentCall, Transcription, log_progress, register


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

# Real report: a fresh Ubuntu box with an NVIDIA driver but the wrong (or no)
# system CUDA Toolkit crashed here instead of raising a catchable ValueError
# -- ctranslate2's dlopen("libcublas.so.12") failure surfaces as whatever
# exception type (or, in the worst case, native abort) its own C++ layer
# happens to produce, not reliably a ValueError with "CUDA" in the message
# the way the missing-CUDA-*support* case below does. Matched by substring
# across exception types instead of a single hardcoded phrase, so any of the
# actual failure shapes ("CUDA", "cuBLAS", "cuDNN") gets the same clear
# message instead of a bare third-party stack trace.
_CUDA_ERROR_MARKERS = ("cuda", "cublas", "cudnn")


def _looks_like_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CUDA_ERROR_MARKERS)


def _load_model(
    model_name: str, backend: str, quantization: str, threads: int, download_root, num_workers: int = 1
):
    """Shared `WhisperModel(...)` construction for run_batch, run_streaming,
    and run_concurrency. `--backend cuda` on a CTranslate2 build without CUDA
    support raises a raw exception deep inside ctranslate2 — caught and
    re-raised as a clear, actionable RuntimeError (ADR-0008: fails
    immediately, before any model weights load, never a silent CPU
    fallback) rather than surfacing a bare third-party stack trace as the
    only explanation.

    `cuda_runtime.preload_installed_cublas()` runs first so that, on Linux,
    a pip-installed `nvidia-cublas-cu12` wheel is already resident under its
    matching SONAME before ctranslate2 ever tries its own dlopen — turning
    what used to be a hard crash on a fresh Ubuntu box (driver present,
    system CUDA Toolkit missing or mismatched) into a working `--backend
    cuda` run with no extra step from the user. A no-op everywhere else
    (nothing pip-installed, or a platform this doesn't cover).

    `num_workers` (default 1, matching faster-whisper's own default and
    today's batch/streaming behavior byte-for-byte) maps straight to
    ctranslate2's `Translator(inter_threads=N)` — the actual concurrency
    slot count. run_concurrency is the only caller that ever passes
    something other than 1."""
    if backend == "cuda":
        cuda_runtime.preload_installed_cublas()

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
            num_workers=num_workers,
            download_root=str(download_root) if download_root is not None else None,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        if backend == "cuda" and _looks_like_cuda_runtime_error(exc):
            raise RuntimeError(
                f"--backend cuda failed: {exc}. Run `goesb doctor` to check what's "
                "missing (on Ubuntu, `pip install \"goesb-runner[cuda]\"` often fixes "
                "a missing cuBLAS), or use --backend cpu."
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
    """Feed each utterance to faster-whisper in `chunk_ms` chunks via
    `streaming.run_windowed_local_agreement_streaming` (faster-whisper has
    no incremental decoder state to resume, so "streaming" here still
    means repeated re-transcription — the same "local agreement" pattern
    used by e.g. whisper_streaming — but of a bounded window, not the
    whole clip). See that shared function's own docstring for the
    real-audio-validated windowing/commit/trim/dedup design (3 real
    correctness bugs found and fixed across 4 rounds against this exact
    engine before it landed) — this adapter only supplies the
    engine-specific decode call.

    Real report: an earlier version of this function re-decoded
    `samples[:end]` — the ENTIRE clip so far, unbounded, every chunk. That
    measured RTF 3.19x on an Apple M1 Pro (whisper-medium, default
    settings) — slower than realtime, and its naive latency math
    understated real latency once behind realtime (see the shared
    function's docstring for why). The bounded-window fix measured RTF
    ~2.5x (a real, reproducible improvement) and WER 0.110 vs the
    original's 0.078 — an honest, expected cost of bounded context, not
    corruption. Still not realtime-capable on this CPU-only hardware — an
    inherent Whisper-architecture cost (fixed ~30s encoder window
    regardless of real audio length), not something this adapter can fix
    — see `whisper-medium-en-streaming`'s own profile comment and
    `cli.py`'s `_MATRIX_STREAMING_EXCLUDED_PROFILE_IDS` for why it stays
    excluded from the wizard.
    """
    try:
        from faster_whisper.audio import decode_audio
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "faster-whisper is not installed; run "
            "`pip install goesb-runner[faster-whisper]`"
        ) from exc

    sample_rate = 16000
    model = _load_model(model_name, backend, quantization, threads, download_root)

    def decode_window(samples_slice) -> list[dict]:
        segments, _info = model.transcribe(
            samples_slice,
            beam_size=beam_size,
            temperature=temperature,
            vad_filter=vad,
            language=language,
            word_timestamps=True,
        )
        return [
            {"word": w.word.strip(), "start": w.start, "end": w.end}
            for segment in segments for w in (segment.words or [])
        ]

    traces: list[StreamTrace] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_audio(str(utterance.audio_path), sampling_rate=sample_rate)
        trace = run_windowed_local_agreement_streaming(
            utterance, samples, sample_rate=sample_rate, chunk_ms=chunk_ms, decode_window=decode_window,
        )
        log_progress(i, len(utterances), utterance.utterance_id, trace.processing_time_s)
        traces.append(trace)
    return traces


@register(
    "faster-whisper", benchmark_type="concurrency",
    applied_parameters=frozenset({"quantization", "beam_size", "temperature", "vad", "threads", "concurrency"}),
    backends=frozenset(_DEVICE_BY_BACKEND),
)
def run_concurrency(
    model_name: str,
    utterances: list[Utterance],
    *,
    concurrency: int = 1,
    duration_s: int = 30,
    quantization: str = "int8",
    beam_size: int = 5,
    temperature: float = 0.0,
    vad: bool = True,
    threads: int = 4,
    download_root: str | Path | None = None,
    language: str | None = None,
    backend: str = "cpu",
) -> list[ConcurrentCall]:
    """Does this hardware stay fast under N simultaneous requests, not just
    one at a time? `concurrency` worker threads repeatedly call
    `model.transcribe()` on one shared, already-loaded model instance for a
    fixed `duration_s` wall-clock window, each call timed independently and
    pooled into one list the caller feeds through `stats.summarize()` for a
    p50/p95 RTF distribution — the actual answer to "does an individual
    request stay fast under load," not just a corpus-aggregate mean.

    No WER/CER: this benchmark type doesn't score accuracy, only
    performance under load, so `reference_text` is never touched.

    `num_workers=concurrency` is passed straight through to `_load_model` —
    it must match the `ThreadPoolExecutor`'s own worker count exactly.
    Fewer workers than `concurrency` would leave ctranslate2's internal
    `inter_threads` pool partially idle; more would trigger its
    `max_queued_batches` backpressure (blocking extra threads) and
    understate what this hardware can actually sustain concurrently.

    Fixed wall-clock duration, not a fixed utterance count per worker:
    comparing concurrency=1 against concurrency=16 needs the same total
    measurement window at every level, or throughput numbers across levels
    aren't comparable — the same reason load-testing tools (wrk, k6) use
    fixed-duration windows rather than fixed-request counts. Each worker
    round-robins through `utterances` at its own offset (`worker_id`,
    `worker_id + concurrency`, ...) so workers never decode the exact same
    utterance in lockstep.

    No warm-up discard: model load (the genuine cold-start cost) already
    happens before the timed window starts, matching the existing
    "model load time deliberately excluded" convention batch/streaming
    already use. Per-call warm-up inside ctranslate2 is a much smaller
    effect a multi-second window with many calls per worker should
    average out — a candidate refinement only if real measurement shows
    first-call skew, not something to build ahead of data."""
    model = _load_model(
        model_name, backend, quantization, threads, download_root, num_workers=concurrency
    )

    deadline = time.perf_counter() + duration_s

    def _worker(worker_id: int) -> list[ConcurrentCall]:
        calls: list[ConcurrentCall] = []
        i = worker_id
        while time.perf_counter() < deadline:
            utterance = utterances[i % len(utterances)]
            start = time.perf_counter()
            segments, _info = model.transcribe(
                str(utterance.audio_path),
                beam_size=beam_size,
                temperature=temperature,
                vad_filter=vad,
                language=language,
            )
            list(segments)  # force the full decode before stopping the clock
            elapsed = time.perf_counter() - start
            calls.append(ConcurrentCall(processing_time_s=elapsed, audio_duration_s=utterance.duration_s))
            i += concurrency
        return calls

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, w) for w in range(concurrency)]
        calls = [c for future in futures for c in future.result()]
    return calls
