"""`whisper.cpp` (via `pywhispercpp`) batch runtime adapter
(docs/02-architecture.md §4).

Optional dependency (`pip install goesb-runner[whisper-cpp]`) —
`pywhispercpp`/`soundfile` are only imported inside `run_batch`, matching the
lazy-import pattern used by every other adapter.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..audio import decode_pcm
from ..pack import Utterance
from ..streaming import StreamTrace, run_windowed_local_agreement_streaming
from . import ConcurrentCall, Transcription, log_progress, register

# whisper.cpp's own build-info string (whisper_print_system_info(), bound
# as Model.system_info() — a static method, no model file needed) is the
# only reliable way to know whether this exact compiled pywhispercpp
# binary actually has a given GPU backend. Its format is NOT uniform —
# confirmed by reading upstream ggml/whisper.cpp source directly
# (whisper_print_system_info in src/whisper.cpp):
#   - COREML and OPENVINO are unconditionally-appended flat pairs:
#     "COREML = 0 | OPENVINO = 0 | " — a plain "KEY = 1" check is correct
#     for these two, and only these two.
#   - CUDA, Metal, and Vulkan are NOT flags — they're ggml's dynamic
#     backend registry entries, each appearing as its own named section:
#     "<registered-name> : <feature>=<value> | ...". The registered name
#     is a compile-time constant per backend (ggml-cuda.cu: GGML_CUDA_NAME
#     is "CUDA" for a real NVIDIA build, "ROCm"/"MUSA" for AMD/Moore
#     Threads builds sharing the same code path; ggml-metal.cpp:
#     GGML_METAL_NAME is "MTL"; ggml-vulkan.cpp: GGML_VK_NAME is
#     "Vulkan") — the section header appears whether or not that backend
#     reports any features at all, so "<NAME> :" presence is the correct,
#     robust check; a "<NAME> = 1" check (what an earlier version of this
#     module used for CUDA) matches no real backend's actual output and
#     would incorrectly report every GPU backend as unavailable, even on
#     a build that genuinely has it.
_GGML_BACKEND_SECTION_RE = {
    "cuda": re.compile(r"\bCUDA\s*:", re.IGNORECASE),
    "metal": re.compile(r"\bMTL\s*:", re.IGNORECASE),
}

# whisper.cpp's models are all trained on and require 16kHz mono input —
# unlike faster-whisper (whose ctranslate2 decode path resamples
# internally), whisper.cpp does not resample, so decode_pcm must be told
# to convert to this rate explicitly (real report: a 48kHz Common Voice
# pack fed in unresampled produced fluent-sounding but 100%+ WER
# hallucinated garbage — audio playing back 3x too fast, not a
# model-quality problem).
_SAMPLE_RATE = 16000


def cuda_available(model_cls: Any) -> bool:
    """True iff `model_cls` (pywhispercpp's `Model`) reports real CUDA
    support in its own compiled-in system_info string. Exposed at module
    level (not adapter-private) so `cli._doctor_engine_line` can give the
    same real answer `goesb doctor` currently can't for this engine,
    instead of "can't be checked without running a real transcription"."""
    return _GGML_BACKEND_SECTION_RE["cuda"].search(model_cls.system_info() or "") is not None


def metal_available(model_cls: Any) -> bool:
    """Same idea as cuda_available, for Apple's Metal backend ("MTL" in
    ggml's own backend registry — confirmed by direct inspection: this
    exact token is what a real Metal-capable build reports, e.g.
    "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | ...")."""
    return _GGML_BACKEND_SECTION_RE["metal"].search(model_cls.system_info() or "") is not None


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
# not vendor-specific the way faster-whisper's `device=` is — it means
# "whichever single GPU backend this binary was compiled with" (a build
# has at most one of CUDA/Metal/Vulkan compiled in, never several), so on
# its own, requesting "cuda" or "metal" would just mean "ask whisper.cpp
# to use its compiled-in GPU backend, whatever that actually is" — a
# no-op rather than a hard error if it's the wrong one, or none at all.
# `run_batch` below checks the matching *_available() function before
# ever setting this, so that gap doesn't reach here anymore — each
# backend value is verified against what this exact build actually has,
# not just "some GPU backend or other."
_USE_GPU_BY_BACKEND = {"cpu": False, "cuda": True, "metal": True}

# Which check function verifies a given --backend value is real on this
# build, before context_params ever gets set from _USE_GPU_BY_BACKEND
# above. cpu has no check — it's always available.
_BACKEND_AVAILABILITY_CHECK = {"cuda": cuda_available, "metal": metal_available}


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

    # One system_info() call, not one per candidate backend — it has a real
    # side effect (observed: initializes the Metal backend on a Metal
    # build, logging several lines) that shouldn't repeat for no reason.
    if backend in _GGML_BACKEND_SECTION_RE:
        info = Model.system_info()
        if not _GGML_BACKEND_SECTION_RE[backend].search(info or ""):
            raise RuntimeError(
                f"--backend {backend} failed: this pywhispercpp build has no {backend} "
                f"support (system_info: {info!r}). Run `goesb doctor` to check what's "
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
        no_context=True,  # ADR-0008: explicit, not left to pywhispercpp's own
        # default. This model instance is loaded once and reused across
        # every utterance in the pack (see the module docstring's "model
        # load time deliberately excluded" convention) — without this,
        # whisper.cpp's own C API defaults to *reusing* the previous
        # transcription as decoder context, which the standalone whisper.cpp
        # CLI never trips over (one file per process) but any harness
        # looping over a corpus on a long-lived model does. pywhispercpp's
        # own Python-level default already happens to be `True` (confirmed
        # empirically: 60 real transcriptions in a row on one instance,
        # byte-identical output before/after) — this makes that safety
        # property an explicit, reviewed profile declaration instead of an
        # implicit library default a reader can't see (see `context_reset`
        # in this engine's own profiles).
        **language_params,
    )

    results: list[Transcription] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, _SAMPLE_RATE, dtype="float32")
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


@register(
    "whisper-cpp", benchmark_type="streaming",
    applied_parameters=frozenset({"threads", "temperature", "chunk_ms"}),
    backends=frozenset(_USE_GPU_BY_BACKEND),
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
    """Feed each utterance to whisper.cpp in `chunk_ms` chunks via
    `streaming.run_windowed_local_agreement_streaming` — same shared,
    real-audio-validated bounded-window design `faster_whisper.run_streaming`
    uses (see that shared function's own docstring), since whisper.cpp,
    like faster-whisper, has no incremental decoder state to resume
    through `pywhispercpp`'s API and must re-decode to produce a
    streaming trace at all. This adapter only supplies the
    engine-specific decode call and word extraction.

    Word-level timestamps come from `token_timestamps=True,
    split_on_word=True, max_len=1` on the `Model` — confirmed by direct
    inspection of pywhispercpp's actual `Segment` output (not assumed
    from docs) that this is what turns whisper.cpp's normal
    sentence/phrase-level segments into one `Segment` per word, each
    with its own `t0`/`t1`. Those timestamps are in whisper.cpp's own
    native unit — centiseconds (hundredths of a second), NOT the same
    unit faster-whisper's `Word.start`/`.end` use (seconds) — confirmed
    by direct measurement (a word's `t1` came back as a small integer
    like 53, not 0.53), so this divides by 100 before handing them to
    the shared windowing function, which expects seconds throughout.
    whisper.cpp also occasionally emits an empty leading `Segment`
    (`t0=0, t1=0, text=''`, confirmed on real audio) — filtered out here
    since it isn't a real word.

    `quantization`/`beam_size`/`vad` are accepted for call-shape parity
    with `run_batch` (see its own docstring for why they're unused) —
    unlike faster-whisper's streaming adapter, `vad` genuinely has
    nothing to attach to here: whisper.cpp's own VAD is a separate,
    optional pass this adapter doesn't wire up.

    Unlike faster-whisper (CPU/CUDA only), `--backend metal` is real
    here on Apple Silicon (`_USE_GPU_BY_BACKEND`) — confirmed live on an
    Apple M1 Pro: Metal actually initializes (`ggml_metal_device_init`)
    and this is the one Whisper-family streaming path with real GPU
    acceleration on that hardware, unlike faster-whisper's CPU-bound
    ctranslate2 backend (no Metal support at all — see
    `faster_whisper.run_streaming`'s own docstring).
    """
    try:
        from pywhispercpp.model import Model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "pywhispercpp is not installed; run "
            "`pip install goesb-runner[whisper-cpp]`"
        ) from exc

    if backend in _GGML_BACKEND_SECTION_RE:
        info = Model.system_info()
        if not _GGML_BACKEND_SECTION_RE[backend].search(info or ""):
            raise RuntimeError(
                f"--backend {backend} failed: this pywhispercpp build has no {backend} "
                f"support (system_info: {info!r}). Run `goesb doctor` to check what's "
                "available, or use --backend cpu."
            )

    resolved_model_id = _resolve_model_id(model_name)
    fixed_language: str | None
    if language:
        fixed_language = language
    elif resolved_model_id.endswith(".en"):
        fixed_language = "en"
    else:
        fixed_language = None  # resolved per utterance below, via auto_detect_language

    # `language=` set explicitly here (never `detect_language=True`) —
    # real bug, confirmed by direct measurement: `detect_language=True`
    # combined with `token_timestamps`/`split_on_word`/`max_len` (needed
    # for word-level timestamps below) silently returned ZERO segments,
    # not a wrong-language guess. `Model.auto_detect_language()` (a
    # dedicated method, not this same code path) doesn't have that
    # problem, so `fixed_language is None` gets resolved once per
    # utterance from that instead, below.
    model = Model(
        resolved_model_id,
        models_dir=str(download_root) if download_root is not None else None,
        n_threads=threads,
        temperature=temperature,
        print_realtime=False,
        print_progress=False,
        token_timestamps=True,
        split_on_word=True,
        max_len=1,
        context_params={"use_gpu": _USE_GPU_BY_BACKEND[backend]},
        no_context=True,  # ADR-0008: explicit, see run_batch's own comment
        # for the general reasoning. Also correct specifically for the
        # windowed re-decode loop below: each `decode_window()` call
        # (whether the same utterance's next window, or a fresh
        # utterance) is meant to be an independent decode of exactly the
        # raw audio slice it's given — `run_windowed_local_agreement_
        # streaming`'s own commit logic handles continuity via text
        # overlap-matching, not decoder-state carry-over, so this doesn't
        # fight the streaming algorithm.
        language=fixed_language or "en",  # placeholder when None; overridden per-call below
    )

    traces: list[StreamTrace] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, _SAMPLE_RATE, dtype="float32")

        # Resolved ONCE per utterance from its own full audio, not
        # re-guessed every chunk — a streaming session re-detecting
        # language from a sub-second first window every chunk would be
        # unreliable even without the bug above, and a real deployment
        # wouldn't flip-flop language mid-conversation either.
        if fixed_language is not None:
            utterance_language = fixed_language
        else:
            (detected, _prob), _all_probs = model.auto_detect_language(samples)
            utterance_language = detected

        def decode_window(samples_slice, _lang=utterance_language) -> list[dict]:
            segments = model.transcribe(samples_slice, language=_lang)
            return [
                {"word": s.text.strip(), "start": s.t0 / 100.0, "end": s.t1 / 100.0}
                for s in segments if s.text.strip()
            ]

        trace = run_windowed_local_agreement_streaming(
            utterance, samples, sample_rate=_SAMPLE_RATE, chunk_ms=chunk_ms, decode_window=decode_window,
        )
        log_progress(i, len(utterances), utterance.utterance_id, trace.processing_time_s)
        traces.append(trace)
    return traces


@register(
    "whisper-cpp", benchmark_type="concurrency",
    applied_parameters=frozenset({"threads", "temperature", "concurrency"}),
    backends=frozenset(_USE_GPU_BY_BACKEND),
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
    """Does this hardware stay fast under N simultaneous requests? Same
    fixed-`duration_s`-window, round-robin-through-utterances harness
    `faster_whisper.run_concurrency` established (ADR-0012) — but a
    genuinely different model-loading strategy underneath.

    pywhispercpp is NOT safe to share one `Model` instance across
    concurrent threads the way ctranslate2 is (confirmed by reading the
    actual bound C API, not assumed): `Model.transcribe()` mutates shared
    instance state (`self._params`, its callback bindings) on every call,
    and pywhispercpp never binds `whisper_init_state`/
    `whisper_full_with_state` — the per-thread-state split whisper.cpp's
    own C API actually has. So this builds `concurrency` full, independent
    `Model` instances up front, one per worker, instead of one shared
    model with a worker-count knob. That's a real N-way memory cost (full
    ggml weights loaded N times) — see ADR-0012's addendum and this
    engine's tighter `overridable.concurrency.range.max` in its
    `*-concurrency` profiles.

    Model construction happens before the timed window starts, same as
    every other adapter's "model load time excluded from RTF" convention
    — N loads here is just N times that same already-excluded cost, not a
    new category of overhead this benchmark type needs to account for.

    Audio is decoded once per utterance up front too (whisper.cpp needs
    raw PCM samples, not a path, unlike faster-whisper) and shared
    read-only across every worker — only the actual `transcribe()` call is
    timed.

    No WER/CER: this benchmark type doesn't score accuracy, so
    `reference_text` is never touched."""
    try:
        from pywhispercpp.model import Model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "pywhispercpp is not installed; run "
            "`pip install goesb-runner[whisper-cpp]`"
        ) from exc

    if backend in _GGML_BACKEND_SECTION_RE:
        info = Model.system_info()
        if not _GGML_BACKEND_SECTION_RE[backend].search(info or ""):
            raise RuntimeError(
                f"--backend {backend} failed: this pywhispercpp build has no {backend} "
                f"support (system_info: {info!r}). Run `goesb doctor` to check what's "
                "available, or use --backend cpu."
            )

    resolved_model_id = _resolve_model_id(model_name)
    if language:
        language_params = {"language": language}
    elif resolved_model_id.endswith(".en"):
        language_params = {"language": "en"}
    else:
        language_params = {"detect_language": True}

    decoded = [
        (utterance, decode_pcm(utterance.audio_path, _SAMPLE_RATE, dtype="float32"))
        for utterance in utterances
    ]

    models = [
        Model(
            resolved_model_id,
            models_dir=str(download_root) if download_root is not None else None,
            n_threads=threads,
            temperature=temperature,
            print_realtime=False,
            print_progress=False,
            context_params={"use_gpu": _USE_GPU_BY_BACKEND[backend]},
            no_context=True,  # ADR-0008: explicit, see run_batch's own
            # comment. This benchmark type never scores accuracy, but each
            # worker's model instance is still reused across many calls
            # (round-robining through the pack) — same explicit-not-implicit
            # reasoning applies regardless.
            **language_params,
        )
        for _ in range(concurrency)
    ]

    deadline = time.perf_counter() + duration_s

    def _worker(worker_id: int) -> list[ConcurrentCall]:
        model = models[worker_id]
        calls: list[ConcurrentCall] = []
        i = worker_id
        while time.perf_counter() < deadline:
            utterance, samples = decoded[i % len(decoded)]
            start = time.perf_counter()
            model.transcribe(samples)
            elapsed = time.perf_counter() - start
            calls.append(ConcurrentCall(processing_time_s=elapsed, audio_duration_s=utterance.duration_s))
            i += concurrency
        return calls

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, w) for w in range(concurrency)]
        calls = [c for future in futures for c in future.result()]
    return calls
