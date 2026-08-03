"""NVIDIA Parakeet-TDT batch runtime adapter (docs/02-architecture.md §4).

Optional dependency (`pip install goesb-runner[parakeet]`) — `transformers`/
`torch` are only imported inside `_load`, matching the lazy-import pattern
every other adapter uses. Uses `transformers`' native
`AutoModelForTDT`/`AutoProcessor` support (confirmed directly against the
HF docs), not `nemo_toolkit` — the NeMo toolkit itself is Apache-2.0 and
the Parakeet-TDT-0.6b-v3 checkpoint is CC-BY-4.0 (commercial use allowed,
attribution required), but `nemo_toolkit['all']` pulls in a much heavier
stack (pytorch-lightning, hydra-core, onnx, ...) this adapter never needs.

`librosa` is a real, required transitive dependency too — confirmed by
direct measurement: `ParakeetFeatureExtractor.from_pretrained` raises
`ImportError` without it (transformers' own `requires_backends` gate),
even though nothing in this adapter calls librosa directly.

Chosen for its multilingual coverage: parakeet-tdt-0.6b-v3 is trained on
NVIDIA's Granary dataset (25 European languages, Dutch included) — unlike
vosk/faster-whisper/whisper-cpp, one checkpoint covers every language this
adapter's profiles declare, so there's no per-language model swap the way
`_resolve_model_id` does for the Whisper family.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..audio import decode_pcm
from ..pack import Utterance
from ..streaming import StreamTrace, run_windowed_local_agreement_streaming
from . import Transcription, log_progress, register

# The Whisper-family adapters resample internally (faster-whisper) or
# require exactly 16kHz (whisper.cpp, vosk) — Parakeet's own feature
# extractor is read at load time instead of hardcoded here, since transformers
# doesn't guarantee every checkpoint on this architecture shares one fixed
# rate the way whisper.cpp's ggml models do.

# ADR-0008: `backend` is always passed as an explicit device string — never
# left to a library's own auto-detection (transformers' own `device_map=
# "auto"` convenience is deliberately not used here for exactly that
# reason). torch's CUDA wheels bundle their own CUDA runtime (unlike
# ctranslate2's separate cuBLAS dlopen dependency — see
# faster_whisper._looks_like_cuda_runtime_error), so no equivalent
# preload/retry dance is needed here: `torch.cuda.is_available()` is a
# reliable, direct answer. "metal" -> torch's own "mps" device string
# (Apple Silicon GPU): unlike whisper.cpp's ggml backend, this isn't a
# separate compile-time flag some builds simply lack — plain `pip install
# torch` already ships MPS support, `torch.backends.mps.is_available()`
# alone is the real, sufficient readiness check. Confirmed genuinely
# faster on a real Apple M1 Pro (full 15-clip fleurs-nl batch pack: RTF
# 0.0395x on metal vs 0.155x on cpu, same 6.2% WER on both — device
# choice didn't change the output) — but only after the JIT/
# first-kernel-compile cost is paid once (measured ~3.4s), which is why
# `_warm_up` exists below.
_DEVICE_BY_BACKEND = {"cpu": "cpu", "cuda": "cuda", "metal": "mps"}

_BACKEND_AVAILABLE_CHECK = {
    "cuda": lambda torch_mod: torch_mod.cuda.is_available(),
    "metal": lambda torch_mod: torch_mod.backends.mps.is_available(),
}


def _resolve_model_id(model_name: str) -> str:
    """Translate GOESB's runtime-agnostic model name ('parakeet-tdt-0.6b-v3')
    into the HF Hub id transformers' `AutoModelForTDT`/`AutoProcessor`
    expect ('nvidia/parakeet-tdt-0.6b-v3') — same translation role as
    faster_whisper._resolve_model_id / whisper_cpp._resolve_model_id."""
    return f"nvidia/{model_name}"


def _warm_up(processor, model, device: str) -> None:
    """One throwaway `generate()` call, on real (silent) audio shaped like
    any other utterance, before the timed loop starts. Real, measured
    cost on Apple Silicon MPS: ~3.4s for the first call vs ~0.5-0.9s
    steady-state on real audio right after — a JIT/kernel-compile cost,
    not decode work — confirmed by direct measurement (5 real Dutch
    FLEURS clips run back to back after this warm-up: RTF 0.05-0.09x
    throughout, no first-clip spike). Letting that land inside
    utterance #1's own `processing_time_s` would silently overstate RTF
    for that one utterance and skew the whole run's aggregate — the same
    "one-off cost, not part of what RTF measures" category model load
    already gets excluded from, just extended to cover the first real
    accelerator kernel launch too. Run unconditionally (not just for
    cuda/metal): no measured warm-up effect on cpu, but a fixed, cheap,
    single code path here is simpler than special-casing by backend."""
    import numpy as np
    import torch

    sample_rate = processor.feature_extractor.sampling_rate
    silence = np.zeros(int(0.5 * sample_rate), dtype="float32")
    inputs = processor(silence, sampling_rate=sample_rate, return_tensors="pt")
    inputs.to(device, dtype=model.dtype)
    with torch.no_grad():
        model.generate(**inputs, return_dict_in_generate=True)


def _load(model_name: str, backend: str, threads: int, download_root: str | Path | None):
    try:
        import torch
        from transformers import AutoModelForTDT, AutoProcessor
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "transformers/torch are not installed; run "
            "`pip install goesb-runner[parakeet]`"
        ) from exc

    check = _BACKEND_AVAILABLE_CHECK.get(backend)
    if check is not None and not check(torch):
        reason = (
            "torch.cuda.is_available() is False (no NVIDIA GPU visible, or a "
            "non-CUDA torch build is installed)"
            if backend == "cuda"
            else "torch.backends.mps.is_available() is False (not Apple Silicon, "
            "or this torch build has no MPS support)"
        )
        raise RuntimeError(
            f"--backend {backend} failed: {reason}. Run `goesb doctor` to check "
            "what's available, or use --backend cpu."
        )
    if backend == "cpu":
        torch.set_num_threads(threads)

    model_id = _resolve_model_id(model_name)
    cache_dir = str(download_root) if download_root is not None else None
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForTDT.from_pretrained(model_id, cache_dir=cache_dir)
    device = _DEVICE_BY_BACKEND[backend]
    model.to(device)
    model.eval()
    _warm_up(processor, model, device)
    return processor, model


@register(
    "parakeet", benchmark_type="batch",
    applied_parameters=frozenset({"threads"}),
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

    `quantization`/`beam_size`/`temperature`/`vad`/`language` are accepted
    for call-shape parity with the other batch adapters
    (docs/03-roadmap.md M2 exit criterion) but unused here: Parakeet-TDT's
    greedy transducer decode has no beam/temperature knob to set, this
    adapter doesn't run a separate VAD pass, and — unlike whisper.cpp,
    which hard-defaults to English and must be told otherwise — the
    multilingual v3 checkpoint auto-detects the spoken language itself, so
    there is nothing genuinely conditioned on the profile's declared
    `language` today. `quantization` here would mean the model's own
    torch dtype (float32/float16/bf16), a real future knob, not yet wired.

    Model load time is deliberately excluded from per-utterance timing,
    matching every other batch adapter's convention — as is the
    accelerator warm-up cost `_load`'s own `_warm_up` call absorbs before
    returning here (real, measured ~3.4s one-off on Apple Silicon MPS).

    `backend` (ADR-0008) is validated (`torch.cuda.is_available()` /
    `torch.backends.mps.is_available()`, not trusted from a profile/CLI
    flag alone) and passed straight through as the device to load the
    model onto.
    """
    processor, model = _load(model_name, backend, threads, download_root)

    import torch

    device = _DEVICE_BY_BACKEND[backend]
    sample_rate = processor.feature_extractor.sampling_rate

    results: list[Transcription] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, sample_rate, dtype="float32")
        start = time.perf_counter()
        inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt")
        inputs.to(device, dtype=model.dtype)
        with torch.no_grad():
            output = model.generate(**inputs, return_dict_in_generate=True)
        hypothesis_text = processor.decode(output.sequences, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - start
        log_progress(i, len(utterances), utterance.utterance_id, elapsed)
        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results


def _merge_tokens_to_words(tokens: list[dict]) -> list[dict]:
    """Parakeet-TDT's own timestamped output (`processor.decode(...,
    durations=...)`) is per-*token* (BPE-style subwords: 'E', 'erst', '
    mo', 'est', 'en', ...), not per-word — `run_windowed_local_agreement_streaming`
    needs word-level `{"word", "start", "end"}` entries, so this merges
    consecutive tokens belonging to the same word first.

    Word-boundary rule confirmed directly against real Dutch audio (not
    assumed from the English docs example): a token whose text starts with
    a literal space marks the start of a new word; one without continues
    the previous word. Punctuation tokens (e.g. ',') also carry no leading
    space and so attach straight onto the preceding word — matching
    `ParakeetProcessor`'s own documented "punctuation is attached to the
    preceding token" behavior for the `tdt` decoder type. A leftover
    trailing comma on a word is harmless here: the shared streaming
    function's own word-matching (`_normalize_word_for_overlap_match`) is
    already punctuation-insensitive."""
    words: list[dict] = []
    for tok in tokens:
        text = tok["token"]
        if text.startswith(" ") or not words:
            words.append({"word": text.lstrip(" "), "start": tok["start"], "end": tok["end"]})
        else:
            words[-1]["word"] += text
            words[-1]["end"] = tok["end"]
    return [w for w in words if w["word"]]


@register(
    "parakeet", benchmark_type="streaming",
    applied_parameters=frozenset({"threads", "chunk_ms"}),
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
    """Feed each utterance to Parakeet-TDT in `chunk_ms` chunks via
    `streaming.run_windowed_local_agreement_streaming` — the same shared,
    real-audio-validated bounded-window design `faster_whisper.run_streaming`
    and `whisper_cpp.run_streaming` use (see that shared function's own
    docstring for the windowing/commit/trim/dedup design).

    Not a hand-off to any genuine incremental/cache-aware streaming path:
    transformers' Parakeet support exposes a decoder-side KV cache
    (`decoder_cache`/`use_decoder_cache`) for the TDT decoder's own
    autoregressive token loop, but there is no documented way to feed the
    Fast Conformer *encoder* new audio incrementally through the
    high-level `generate()` API this adapter uses — NeMo's real
    cache-aware-streaming Parakeet needs its own purpose-trained streaming
    checkpoints and NeMo-side buffer/context-size plumbing this
    `transformers` port doesn't carry. Bounded re-decode is the same
    honest fallback the other two Whisper-family engines already use here
    — and, unlike them, it actually lands inside realtime, on both
    backends measured: real Dutch FLEURS audio, full 15-clip fleurs-nl
    pack, chunk_ms=1000 — `--backend cpu` RTF 0.876x, `--backend metal`
    (real Apple Silicon MPS) RTF 0.315x, ~2.8x faster — vs faster-
    whisper's non-realtime 2.5-3.19x on the same class of hardware,
    because Parakeet's batch RTF is so much faster to begin with (~0.15x
    cpu / ~0.04x metal) that repeated bounded-window re-decode still has
    real headroom left. WER: 6.57%, byte-identical between cpu and metal
    (same weights, same greedy decode, same windowing — device choice
    doesn't change the output here) — with a couple of spurious extra
    words at window-boundary cut points (e.g. "instemmen" -> "instemmen
    Stemmen") — a genuinely different failure mode than faster-whisper's
    original same-word duplication bugs (those were the same committed
    word reappearing, catchable by dedup; this is the model producing a
    new, wrong word once its context is truncated mid-word at the window
    edge, which no post-hoc word-level dedup can catch since it never
    matches anything already committed). Consistent with the established
    "honest cost of bounded context, not corruption" framing, not a bug
    in this adapter's own merge logic.

    `quantization`/`beam_size`/`temperature`/`vad`/`language` unused — see
    `run_batch`'s own docstring for why.
    """
    processor, model = _load(model_name, backend, threads, download_root)

    import torch

    device = _DEVICE_BY_BACKEND[backend]
    sample_rate = processor.feature_extractor.sampling_rate

    def decode_window(samples_slice) -> list[dict]:
        inputs = processor(samples_slice, sampling_rate=sample_rate, return_tensors="pt")
        inputs.to(device, dtype=model.dtype)
        with torch.no_grad():
            output = model.generate(**inputs, return_dict_in_generate=True)
        _text, token_timestamps = processor.decode(
            output.sequences, durations=output.durations, skip_special_tokens=True,
        )
        return _merge_tokens_to_words(token_timestamps[0])

    traces: list[StreamTrace] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, sample_rate, dtype="float32")
        trace = run_windowed_local_agreement_streaming(
            utterance, samples, sample_rate=sample_rate, chunk_ms=chunk_ms, decode_window=decode_window,
        )
        log_progress(i, len(utterances), utterance.utterance_id, trace.processing_time_s)
        traces.append(trace)
    return traces
