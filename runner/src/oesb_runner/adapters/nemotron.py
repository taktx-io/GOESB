"""NVIDIA Nemotron 3.5 ASR streaming runtime adapter (docs/02-architecture.md
§4, ADR-0013) — the project's first genuinely cache-aware streaming engine
that isn't Kaldi.

Optional dependency (`pip install goesb-runner[nemotron]`) — `transformers`/
`torch` are only imported inside `_load`, matching the lazy-import pattern
every other adapter uses. Uses `transformers`' native `AutoModelForRNNT`/
`AutoProcessor` support, not `nemo_toolkit`: same reasoning as
`parakeet.py`'s own module docstring (`nemo_toolkit['all']` drags in
pytorch-lightning/hydra-core/onnx this adapter never touches). `librosa` is
a real transitive dependency of the feature extractor for the same
`requires_backends` reason Parakeet's is.

Note the auto class is `AutoModelForRNNT`, NOT Parakeet's `AutoModelForTDT`:
`nemotron3_5_asr` registers into `MODEL_FOR_RNNT_MAPPING_NAMES`, a different
mapping from the TDT one (`transformers/models/auto/modeling_auto.py`).

One checkpoint (`nvidia/nemotron-3.5-asr-streaming-0.6b`) covers every
language GOESB has packs for — en/nl/de/fr/es/pt are all in the model card's
top "transcription-ready" tier — so there is no per-language model swap the
way `_resolve_model_id` does for the Whisper family and vosk. Unlike
Parakeet, though, `language` is NOT ignored here: this processor turns it
into a real `prompt_ids` model input that conditions the decode (see
`_resolve_language` below for the fallback chain, needed because GOESB's
`es-419` is not a key this checkpoint carries).

Measured, not assumed, on BOTH declared backends — Apple M-series MPS
(torch 2.13 / transformers 5.14.1) and a real NVIDIA RTX A6000 (torch
2.13+cu126 / transformers 5.15.1, driver 555.58) — against
`packs/fleurs-nl`, 15 clips / 119.5s. See the ADR-0013 addendum for the
full tables:

- `processor.supported_streaming_latencies_ms` for this checkpoint is FOUR
  modes, `{0: 80, 3: 320, 6: 560, 13: 1120}` — not the five the model card
  lists. There is no 160 ms / right-context-1 mode on this checkpoint.
  Identical on transformers 5.14.1 and 5.15.1.
- Both backends work and produce the same text. Batch RTF 0.035x on cuda /
  0.072x on metal, WER 0.1314 on both (and on cpu, which this adapter
  refuses to run — see `_DEVICE_BY_BACKEND`).
- ~2.56 GB of fp32 weights; ~2.67 GB of device memory per loaded, warmed
  instance measured on cuda (`torch.cuda.mem_get_info` delta).
- The batch and streaming paths produce *identical* text at the 320 ms mode,
  not merely similar — this checkpoint's offline path is the same
  cache-aware limited-right-context decode run in one shot. See
  `run_batch`'s docstring; it changes what the batch profiles are for.
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..audio import decode_pcm
from ..pack import Utterance
from ..streaming import PartialUpdate, StreamTrace
from . import ConcurrentCall, Transcription, log_progress, register

# ADR-0008: `backend` is always an explicit device string, never left to a
# library's own auto-detection. transformers' own documentation example for
# this model uses `device_map="auto"` — deliberately NOT used here, exactly
# as `parakeet.py` refuses it for the same reason. Do not "fix" this back:
# a run that silently picked its own device would sign a result whose
# declared `runtime.backend` isn't the one that produced it.
#
# No "cpu" entry, on purpose (ADR-0013 §4): NVIDIA's model card lists GPU
# architectures only (Turing -> Blackwell, Jetson), and `--backend cpu`
# against a nemotron profile is a hard error from the shared
# `get_supported_backends` gate rather than a slow path that quietly
# produces real signed results. Recorded honestly: torch CPU is not in fact
# unusably slow on this checkpoint (measured batch RTF 0.227x on an Apple
# M-series CPU — faster than realtime), so this is a scoping decision about
# what GOESB publishes for this engine, not a performance cliff. See the
# ADR-0013 addendum.
_DEVICE_BY_BACKEND = {"cuda": "cuda", "metal": "mps"}

_BACKEND_AVAILABLE_CHECK = {
    "cuda": lambda torch_mod: torch_mod.cuda.is_available(),
    "metal": lambda torch_mod: torch_mod.backends.mps.is_available(),
}


def _resolve_model_id(model_name: str) -> str:
    """Translate GOESB's runtime-agnostic model name
    ('nemotron-3.5-asr-streaming-0.6b') into the HF Hub id transformers'
    `AutoModelForRNNT`/`AutoProcessor` expect — same translation role as
    `parakeet._resolve_model_id` / `faster_whisper._resolve_model_id`."""
    return f"nvidia/{model_name}"


def _resolve_language(processor, language: str | None) -> str:
    """A profile's BCP-47 `language` -> a key this checkpoint's
    `prompt_dictionary` actually has.

    Needed because the two vocabularies don't line up: the processor raises
    `ValueError: Unknown language=...` for anything missing, and GOESB
    already ships `es-419` (Latin-American Spanish), which this checkpoint
    does not carry — confirmed by direct lookup, alongside `en-US`/`nl-NL`/
    `de-DE`/`fr-FR`/`pt-BR`, which it does. Falls back full tag -> base
    subtag ('es-419' -> 'es') -> 'auto' (the checkpoint's own
    language-detection prompt), so a profile can never fail the run over a
    regional subtag while still getting language conditioning."""
    dictionary = processor.prompt_dictionary
    if language:
        if language in dictionary:
            return language
        base = language.split("-")[0]
        if base in dictionary:
            return base
    return "auto"


def _generate(model, **kwargs):
    """`generate()` without an explicit `max_new_tokens` emits `UserWarning:
    Using the model-agnostic default max_length=...` — cosmetic noise, not a
    real truncation risk, for the same reason `parakeet._generate` documents:
    the RNNT mixin's own `_prepare_generated_length` sizes that buffer
    generously (and, in streaming mode, pins it to 1e9 outright) while the
    real stop condition is encoder exhaustion. Suppressed by exact message
    match so an unrelated future warning still surfaces."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using the model-agnostic default", category=UserWarning)
        return model.generate(**kwargs, return_dict_in_generate=True)


def _warm_up(processor, model, device: str, language: str) -> None:
    """One throwaway `generate()` on real (silent) audio before the timed
    loop, for the same measured reason `parakeet._warm_up` exists: the first
    accelerator call pays a one-off JIT/kernel-compile cost that would
    otherwise land inside utterance #1's own `processing_time_s` and skew
    the run's RTF. Measured here on Apple Silicon MPS: 2.22s for the first
    call vs 0.09s for the identical second one."""
    import numpy as np
    import torch

    sample_rate = processor.feature_extractor.sampling_rate
    silence = np.zeros(int(0.5 * sample_rate), dtype="float32")
    inputs = processor(silence, sampling_rate=sample_rate, language=language, return_tensors="pt")
    inputs.to(device, dtype=model.dtype)
    with torch.no_grad():
        _generate(model, **inputs)


def _load(model_name: str, backend: str, threads: int, download_root: str | Path | None,
          language: str | None = None, streaming_latency_ms: int | None = None):
    try:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "transformers/torch are not installed; run "
            "`pip install goesb-runner[nemotron]`"
        ) from exc

    check = _BACKEND_AVAILABLE_CHECK.get(backend)
    if check is None:
        raise RuntimeError(
            f"--backend {backend} is not supported by the nemotron adapter "
            f"(supported: {', '.join(sorted(_DEVICE_BY_BACKEND))}). This engine is "
            "GPU-only by design (ADR-0013 §4). Run `goesb doctor` to see what this "
            "machine can run."
        )
    if not check(torch):
        reason = (
            "torch.cuda.is_available() is False (no NVIDIA GPU visible, or a "
            "non-CUDA torch build is installed)"
            if backend == "cuda"
            else "torch.backends.mps.is_available() is False (not Apple Silicon, "
            "or this torch build has no MPS support)"
        )
        raise RuntimeError(
            f"--backend {backend} failed: {reason}. Run `goesb doctor` to check "
            "what's available."
        )

    # `threads` is genuinely applied here, GPU-only engine notwithstanding —
    # this is not the "accepted for call-shape parity" case (ADR-0009 §2
    # forbids declaring a parameter `overridable` that the adapter drops).
    # Feature extraction is `torch.stft` on the CPU device
    # (`NemotronAsrStreamingFeatureExtractor._torch_extract_fbank_features`,
    # confirmed by reading it, not assumed from the name), and it runs inside
    # every timed call: once per utterance in `run_batch`/`run_concurrency`,
    # once per chunk inside the streaming generator. So torch's CPU intra-op
    # pool is real work this benchmark measures, even when the encoder and
    # decoder are on the GPU. Set unconditionally rather than only for a cpu
    # backend the way `parakeet._load` does — there is no cpu backend here,
    # and the knob would otherwise never do anything.
    torch.set_num_threads(threads)

    model_id = _resolve_model_id(model_name)
    cache_dir = str(download_root) if download_root is not None else None
    try:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        # This checkpoint's feature extractor calls `librosa.filters.mel`,
        # and librosa imports numba at module scope. numba refuses to load
        # against a numpy it doesn't support, and the resulting ImportError
        # surfaces thousands of frames below anything recognisable — the
        # first hint anything is wrong arrives at first inference, after a
        # model download. Real report from a pipx install:
        # `ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.`
        # The extra now pins `numpy<2.5`, but an environment assembled
        # incrementally can still drift into this state, so name the repair
        # rather than let the raw traceback speak for itself.
        raise RuntimeError(
            f"nemotron's audio feature extractor failed to import its own "
            f"dependencies: {exc}. This is usually a numpy/numba version "
            f"conflict in this environment rather than a problem with the "
            f"model. Check it with `pip check`; if it reports a numba/numpy "
            f"mismatch, `pip install \"numpy<2.5\"` (or, for a pipx install, "
            f"`pipx runpip goesb-runner install \"numpy<2.5\"`) repairs it."
        ) from exc
    if streaming_latency_ms is not None:
        _select_streaming_latency(processor, streaming_latency_ms)
    model = AutoModelForRNNT.from_pretrained(model_id, cache_dir=cache_dir)
    device = _DEVICE_BY_BACKEND[backend]
    model.to(device)
    model.eval()
    resolved_language = _resolve_language(processor, language)
    _warm_up(processor, model, device, resolved_language)
    return processor, model, resolved_language


def _select_streaming_latency(processor, streaming_latency_ms: int) -> None:
    """Point the processor at the right-attention-context mode whose latency
    is exactly `streaming_latency_ms`, or raise.

    Validated against *the checkpoint's own* `supported_streaming_latencies_ms`
    at run time, never against a list baked into a profile or this module —
    the property is computed from the checkpoint's
    `supported_num_lookahead_tokens`, so the checkpoint is the authority and
    a future revision that adds or drops a mode is picked up for free.

    Never snapped to the nearest supported mode. The same stance ADR-0008
    takes on `--backend`, for the same reason: a signed result whose declared
    `streaming_latency_ms` is not the value that ran is worse than a run that
    refuses to start. Measured for this checkpoint: four modes,
    `{80, 320, 560, 1120}` ms — the model card's fifth (160 ms) does not
    exist here.
    """
    supported = processor.supported_streaming_latencies_ms  # {right_context: latency_ms}
    by_latency = {latency: right for right, latency in supported.items()}
    if streaming_latency_ms not in by_latency:
        raise ValueError(
            f"streaming_latency_ms={streaming_latency_ms} is not supported by this "
            f"checkpoint. Supported: {sorted(by_latency)} ms "
            f"(right attention context {dict(sorted(supported.items()))}). "
            "This adapter never snaps to the nearest supported mode — pick one of "
            "these exactly."
        )
    processor.set_num_lookahead_tokens(by_latency[streaming_latency_ms])


def _mel_chunks(processor, samples):
    """Yield `(input_features, audio_consumed_s)` for cache-aware streaming —
    exactly the fixed chunk geometry `NemotronAsrStreamingGenerationMixin.
    _validate_stream_chunk` demands (`1 + subsampling_factor * right` mel
    frames for the first chunk, `subsampling_factor * (right + 1)` for every
    subsequent one; anything else, including a short final chunk, raises).

    Does NOT use the processor's own `num_samples_first_audio_chunk` /
    `num_samples_per_audio_chunk` helpers, which document themselves as
    "the number of raw audio samples to feed the processor so it returns
    exactly N frames" but are **off by one frame** for this checkpoint —
    confirmed by direct measurement at every one of the four supported
    modes (e.g. right=3 asks for 25/32 frames; its advertised 4040/5520
    samples produce 26/33, and `_validate_stream_chunk` then rejects them
    outright). The properties assume `win_length` (400) where the feature
    extractor actually windows at `n_fft` (512). Reported upstream-worthy,
    worked around here rather than pinned to a transformers patch release.

    So this feeds each chunk enough samples to cover its frames' full STFT
    windows and slices the resulting mel to the exact required frame count:

    - first chunk (`center=True`, which left-pads by `n_fft // 2` the same
      way a full-utterance pass does): `(F_first - 1) * hop + n_fft // 2`
      samples, sliced to `F_first` frames.
    - chunk k>0 (`center=False`): starts at
      `(F_first + (k-1) * F_sub) * hop - n_fft // 2`, takes
      `(F_sub - 1) * hop + n_fft` samples, sliced to `F_sub` frames — a
      352-sample (`n_fft - hop`) overlap with the previous chunk, which is
      STFT window context, not re-decoded audio.

    Verified, not assumed: concatenating every chunk's mel output this way
    reproduces the offline full-utterance pass bit-for-bit on real Dutch
    audio at all four modes — the only differing frame is the very last one
    of the clip, a zero-padding tail artefact the offline path has too.
    That matters because a chunk geometry that merely *validates* can still
    feed the encoder subtly wrong features; this one demonstrably doesn't.

    The tail is zero-padded to a full chunk (the library requires it). No
    fabricated trailing words were observed from that padding on real audio
    — consistent with `center=True`'s own right-edge zero padding in the
    offline path, which the model was trained through.

    Each chunk advances the stream by exactly `F_sub * hop` samples, which is
    `(right + 1) * encoder_frame_ms` — i.e. the chunk's audio advance and the
    mode's declared `streaming_latency_ms` are the same number by
    construction (320 ms of audio per chunk at the 320 ms mode).
    """
    import numpy as np

    extractor = processor.feature_extractor
    hop, n_fft, sample_rate = extractor.hop_length, extractor.n_fft, extractor.sampling_rate
    frames_first = processor.num_mel_frames_first_audio_chunk
    frames_sub = processor.num_mel_frames_per_audio_chunk
    total = len(samples)

    chunk_index = 0
    while True:
        is_first = chunk_index == 0
        if is_first:
            offset, want, need = 0, (frames_first - 1) * hop + n_fft // 2, frames_first
        else:
            offset = (frames_first + (chunk_index - 1) * frames_sub) * hop - n_fft // 2
            want, need = (frames_sub - 1) * hop + n_fft, frames_sub
            if offset >= total:
                return
        window = samples[offset:offset + want]
        if len(window) < want:
            window = np.pad(window, (0, want - len(window)))
        features = processor(
            window, sampling_rate=sample_rate, is_streaming=True,
            is_first_audio_chunk=is_first, return_tensors="pt",
        )["input_features"]
        consumed_s = (frames_first + chunk_index * frames_sub) * hop / sample_rate
        yield features[:, :need], min(consumed_s, total / sample_rate)
        chunk_index += 1


def _words_with_completion_times(tokens: list[dict], clip_duration_s: float) -> list[tuple[str, float]]:
    """`(word, the audio position at which that word is known to be finished)`.

    `processor.decode(..., durations=...)` hands back *tokens* (BPE-style
    subwords: 'E', 'er', 'st', ' alle', ...), each stamped with the encoder
    frame it was emitted at, converted to seconds. A token starting with a
    literal space begins a new word — the same boundary rule
    `parakeet._merge_tokens_to_words` confirmed against real Dutch audio,
    with punctuation attaching to the preceding word for the same reason.

    A word is *finished* only once the next word's first token appears (or
    the clip's audio runs out): an RNNT never revises an emitted token, but
    it can still be mid-word, and a partial hypothesis ending in half a word
    would be counted as a rewritten word by `partial_stability` when the
    rest of it lands. Holding the trailing incomplete word back keeps every
    published partial made of whole, never-revised words — which is what
    makes this engine's ~1.0 stability a true statement rather than a
    tokenizer artefact.
    """
    words: list[list] = []  # [text, first_token_start_s]
    for token in tokens:
        text = token["token"]
        if text.startswith(" ") or not words:
            words.append([text.lstrip(" "), float(token["start"])])
        else:
            words[-1][0] += text
    words = [w for w in words if w[0]]
    return [
        (text, words[i + 1][1] if i + 1 < len(words) else clip_duration_s)
        for i, (text, _start) in enumerate(words)
    ]


@register(
    "nemotron", benchmark_type="batch",
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
    """Transcribe every utterance once, offline (non-streaming), and time
    each call — the same-model baseline Nemotron's streaming latency numbers
    have to be read against.

    `quantization`/`beam_size`/`temperature`/`vad` are accepted for
    call-shape parity with the other batch adapters (docs/03-roadmap.md M2
    exit criterion) but unused: the RNNT greedy transducer decode has no
    beam/temperature knob, this adapter runs no separate VAD pass, and
    `quantization` would mean the model's own torch dtype — a real future
    knob, not yet wired. `threads` IS applied (`torch.set_num_threads` in
    `_load`) despite this being a GPU-only engine: mel feature extraction is
    `torch.stft` on CPU and runs inside every timed call. See `_load`.

    `language` IS applied, unlike Parakeet's: this processor resolves it
    into a real `prompt_ids` model input that conditions the decode. See
    `_resolve_language` for the fallback chain a regional subtag the
    checkpoint doesn't carry (e.g. GOESB's `es-419`) takes.

    **This is not a full-right-context baseline, and it was measured, not
    assumed.** The processor stamps its own `default_num_lookahead_tokens`
    onto every offline call too, so "batch" here is the same cache-aware
    encoder run in one shot at whatever mode the processor is in — this
    adapter leaves it at the checkpoint's default (right context 3, the
    320 ms mode). Measured on `packs/fleurs-nl`: batch WER tracks that
    setting exactly (0.1204 / 0.1314 / 0.1058 / 0.1095 at right context
    0 / 3 / 6 / 13), and at right context 3 the batch and streaming paths
    produce normalize-identical text on all 15 clips — WER 0.1314 both
    ways, zero divergence. So what this profile contributes next to
    `nemotron-3-5-<lang>-streaming` is the compute baseline, not an
    accuracy one: 0.072x vs 0.216x RTF on metal, 0.035x vs 0.130x on cuda,
    for the same output.

    Model load and accelerator warm-up are excluded from per-utterance
    timing, matching every other batch adapter.

    `backend` (ADR-0008) is validated against the real
    `torch.cuda.is_available()`/`torch.backends.mps.is_available()` answer,
    not trusted from a profile or CLI flag, and passed through as the device.
    """
    processor, model, resolved_language = _load(model_name, backend, threads, download_root, language)

    import torch

    device = _DEVICE_BY_BACKEND[backend]
    sample_rate = processor.feature_extractor.sampling_rate

    results: list[Transcription] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, sample_rate, dtype="float32")
        start = time.perf_counter()
        inputs = processor(samples, sampling_rate=sample_rate, language=resolved_language, return_tensors="pt")
        inputs.to(device, dtype=model.dtype)
        with torch.no_grad():
            output = _generate(model, **inputs)
        hypothesis_text = processor.decode(output.sequences, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - start
        log_progress(i, len(utterances), utterance.utterance_id, elapsed)
        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results


@register(
    "nemotron", benchmark_type="streaming",
    applied_parameters=frozenset({"threads", "streaming_latency_ms"}),
    backends=frozenset(_DEVICE_BY_BACKEND),
)
def run_streaming(
    model_name: str,
    utterances: list[Utterance],
    *,
    streaming_latency_ms: int | None = None,
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
    """Genuinely incremental, cache-aware streaming — NOT
    `streaming.run_windowed_local_agreement_streaming`.

    That shared bounded-window function exists for engines with no
    incremental path (faster-whisper, whisper.cpp, parakeet), which have to
    re-decode a growing buffer to produce a streaming trace at all. This
    checkpoint has a real one: `NemotronAsrStreamingGenerationMixin.generate`
    accepts `input_features` as a *generator* of fixed-size mel chunks,
    encodes each into the encoder frame buffer as the decoder consumes it,
    and carries a `padding_cache` + `encoder_past_key_values` across chunks
    (NeMo's `chunked_limited` cache-aware streaming, reached through plain
    transformers). Each second of audio is encoded exactly once. Routing
    this engine through the re-decode path would throw away the entire
    reason it was added.

    **Finality is real here**, the same way `vosk.run_streaming`'s docstring
    establishes it for Kaldi endpointing, and for a stronger reason: a
    streaming RNNT emits tokens strictly left-to-right and never revises
    one. So `committed_word_count` is every whole word emitted so far — no
    local-agreement approximation, no `_COMMIT_SAFETY_MARGIN`. The one
    subtlety is sub-word: a word's final BPE token may not have landed yet,
    so `_words_with_completion_times` holds the trailing incomplete word
    back until the next word's first token appears (or the clip ends). Every
    word this adapter publishes in a partial is therefore whole and final,
    which makes two documented consequences exact rather than approximate:
    `partial_stability` is 1.0, and `first_partial_latency` equals
    `first_final_latency`. Both are true statements about the engine, not
    flattering artefacts — see docs/specs/metrics.md, which records that
    those metrics only discriminate among the re-decode engines.

    `streaming_latency_ms` is the encoder's right attention context, NOT a
    re-decode window (ADR-0013 §3) — which is why this adapter declares it
    instead of `chunk_ms`. It is validated against the checkpoint's own
    `supported_streaming_latencies_ms` and never snapped
    (`_select_streaming_latency`). Passing `None` uses the checkpoint's own
    default mode (320 ms for this one).

    `chunk_ms` is accepted and IGNORED — `cli.py`'s streaming dispatch passes
    a fixed kwarg set to every streaming adapter, so this adapter takes it for
    call-shape parity exactly as `vosk.run_streaming` takes `beam_size`/
    `temperature`. Nothing in this adapter reads it: the chunk size is fixed
    by the checkpoint's chunk geometry, not chosen. `quantization`/
    `beam_size`/`temperature`/`vad` are unused for the reasons `run_batch`
    gives; `threads` and `language` are both genuinely applied, also as
    there — `threads` matters more here than in `run_batch`, since the CPU
    mel extraction it bounds runs once per chunk rather than once per
    utterance.

    Timing convention, matching `vosk.run_streaming` and the shared
    re-decode loop so the numbers stay comparable across engines: chunk k's
    audio "arrives" at `chunk_end_s` and its hypothesis is available
    `decode wall-clock` later. A chunk's wall-clock is the interval between
    handing it to the model and handing over the next one — which is
    precisely the encoder forward pass plus every decoder step that chunk's
    frames unlocked. Measured on Apple Silicon MPS at the 320 ms mode: ~52-69
    ms of compute per 320 ms chunk (mean inter-pull interval directly, and
    0.216x RTF over the pack, which is the same number per chunk) — so no
    backlog accumulates behind realtime and the naive
    `chunk_end_s + decode_wall_s` arithmetic every
    other streaming adapter uses is exactly right here rather than merely
    conventional. Measured streaming RTF through this adapter, same 15-clip
    fleurs-nl pack, metal / cuda: 80 ms -> 0.539x / 0.435x, 320 ms ->
    0.216x / 0.130x, 560 ms -> 0.160x / 0.104x, 1120 ms -> 0.123x / 0.066x.
    WER by mode is identical on both backends: 0.1204 / 0.1314 / 0.1095 /
    0.1095 — the 80 ms mode costs ~3x the compute of 320 ms and is not more
    accurate. Measured p50 first-partial latency on cuda: 359 / 680 / 733 /
    995 ms, and `partial_stability` was exactly 1.0000 at every mode.

    Measured (ADR-0013 §2): at the 320 ms mode this
    produces text *identical* to `run_batch`'s on all 15 clips of
    `packs/fleurs-nl` (WER 0.1314 both ways, streaming-vs-batch divergence
    0.0000). The offline path is the same cache-aware limited-right-context
    decode run in one shot, so there is no full-context baseline to diverge
    from — see `run_batch`'s own docstring and the ADR-0013 addendum.
    """
    processor, model, resolved_language = _load(
        model_name, backend, threads, download_root, language, streaming_latency_ms,
    )

    import numpy as np
    import torch

    device = _DEVICE_BY_BACKEND[backend]
    sample_rate = processor.feature_extractor.sampling_rate
    num_lookahead_tokens = processor.default_num_lookahead_tokens
    # Resolved once, off the same processor state the chunk geometry comes
    # from, so a trace can never be signed against a different mode than the
    # one that produced it.
    prompt_ids = processor(
        np.zeros(sample_rate // 10, dtype="float32"), sampling_rate=sample_rate,
        language=resolved_language, return_tensors="pt",
    )["prompt_ids"].to(device)

    traces: list[StreamTrace] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, sample_rate, dtype="float32")
        clip_duration_s = len(samples) / sample_rate

        # (wall clock at hand-off, audio consumed by then) per chunk. Appended
        # from inside the generator, so it records when the model actually
        # pulled each chunk — the model drives the pull rate (it only asks for
        # the next chunk once the decoder has consumed the previous one's
        # frames), which is what makes these intervals real per-chunk compute.
        marks: list[tuple[float, float]] = []

        def chunk_stream(samples=samples, marks=marks):
            for features, consumed_s in _mel_chunks(processor, samples):
                marks.append((time.perf_counter(), consumed_s))
                yield features

        start = time.perf_counter()
        with torch.no_grad():
            output = _generate(
                model, input_features=chunk_stream(),
                num_lookahead_tokens=num_lookahead_tokens, prompt_ids=prompt_ids,
            )
        finished = time.perf_counter()
        processing_time_s = finished - start

        _text, token_timestamps = processor.decode(
            output.sequences, durations=output.durations, skip_special_tokens=True,
        )
        words = _words_with_completion_times(token_timestamps[0], clip_duration_s)

        # A word's own first-token timestamp is the encoder frame it was
        # emitted at, i.e. real detected-speech position — the same
        # speech-onset zeroing every StreamTrace uses (see streaming.py's
        # module docstring), read here off the model's own output rather
        # than a separate VAD pass.
        speech_onset_s = float(token_timestamps[0][0]["start"]) if token_timestamps[0] else 0.0
        speech_offset_s = (
            min(float(token_timestamps[0][-1]["end"]), clip_duration_s)
            if token_timestamps[0] else clip_duration_s
        )

        updates: list[PartialUpdate] = []
        for k, (mark_time, chunk_end_s) in enumerate(marks):
            next_time = marks[k + 1][0] if k + 1 < len(marks) else finished
            decode_wall_s = next_time - mark_time
            is_last_chunk = k + 1 == len(marks)
            visible = [
                text for text, complete_at_s in words
                if is_last_chunk or complete_at_s < chunk_end_s
            ]
            text = " ".join(visible)
            updates.append(PartialUpdate(
                chunk_end_s=chunk_end_s - speech_onset_s,
                emit_time_s=chunk_end_s + decode_wall_s - speech_onset_s,
                text=text,
                # Every word published above is whole and, for an RNNT,
                # never revised — so committed == emitted, exactly.
                committed_word_count=len(visible),
            ))

        log_progress(i, len(utterances), utterance.utterance_id, processing_time_s)
        traces.append(StreamTrace(
            utterance_id=utterance.utterance_id,
            audio_duration_s=max(0.0, speech_offset_s - speech_onset_s),
            processing_time_s=processing_time_s,
            updates=updates,
            final_text=updates[-1].text if updates else "",
        ))
    return traces


@register(
    "nemotron", benchmark_type="concurrency",
    applied_parameters=frozenset({"threads", "concurrency"}),
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
    """Does this GPU stay fast under N simultaneous requests, not just one at
    a time? Same fixed-`duration_s`-window, round-robin-through-utterances
    harness `faster_whisper.run_concurrency` established (ADR-0012).

    Nemotron is NOT safe to share one model instance across concurrent
    threads. ADR-0012 reached three different answers for three engines and
    this is a fourth reached the same way — by reading what the library
    actually does, not by assuming the Parakeet answer transfers. Two
    independent races, both in transformers' own generate() overrides for
    this model:

    1. `NemotronAsrStreamingGenerationMixin.generate` sets `self._streaming`,
       `self._stream_exhausted` and `self._streaming_num_lookahead_tokens` as
       plain instance attributes and `delattr`s them in its `finally` — so
       one call finishing mid-flight strips the state another call is still
       decoding against.
    2. `Nemotron3_5AsrGenerationMixin.generate` goes further and rebinds
       `self.get_audio_features` to a closure over `self._prompt_ids`, then
       `del`s the attribute in `finally`. Two concurrent calls don't merely
       race — they cross-contaminate language conditioning, and whichever
       finishes first leaves the other calling a deleted attribute.

    Both sit on top of the inherited `ParakeetRNNTGenerationMixin` state
    (`_encoder_finished`, `_symbols_at_frame`, `_step_durations`) that
    `parakeet.run_concurrency` already documents. So: `concurrency` full,
    independent model instances, one per worker — whisper.cpp's and vosk's
    shape, not faster-whisper's genuinely-thread-safe shared ctranslate2
    `Translator`.

    On a GPU that N-way cost is VRAM — but measured on a real RTX A6000,
    VRAM is not what limits this engine. Peak device memory across a whole
    timed window: 3.2 GiB at concurrency 1, 5.6 at 2, 10.5 at 4, 20.3 at 8,
    so 8 workers fit on a 24 GB card.

    What limits it is throughput, and the shape is worth knowing before
    anyone plans capacity around this engine: throughput peaks at
    concurrency **2** (~50 audio-s/s) and then collapses — ~21 at 3, ~20 at
    4, ~18-21 at 8 — measured twice independently on the same box, with the
    cliff between 2 and 3 reproducing both times. It is not CPU-thread
    contention: re-running 2/3/8 at `threads=1` vs `threads=4` moved
    throughput less than the ~11-15% run-to-run noise. N independent
    instances contend for the same SMs and nothing batches across requests.
    The profile's ceiling of 8 exists to keep that collapse sweepable, not
    because 8 is a sensible operating point.

    ADR-0012's deferred pre-run OOM check is deliberately still deferred
    here and NOT silently omitted: a cheap `torch.cuda.mem_get_info()`
    precheck would only cover CUDA (MPS has no equivalent free-VRAM query),
    would still race any other process on the card, and — since model
    construction below is sequential and before the timed window — an OOM
    today already surfaces during load with a torch OOM message naming the
    allocation, not midway through a timed run. It is also, on these
    numbers, guarding a failure mode this engine reaches late if at all.

    Model construction — including each instance's own `_warm_up` — is
    sequential and happens before the timed window starts, same as every
    other adapter's "model load excluded from RTF" convention.

    Audio is decoded once per utterance up front and shared read-only across
    workers; feature extraction and `generate()` are both inside the timed
    region, matching `run_batch`'s convention of timing the full
    decode-to-tokens pipeline.

    No WER/CER: this benchmark type doesn't score accuracy, so
    `reference_text` is never touched.
    """
    loaded = [_load(model_name, backend, threads, download_root, language) for _ in range(concurrency)]

    import torch

    device = _DEVICE_BY_BACKEND[backend]
    sample_rate = loaded[0][0].feature_extractor.sampling_rate

    decoded = [
        (utterance, decode_pcm(utterance.audio_path, sample_rate, dtype="float32"))
        for utterance in utterances
    ]

    deadline = time.perf_counter() + duration_s

    def _worker(worker_id: int) -> list[ConcurrentCall]:
        processor, model, resolved_language = loaded[worker_id]
        calls: list[ConcurrentCall] = []
        i = worker_id
        while time.perf_counter() < deadline:
            utterance, samples = decoded[i % len(decoded)]
            start = time.perf_counter()
            inputs = processor(samples, sampling_rate=sample_rate, language=resolved_language, return_tensors="pt")
            inputs.to(device, dtype=model.dtype)
            with torch.no_grad():
                _generate(model, **inputs)
            elapsed = time.perf_counter() - start
            calls.append(ConcurrentCall(processing_time_s=elapsed, audio_duration_s=utterance.duration_s))
            i += concurrency
        return calls

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, w) for w in range(concurrency)]
        calls = [c for future in futures for c in future.result()]
    return calls
