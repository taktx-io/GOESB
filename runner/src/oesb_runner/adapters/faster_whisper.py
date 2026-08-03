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
from ..streaming import PartialUpdate, StreamTrace
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


# How many of a window's most-recent agreeing words to hold back from
# committing even once they agree across two consecutive decodes — the
# same reason whisper_streaming's own LocalAgreement-2 policy never
# commits its freshest words: a decode's last word or two is the part
# most likely to be revised once more audio (more context) lands, since
# it's closest to wherever the decode's own attention window currently
# ends. 2 is a conservative starting point, not a measured optimum.
_COMMIT_SAFETY_MARGIN = 2

# Real report, confirmed by directly measuring real audio: trimming the
# window flush to the last committed word's own `end` timestamp silently
# dropped single words at the seam on 5+ of 15 real LibriSpeech
# utterances ("...OF ART MISTER QUILTER WRITES..." -> "...of Mr. Quilter
# writes...", "ART" gone) -- Whisper's per-word end timestamps aren't
# precise enough to cut flush against, and VAD (re-run fresh on every
# window) reads a hard mid-phoneme cut as leading silence and skips real
# speech at the new window's own start. This cushion, subtracted from the
# first still-uncommitted word's own START (a real word boundary, not an
# imprecise end-time) before trimming, gives the next window's VAD/
# encoder genuine leading acoustic context instead of a splice. Not
# tuned against real audio beyond confirming it stops the word-dropping
# observed above -- not a measured optimum.
_TRIM_CUSHION_S = 0.3


def _normalize_word_for_overlap_match(word: str) -> str:
    """Real report: comparing raw words for the committed/window overlap
    check below missed real duplicates on actual audio ("and we are We
    are glad", "finish in art Art is") -- Whisper capitalizes a window's
    own first word as if it were a fresh sentence start, even when that
    word was already committed mid-sentence, lowercase, from the
    previous window. Trailing punctuation (a comma landing on one
    decode's version of a word but not the other's) caused the same kind
    of miss. Comparison-only -- `committed_words` itself keeps whatever
    casing/punctuation it was first committed with."""
    return word.strip(".,!?;:\"'").lower()


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
    only the audio *since the last commit* after every chunk (faster-whisper
    has no incremental decoder state to resume, so "streaming" here still
    means repeated re-transcription — the same "local agreement" pattern
    used by e.g. whisper_streaming — but bounded, not whole-buffer).

    Real report: the original version of this function re-decoded
    `samples[:end]` — the ENTIRE clip so far, unbounded, every chunk. That
    measured RTF 3.19x on an Apple M1 Pro (whisper-medium, default
    settings) — slower than realtime, and not just slow: `emit_time_s`
    below assumes each chunk's decode starts the instant its audio
    "arrives," which is only honest if the system isn't still catching up
    on a backlog from earlier chunks. Once RTF > 1 that assumption breaks,
    so first_final_latency understated what a real deployment would
    actually experience. See `whisper-medium-en-streaming`'s own profile
    comment and `cli.py`'s `_MATRIX_STREAMING_EXCLUDED_PROFILE_IDS` for
    where this got flagged and pulled from the wizard pending this fix.

    The fix: track `window_start_sample`, advanced forward to just after
    the last COMMITTED word (via `word_timestamps=True`) every time
    local agreement locks one in. Only `samples[window_start_sample:end]`
    — audio since the last commit, not the whole clip — gets re-decoded
    each chunk, so per-chunk decode cost stays roughly bounded instead of
    growing with clip length. `_COMMIT_SAFETY_MARGIN` holds back each
    window's freshest words from being committed even once they agree,
    since those are the ones most likely to be revised by the next
    chunk's extra context.

    Validated against all 15 real utterances in librispeech-en-vosk-streaming
    (4 rounds, each a real correctness bug found and fixed on real audio,
    not assumed — see `_TRIM_CUSHION_S` and `_normalize_word_for_overlap_match`
    for the two that needed their own fix): RTF held at a consistent
    ~2.5x across all 4 runs (vs ~3.19x for the original whole-buffer
    version — a real, reproducible improvement, not noise), and WER
    landed at 0.110 vs the original's 0.078. That WER gap is real and
    expected — bounded context genuinely costs some accuracy relative to
    whole-clip decoding, the same tradeoff every windowed streaming ASR
    system makes — but the errors it costs are boundary-adjacent
    hallucinations (an occasional short inserted word right at a window
    seam), not the silent word-DROPS ("...OF ART MISTER..." -> "...of
    Mr....") or wholesale DUPLICATION ("and we are We are glad") the
    first two attempts at this fix produced. RTF is still >1 (not
    realtime-capable on this CPU-only hardware — see the profile's own
    comment and cli.py's `_MATRIX_STREAMING_EXCLUDED_PROFILE_IDS` for
    why that's an inherent Whisper-architecture cost this rewrite can't
    fix, not a bug), so this stays excluded from the wizard.
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
        # from faster-whisper's own word timestamps — no new dependency,
        # no separate detection pass, since word_timestamps is already on
        # for the bounded-window trim logic below. onset locks in at first
        # detection (the earliest chunk that produces any word); offset
        # keeps updating to the latest chunk's last-word end, so by the
        # final chunk it reflects where speech actually stopped.
        speech_onset_s: float | None = None
        speech_offset_s = clip_duration_s  # fallback if no words ever land

        # Bounded-window state — see this function's own docstring.
        window_start_sample = 0
        committed_words: list[str] = []  # locked in from PRIOR windows, never revised
        previous_window_words: list[dict] = []  # this window's previous decode, for local agreement

        end = 0
        while end < total_samples:
            end = min(end + chunk_samples, total_samples)
            is_last_chunk = end >= total_samples
            chunk_end_s = end / sample_rate
            window_offset_s = window_start_sample / sample_rate

            start = time.perf_counter()
            segments, _info = model.transcribe(
                samples[window_start_sample:end],
                beam_size=beam_size,
                temperature=temperature,
                vad_filter=vad,
                language=language,
                word_timestamps=True,
            )
            segments = list(segments)
            decode_wall_s = time.perf_counter() - start
            processing_time_s += decode_wall_s

            # Absolute clip-relative time, not window-relative — the words
            # faster-whisper hands back are timestamped from position 0 of
            # whatever buffer was passed in, which is `window_offset_s`
            # into the real clip once the window has trimmed forward.
            window_words = [
                {"word": w.word.strip(), "start": w.start + window_offset_s, "end": w.end + window_offset_s}
                for segment in segments for w in (segment.words or [])
            ]

            # `_TRIM_CUSHION_S` deliberately keeps a little already-committed
            # audio in the new window (real report: cutting flush against
            # it dropped words instead — see that constant's own
            # docstring) — which means this decode can re-transcribe the
            # tail end of `committed_words` a second time. Strip however
            # many of `window_words`' own leading entries exactly match
            # the tail of `committed_words`, in order, before treating the
            # rest as this window's real (new, uncommitted) content —
            # otherwise that re-transcribed overlap gets appended AGAIN,
            # duplicating words in the final text ("and we are We are
            # glad", confirmed on real audio without this check).
            if committed_words:
                overlap = 0
                committed_tail_norm = [_normalize_word_for_overlap_match(w) for w in committed_words]
                for k in range(min(len(window_words), len(committed_words)), 0, -1):
                    window_head_norm = [_normalize_word_for_overlap_match(w["word"]) for w in window_words[:k]]
                    if window_head_norm == committed_tail_norm[-k:]:
                        overlap = k
                        break
                window_words = window_words[overlap:]

            if window_words:
                if speech_onset_s is None:
                    speech_onset_s = window_words[0]["start"]
                # Clamp to this chunk's own buffer end — same known-quirk
                # clamp the previous version used (Whisper's predicted
                # timestamps can slightly overshoot the real audio fed in).
                speech_offset_s = min(window_words[-1]["end"], chunk_end_s)

            # Local agreement WITHIN this window: a word only counts as
            # agreeing once it matches, by position, across two
            # consecutive decodes of the SAME (untrimmed) window — same
            # rule the previous whole-buffer version used, just scoped to
            # the current window instead of the whole clip. Real report
            # that shaped the original version of this rule, still true
            # here: a cruder "every word but the last" rule had committed
            # text revised in 14 of 30 chunks on real audio, since
            # VAD/segmentation isn't stable just because a word isn't the
            # decode's last one.
            agreeing = 0
            for a, b in zip(previous_window_words, window_words):
                if a["word"] != b["word"]:
                    break
                agreeing += 1
            newly_stable = len(window_words) if is_last_chunk else max(0, agreeing - _COMMIT_SAFETY_MARGIN)
            previous_window_words = window_words

            if newly_stable > 0:
                newly_committed = window_words[:newly_stable]
                committed_words.extend(w["word"] for w in newly_committed)
                # Trim: advance the window, bounding decode cost (the
                # actual point of this rewrite) — but NOT flush against
                # the last committed word's own `end` timestamp; cut
                # there instead of `_TRIM_CUSHION_S` before the first
                # still-uncommitted word's own START (see the constant's
                # own docstring for the real-audio word-dropping this
                # fixes). Monotonic: never moves backward past where the
                # window already is.
                next_word_start_s = (
                    window_words[newly_stable]["start"] if newly_stable < len(window_words)
                    else newly_committed[-1]["end"]
                )
                cushioned_start_s = max(0.0, next_word_start_s - _TRIM_CUSHION_S)
                window_start_sample = max(
                    window_start_sample,
                    min(total_samples, round(cushioned_start_s * sample_rate)),
                )
                # The window just moved out from under `previous_window_words`
                # (its offsets no longer correspond to any future decode's
                # window) — reset so the next decode's agreement check
                # starts fresh against the new window, same as the very
                # first chunk's own bootstrap (empty previous_window_words).
                previous_window_words = []

            tail_words = [w["word"] for w in window_words[newly_stable:]]
            text = " ".join(committed_words + tail_words).strip()

            updates.append(PartialUpdate(
                chunk_end_s=chunk_end_s,
                emit_time_s=chunk_end_s + decode_wall_s,
                text=text,
                committed_word_count=len(committed_words),
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
