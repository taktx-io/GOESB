"""`vosk` batch runtime adapter (docs/02-architecture.md §4).

Optional dependency (`pip install goesb-runner[vosk]`) — `vosk`/`soundfile`
are only imported inside `run_batch`, matching the lazy-import pattern used
by every other adapter.
"""
from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..audio import decode_pcm
from ..pack import Utterance
from ..streaming import PartialUpdate, StreamTrace
from . import ConcurrentCall, Transcription, log_progress, register

# "Small" (~40-50MB) alongside each language's own larger, more accurate
# model (~0.9-2.3GB) — confirmed present for every language already listed
# here by checking https://alphacephei.com/vosk/models directly (this repo
# only ever wired up the small tier before; Vosk itself always had both).
_MODEL_URLS = {
    "vosk-model-small-en-us-0.15":
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "vosk-model-en-us-0.22":
        "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
    "vosk-model-small-es-0.42":
        "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "vosk-model-es-0.42":
        "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip",
    "vosk-model-small-fr-0.22":
        "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "vosk-model-fr-0.22":
        "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip",
    "vosk-model-small-de-0.15":
        "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "vosk-model-de-0.21":
        "https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip",
    "vosk-model-small-pt-0.3":
        "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "vosk-model-pt-fb-v0.1.1-20220516_2113":
        "https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip",
    "vosk-model-small-nl-0.22":
        "https://alphacephei.com/vosk/models/vosk-model-small-nl-0.22.zip",
    "vosk-model-nl-spraakherkenning-0.6":
        "https://alphacephei.com/vosk/models/vosk-model-nl-spraakherkenning-0.6.zip",
}
_SAMPLE_RATE = 16000


def _resolve_model_dir(model_name: str, download_root: Path) -> Path:
    """Download + unzip the pinned vosk model into `download_root` if not
    already there. `vosk.Model(model_path=...)` needs an already-extracted
    local directory — unlike faster-whisper's HF snapshot download, vosk has
    no built-in fetch-by-pinned-version — so this does it explicitly: a
    declarative fetch of a named, version-pinned asset, not arbitrary code
    (ADR-0004)."""
    model_dir = download_root / model_name
    if model_dir.exists():
        return model_dir
    try:
        url = _MODEL_URLS[model_name]
    except KeyError:
        raise ValueError(
            f"unknown vosk model {model_name!r}; known models: {sorted(_MODEL_URLS)}"
        ) from None

    download_root.mkdir(parents=True, exist_ok=True)
    zip_path = download_root / f"{model_name}.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(download_root)
    zip_path.unlink()
    return model_dir


@register("vosk", benchmark_type="batch", applied_parameters=frozenset())
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

    `quantization`/`beam_size`/`temperature`/`vad`/`threads`/`language` are
    accepted for call-shape parity with the other batch adapters
    (docs/03-roadmap.md M2 exit criterion: adapters swap without core
    changes) but unused — vosk's Kaldi decoder has no equivalent tunables
    exposed through its Python API, and each vosk model is already
    per-language (the profile's `model.name` picks the language, not a
    runtime parameter). `backend` (ADR-0008) is accepted for the same
    call-shape-parity reason but unused — this adapter is genuinely
    CPU-only (no `backends` declared at registration means it defaults to
    cpu-only, so the CLI never calls in with anything else)."""
    try:
        import vosk
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "vosk is not installed; run `pip install goesb-runner[vosk]`"
        ) from exc
    vosk.SetLogLevel(-1)  # silence vosk's own stderr logging

    root = Path(download_root) if download_root is not None else Path.home() / ".goesb" / "models" / model_name
    model_dir = _resolve_model_dir(model_name, root)
    model = vosk.Model(model_path=str(model_dir))

    results: list[Transcription] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, _SAMPLE_RATE, dtype="int16")
        recognizer = vosk.KaldiRecognizer(model, _SAMPLE_RATE)

        start = time.perf_counter()
        recognizer.AcceptWaveform(samples.tobytes())
        hypothesis_text = json.loads(recognizer.FinalResult()).get("text", "")
        elapsed = time.perf_counter() - start

        log_progress(i, len(utterances), utterance.utterance_id, elapsed)
        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results


@register(
    "vosk", benchmark_type="streaming",
    applied_parameters=frozenset({"chunk_ms"}),
)
def run_streaming(
    model_name: str,
    utterances: list[Utterance],
    *,
    chunk_ms: int = 1000,
    streaming_latency_ms: int | None = None,
    quantization: str = "int8",
    beam_size: int = 5,
    temperature: float = 0.0,
    vad: bool = True,
    threads: int = 4,
    download_root: str | Path | None = None,
    language: str | None = None,
    backend: str = "cpu",
) -> list[StreamTrace]:
    """Feed each utterance to vosk in `chunk_ms` slices, one `KaldiRecognizer`
    per utterance carrying real incremental decoder state across calls —
    unlike `faster_whisper.run_streaming`, which has no decoder state to
    resume and must re-decode the whole growing buffer every chunk, vosk's
    `AcceptWaveform` only ever sees the *new* audio, and Kaldi's own
    endpointing decides when a stretch of speech is genuinely final. That
    finality is real, not a heuristic: once `AcceptWaveform` returns True
    (or the utterance's audio runs out and `FinalResult()` force-flushes
    whatever's pending), the words in that `Result()` are committed and
    this adapter never revises them — no local-agreement approximation
    needed, unlike the whisper-style whole-buffer re-decode this mirrors
    the call shape of.

    `quantization`/`beam_size`/`temperature`/`vad`/`threads`/`language`/
    `backend` are accepted for call-shape parity with every other vosk
    adapter (see `run_batch`'s own docstring) but unused.

    `streaming_latency_ms` is accepted and IGNORED: it is a cache-aware
    engine's encoder right-attention context (ADR-0013 §3), a different
    physical quantity from this adapter's `chunk_ms` re-decode window, and
    `cli.py`'s streaming dispatch passes one fixed kwarg set to every
    streaming adapter. Profiles for this engine never declare it, so it
    arrives as None.
    """
    try:
        import vosk
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "vosk is not installed; run `pip install goesb-runner[vosk]`"
        ) from exc
    vosk.SetLogLevel(-1)

    root = Path(download_root) if download_root is not None else Path.home() / ".goesb" / "models" / model_name
    model_dir = _resolve_model_dir(model_name, root)
    model = vosk.Model(model_path=str(model_dir))

    chunk_samples = max(1, int(chunk_ms / 1000 * _SAMPLE_RATE))

    traces: list[StreamTrace] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, _SAMPLE_RATE, dtype="int16")
        total_samples = len(samples)
        clip_duration_s = total_samples / _SAMPLE_RATE

        recognizer = vosk.KaldiRecognizer(model, _SAMPLE_RATE)
        # Word-level timings on each final result — the same "where did
        # detected speech actually start/end" signal
        # faster_whisper.run_streaming reads off VAD segment timestamps,
        # needed here for the same speech-onset-zeroing this StreamTrace
        # convention requires (see streaming.py's own docstring).
        recognizer.SetWords(True)

        updates: list[PartialUpdate] = []
        processing_time_s = 0.0
        speech_onset_s: float | None = None
        speech_offset_s = clip_duration_s  # fallback if no words ever land
        committed_segments: list[str] = []
        pos = 0
        while pos < total_samples:
            end = min(pos + chunk_samples, total_samples)
            is_last_chunk = end >= total_samples
            chunk_end_s = end / _SAMPLE_RATE
            chunk_bytes = samples[pos:end].tobytes()

            start = time.perf_counter()
            finalized = recognizer.AcceptWaveform(chunk_bytes)
            if finalized:
                result = json.loads(recognizer.Result())
            elif is_last_chunk:
                # Audio's exhausted — force-flush whatever Kaldi still has
                # buffered rather than leaving it stranded as an
                # never-committed partial (same "commit everything once
                # the clip ends" convention faster_whisper's is_last_chunk
                # branch uses).
                result = json.loads(recognizer.FinalResult())
                finalized = True
            else:
                result = None
            decode_wall_s = time.perf_counter() - start
            processing_time_s += decode_wall_s

            if finalized and result is not None:
                words = result.get("result", [])
                if words:
                    if speech_onset_s is None:
                        speech_onset_s = words[0]["start"]
                    speech_offset_s = min(words[-1]["end"], chunk_end_s)
                segment_text = result.get("text", "")
                if segment_text:
                    committed_segments.append(segment_text)
                partial_text = ""
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "")

            text = " ".join([*committed_segments, partial_text]).strip() if partial_text else " ".join(committed_segments).strip()
            committed_word_count = sum(len(seg.split()) for seg in committed_segments)

            updates.append(PartialUpdate(
                chunk_end_s=chunk_end_s,
                emit_time_s=chunk_end_s + decode_wall_s,
                text=text,
                committed_word_count=committed_word_count,
            ))
            pos = end

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
    "vosk", benchmark_type="concurrency",
    applied_parameters=frozenset({"concurrency"}),
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
    """Does this hardware stay fast under N simultaneous requests?
    `concurrency` worker threads each repeatedly transcribe utterances for
    a fixed `duration_s` wall-clock window (same fixed-window,
    round-robin-through-utterances harness `faster_whisper.run_concurrency`
    established) -- but unlike that adapter, this builds one full
    `vosk.Model` instance PER WORKER rather than sharing one.

    That's a real, documented risk, not a theoretical one:
    alphacep/vosk-api#606 reports SIGSEGV crashes from creating
    `KaldiRecognizer` instances in parallel threads against one shared
    `Model` -- a race in the model's own C++ reference counting
    (concurrent `UnRef()` calls corrupting the count, freeing the model
    while another thread is still using it). The Python binding
    (python/vosk/__init__.py) adds no locking of its own around
    construction/destruction to guard against this. Whether the
    underlying C++ bug was ever actually fixed isn't verifiable from the
    Python layer, so this takes the same stance ADR-0012's whisper-cpp
    addendum already established for a similar not-independently-
    verifiable safety question: don't share what can't be confirmed safe
    to share, even though Vosk's own reference server usage (one shared
    Model, one Recognizer per connection/thread) suggests this pattern is
    *intended* to be safe. Vosk's models are also small enough (the
    `small` tier is ~40-50MB; even the larger per-language models top out
    around 1-2GB) that N full instances is a real but bounded cost, not
    whisper-cpp's forced-into-a-much-tighter-ceiling problem -- see the
    `vosk-*-concurrency` profiles' own `overridable.concurrency.range.max`
    values, tighter for the larger model tier.

    `quantization`/`beam_size`/`temperature`/`vad`/`threads`/`language`/
    `backend` are accepted for call-shape parity with every other
    concurrency adapter (cli.py's `_do_run_concurrency` closure calls all
    of them identically regardless of engine) but unused, same as
    `run_batch`'s own stance above."""
    try:
        import vosk
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "vosk is not installed; run `pip install goesb-runner[vosk]`"
        ) from exc
    vosk.SetLogLevel(-1)

    root = Path(download_root) if download_root is not None else Path.home() / ".goesb" / "models" / model_name
    model_dir = _resolve_model_dir(model_name, root)
    models = [vosk.Model(model_path=str(model_dir)) for _ in range(concurrency)]

    deadline = time.perf_counter() + duration_s

    def _worker(worker_id: int) -> list[ConcurrentCall]:
        model = models[worker_id]
        calls: list[ConcurrentCall] = []
        i = worker_id
        while time.perf_counter() < deadline:
            utterance = utterances[i % len(utterances)]
            samples = decode_pcm(utterance.audio_path, _SAMPLE_RATE, dtype="int16")
            recognizer = vosk.KaldiRecognizer(model, _SAMPLE_RATE)

            start = time.perf_counter()
            recognizer.AcceptWaveform(samples.tobytes())
            recognizer.FinalResult()
            elapsed = time.perf_counter() - start

            calls.append(ConcurrentCall(processing_time_s=elapsed, audio_duration_s=utterance.duration_s))
            i += concurrency
        return calls

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, w) for w in range(concurrency)]
        calls = [c for future in futures for c in future.result()]
    return calls
