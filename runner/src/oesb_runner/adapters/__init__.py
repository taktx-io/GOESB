"""Runtime adapter registry (FR-11.1): how a model is driven for a benchmark
type. Each adapter is reviewed, in-tree code (ADR-0004) — never a plugin
supplied at run time. Core dispatch here is runtime-agnostic; adapter modules
lazy-import their actual ML dependency so importing this package never
requires it to be installed.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass


def log_progress(index: int, total: int, utterance_id: str, elapsed_s: float) -> None:
    """One line per finished utterance, to stderr. Batch/streaming loops are
    otherwise silent for an entire repeat — on anything past a handful of
    short clips that reads as a hang rather than progress."""
    print(f"  [{index}/{total}] {utterance_id} ({elapsed_s:.2f}s)", file=sys.stderr)


@dataclass(frozen=True)
class Transcription:
    """One utterance's batch-adapter output — shared shape every batch
    adapter (faster-whisper, vosk, whisper.cpp, ...) returns, so the CLI's
    batch run loop stays adapter-agnostic."""
    utterance_id: str
    hypothesis_text: str
    processing_time_s: float


@dataclass(frozen=True)
class ConcurrentCall:
    """One completed transcription call inside a `concurrency` benchmark's
    worker-thread harness — no `utterance_id`/`hypothesis_text` (unlike
    `Transcription`) since this benchmark type doesn't score accuracy, only
    per-call timing pooled across every worker into one RTF distribution."""
    processing_time_s: float
    audio_duration_s: float


# Keyed by (runtime_name, benchmark_type) — a runtime can implement more than
# one benchmark type (e.g. faster-whisper's "batch" and "streaming" loops are
# different callables, same underlying runtime).
_ADAPTERS: dict[tuple[str, str], Callable] = {}

# Which compute backends this (runtime, benchmark_type) adapter actually
# accepts (ADR-0008): the set `--backend` is validated against before the
# adapter is ever called — requesting one outside this set is a hard error,
# never a silent fallback to whatever the underlying library would have
# auto-selected. Defaults to cpu-only, matching every adapter that doesn't
# declare otherwise (vosk is genuinely CPU-only and needs no entry here).
_SUPPORTED_BACKENDS: dict[tuple[str, str], frozenset[str]] = {}

# "No silent knobs" (ADR-0009 §2): every batch adapter shares one call
# signature for call-shape parity (docs/03-roadmap.md M2 exit criterion —
# adapters swap without core changes), but several accept kwargs they never
# actually apply (whisper-cpp: beam_size/vad/quantization; vosk: all of
# them — see their own docstrings). A profile may only declare a parameter
# `overridable` if it's in *this* set for its runtime — declaring one the
# adapter silently ignores would sign a result asserting a value that had
# no effect, which is worse than not having the feature. Reviewed alongside
# each adapter's own registration, not derived from its signature (a kwarg
# existing there proves nothing about whether the code underneath it does).
_APPLIED_PARAMETERS: dict[tuple[str, str], frozenset[str]] = {}


def register(
    runtime_name: str,
    benchmark_type: str = "batch",
    applied_parameters: frozenset[str] = frozenset(),
    backends: frozenset[str] = frozenset({"cpu"}),
) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        key = (runtime_name, benchmark_type)
        if key in _ADAPTERS:
            raise ValueError(f"runtime adapter already registered: {key!r}")
        _ADAPTERS[key] = fn
        _APPLIED_PARAMETERS[key] = applied_parameters
        _SUPPORTED_BACKENDS[key] = backends
        return fn

    return decorator


def get_adapter(runtime_name: str, benchmark_type: str = "batch") -> Callable:
    try:
        return _ADAPTERS[(runtime_name, benchmark_type)]
    except KeyError:
        raise ValueError(
            f"unknown runtime adapter: {runtime_name!r} for benchmark_type {benchmark_type!r}"
        ) from None


def get_applied_parameters(runtime_name: str, benchmark_type: str = "batch") -> frozenset[str]:
    """Which parameters this (runtime, benchmark_type) adapter genuinely
    applies — the universe a profile's `overridable` block must be a
    subset of (ADR-0009 §2). Unknown pairs default to the empty set (fail
    closed): a future adapter that forgets to declare this applies
    nothing, rather than silently allowing everything."""
    return _APPLIED_PARAMETERS.get((runtime_name, benchmark_type), frozenset())


def registered_adapters() -> list[tuple[str, str]]:
    """Every (runtime_name, benchmark_type) pair currently registered —
    `goesb doctor` iterates this to report per-engine backend readiness
    without hardcoding the runtime list a second time."""
    return sorted(_ADAPTERS)


def get_supported_backends(runtime_name: str, benchmark_type: str = "batch") -> frozenset[str]:
    """Which `--backend` values this (runtime, benchmark_type) adapter
    accepts (ADR-0008). Unknown pairs default to cpu-only — fail toward the
    one backend every machine has, not toward permitting an unregistered
    combination to claim GPU support it never declared."""
    return _SUPPORTED_BACKENDS.get((runtime_name, benchmark_type), frozenset({"cpu"}))


# Built-in adapters register themselves on import.
from . import faster_whisper, vosk, whisper_cpp  # noqa: F401
