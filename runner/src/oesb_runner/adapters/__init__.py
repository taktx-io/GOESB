"""Runtime adapter registry (FR-11.1): how a model is driven for a benchmark
type. Each adapter is reviewed, in-tree code (ADR-0004) — never a plugin
supplied at run time. Core dispatch here is runtime-agnostic; adapter modules
lazy-import their actual ML dependency so importing this package never
requires it to be installed.
"""
from __future__ import annotations

from typing import Callable

# Keyed by (runtime_name, benchmark_type) — a runtime can implement more than
# one benchmark type (e.g. faster-whisper's "batch" and "streaming" loops are
# different callables, same underlying runtime).
_ADAPTERS: dict[tuple[str, str], Callable] = {}


def register(runtime_name: str, benchmark_type: str = "batch") -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        key = (runtime_name, benchmark_type)
        if key in _ADAPTERS:
            raise ValueError(f"runtime adapter already registered: {key!r}")
        _ADAPTERS[key] = fn
        return fn

    return decorator


def get_adapter(runtime_name: str, benchmark_type: str = "batch") -> Callable:
    try:
        return _ADAPTERS[(runtime_name, benchmark_type)]
    except KeyError:
        raise ValueError(
            f"unknown runtime adapter: {runtime_name!r} for benchmark_type {benchmark_type!r}"
        ) from None


# Built-in adapters register themselves on import.
from . import faster_whisper  # noqa: E402,F401
