"""Runtime adapter registry (FR-11.1): how a model is driven for a benchmark
type. Each adapter is reviewed, in-tree code (ADR-0004) — never a plugin
supplied at run time. Core dispatch here is runtime-agnostic; adapter modules
lazy-import their actual ML dependency so importing this package never
requires it to be installed.
"""
from __future__ import annotations

from typing import Callable

_ADAPTERS: dict[str, Callable] = {}


def register(runtime_name: str) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        if runtime_name in _ADAPTERS:
            raise ValueError(f"runtime adapter already registered: {runtime_name!r}")
        _ADAPTERS[runtime_name] = fn
        return fn

    return decorator


def get_adapter(runtime_name: str) -> Callable:
    try:
        return _ADAPTERS[runtime_name]
    except KeyError:
        raise ValueError(f"unknown runtime adapter: {runtime_name!r}") from None


# Built-in adapters register themselves on import.
from . import faster_whisper  # noqa: E402,F401
