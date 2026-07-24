"""Per-language normalization ruleset registry (FR-2.6, FR-6.1).

The core stays language-agnostic: this module only defines the plugin
interface (`register`/`normalize`) and dispatch. All language-specific text
handling lives in ruleset modules like `oesb_nl_v1`, never here.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Normalizer(Protocol):
    def __call__(self, text: str, **options: object) -> str: ...


_REGISTRY: dict[str, Normalizer] = {}


def register(ruleset_id: str) -> Callable[[Normalizer], Normalizer]:
    """Class/function decorator registering a normalizer under `ruleset_id`."""

    def decorator(fn: Normalizer) -> Normalizer:
        if ruleset_id in _REGISTRY:
            raise ValueError(f"ruleset_id already registered: {ruleset_id!r}")
        _REGISTRY[ruleset_id] = fn
        return fn

    return decorator


def get_normalizer(ruleset_id: str) -> Normalizer:
    try:
        return _REGISTRY[ruleset_id]
    except KeyError:
        raise ValueError(f"unknown normalization ruleset_id: {ruleset_id!r}") from None


def normalize(ruleset_id: str, text: str, **options: object) -> str:
    return get_normalizer(ruleset_id)(text, **options)


# Built-in rulesets register themselves on import.
from . import oesb_en_v1, oesb_nl_v1  # noqa: F401
