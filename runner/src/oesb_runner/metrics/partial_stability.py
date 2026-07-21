"""Partial Stability (docs/specs/metrics.md: `partial_stability`)."""
from __future__ import annotations

from typing import Sequence

from ..streaming import StreamTrace

METRIC_ID = "partial_stability"
UNIT = "ratio"


def compute(traces: Sequence[StreamTrace]) -> float:
    """Corpus-level fraction of partial-hypothesis words that survive
    unchanged into the next update ("flicker"). For each consecutive pair of
    updates, words in the previous hypothesis beyond the common prefix with
    the new one are counted as rewritten; 1.0 - rewritten/emitted, pooled over
    the whole pack (like WER, not averaged per-utterance).
    """
    total_words_emitted = 0
    total_rewritten = 0
    for trace in traces:
        previous_words: list[str] = []
        for update in trace.updates:
            words = update.text.split()
            common_prefix = 0
            for a, b in zip(previous_words, words):
                if a != b:
                    break
                common_prefix += 1
            total_rewritten += len(previous_words) - common_prefix
            total_words_emitted += len(words)
            previous_words = words
    if total_words_emitted == 0:
        raise ValueError("no partial hypotheses to compute stability from")
    return max(0.0, min(1.0, 1.0 - (total_rewritten / total_words_emitted)))
