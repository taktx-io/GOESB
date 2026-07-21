"""Shared edit-distance alignment used by WER (word) and CER (character)."""
from __future__ import annotations

from typing import Sequence


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    """Minimum substitutions+deletions+insertions to turn hypothesis into reference."""
    n, m = len(reference), len(hypothesis)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = curr
    return prev[m]
