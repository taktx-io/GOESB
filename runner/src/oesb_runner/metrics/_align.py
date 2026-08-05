"""Shared edit-distance alignment used by WER (word) and CER (character)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EditCounts:
    """Substitution/deletion/insertion breakdown of a minimum-edit
    alignment — `deletions` is a reference item missing from the
    hypothesis, `insertions` is an extra hypothesis item not in the
    reference, same convention every WER tool (sclite, jiwer, ...) uses."""
    substitutions: int
    deletions: int
    insertions: int

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> EditCounts:
    """Minimum-edit alignment between hypothesis and reference, broken down
    into substitutions/deletions/insertions, not just the total.

    A WER/CER ratio alone can't distinguish "hears worse" (substitutions/
    deletions rising) from "runs away" (insertions rising) — two different
    failure modes with two different fixes (real report: a benchmark
    harness bug inflated one engine's insertions ~3x while its
    substitutions and deletions both *improved* over the comparison
    engine — WER alone made it look like the straightforwardly worse
    model, when the actual diagnosis was "hears fine, hallucinates").

    Full `(n+1) x (m+1)` DP table (not the previous rolling-two-rows
    version) since backtrace needs the whole table — real utterance/
    character sequences here are short enough (words per utterance,
    chars per utterance) that this is not a meaningful cost. Backtrace
    prefers the diagonal (match/substitution) over deletion/insertion
    when multiple minimal-cost paths exist, matching the standard
    WER-tool tie-breaking convention.
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost,  # substitution / match
            )

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1):
            if reference[i - 1] != hypothesis[j - 1]:
                substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return EditCounts(substitutions=substitutions, deletions=deletions, insertions=insertions)
