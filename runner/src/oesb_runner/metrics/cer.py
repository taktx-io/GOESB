"""Character Error Rate (docs/specs/metrics.md: `cer`) — same as WER at
character granularity, over already-normalized text."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ._align import edit_distance

METRIC_ID = "cer"
UNIT = "ratio"
SUBSTITUTIONS_METRIC_ID = "cer_substitutions"
DELETIONS_METRIC_ID = "cer_deletions"
INSERTIONS_METRIC_ID = "cer_insertions"
BREAKDOWN_UNIT = "count"


@dataclass(frozen=True)
class CerBreakdown:
    """Same shape as `wer.WerBreakdown`, at character granularity — see
    its own docstring for why `per_utterance` exists and is never
    averaged into `value`."""
    value: float
    substitutions: int
    deletions: int
    insertions: int
    ref_chars: int
    per_utterance: tuple[float, ...]


def compute(pairs: Sequence[tuple[str, str]]) -> float:
    """Corpus-level CER: total character edit distance / total reference chars."""
    return compute_detailed(pairs).value


def compute_detailed(pairs: Sequence[tuple[str, str]]) -> CerBreakdown:
    """Same corpus-level CER as `compute`, plus the substitution/deletion/
    insertion totals and the per-utterance ratio list — one alignment
    pass per pair, not two or three."""
    total_substitutions = total_deletions = total_insertions = 0
    total_ref_chars = 0
    per_utterance: list[float] = []
    for reference, hypothesis in pairs:
        counts = edit_distance(list(reference), list(hypothesis))
        total_substitutions += counts.substitutions
        total_deletions += counts.deletions
        total_insertions += counts.insertions
        total_ref_chars += len(reference)
        if reference:
            per_utterance.append(counts.total / len(reference))
    if total_ref_chars == 0:
        raise ValueError("at least one non-empty reference required to compute CER")
    total_edits = total_substitutions + total_deletions + total_insertions
    return CerBreakdown(
        value=total_edits / total_ref_chars,
        substitutions=total_substitutions,
        deletions=total_deletions,
        insertions=total_insertions,
        ref_chars=total_ref_chars,
        per_utterance=tuple(per_utterance),
    )
