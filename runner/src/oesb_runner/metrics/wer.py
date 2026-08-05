"""Word Error Rate (docs/specs/metrics.md: `wer`).

Operates on already-normalized text (profile normalization is applied
upstream, per-language, before this ever runs) — this module is
language-agnostic alignment only.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ._align import edit_distance

METRIC_ID = "wer"
UNIT = "ratio"
# One module producing multiple metric ids, same pattern cpu_ram.py uses
# for CPU_METRIC_ID/RAM_METRIC_ID -- a WER ratio alone can't distinguish
# "hears worse" from "runs away" (see WerBreakdown/edit_distance's own
# docstrings), so the breakdown is published as its own declared metrics,
# not silently folded into the `wer` value.
SUBSTITUTIONS_METRIC_ID = "wer_substitutions"
DELETIONS_METRIC_ID = "wer_deletions"
INSERTIONS_METRIC_ID = "wer_insertions"
BREAKDOWN_UNIT = "count"


@dataclass(frozen=True)
class WerBreakdown:
    """Corpus-level WER plus the substitution/deletion/insertion counts
    the alignment already computes — see `_align.edit_distance`'s own
    docstring for why a bare ratio isn't enough to diagnose a failure.

    `per_utterance` is the same alignment's *other* under-used output:
    one WER ratio per input pair, in input order — never averaged into
    `value` (that would reintroduce the short-utterance bias `compute`'s
    own docstring warns about), but real signal on its own. A corpus
    mean can look unremarkable while a handful of individual recordings
    sit at 60-90%+ — invisible in a mean, obvious as a p95 pooled over
    this list (see docs/specs/metrics.md "Reporting"). Utterances with
    an empty reference are excluded (WER is undefined with a zero
    denominator), not included as 0.0 or infinity."""
    value: float
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    per_utterance: tuple[float, ...]


def compute(pairs: Sequence[tuple[str, str]]) -> float:
    """Corpus-level WER: total edit distance / total reference word count.

    Aggregated over the whole pack (sum of edits / sum of ref words), not the
    mean of per-utterance ratios — averaging ratios biases short utterances.
    """
    return compute_detailed(pairs).value


def compute_detailed(pairs: Sequence[tuple[str, str]]) -> WerBreakdown:
    """Same corpus-level WER as `compute`, plus the substitution/deletion/
    insertion totals and the per-utterance ratio list — one alignment
    pass per pair, not two or three."""
    total_substitutions = total_deletions = total_insertions = 0
    total_ref_words = 0
    per_utterance: list[float] = []
    for reference, hypothesis in pairs:
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        counts = edit_distance(ref_tokens, hyp_tokens)
        total_substitutions += counts.substitutions
        total_deletions += counts.deletions
        total_insertions += counts.insertions
        total_ref_words += len(ref_tokens)
        if ref_tokens:
            per_utterance.append(counts.total / len(ref_tokens))
    if total_ref_words == 0:
        raise ValueError("at least one non-empty reference required to compute WER")
    total_edits = total_substitutions + total_deletions + total_insertions
    return WerBreakdown(
        value=total_edits / total_ref_words,
        substitutions=total_substitutions,
        deletions=total_deletions,
        insertions=total_insertions,
        ref_words=total_ref_words,
        per_utterance=tuple(per_utterance),
    )
