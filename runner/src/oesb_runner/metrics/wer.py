"""Word Error Rate (docs/specs/metrics.md: `wer`).

Operates on already-normalized text (profile normalization is applied
upstream, per-language, before this ever runs) — this module is
language-agnostic alignment only.
"""
from __future__ import annotations

from typing import Sequence

from ._align import edit_distance

METRIC_ID = "wer"
UNIT = "ratio"


def compute(pairs: Sequence[tuple[str, str]]) -> float:
    """Corpus-level WER: total edit distance / total reference word count.

    Aggregated over the whole pack (sum of edits / sum of ref words), not the
    mean of per-utterance ratios — averaging ratios biases short utterances.
    """
    total_edits = 0
    total_ref_words = 0
    for reference, hypothesis in pairs:
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        total_edits += edit_distance(ref_tokens, hyp_tokens)
        total_ref_words += len(ref_tokens)
    if total_ref_words == 0:
        raise ValueError("at least one non-empty reference required to compute WER")
    return total_edits / total_ref_words
