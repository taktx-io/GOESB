"""Character Error Rate (docs/specs/metrics.md: `cer`) — same as WER at
character granularity, over already-normalized text."""
from __future__ import annotations

from typing import Sequence

from ._align import edit_distance

METRIC_ID = "cer"
UNIT = "ratio"


def compute(pairs: Sequence[tuple[str, str]]) -> float:
    """Corpus-level CER: total character edit distance / total reference chars."""
    total_edits = 0
    total_ref_chars = 0
    for reference, hypothesis in pairs:
        total_edits += edit_distance(list(reference), list(hypothesis))
        total_ref_chars += len(reference)
    if total_ref_chars == 0:
        raise ValueError("at least one non-empty reference required to compute CER")
    return total_edits / total_ref_chars
