"""Repeat-run statistics (FR-5.3): spread must be surfaced, never hidden."""
from __future__ import annotations


def _percentile(sorted_values: list[float], p: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * p
    f, c = int(k), min(int(k) + 1, n - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summarize(values: list[float]) -> dict[str, float]:
    """Mean plus a spread block: std/min/max/p50/p95, per benchmark-result.schema.json."""
    if not values:
        raise ValueError("at least one value required")
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    values_sorted = sorted(values)
    return {
        "value": mean,
        "std": variance ** 0.5,
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "p50": _percentile(values_sorted, 0.5),
        "p95": _percentile(values_sorted, 0.95),
    }


def relative_std(summary: dict[str, float]) -> float:
    """std / |mean|, used to flag drift against a documented tolerance (FR-5.3)."""
    if summary["value"] == 0:
        return 0.0 if summary["std"] == 0 else float("inf")
    return summary["std"] / abs(summary["value"])
