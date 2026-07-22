"""ADR-0002's M2 charge: characterise whether the runner's own measurement
harness biases the per-utterance timing RTF/latency are built from.

The thing that could actually bias a metric isn't Python function-call
overhead around the timed work (each adapter's own `time.perf_counter()`
call sits immediately around just the model call, unaffected by anything
outside it) — it's whether `_sample_during`'s background CPU/RAM/temperature
sampler thread measurably slows down the *foreground* work it wraps (GIL
contention from a thread waking every `interval_s` to call psutil/sysfs
reads). This measures exactly that: a fixed, real (not `sleep`-based) CPU
workload, timed by its own internal `perf_counter()` exactly like an
adapter times a transcribe() call, run with vs without the sampler thread
active. Real and provable on any machine — not Linux/RAPL-dependent, unlike
the energy/thermal probes themselves.

See docs/specs/runner-overhead.md for the measured verdict from this run.
"""
from __future__ import annotations

import statistics
import time

from oesb_runner.cli import _sample_during

# Enough busywork (~400ms on a modern machine) that OS scheduling jitter —
# which is roughly fixed-cost, not proportional to workload size — doesn't
# dominate the comparison. A shorter workload made this test flaky: a single
# scheduler hiccup in one bare-run rep could swing the mean by 2x.
_N_ITERATIONS = 10_000_000
_REPEATS = 9


def _busywork() -> float:
    """A fixed amount of real CPU work, self-timed exactly like an adapter
    times its own transcribe() call — this return value is what RTF/latency
    are actually computed from in production, so it's the right thing to
    compare bare vs sampled."""
    start = time.perf_counter()
    total = 0
    for i in range(_N_ITERATIONS):
        total += i * i
    _ = total  # keep the loop from being optimized away by intent, not effect
    return time.perf_counter() - start


def test_sampler_thread_overhead_is_small_relative_to_workload():
    bare_times = [_busywork() for _ in range(_REPEATS)]
    sampled_times = []
    for _ in range(_REPEATS):
        # Production default interval (cli.py's `_sample_during(fn)` call
        # sites never override it) — this is the overhead a real run
        # actually carries, not a worst-case finer-grained hypothetical.
        elapsed, _samples, _temp, _rapl = _sample_during(_busywork)
        sampled_times.append(elapsed)

    # Median, not mean: robust to the occasional scheduler-jitter outlier
    # (see the comment on _N_ITERATIONS) — the same reasoning as p50 being
    # reported for latency metrics rather than a noise-sensitive mean.
    bare_median = statistics.median(bare_times)
    sampled_median = statistics.median(sampled_times)
    overhead_pct = (sampled_median - bare_median) / bare_median * 100

    print(
        f"\nrunner overhead: bare={bare_median * 1000:.2f}ms "
        f"sampled={sampled_median * 1000:.2f}ms overhead={overhead_pct:+.2f}%"
    )

    # At the production default interval, measured overhead is noise-level
    # (straddles zero across repeated trials — see
    # docs/specs/runner-overhead.md) — this generous bound exists to catch a
    # regression that makes the sampler meaningfully expensive, not to pin
    # an exact %.
    assert overhead_pct < 20.0
