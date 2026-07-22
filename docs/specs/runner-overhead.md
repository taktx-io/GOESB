# Runner overhead characterization

[ADR-0002](../adr/0002-tech-stack.md) commits to characterising the Python
runner's own measurement overhead by M2, so it's known whether the harness
itself biases RTF/latency numbers rather than purely reflecting the runtime
under test. This is that characterization.

## What could actually bias a metric

Each adapter times only its own model call with `time.perf_counter()`
(e.g. `faster_whisper.run_batch`'s `start = time.perf_counter(); ...
elapsed = time.perf_counter() - start`), immediately around just the work
being measured. Nothing about pack loading, schema validation, or CLI
argument parsing sits inside that window, so none of that can bias a
result — it's fixed one-off setup cost, not per-utterance measurement noise.

The one thing that genuinely runs *concurrently* with the timed work is
`cli.py`'s `_sample_during()` background thread — it wakes on a fixed
interval to sample CPU/RAM/temperature while the adapter call is in flight.
A background thread contending for the GIL could, in principle, measurably
slow down the foreground work it's timing.

## Method

`runner/tests/test_overhead.py`: a fixed, real (not `sleep`-based) CPU
workload (~400ms of pure-Python arithmetic on this machine), self-timed
exactly like an adapter times a `transcribe()` call, run 9 times bare and 9
times wrapped in `_sample_during()` at its production-default sampling
interval (0.2s — the interval every real `cli.py run` call site actually
uses). Median (not mean) is compared, since a single OS-scheduler jitter
event in either group can swing a short workload's mean by 2x — matching
the same reasoning that leads streaming latency to report p50/p95 rather
than mean (docs/specs/metrics.md "Reporting").

## Result (measured on this machine: Apple M1 Pro, macOS, arm64)

At the production default (`interval_s=0.2`), overhead is **noise-level** —
repeated trials straddle zero (e.g. -12%, +2%, -7%, +1.5% across separate
runs), meaning the sampler thread's cost is indistinguishable from ordinary
run-to-run measurement jitter at this sampling cadence.

A deliberately more aggressive interval (`interval_s=0.05`, 4x more frequent
wakeups than production ever uses) does produce a measurable, consistent
~7-11% overhead — informative as an upper bound if a future need arises for
finer-grained sampling (e.g. higher-resolution power curves), but not
representative of any real run today.

## Verdict

**Not material at production settings.** No RTF/latency correction is
applied. `test_overhead.py`'s regression guard (generous <20% bound, since
noise straddles zero) exists to catch the sampler becoming meaningfully more
expensive in the future — e.g. if a probe added to the sampler loop turns
out to be slow (a hwmon/RAPL read is a cheap sysfs `read()`, but a future
probe might not be) — not to pin an exact percentage.

## Caveats

- Measured on one machine (Apple Silicon, macOS). Not yet re-run on Linux or
  Windows, or with the sampler thread additionally reading real RAPL/hwmon
  sysfs files under load (this measurement predates those probes actually
  finding hardware to read on this machine — see
  [environment-capture.md](environment-capture.md)'s own roadmap note on the
  same gap). Revisit if cross-platform CI or real Linux hardware surfaces a
  different picture.
- This characterizes the *sampler thread's* cost specifically, not Python's
  general interpreter overhead relative to a natively-compiled harness
  (ADR-0002's "Rust runner" alternative) — that broader question stays open,
  revisited only if this or a future measurement shows it's material.
