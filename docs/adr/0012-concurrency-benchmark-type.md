# ADR-0012 — `concurrency` benchmark_type: load, not language

- **Status:** Accepted (Eric, 2026-07-31).
- **Date:** 2026-07-31
- **Builds on / relates to:** [ADR-0008](0008-explicit-compute-backend.md)
  (explicit `--backend`, the same primitive this reuses), [ADR-0009](0009-parameterized-profile-configuration.md)
  (`--param key=value` sweeps — concurrency levels reuse this verbatim, no new
  sweep mechanism), [ADR-0011](0011-decouple-packs-from-profiles.md) (language-based
  pack eligibility — this ADR adds the one sanctioned exception to it).

## Context

Comparing an RTX A4000 against an RTX 4090 on the live leaderboard showed
near-identical RTF at every model size, with no way to tell whether that meant
the two cards are genuinely equivalent or whether the benchmark simply wasn't
stressing the GPU enough to show a difference. It's the latter: every existing
benchmark_type (`batch`, `streaming`) processes one utterance at a time,
sequentially — small-batch Whisper decoding at that concurrency is latency-bound
(per-step kernel-launch/host-round-trip overhead), not GPU-compute-bound, so a
5-6x more powerful card doesn't show up in RTF at all. The real question — does
this hardware stay fast when several requests hit it simultaneously — has no
benchmark to answer it, and the runner harness has no concurrent-request code
path anywhere to build one on top of.

Confirmed (against the installed library, not assumed): ctranslate2/faster-whisper
already supports genuine concurrent inference. `WhisperModel(num_workers=N)` maps
to ctranslate2's `Translator(inter_threads=N)` — a real internal thread-pool of N
execution slots, with `max_queued_batches` providing real backpressure once all
slots are busy. GOESB's adapter never set `num_workers` (implicitly 1).

Also confirmed useful but separately unimplemented: `docs/specs/metrics.md`
documented `gpu_pct`/`throughput`/`watt_per_stream`/`eur_per_stream` from early
on, none with an actual metric plugin — this benchmark type was anticipated in
the spec well before it existed.

## Decision

**A new `benchmark_type: "concurrency"` measures performance under N
simultaneous requests. It never scores transcription accuracy — no `wer`/`cer`,
no `language`, no `normalization` block — only timing/resource metrics under
load.**

### Runner

- `benchmark-profile.schema.json`: `benchmark_type` enum gains `"concurrency"`;
  `configuration` gains `concurrency`/`duration_s` (both already usable via the
  existing, un-`additionalProperties`-restricted `configuration` block, but
  declared explicitly per the file's own convention).
- `faster_whisper.py`: `_load_model` gains `num_workers` (default 1, unchanged
  behavior for batch/streaming); new `run_concurrency` spins up
  `ThreadPoolExecutor(max_workers=concurrency)` against one shared, already-loaded
  model with `num_workers=concurrency` — the two must match exactly, or
  ctranslate2's own `inter_threads`/`max_queued_batches` would either sit
  partially idle or throttle extra threads, understating what the hardware can
  actually sustain. Each worker round-robins through the pack's utterances for a
  **fixed wall-clock `duration_s`**, not a fixed call count — comparing
  concurrency=1 against concurrency=16 needs the same total measurement window
  at every level for throughput numbers to be comparable, the same reason
  load-testing tools (wrk, k6) use fixed-duration windows. No warm-up discard:
  model load already happens before the timed window starts, matching batch's
  own "load time excluded" convention; per-call warm-up inside ctranslate2 is a
  smaller effect a multi-second window with many calls should average out on
  its own — a refinement to add only if real measurement shows otherwise, not
  ahead of data.
- `cli.py`: new `elif benchmark_type == "concurrency":` branch in `run()`.
  `real_time_factor` is **pooled as a per-call distribution** (p50/p95 always
  attached) for this benchmark_type specifically — same metric id batch/streaming
  already report, but as a single corpus-aggregate scalar there; concurrency's
  whole point is showing whether *individual* requests degrade under load, which
  a mean alone would hide. Reuses the exact pooling machinery streaming's own
  latency metrics (`first_partial_latency` etc.) already use, gated on
  `benchmark_type == "concurrency"` rather than a fixed metric-id set. New
  `throughput` metric (`docs/specs/metrics.md`'s already-documented, previously
  unimplemented id) computed as total audio-seconds processed across every
  worker divided by `duration_s`.
- **The ADR-0011 exception**: a concurrency profile declares no `language` at
  all, but still resolves a real, hash-verified, already-public pack through
  the ordinary `load_pack()` path — the ADR-0011 language-equality gate is
  skipped only for `benchmark_type == "concurrency"`, not bypassed structurally.
  This was the one real design fork:
  - **Bypass packs entirely** (bundle a fixed audio fixture inside the runner
    package, skip `load_pack`/pack.yaml/the gate) would need three coordinated
    breaking changes: `benchmark-result.schema.json`'s `pack` is in the
    document's top-level `required` list with its own required sub-fields (no
    null escape anywhere), the platform DB's `pack_id` column is `NOT NULL`,
    and `ingest.py` reads `body["pack"]` unconditionally in two places.
  - **Reuse a real pack, skip only the language check** (chosen): the submitted
    result has an ordinary, already-nullable-`language`/non-null-`pack_id`
    shape identical to what the platform already accepts — zero schema
    migration, zero ingest.py change, and `load_pack`'s hash-integrity
    verification comes for free instead of needing its own mechanism.

  Any open pack works equally well as filler audio (content is irrelevant, only
  its presence/duration matters) — the wizard picks one deterministically
  (`librispeech-en` when present, otherwise the alphabetically-first open pack)
  instead of asking the user to choose something with no effect on the result.

- Wizard: `_wizard_run_concurrency()` is a new, separate flow — not a variant of
  `_wizard_run()`'s language×engine matrix picker, which cannot handle a
  language-less profile at all (`profiles_by_id[profile_id]["language"]` is a
  plain dict-index there, not `.get`). Picks profile(s) → auto-picks a pack →
  shares `_wizard_confirm_and_run` (backend/hardware picking, the
  `_wizard_engine_parameters` sweep step, confirm, `_reexec` queue) with
  `_wizard_run`, factored out since that tail was previously duplicated
  line-for-line between the two flows.

### Platform

No migration. `language` was already nullable end-to-end; `pack_id` stays
required and a concurrency result always has one (see above); `benchmark_type`
is an unconstrained string column everywhere except `routes/answers.py`'s
`Literal["batch", "streaming"]` query-param type, widened to include
`"concurrency"` as cheap insurance (the answers/single-recommendation feature
has no reason to actually be queried with it). Metrics are stored/returned as a
fully generic JSONB passthrough — `throughput`/`gpu_pct` need no schema change
to appear.

## Non-goals / explicit constraints

- **Not every engine.** Only `faster-whisper` implements `run_concurrency` for
  now — vosk's pattern (shared `Model`, one `KaldiRecognizer` per stream) and
  whisper.cpp/pywhispercpp's concurrency contract are different and unverified;
  extending this to them is separate future work, not assumed to be a drop-in.
- **Not a web chart yet.** The leaderboard UI for viewing a concurrency sweep
  (RTF/throughput vs concurrency level, per hardware) is tracked separately —
  this ADR covers the runner/platform data model, not its presentation.
- **Not touching `_wizard_pick_backends`'s pre-existing `"batch"`-hardcoded
  readiness probe.** It happens to give the right answer for faster-whisper
  today (its declared backend set is identical across `batch`/`streaming`/
  `concurrency`), but this is pre-existing behavior this ADR doesn't change or
  rely on being correct in general.

## Consequences

- A benchmark result can now legitimately have no `language` — any code that
  assumed every result belongs to a language (dashboards, exports, ad hoc
  queries) should treat `language IS NULL` as "not applicable," not "unknown."
- `real_time_factor` now has two different aggregation semantics depending on
  `benchmark_type` — a mean for batch/streaming, a pooled p50/p95 distribution
  for concurrency. Anything reading this metric id generically (the web table
  already does, via a fully generic `Object.entries(metrics)`) keeps working
  either way, but anything that assumes "one scalar per result" specifically
  for this id needs to check `benchmark_type` first.
- Establishes the pattern for any future non-accuracy-scored benchmark
  dimension: reuse packs/profiles/results as-is, add a narrow, named exception
  to whichever accuracy-specific gate doesn't apply, rather than building a
  parallel data path.

## Addendum (2026-07-31): whisper-cpp/metal support, more sizes, wizard nudge

Extended to a second engine and more model sizes:

- **`profiles/whisper-{tiny,base,small,large-v3}-concurrency`** join the
  existing `whisper-medium-concurrency` (faster-whisper, cpu/cuda) — same
  minimal-diff pattern batch profiles already use across sizes (only `id`,
  `title`, `model.name` differ).
- **`profiles/whispercpp-{tiny,base,small,medium,large-v3}-concurrency`**
  (whisper-cpp, cpu/cuda/metal — this engine's own already-declared backend
  set) are new. This needed a genuinely different `run_concurrency` than
  faster-whisper's, for a reason confirmed against the actual bound C API,
  not assumed: **pywhispercpp is not safe to share one `Model` instance
  across concurrent threads.** `Model.transcribe()` mutates shared instance
  state (`self._params`, its callback bindings) on every call, and
  pywhispercpp never binds `whisper_init_state`/`whisper_full_with_state` —
  the per-thread-state split whisper.cpp's own C API actually has, the
  mechanism that would otherwise let one set of loaded weights serve many
  threads (the same shape as vosk's shared-`Model`-plus-per-thread-
  `KaldiRecognizer` pattern, or ctranslate2's `inter_threads`). Without that
  split, `whisper_cpp.run_concurrency` builds `concurrency` full, independent
  `Model` instances up front (one per worker), instead of one shared model
  with a worker-count knob — a real N-way memory cost (full ggml weights
  loaded N times), which is why the whisper-cpp `*-concurrency` profiles
  declare a tighter `overridable.concurrency.range.max` (16) than
  faster-whisper's (64): the same nominal concurrency level is a much
  heavier ask on this engine, and the ceiling says so.
- **Wizard nudge, not a default change.** `configuration.concurrency: 1`
  stays the correct schema default (matches `num_workers=1`'s own default,
  matches every other parameter's "Enter = today's behavior" convention) —
  but Enter-through on every other parameter means "run once, at the
  default," which is uniquely unhelpful for `concurrency` specifically: a
  single point shows nothing about load behavior, the entire reason this
  benchmark_type exists. `_wizard_engine_parameters` now special-cases just
  this one parameter: blank input expands to a suggested sweep (`1,4,8,16`,
  clamped to that profile's own range — e.g. `1,4,8` for whisper-cpp),
  stated in the prompt text itself so nothing happens that isn't visible
  before it happens.
