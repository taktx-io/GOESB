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
  this one parameter, stated in the prompt text itself so nothing happens
  that isn't visible before it happens. See the 2026-08-01 addendum below
  for what blank input actually does now — superseding the static
  `1,4,8,16` sweep this paragraph originally described.

## Addendum (2026-08-01): auto-detecting the useful max, instead of guessing it

The static `1,4,8,16` sweep from the previous addendum was still a guess —
picked to look reasonable across engines, not measured against any real
hardware. Two problems: it can undershoot (real capacity is past 16, the
sweep never finds the knee) or overshoot (wastes a run well past the point
throughput stopped improving, and for whisper-cpp specifically, each extra
level is a full additional model load per ADR-0012's own note above). The
knee is exactly what the whole `metric vs concurrency` web chart (Phase 3)
displays — no reason to leave finding it to a human's static guess when the
runner already measures throughput at every level it runs.

- **`_run_concurrency_auto_sweep` is now the wizard's default** for the
  `concurrency` parameter — blank Enter resolves to an `"auto"` sentinel
  (not a real `--param` value; `_resolve_one_param` never sees it) that
  `_wizard_confirm_and_run` expands at *run time* instead of upfront: run
  concurrency=1, read back that run's own `throughput` metric from the
  result JSON `run` just wrote to `runs/results/`, double to 2, compare —
  if the doubling bought less than 15% more throughput, the previous level
  was the knee and the sweep stops there; otherwise keep doubling. Still
  clamped to the profile's own `overridable.concurrency.range.max` as a
  hard ceiling regardless of how the throughput curve looks (whisper-cpp's
  16 vs faster-whisper's 64) — the auto-sweep finds the knee *within* that
  ceiling, it doesn't replace it.
- **15% is a floor, not a tuned constant** — comfortably above normal
  run-to-run measurement noise (FR-5.3's own tolerance check already flags
  >15% relative std as noteworthy) but well under a real scaling step
  (throughput roughly doubles with concurrency until contention sets in).
  Revisit if it proves too twitchy in practice; not worth a config knob
  until it does.
- **Typing your own comma list still works and always has priority** — the
  prompt says so explicitly (`"Enter to auto-detect the useful max, or your
  own comma list e.g. 1,4,8,16"`). A manual sweep skips the auto-sweep path
  entirely and behaves exactly as before (upfront expansion via
  `_parse_param_sweep`, one `_reexec` per listed value).
- **Every level the auto-sweep explores is still a normal, independently
  signed `run` result** — no new schema, no new submission path. Only the
  *set* of concurrency values to run is decided adaptively; each one still
  goes through the exact same `goesb run ... --param concurrency=N` path a
  manually-typed sweep would.
- **Not implemented: a pre-run OOM safety check.** A separate, narrower
  problem from "what's the useful max" — whether a *requested* level would
  exceed available RAM/VRAM before even starting it (relevant mainly to
  whisper-cpp's N-full-model-instances harness). Flagged during design as
  worth adding alongside this, deliberately deferred — the auto-sweep's own
  doubling-from-1 approach means a runaway level is reached gradually, not
  requested outright, which is a real (if softer) mitigation until a
  dedicated check exists.

## Addendum (2026-08-01): vosk concurrency support — a third, different thread-safety answer

Extended to a third engine, following the same "verify, don't assume"
discipline the whisper-cpp addendum above established — and landing on a
third distinct answer, after faster-whisper's "safe to share" and
whisper-cpp's "confirmed unsafe to share":

- **`profiles/vosk-{small,medium}-concurrency`** are new (2 profiles, not
  split per language) — concurrency measures hardware capacity, not
  transcription accuracy, so which language's model happens to back it is
  irrelevant to the answer; these follow the exact same naming shape
  `whisper-medium-concurrency` already uses, no language segment. A vosk
  *model* is itself language-specific (unlike whisper's one multilingual
  model), so each profile still has to pick one concrete model to
  actually load — English, arbitrarily, matching this codebase's existing
  "prefer librispeech-en when present" default elsewhere
  (`_wizard_run_concurrency`'s own filler-pack choice). An earlier version
  of this addendum shipped 12 profiles (one per language, matching vosk's
  batch profiles) before this was caught and corrected — the batch
  profiles genuinely need a language axis (that's what they're measuring),
  concurrency profiles never did.
- **`profiles/vosk-medium-{lang}-batch`** are also new (6 profiles) — this
  codebase had only ever wired up vosk's small (~40-50MB) model tier into
  `_MODEL_URLS`; Vosk itself has always published a larger, more accurate
  model per language too (confirmed against alphacephei.com/vosk/models
  directly: en-us 1.8GB/WER 5.69 vs small's 40MB/WER 9.85, and the other
  five languages each have a comparable larger tier). `_MATRIX_COLUMNS`
  gains `("vosk", "medium")` so the wizard's batch matrix shows both.
- **vosk's `run_concurrency` builds one full `vosk.Model` instance per
  worker, not a shared model** — the same shape whisper-cpp's addendum
  landed on, but for a different, still-real reason: alphacep/vosk-api#606
  documents a SIGSEGV crash from creating `KaldiRecognizer` instances in
  parallel threads against one shared `Model`, traced to a race in the
  model's own C++ reference counting (concurrent `UnRef()` calls). The
  Python binding adds no locking around construction/destruction to guard
  against it. Vosk's own reference server implementation *does* use one
  shared Model with one Recognizer per connection/thread — strongly
  suggesting the pattern is intended to be safe and the 2021 bug was
  meant to be fixed — but that couldn't be independently confirmed from
  the Python layer or the (closed, resolution-unclear) issue thread alone.
  Given a real, specific crash report and no way to verify it's actually
  resolved, this takes the same stance as whisper-cpp: don't share what
  can't be confirmed safe to share. Cheaper here than it was for
  whisper-cpp, though — vosk's models are small enough (small tier
  40-50MB; the larger per-language tier tops out around 1-2GB) that N
  full instances is a bounded cost, not a forced-into-a-tiny-ceiling one —
  reflected in `range.max`: 32 for the small tier, 8 for the larger one
  (tighter, matching the same "heavier per-instance memory cost" logic
  whisper-cpp's own tighter ceiling already established).

## Addendum (2026-08-01): auto-sweep robustness — two-strike stop, doubling-then-linear steps

Two refinements to the auto-sweep from the earlier same-day addendum,
prompted by looking at real sweep data collected on real hardware this
session:

- **Two consecutive low-gain levels required to stop, not one.** A single
  `duration_s` window's throughput reading carries real run-to-run
  measurement noise — stopping on the very first doubling that comes in
  under `_CONCURRENCY_PLATEAU_GAIN` risked calling a false plateau from a
  single noisy reading rather than the hardware's actual ceiling. A level
  that recovers back above the gain floor resets the streak (it wasn't a
  real plateau, just a dip); only two low-gain levels *in a row* stop the
  sweep now. Costs one extra level on every sweep, including the clean,
  non-noisy ones — accepted, since a false-early-stop undermines the
  entire point of the feature more than one extra `duration_s` window
  costs.
- **Doubling up to concurrency=8, then +4 a level, not doubling forever.**
  Pure doubling is cheap for the uninteresting low end (1, 2, 4, 8 in just
  four levels) but by the time levels are large enough to be near a real
  knee, doubling's own step has gotten coarse exactly where resolution
  matters most — 8 to 16 is already a +100% jump, easily hiding a real
  ceiling at 10, 11, or 12 between them. `_next_concurrency_level` keeps
  doubling below 8, then switches to a flat +4 step at and above it:
  1, 2, 4, 8, 12, 16, 20, 24, .... Trades away doubling's logarithmic
  reach on a hypothetical very-high-ceiling machine for real resolution
  near the plateau on this tool's actual target hardware — every real
  sweep run collected so far this session plateaued in the single-to-
  low-double-digit range, never anywhere near needing doubling's reach
  past 24. A hard ceiling (`range.max`) still bounds the worst case
  regardless.
