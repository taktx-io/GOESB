# Spec: parameterized profile configuration

**Status:** Resolved design — decided in design discussion 2026-07-26, recorded as
ADR-0009 (`docs/adr/0009-parameterized-profile-configuration.md`). This spec is the
implementation-level detail behind that ADR. Supersedes the earlier proposal version of
this document (the open question in its §4 is resolved here). §6 (parameter catalog) and
§5's UI design were added after the initial resolution; see also the runner handoff
addendum (`docs/handoffs/2026-07-26-adr-0009-runner-addendum.md`).

## 1. Problem

A **profile** (`profiles/<id>/profile.yaml`) is the immutable, git-reviewed definition of
exactly how a benchmark runs. That immutability is what makes a leaderboard row
trustworthy — but it also means that sweeping a *configuration* parameter (e.g.
`beam_size` at 1, 4, 8) requires minting a distinct profile id per value: a new
git-reviewed file, pack pairing, and matrix column for every point in the sweep. One
workload with one varying dial should not need five near-duplicate profile files.

ADR-0008 already solved this exact tension for a different axis (compute backend):
backend is a runner flag, recorded verbatim in the signed result, faceted on the
platform — not a profile-level distinction. This spec generalizes that pattern to
configuration parameters.

## 2. Profile: declaring which dials may be turned

A profile may declare a bounded, reviewed set of **override-eligible** parameters via a
new top-level `overridable` block in profile.yaml:

```yaml
model:
  name: whisper-base
  beam_size: 5        # stays the single source of truth for the default
  vad: true
overridable:
  beam_size:
    allowed: [1, 2, 4, 5, 8]     # explicit domain…
  vad: {}                        # …or implicit (boolean: both values)
```

Rules:

- Every `overridable` key must reference an existing key in the profile's `model` or
  `configuration` block. **The default is the value already in the profile body** — it is
  not repeated inside `overridable`, so there is exactly one source of truth.
- Each entry carries a bounded domain: `allowed` (enum list) or `range: {min, max}`.
  Booleans may omit the domain. Free-form/unbounded parameters are not permitted — the
  domain is part of what review approves.
- **No silent knobs:** a profile may declare a parameter overridable only if its adapter
  *demonstrably applies* it. Several adapters accept kwargs purely for call-shape parity
  and ignore them (whisper-cpp: `beam_size`, `vad`, `quantization`; vosk: all of them —
  see their docstrings). Declaring one of those would produce a signed result asserting
  a value that had no effect on the run — strictly worse than not having the feature.
  Each adapter therefore declares its set of **applied parameters** (a registry constant
  next to the adapter, reviewed with it), and profile validation enforces
  `overridable ⊆ applied` for the profile's `runtime.name`. `--param` targeting an
  unapplied parameter is a hard error even if a profile were to mis-declare it.
- Adding, removing, or widening an `overridable` entry changes what a leaderboard row
  under this profile id can mean, so it is a **profile version bump with a changelog
  entry**, reviewed like any other profile change (ADR-0004 spirit: review-gated, not a
  free-form escape hatch).
- Layering, mirroring ADR-0008's adapter/profile split exactly: the **adapter** owns the
  parameter universe and its validation (what `beam_size` means for this engine, type
  checking, hard limits — reviewed adapter code); the **profile** owns eligibility and
  the domain for this workload. Requesting a parameter the profile doesn't declare, or a
  value outside its domain, is a hard error — never a silent fallback or clamp.

Schema change: `benchmark-profile.schema.json` gains the optional `overridable` object
(keys: parameter names; values: `{allowed: [...]}` | `{range: {min, max}}` | `{}`).

## 3. Runner: `--param`, and what gets signed

- `goesb run <profile> <pack> --param beam_size=8` (repeatable per parameter) overrides
  the profile default **for that run only**. Validation happens before anything runs:
  key declared overridable by the profile, value inside the declared domain, adapter
  applies it (§2 "no silent knobs").
- The signed result document gains a `parameters` object recording the **resolved value
  of every override-eligible parameter — including untouched defaults** — plus the
  profile default it resolved against:

```json
"parameters": {
  "beam_size": { "value": 8, "default": 5 },
  "vad":       { "value": true, "default": true }
}
```

  Recording defaults too keeps every result self-describing: grouping and badge
  rendering never require a lookup into the profile catalog, and results remain
  interpretable across later profile versions. This is a versioned result-schema change
  (`additionalProperties: false`); it should ride the same schema bump that adds
  `runtime.backend` (ADR-0008) if that is still pending.
- Provenance needs no new mechanism: `config_sha256` already hashes the fully-resolved
  configuration ("profile.configuration + any explicit run overrides" — its existing
  description), and any post-hoc edit to `parameters` breaks `payload_sha256`/signature
  verification.

Comparability therefore shifts from "same `profile_id` ⇒ same method" to "same
`profile_id` + same parameter values ⇒ same method" — the same grouping discipline
backend already requires.

## 4. Wizard: per-engine parameter step (resolves the former open question)

Between hardware selection/engine preflight and the repeats prompt, the batch wizard
gains one parameter step, **grouped by engine**:

- The selected matrix cells are grouped by engine. For each engine whose selected
  profiles declare overridable parameters, the wizard prompts once per parameter, with
  the profile default pre-filled. (Within an engine, prompt only for parameters eligible
  in *all* of that engine's selected cells.) An engine with nothing overridable — e.g.
  vosk — is not prompted for at all.
- **Enter accepts the default.** Pressing Enter through every prompt reproduces today's
  behavior exactly: defaults everywhere, zero added friction.
- **A single value overrides** that parameter for all of that engine's selected cells.
- **A comma-separated list sweeps**: `1,4,8` expands each of that engine's cells into
  one run per value. The prompt's pre-filled default doubles as the hint to include the
  baseline (sweeping `1,4,8` when the default is 5 disconnects the sweep from existing
  leaderboard rows; sweeping `1,4,5,8` keeps the anchor).
- Values are validated against each profile's declared domain **up front**, alongside
  `_preflight_engines` — a bad value must fail before run 1, not at run 12 of 15.
- Engine-specific parameters never leak: a mixed whisper + vosk selection asks the
  whisper questions, runs vosk cells as-is, and no combination is ever partially or
  ambiguously parameterized.

Example: whisper-base (faster-whisper) selected for en + de, vosk-small for en; user
answers `beam_size: 1,4,8`, Enter for `vad`; repeats 2.

```
About to run 7 benchmark(s) (14 runs incl. 2 repeats):
  whisper-base-en-batch  x  <pack>   beam_size=1
  whisper-base-en-batch  x  <pack>   beam_size=4
  whisper-base-en-batch  x  <pack>   beam_size=8
  whisper-base-de-batch  x  <pack>   beam_size=1
  whisper-base-de-batch  x  <pack>   beam_size=4
  whisper-base-de-batch  x  <pack>   beam_size=8
  vosk-small-en-batch    x  <pack>
Proceed?
```

Confirmation requirements:

- Enumerate the **full expansion**, one line per (profile × pack × parameter values),
  non-default values shown inline.
- State the honest total **including repeats**. (Today's confirmation counts combos and
  ignores repeats — `--repeats 2` silently doubles the work. Fix that here.)
- Soft warning above a threshold (~20 expanded combos) for the fat-fingered
  whole-column × 5-values case.

The batch loop itself is unchanged: each expanded combo `_reexec`s
`run <profile> <pack> --repeats N --hardware H [--param k=v ...]` in isolation, and the
batch summary echoes the parameter values per line the same way the confirmation does.

## 5. Platform: honest leaderboard, expressive filters

**Decision: non-default runs are first-class leaderboard rows — nothing is hidden or
pinned away.** An earlier draft of this design proposed pinning the default leaderboard
view to profile-default parameter values. That was rejected: hiding rows is a form of
discretionary curation, and the platform's own leaderboard principle (see
`routes/leaderboards.py`: curated views are URL-visible presets of mechanical filters,
never a separate mechanism) says the truth is shown and the criteria are always visible.
If a `beam_size=1` run is the fastest thing ever measured on an N100, it tops the speed
sort — and its WER column makes the accuracy cost self-evident.

API additions:

- **Parameter facets:** filter/group by parameter value, as backend became a facet under
  ADR-0008 — e.g. `?param.beam_size=8`, matching `document.parameters.<key>.value`.
- **Metric-threshold filters**, so honest tradeoff queries are first-class:
  "fastest with WER < 10%" = `?sort=real_time_factor&order=asc&max_wer=10`. Thresholds
  (`min_<metric>` / `max_<metric>`) apply to any metric id, implemented with the same
  JSONB-path cast the `sort` parameter already uses; an unknown metric id is a 400 with
  a reason (match the existing `max_price_eur` explicit-rejection style). All new
  filters are echoed in the response `filters` block.
- **Response shape:** `LeaderboardEntry` gains `parameters` — the result's
  `{value, default}` map — so the web renders values and default markers without any
  catalog lookup.

### UI design (decided 2026-07-26; mockup iterated in design session)

The display must not editorialize: the profile default is the reviewed *anchor point*,
not "the correct value" — whether it is sensible for a given hardware/language/audio is
exactly the empirical question the leaderboard answers.

- **Vary-in-view parameter columns.** A parameter renders as a table column whenever its
  values *vary among the currently visible rows*; when uniform, it collapses into a
  "Constant across this view: quantization int8 · threads 4" line under the table —
  always visible, never repeated as noise. Engines the parameter doesn't apply to show
  "—". The rule is computed over the current (filtered, paged) result set, so the table
  always shows exactly what distinguishes its rows; columns appearing/disappearing as
  filters change is intended behavior.
- **Neutral default marker.** Values are styled neutrally; the profile default carries
  only a muted asterisk (legend: "* profile default"). No warning colors on non-default
  values — labeled, never judged. (An earlier badge-on-deviation design was rejected for
  implying default = correct.)
- **Expandable row detail** shows the complete resolved configuration — every parameter
  including defaults, `(default N)` annotations, plus profile version, pack, repeats,
  backend — straight from the signed result's `parameters` map.
- **Quality gates as preset chips + custom input.** A chip row above the table: presets
  for common thresholds (WER < 5 / 10 / 15%), plus "custom…" accepting any metric and
  min/max bound. Active gates are removable chips like any other filter.
- **The nudge, not a gate.** When sorted by a gameable metric (speed, energy) with no
  accuracy gate active, show one dismissible hint ("Sorting by speed with no accuracy
  gate — add WER < 10%?") with one-click apply. The truth always shows either way.
- **Everything in the URL.** Every chip, gate, and sort maps to a query parameter, so
  any view is shareable and curated views ("Fastest usable") are plain preset links —
  the mechanical-curation principle extended to the new controls.

Ingest stores `parameters` with the result document (JSONB); no new integrity checks
beyond existing signature verification.

## 6. Parameter catalog — current adapters (verified against adapter code 2026-07-26)

What each adapter *actually applies* today (the §2 "no silent knobs" registry), and what
the bulk-generated profiles should declare. Sources: the adapters' own signatures and
docstrings in `runner/src/oesb_runner/adapters/`.

### faster-whisper (batch)

Applied: `quantization` (→ ctranslate2 `compute_type`), `beam_size`, `temperature`,
`vad` (→ `vad_filter`), `threads` (→ `cpu_threads`). Declare on all faster-whisper batch
profiles:

```yaml
overridable:
  beam_size:    { allowed: [1, 2, 4, 5, 8] }
  vad:          {}
  quantization: { allowed: [int8, float32] }   # universally valid on the cpu backend;
                                               # widen per-backend later under ADR-0008
  threads:      { range: { min: 1, max: 16 } }
```

Deliberately **not** declared: `temperature` — any value > 0 introduces sampling
nondeterminism, which conflicts with the repeat-tolerance check (FR-5.3) and makes
"same parameters ⇒ same experiment" false. Stays locked at 0.0. `language` is a profile
axis, never a parameter.

### faster-whisper (streaming profile)

Additionally applied: `chunk_ms`. Declare
`chunk_ms: { allowed: [250, 500, 1000, 2000] }` (default 1000) on the streaming
profile, alongside the batch set above.

### whisper-cpp (batch)

Applied today: `threads` (→ `n_threads`), `temperature`. **Not applied** despite being
accepted (call-shape parity only, per the adapter's docstring): `beam_size` (pywhispercpp
leaves whisper.cpp on its greedy strategy — the nested `beam_search` param is not wired),
`vad` (no VAD in this adapter), `quantization` (a ggml *model-file* choice, i.e. a model
artifact question, not a runtime flag). Declare only:

```yaml
overridable:
  threads: { range: { min: 1, max: 16 } }
```

`temperature` excluded for the same nondeterminism reason as above. Wiring pywhispercpp's
`beam_search` parameters so `beam_size` can honestly be declared here is the natural
follow-up adapter task — until then, whisper-cpp profiles must not declare it.

### vosk (batch)

Applied: nothing — vosk's Kaldi decoder exposes no equivalent tunables through its
Python API, and each model is per-language by artifact. Vosk profiles get **no
`overridable` block**, and the wizard never prompts for vosk (§4 falls out naturally).

### Catalog consequences

- The cross-engine sweep story at launch is: `threads` sweeps across faster-whisper and
  whisper-cpp together; `beam_size`/`vad`/`quantization` sweep on faster-whisper only.
- `goesb doctor` / the platform's most-wanted view can treat unfilled (profile ×
  parameter value) cells as contribution prompts, same as backend under ADR-0008.

## 7. What this deliberately does not do

- No parameter changes what is *scored* — normalization, scoring, metrics stay locked in
  the profile, non-overridable by construction (they are not in `model`/`configuration`).
- No unbounded overrides: a parameter without a reviewed domain cannot be swept.
- No parameters that the adapter accepts but ignores (§2 "no silent knobs").
- No wizard cross-product across engines with mismatched parameter sets — the per-engine
  grouping makes that state unrepresentable rather than warned-about.

## Relationships

Decided by ADR-0009; generalizes ADR-0008 (compute backend as a result-level axis) and
leans on ADR-0004 (reviewed adapters; review-gated override lists) and the existing
signing mechanism (ADR-0004/0005) for provenance. Touches: the wizard
(`_wizard_run`, `_preflight_engines` in `runner/src/oesb_runner/cli.py`), the `run`
command (`--param`), `benchmark-profile.schema.json` and
`benchmark-result.schema.json` (`runner/src/oesb_runner/schemas/`), the adapters'
applied-parameter registry (`runner/src/oesb_runner/adapters/`), and oesb-platform's
ingest + `routes/leaderboards.py` (parameter facets, metric-threshold filters) + web
leaderboard UI (§5 UI design).
