# Handoff brief: implement ADR-0009 (parameterized profile configuration)

For Claude Code. Context documents (read first):
- `docs/adr/0009-parameterized-profile-configuration.md` — the decision
- `docs/specs/parameterized-profile-configuration.md` — implementation-level detail
- `docs/adr/0008-explicit-compute-backend.md` — the pattern being generalized

Two repos are touched: **OESB** (runner, schemas, profiles) and **oesb-platform**
(API, web). Suggested order below; each numbered block is independently landable.

## 1. OESB — profile schema + catalog

- `runner/src/oesb_runner/schemas/benchmark-profile.schema.json`: add optional
  top-level `overridable` object. Keys: parameter names; values: one of
  `{"allowed": [...]}`, `{"range": {"min": n, "max": n}}`, or `{}` (boolean
  params only). Validation rule (runner-side, not expressible in JSON Schema
  alone): every `overridable` key must exist in the profile's `model` or
  `configuration` block — its value there is the default.
- Add `overridable` blocks to the bulk generator
  (`scripts/generate_bulk_assets.py`) for the whisper engines — start with
  `beam_size` (suggest `allowed: [1, 2, 4, 5, 8]`) and `vad`. Vosk profiles get
  none. Regenerating is a profile version bump (1.0.0 → 1.1.0) with a changelog
  entry per profile ("declare beam_size/vad override-eligible").

## 2. OESB — runner `run` command

- New repeatable flag on `run`: `--param KEY=VALUE`. Resolution order: profile
  default, overridden by `--param`. Hard errors (before loading anything heavy):
  key not in the profile's `overridable` set; value outside the declared
  domain; adapter rejects the value. Mirror the backend-flag error style
  (no silent fallback, no clamping).
- Adapters (`adapters/faster_whisper.py`, `whisper_cpp.py`, `vosk.py`): accept
  resolved parameters explicitly; adapter-side type/limit validation stays the
  authority on what a parameter means (ADR-0004).
- Result document (`benchmark-result.schema.json`, `signing.py` payload
  construction): add required-when-profile-has-overridables `parameters` object:
  `{"<key>": {"value": <resolved>, "default": <profile default>}}` for **every**
  eligible parameter, overridden or not. This is a versioned schema change under
  `additionalProperties: false` — coordinate with the pending `runtime.backend`
  field from ADR-0008 so both land in one `schema_version` bump if 0008's schema
  work is still open. `config_sha256` already covers resolved config — verify the
  hash input actually includes `--param` overrides.
- Tests: `test_cli.py` (flag parsing, validation errors), `test_signing.py`
  (parameters in signed payload), `test_schema_validation.py`, adapter tests for
  parameter pass-through, plus an end-to-end run with an override asserting the
  result records `{value, default}` correctly.

## 3. OESB — batch wizard (`_wizard_run` in `runner/src/oesb_runner/cli.py`)

New step between engine preflight and the repeats prompt:

- Group selected profiles by engine (`runtime.name`). For each engine, compute
  the overridable parameters common to all of its selected profiles; skip the
  engine entirely if empty (vosk today).
- One `questionary.text` prompt per (engine, parameter), default pre-filled from
  the profile value. Semantics: **Enter** = default; **single value** = override
  for that engine's cells; **comma list** (`1,4,8`) = sweep, expanding each of
  that engine's cells into one combo per value.
- Preflight-validate every entered value against each affected profile's domain
  immediately (same philosophy as `_preflight_engines` — fail before run 1).
- Confirmation changes:
  - enumerate the full expansion, one line per (profile × pack × param values),
    non-default values shown inline;
  - headline shows expanded combo count AND total runs including repeats, e.g.
    `About to run 7 benchmark(s) (14 runs incl. 2 repeats)` — note this also
    fixes the existing undercount where repeats silently multiply the batch;
  - soft warning (extra confirm) above ~20 expanded combos.
- Batch loop: unchanged structure; append `--param k=v` per expanded combo to the
  existing `_reexec(["run", ...])`. Batch summary lines echo param values.
- Tests: `test_cli.py` — grouping, Enter-through == today's behavior (regression
  guard), sweep expansion counts, mixed whisper+vosk selection never passes a
  param to vosk, validation preflight, confirmation totals.

## 4. oesb-platform — API

- Ingest: accept/verify the new result `schema_version`; `parameters` lands in
  the stored JSONB document. No new columns required initially (facet via JSONB);
  add generated columns later only if query performance demands it.
- `routes/leaderboards.py`:
  - **Parameter facets:** filter by parameter value, e.g. `?param.beam_size=8`
    (or `param_beam_size=8` if dotted query keys are awkward in FastAPI — pick
    one, document it) matching `document.parameters.<key>.value`.
  - **Metric-threshold filters:** `min_<metric>` / `max_<metric>` style (e.g.
    `?max_wer=10`), any metric id, implemented with the same
    `cast(...astext, Float)` pattern the `sort` param uses. Unknown metric id →
    400 with reason (match the existing `max_price_eur` explicit-rejection
    style, not silent ignoring).
  - Response: `LeaderboardEntry` gains `parameters` (the `{value, default}` map)
    so the web can render values + non-default badges without a catalog lookup.
  - Echo new filters in the response `filters` block (curation-transparency
    principle in the route docstring).
- **Explicitly NOT wanted:** no hiding or default-pinning of non-default runs in
  any view. The default sort stays as-is; fairness comes from visible parameter
  values + threshold filters (ADR-0009 §4 — this was considered and rejected).
- Tests: `tests/test_routes.py` — param facet, threshold filter (inclusive
  bounds, combined with sort), 400 on unknown metric, parameters in response.

## 5. oesb-platform — web

- Leaderboard rows: show parameter values; visually mark non-default ones
  (compare `value` vs `default` from the API — no catalog fetch). A fast row must
  never be mysteriously fast.
- Filter UI: parameter facets (driven by which profiles declare `overridable` —
  readable from the profiles the API already serves) and metric-threshold inputs
  ("fastest with WER < 10%" as a first-class query). Curated preset links remain
  plain URLs over these params.

## Acceptance criteria

1. A profile with no `overridable` block behaves byte-identically to today
   (runner, wizard, API).
2. Wizard Enter-through on any selection produces the same runs as before this
   change.
3. `goesb run p pack --param beam_size=8` → signed result contains
   `parameters.beam_size = {value: 8, default: 5}` and verifies; same run with
   `beam_size=3` (not in domain) fails before model load with a clear error.
4. Mixed whisper+vosk batch with a beam_size sweep: whisper cells expand, vosk
   runs once, confirmation total (incl. repeats) matches actual run count.
5. `GET /leaderboards?sort=real_time_factor&max_wer=10` returns only rows with
   WER < 10 (and includes non-default-parameter rows).
6. No leaderboard view filters out non-default runs implicitly.

## Constraints / reuse

- Reuse the existing signing path (`signing.py`) — do not add a separate
  integrity mechanism for parameters.
- Keep the ADR-0008 error philosophy everywhere: explicit, early, never silent.
- Schema bumps: one coordinated `schema_version` change; migrate existing stored
  results as "no overridables declared" (absent `parameters` = pre-0009 result,
  treated as all-defaults for its profile version).

When done, report back with: files changed per repo, the final schema_version
chosen, any deviations from this brief (and why), and anything discovered that
should flow back into the spec/ADR.
