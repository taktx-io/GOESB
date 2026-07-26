# ADR-0009 — Configuration parameters are profile-declared dials, resolved per run and shown honestly

- **Status:** Proposed
- **Date:** 2026-07-26

## Context

Profiles are immutable, git-reviewed definitions of exactly how a benchmark runs —
that is what makes "same `profile_id`" mean "same experiment" (NFR-1). But it also
means exploring a configuration parameter (`beam_size` at 1, 4, 8) requires a
near-duplicate profile file per value: for a 5-value sweep across several engines,
a pile of reviewed files for what is really one workload with one varying dial.

ADR-0008 faced this identical tension for compute backend and resolved it without
touching profile identity: backend is an explicit runner flag, recorded verbatim in
the signed result, and compared *across* on the platform. The question here is
whether that pattern generalizes to configuration parameters — and, since the batch
wizard lets users select many (language × engine × size) cells at once, what a
parameter sweep means inside a multi-cell batch.

A leaderboard question rides along: if `beam_size=1` runs land next to `beam_size=8`
runs, does the default view hide non-default rows to keep rankings "fair"?

## Decision

### 1. Profiles declare override-eligible parameters; adapters validate them

A profile may mark specific `model`/`configuration` keys as overridable, each with a
bounded, reviewed domain (`overridable: {beam_size: {allowed: [1, 2, 4, 5, 8]}}`).
The default stays the value in the profile body — one source of truth. Changing the
overridable set or a domain is a profile version bump with a changelog entry.

The layering mirrors ADR-0008 exactly: the **adapter** (reviewed code, ADR-0004)
owns what parameters exist and validates values; the **profile** owns which are
eligible for this workload and within what domain. An undeclared parameter or
out-of-domain value is a hard error, never a silent fallback.

Eligibility lives in profile.yaml — not a runner-side allowlist — because declaring
a dial turnable changes what a leaderboard row under that id means (profile-
governance territory), and because the platform reads profile.yaml to know which
facets to offer, without introspecting runner code.

### 2. The run records every resolved dial, signed

`goesb run ... --param beam_size=8` overrides for that run only. The signed result
gains a `parameters` object recording the resolved value of **every** eligible
parameter — untouched defaults included — alongside the profile default
(`{"beam_size": {"value": 8, "default": 5}}`). Results stay self-describing:
grouping and "non-default" badges never need the profile catalog. Provenance needs
no new mechanism — `config_sha256` already hashes the fully-resolved configuration,
and the existing signature covers the document.

Comparability becomes "same `profile_id` + same parameter values ⇒ same method" —
the same grouping discipline backend already introduced.

### 3. The wizard asks per engine; Enter means today's defaults

In a multi-cell batch, parameters are prompted **per engine**, not per batch: each
engine in the selection is asked only about its own overridable parameters, with
defaults pre-filled. Enter accepts the default (a full Enter-through reproduces
today's behavior exactly); a single value overrides that engine's cells; a
comma-separated list (`1,4,8`) sweeps them — cells × values. Engines with no dials
(vosk) are never prompted. This makes ill-formed states unrepresentable: an
engine-specific parameter cannot leak onto an engine that lacks it, so the
cross-product-over-mixed-selections problem never arises.

Guardrails: values are validated against declared domains before run 1 (preflight,
next to engine preflight); the confirmation enumerates the full expansion with
non-default values inline and states the honest total **including repeats** (fixing
the existing undercount where `--repeats 2` silently doubles the batch); a soft
warning fires above ~20 expanded combos.

### 4. The leaderboard shows the honest truth; filters do the fairness

Non-default runs are first-class rows in every view. We explicitly considered and
**rejected** pinning the default view to profile-default values: hiding rows is
discretionary curation, which the platform's leaderboard principle already forbids
(curated views are URL-visible presets of mechanical filters, nothing more). If a
`beam_size=1` run is the fastest on some hardware, it tops the speed sort — its WER
column makes the cost self-evident.

Instead the platform gains expressiveness: every row displays its parameter values
with non-default ones visibly marked; parameters become filter/group facets like
backend; and **metric-threshold filters** (`?sort=real_time_factor&max_wer=10` —
"fastest with WER under 10%") make honest tradeoff queries first-class, implemented
with the same JSONB-path cast sorting already uses.

## Consequences

- **+** Parameter sweeps need zero new profile files; profile identity stays
  engine × model × language (× backend, per ADR-0008).
- **+** Every result is self-describing about its dials; any post-hoc edit breaks
  the signature. No new integrity mechanism.
- **+** The wizard's common path is unchanged (Enter = defaults), and sweeps are
  legible before they run — count, expansion, and repeats all stated.
- **+** The leaderboard stays honest and mechanically curated; "fast but sloppy"
  configurations are visible, labeled, and filterable rather than hidden.
- **−** Comparability now requires grouping by parameter values, not just
  profile id — consumers of the API must be told (docs + response shape).
- **−** Schema/migration work: profile schema gains `overridable`, result schema
  gains `parameters` (a versioned change under `additionalProperties: false` —
  ride the same bump as ADR-0008's pending `runtime.backend` if possible), ingest
  and leaderboards gain parameter facets and threshold filters.
- **−** Review burden shifts to domains: approving `allowed: [1..8]` is approving
  8 leaderboard variants of the workload. Deliberate — bounded and reviewed is the
  point (ADR-0004).

## Relationships

Generalizes ADR-0008 (explicit compute backend); builds on ADR-0004 (reviewed
adapters; review-gated allowlists) and ADR-0005 (signing/trust). Detailed in
`docs/specs/parameterized-profile-configuration.md`. Touches the wizard and `run`
command (`runner/src/oesb_runner/cli.py`), both schemas
(`runner/src/oesb_runner/schemas/`), and oesb-platform ingest,
`routes/leaderboards.py`, and the web leaderboard UI.
