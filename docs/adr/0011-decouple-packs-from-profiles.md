# ADR-0011 — Decouple packs from profiles; join on language

- **Status:** Accepted (Eric, 2026-07-28).
- **Date:** 2026-07-28
- **Builds on / relates to:** [ADR-0004](0004-runner-security-model.md) (explicit,
  never implicit — the standard this decision is held to), [ADR-0008](0008-explicit-compute-backend.md)
  (same "hard error before anything runs" posture, applied here to pack/profile
  eligibility instead of backend selection), `scripts/generate_bulk_assets.py`
  (the sibling-pack pattern this ADR retires), `docs/04-glossary.md` (Pack/Profile
  definitions, updated alongside this decision).

## Context

`pack.yaml` has always carried a required `profile_id` field, and `runner/src/oesb_runner/cli.py`'s
`run` command hard-refused any pack whose `profile_id` didn't exactly equal the
profile being run. In practice this welded one pack to exactly one (engine, size,
language) combination — a real Dutch Common Voice pack authored for
`whisper-medium-nl-batch` could not be used to benchmark `vosk-small-nl-batch` or
`whispercpp-base-nl-batch`, even though all three profiles are equally "Dutch" and
the audio itself has nothing engine-specific about it.

The existing workaround, `scripts/generate_bulk_assets.py`'s "sibling pack"
pattern, faked a full (engine × size) matrix per language by writing one real pack
plus ~10 near-identical duplicate `pack.yaml` files per language — same
`manifest.jsonl`, same `audio.source`, different `id`/`profile_id`, relying on
`goesb run`'s shared-audio-cache auto-fetch to avoid re-downloading anything. It
worked, but every new language meant authoring ~11 packs to describe one dataset,
and it doesn't scale to the actual ask: multiple *distinct* packs per locale (e.g.
Dutch FLEURS, Dutch Common Voice, a future Dutch Jasmin corpus), each independently
selectable against every engine profile in that language, with the leaderboard
showing every (engine × pack) result as its own row rather than picking one
winner per profile.

Both schemas already carried the field needed to do this properly:
`profile.language` and `pack.metadata.language` are both present, both already
written as full BCP-47 tags (`nl-NL`, `de-DE`, `en-US`, `es-419`, `fr-FR`,
`pt-BR`) — confirmed against every committed profile and pack, not assumed. The
`profile_id` pin was enforcing a narrower relationship than the data itself
implied.

## Decision

**A pack is eligible for a profile when `pack.metadata.language` exactly equals
`profile.language` — not when `pack.profile_id` equals the profile's id.**

- `benchmark-pack.schema.json`: `profile_id` removed from `required`. The field
  itself is *not* deleted from the schema — packs that still declare it (every
  pack committed before this ADR) stay valid unchanged. It's informational only
  from here on; new packs should omit it.
- `cli.py`'s `run` command: the hard `pack_yaml["profile_id"] != profile_id` check
  is replaced with a language check, in the same place (before `_resolve_pack_audio`,
  same "explicit, early, never silent" posture ADR-0008 established for
  `--backend`):
  - No `language` on the profile at all → hard error. A profile schema allows
    omitting `language`; this ADR does not, in practice, since eligibility can't
    be verified without it.
  - `pack.metadata.language != profile.language` → hard error, both values quoted
    in the message.
  - Otherwise: proceeds.
- `_matching_packs` (the wizard's pack-picker filter) takes a `language` now
  instead of a `profile_id`, matching the same rule.
- `_pack_rows` (shared by `list-packs` and the wizard) now carries `language`
  instead of `profile_id` in each row; `list-packs` gained a LANGUAGE column
  since it's now the field that actually explains eligibility.
- Exact-tag match only, deliberately no fuzzy/prefix fallback. `nl-NL`, `nl-BE`,
  and `fy-NL` are three different things — two real dialect differences and one
  entirely different ISO-639 language sharing a province — and silently letting
  one satisfy another would quietly break WER comparability across a leaderboard.
  A genuinely new locale gets the same onboarding any language already gets: a
  profile declaring that `language` (it also drives `normalization.ruleset_id`),
  a pack with a matching `metadata.language`. `pack.metadata.dialect` (already
  schema-defined, already used by `librispeech-en-batch`) is the right place for
  a same-tag accent variant that should still count as the same locale for
  eligibility purposes.

## What this replaces

`generate_bulk_assets.py`'s sibling-pack generation is no longer needed for new
languages — one real pack per (language × audio source) is enough; every existing
profile in that language becomes eligible for it immediately, no per-profile
duplication. The ~58 sibling packs this pattern already produced (10 per language
across `de`/`es`/`fr`/`pt`/`nl`'s FLEURS set, plus 8 for `en`'s LibriSpeech set —
recounted directly against `packs/`, not estimated) are retired outright as part
of this same change, not deprecated in place: every result referencing them is our
own internal benchmark run, reproducible on the same hardware, so there is no
external consumer whose data would be orphaned by removing them. The audio-source
packs they duplicated (`fleurs-de-batch`, `fleurs-es-batch`, `fleurs-fr-batch`,
`fleurs-nl-batch`, `fleurs-pt-batch`, `librispeech-en-batch`, plus the
hand-authored `librispeech-en-whispercpp-batch` / `librispeech-en-vosk-batch` /
`librispeech-en-streaming`) are untouched and now directly cover every batch
profile in their language on their own.

The corresponding platform-side maintenance-mode gate, results reset, and
canonical-pack-set rollout are tracked in oesb-platform, not here — this ADR
covers the schema and runner decision; that repo's own change log covers the
production cutover built on top of it.

## Non-goals / explicit constraints

- **Not touching `benchmark_type`.** A pack has no `benchmark_type` field today,
  so nothing currently stops a batch-curated pack from also matching a streaming
  profile purely on language. Left as-is for this ADR — audio is audio, and
  narrowing this further is a separate decision if it turns out to matter in
  practice.
- **Not changing how results are scored, signed, or grouped.** `Result.profile_id`
  and `Result.pack_id` were already independent, already-indexed fields before
  this ADR; nothing about a signed result document changes shape.
- **Not relaxing language matching.** Exact BCP-47 tag equality only, see above —
  this ADR deliberately does not add a "close enough" mode.

## Consequences

- Every profile in a language becomes eligible for every pack in that language
  the moment both exist — no sibling-pack authoring step for new content.
- The wizard's pack-picker checkbox (already built for the "more than one pack
  matches" case) now fires far more often — most languages go from exactly one
  match to several (the audio-source pack plus every Common Voice pack in that
  language) — this is the intended UX, not a regression; `_choose_packs_for_profile`
  needed no changes to handle it.
- `pack.yaml`'s `profile_id` becomes vestigial. Left in the schema rather than
  removed so existing packs don't need a version bump or re-hash; new packs
  should simply not declare it.
- Establishes the pattern for every future non-FLEURS/non-LibriSpeech corpus
  (a Dutch Jasmin pack, a second English source, etc.) — they declare
  `metadata.language`, they're immediately usable everywhere that language is
  benchmarked, no wizard or schema changes needed per-pack.
