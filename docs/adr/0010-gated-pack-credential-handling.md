# ADR-0010 — Gated-pack credential handling in the wizard

- **Status:** Accepted (Eric, 2026-07-27).
- **Date:** 2026-07-27
- **Builds on / relates to:** [ADR-0004](0004-runner-security-model.md) (declarative
  inputs only, least privilege), [ADR-0005](0005-signing-token-distribution-and-trust-limits.md)
  (the existing `~/.goesb/keys/` local-secret pattern this reuses), `packs/README.md`
  (open/community/private visibility), `runner/src/oesb_runner/audio_sources.py`'s
  existing `fetch_instructions` fallback (the mechanism this extends, not replaces).

## Context

Some otherwise-free, high-value corpora (Mozilla Common Voice via the Mozilla Data
Collective platform; likely others later — e.g. the Speech Accessibility Project)
are access-gated: free to use, but only via a personal API key tied to an account,
because the platform's own terms forbid goesb re-hosting or re-serving the audio
itself. That's a hard constraint, not a preference — a shared/embedded goesb
credential would (a) violate those terms, since usage is meant to be attributable
to one account, and (b) leak immediately, since the runner ships as plain,
pip-installable source (the same problem ADR-0005 already identified for a
runner-embedded signing key).

Today the runner already has a graceful fallback for exactly this shape of pack:
`audio.source.fetch_instructions` is free text, printed verbatim when a pack isn't
auto-fetchable. That's sufficient to *inform* the user, but it means every gated
pack still requires them to leave the wizard, go set up credentials by hand, and
re-run — clunky, and it treats every gated pack as a one-off special case rather
than a pattern the wizard understands.

## Decision

**Make "this source needs a credential" a declarative, schema-level property of a
pack**, not something encoded only in prose. Add an optional `credential` object
under `audio.source`:

```json
"credential": {
  "type": "object",
  "properties": {
    "env_var":      { "type": "string", "description": "e.g. MDC_API_KEY" },
    "signup_url":   { "type": "string", "format": "uri" },
    "instructions": { "type": "string", "description": "Short human-readable steps to obtain it." }
  },
  "required": ["env_var", "signup_url", "instructions"]
}
```

This stays entirely within ADR-0004's "declarative inputs only" model — the
requirement is data in the manifest, not a hook or a script.

**The wizard gets a new preflight step**, `_preflight_pack_credentials`, run
immediately after pack selection (same place `_preflight_engines` runs today —
before the batch subprocess loop, so nothing stalls hours into an unattended run
waiting on a prompt). For every selected pack:

1. Collect distinct `(env_var, signup_url, instructions)` triples across the
   whole batch — dedup by `env_var`, so a batch touching several Common-Voice-backed
   packs asks once, not once per pack.
2. For each one not already resolvable (checked in order: process environment,
   then the local credential store below), print `instructions` + `signup_url`,
   then prompt with `questionary.password(...)` — **masked input**, never
   `questionary.text`, since this is a secret.
3. An empty/declined answer drops only the combos needing that credential from
   the batch — same continue-past-failure spirit `_preflight_engines` already
   uses for an uninstalled engine — never aborts the whole run.
4. A non-empty answer is both (a) set into `os.environ[env_var]` for this
   process — `_reexec`'s `subprocess.run` inherits the parent environment by
   default, so every batch combo's `run` subprocess picks it up automatically,
   no plumbing needed — and (b) persisted to a new local credential store so
   this is asked **at most once per machine**, not once per wizard invocation.

**Local credential store: `~/.goesb/credentials.json`, mode `0600`.** This is a
new sibling to the existing `~/.goesb/keys/` signing-key convention
(`signing.py`'s `DEFAULT_KEY_DIR`) — same directory root, same file-permission
discipline, so there's one obvious place on disk a security-conscious user finds,
audits, or deletes all of goesb's local secrets. A small new module,
`oesb_runner/credentials.py`, owns `load_credential(env_var)` /
`save_credential(env_var, value)` so both the wizard step and the fetch provider
below share one implementation.

**A real fetch provider, not just better prompting.** Collecting the credential
only pays off if something then uses it. Add `fetch_common_voice_audio` (or a
more generic `fetch_gated_audio` keyed by the pack's declared provider) to
`audio_sources.py`, added to `_PROVIDERS` and `AUTO_FETCH_SOURCE_TYPES` under a
new `source.type: "mozilla_data_collective"`. It calls the official
`datacollective` package (`download_dataset(...)`), authenticated via the
env var the credential preflight just populated. If the fetch call itself fails
(bad/expired key, revoked, network), fail that pack the same way a missing-model
auto-fetch fails today (clear stderr message, non-zero exit, batch continues) —
**never a raw traceback**, since `_resolve_pack_audio`'s `auto_fetch_audio` call
site (`cli.py` ~line 736) has no try/except today and this is new failure surface
being introduced there.

## Non-goals / explicit constraints

- **The credential is never sent to goesb's own servers.** It authenticates
  directly against the gated source's own API, from the user's own machine —
  consistent with ADR-0004's "audio never leaves the machine" / least-privilege
  framing, just applied to inbound fetching instead of outbound submission.
- **Never captured in the environment fingerprint.** `environment.py`'s
  `capture_environment()` probes specific hardware fields today (CPU/RAM/GPU/etc.)
  and does not dump `os.environ` — this must stay true; a credential leaking into
  a signed, publicly-submitted result would be a real incident. Add a test
  asserting `capture_environment()`'s output never contains a declared credential
  env var's value.
- **Never echoed to the terminal or logs.** Masked prompt only; the value must
  not appear in any `typer.echo`, exception message, or `_reexec` argv (it
  travels via environment, not command-line arguments, which is also why it
  never shows up in `ps`-visible process listings the way an argv value would).
- Declining a credential prompt is a normal, supported outcome (some contributors
  will only ever run open, ungated packs) — it must never be treated as an error
  for the rest of the batch.

## Consequences

- New schema field (additive, optional — existing packs validate unchanged).
- New `oesb_runner/credentials.py` module and a new wizard preflight step.
- New fetch provider + a first real consumer of it: the Common Voice NL elderly
  pack (separately scoped — this ADR is the mechanism, not that pack's authoring).
- Establishes the pattern for every future gated corpus (e.g. the Speech
  Accessibility Project, if it's ever wired in) — they declare `credential`,
  they get this exact UX for free, no wizard changes needed per-pack.
- `fetch_instructions` (free text) remains as the fallback for sources that are
  gated in some way this structured field doesn't capture (e.g. a multi-step
  manual process with no single env-var credential) — this ADR adds a
  structured *common case*, it doesn't replace the general escape hatch.
