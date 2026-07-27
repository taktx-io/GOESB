# Handoff brief: implement ADR-0010 (gated-pack credential handling)

For Claude Code. Context documents (read first):
- `docs/adr/0010-gated-pack-credential-handling.md` — the decision
- `docs/adr/0004-runner-security-model.md` / `0005-signing-token-distribution-and-trust-limits.md` —
  the precedent this reuses (`~/.goesb/keys/`, chmod 0600, declarative-inputs-only)
- `runner/src/oesb_runner/audio_sources.py` — the existing auto-fetch/fetch_instructions
  mechanism this extends
- `runner/src/oesb_runner/cli.py` `_wizard_run` / `_preflight_engines` (~line 320-500) —
  the pattern the new preflight step mirrors

All changes are in **OESB** (runner + schema). No `oesb-platform` changes needed —
credentials are entirely local/client-side, never touch the API.

**Staged rollout (Eric, 2026-07-27):** this pass builds the mechanism (§1-4)
*and* exactly one pack proving it out (§5) — the Common Voice NL elderly pack,
nothing else. Stop there. Expanding to further Common-Voice packs (other
languages/age slices) or other gated corpora is an explicit later decision
once this one pack has been through both automated and manual testing and
Eric has confirmed it's good — do not treat "the mechanism works" as license
to keep building more packs in this same pass.

## 1. Schema

`runner/src/oesb_runner/schemas/benchmark-pack.schema.json`: add optional
`credential` object under `audio.source.properties` (see ADR-0010 for the exact
shape — `env_var`, `signup_url`, `instructions`, all required within the object
if it's present at all). Additive/optional — run `scripts/validate_assets.py`
against every existing pack to confirm nothing regresses.

## 2. New module: `oesb_runner/credentials.py`

- `DEFAULT_CREDENTIALS_PATH = Path.home() / ".goesb" / "credentials.json"` —
  sibling to `signing.DEFAULT_KEY_DIR`, same root.
- `load_credential(env_var: str, *, path=DEFAULT_CREDENTIALS_PATH) -> str | None`:
  check `os.environ` first (an explicitly-exported var always wins — don't
  shadow a user's own shell config), then the JSON store if the file exists.
- `save_credential(env_var: str, value: str, *, path=DEFAULT_CREDENTIALS_PATH) -> None`:
  read-modify-write the JSON object (`{env_var: value, ...}`), `path.parent.mkdir(parents=True, exist_ok=True)`,
  write, then `path.chmod(0o600)` — mirror `signing.load_or_create_keypair`'s
  exact chmod discipline.
- Tests: round-trip save/load; a pre-existing `os.environ` value takes priority
  over the stored one; file permissions are 0600 after save; corrupt/missing
  JSON degrades to `None` rather than raising (same "never crash the wizard on
  a local-state problem" philosophy as `_load_profile_for_wizard`).

## 3. Wizard preflight step (`cli.py`)

New function `_preflight_pack_credentials(combos, packs) -> list[tuple[str, str]]`,
called in `_wizard_run` right where `combos` is finalized (~line 496), before
`_preflight_engines` (order between the two doesn't matter functionally, but
credentials-then-engines reads more naturally — ask about access before asking
about install).

- Look up each combo's pack dict (already loaded as `packs` in `_wizard_run`) for
  `audio.get("source", {}).get("credential")`.
- Dedup by `env_var` across the whole batch.
- For each `env_var` not resolvable via `credentials.load_credential`: print
  `instructions` + `signup_url` (`typer.echo`, err=True — matches existing
  wizard-message channel), then `questionary.password(f"Paste your {env_var} (leave blank to skip these packs):").ask()`.
  - `None` (Ctrl-C/abort) → treat like the rest of the wizard's abort convention
    (return `None` from the whole preflight, `_wizard_run` bails, same as an
    aborted matrix/hardware pick).
  - Empty string → drop only the combos whose pack needs this `env_var`, echo
    which ones and why (mirror `_preflight_engines`'s
    `"  {profile_id} x {pack_id} — skipping (...)"` line style exactly).
  - Non-empty → `os.environ[env_var] = value` (so `_reexec`'s subprocess
    inherits it) **and** `credentials.save_credential(env_var, value)`.
- Return the surviving combo list, same shape/contract as `_preflight_engines`.

Wire it into `_wizard_run`: `combos = _preflight_pack_credentials(combos, packs)`
then `if not combos: return`, right alongside the existing
`combos = _preflight_engines(...)` call.

## 4. Fetch provider (`audio_sources.py`)

- New `AUTO_FETCH_SOURCE_TYPES` entry: `"mozilla_data_collective"`.
- New `fetch_common_voice_audio(params, wanted_names, audio_dir)`: import
  `datacollective` lazily inside the function (same guarded-optional-dependency
  style the STT adapters use for `faster_whisper`/`vosk`/`pywhispercpp` — don't
  make it a hard dependency of the base install), call
  `datacollective.download_dataset(params["dataset_id"])`, then reuse this
  module's existing extraction/filtering approach to pull out only
  `wanted_names` into `audio_dir` (match `_stream_extract`'s return contract:
  the set of filenames actually obtained).
- Register it in `_PROVIDERS`.
- **Error handling at the call site matters most here.** `cli.py`'s
  `_resolve_pack_audio` (~line 736) calls `auto_fetch_audio(...)` with no
  try/except today — a bad/expired/revoked key must not surface as a raw
  traceback. Wrap that call (or catch inside the provider and return a
  clearly-empty result) so a credential failure reports the same way a
  partial/missing auto-fetch already does (`"auto-fetch only found 0/N clips"`
  today reads oddly for an auth failure specifically — worth a slightly more
  specific message when the provider can distinguish "auth rejected" from
  "network/other error", but the important acceptance bar is: **no
  traceback, clear message, that one pack's combo fails, the rest of the batch
  continues**).

## 5. First real consumer: one Common Voice NL elderly pack — IN SCOPE for this PR

Eric's call: build and test **exactly one** pack against this mechanism before
any other gated pack gets built. Do not author additional Common-Voice packs
(other languages, other age slices, other corpora) in this pass — that's a
deliberate follow-up, gated on Eric's sign-off of this one, not something to
continue into automatically once this pack looks fine.

- Author `packs/common-voice-nl-elderly-batch/pack.yaml` (or similar id) with
  `audio.source.type: "mozilla_data_collective"` and a `credential` block:
  sign-up URL `https://mozilladatacollective.com`, `env_var: "MDC_API_KEY"`,
  instructions covering the account/API-key/`pip install datacollective` steps
  (copy already drafted in the Cowork thread this ADR came from — reuse it,
  adjust only if the actual `datacollective` API shape requires it).
- "Elderly" = Common Voice's own self-reported `age` field, filtered to its
  oldest available bucket(s) for `nl`. Don't assume the exact bucket labels —
  inspect the real Common Voice NL metadata/validated split at authoring time
  and document in the pack's own `README.md`/`metadata` exactly which age
  buckets ended up included and how many speakers/clips that yielded (this
  determines whether the pack is even viable — if the oldest bucket(s) for
  `nl` turn out to be too small to be a meaningful eval set, say so and report
  back rather than padding it with younger speakers silently).
- Mirror `scripts/fetch_fleurs_subset.py` / `fetch_librispeech_subset.py`'s
  existing pattern: a new one-time authoring script (e.g.
  `scripts/build_common_voice_nl_elderly_pack.py`) that uses the
  `datacollective` package (needs *your own* `MDC_API_KEY` to run — this
  authoring step is separate from, and in addition to, the runtime fetch
  provider in §4) to pull the Dutch validated split, filter by age, and emit
  `pack.yaml` + `manifest.jsonl` — same division of labor as FLEURS/LibriSpeech
  already have: an authoring-time script picks and freezes the subset, the
  runtime provider (§4) just re-fetches those exact already-known filenames
  for every later user.
- Set `metadata.age_group`, `metadata.speech_style` (Common Voice is read-aloud
  — same "easy" register as FLEURS, per the Cowork thread's earlier finding
  that read speech understates real spontaneous-speech difficulty; say this
  plainly in the pack's own `README.md` so nobody mistakes its WER for the
  harder JASMIN-style number later) and any other schema-required metadata
  honestly from what's actually in the corpus.

## Acceptance criteria

1. A pack with no `audio.source.credential` behaves byte-identically to today —
   schema change is additive, existing packs (FLEURS, LibriSpeech) untouched.
2. Running the wizard against a batch with one gated pack, with no credential
   set anywhere: prompted once, masked (no plaintext echo — assert via a
   mocked `questionary.password` in tests, not a real terminal capture),
   correct `instructions`/`signup_url` shown.
3. A batch with **two** packs sharing the same `env_var`: prompted exactly
   once, not twice.
4. Declining (empty answer) drops only the affected combos; a batch that also
   has ungated packs still runs those.
5. Second wizard invocation, same machine, credential previously saved: no
   prompt at all — silently resolved from `~/.goesb/credentials.json`.
6. `~/.goesb/credentials.json` is mode `0600` after any save.
7. `capture_environment()`'s output, and any signed result document, never
   contains a credential value — add an explicit test asserting this (search
   the fingerprint/result dict for the literal secret value used in the test
   fixture and assert absence).
8. A revoked/invalid key during actual fetch produces a clear stderr message
   and a normal non-zero-exit combo failure — never an uncaught traceback —
   and the rest of the batch still runs.

## Constraints / reuse

- Reuse `~/.goesb/` as the root (sibling to `keys/` and `cache/`) — don't
  invent a new dotfile location.
- Keep the ADR-0004 "declarative inputs only" line intact: nothing here
  introduces code-as-input; `credential` is pure manifest data.
- Match `_preflight_engines`'s exact continue-past-failure UX (echoed
  skip-reason lines, never a hard abort for the whole batch over one pack).

When done, report back with: files changed, the chosen schema field names if
anything deviated from ADR-0010, exactly which Common Voice NL age bucket(s)
ended up in the pack and how many speakers/clips that is, and explicit
confirmation this is a single-pack pilot awaiting Eric's manual test pass
before anything else gets built on top of it.
