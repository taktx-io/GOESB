# Changelog

All notable changes to GOESB are documented here. Format loosely follows
Keep a Changelog; the project uses semantic versioning once it ships releases.

## [Unreleased]

## [0.3.3] - 2026-07-26
### Fixed
- **`_pick_hardware_id` silently fell back to `custom` on any unmatched
  answer, with no warning.** Typing the catalog id/slug (e.g. `intel-n150`)
  instead of picking the shown label (`Intel N150 (Intel)`) from the
  autocomplete menu resolved to `hardware_id: "custom"` with zero
  indication anything was off — the mistake only surfaced later, on the
  leaderboard, filed under the wrong hardware entirely. Still falls back
  to `custom` (no dead end), but now prints `'<answer>' doesn't match a
  catalog entry — recording hardware as 'custom' instead` unless `custom`
  was actually chosen on purpose via the explicit "Other / not yet in the
  catalog" option.

## [0.3.2] - 2026-07-26
### Changed
- **`goesb submit` now batches under one call-home token instead of one per
  file.** The API's per-IP rate limit on token issuance (20/hour) counts
  *tokens*, not results — submitting one at a time meant a legitimate
  multi-result batch (e.g. the wizard's multi-select submit, or a sweep
  across many profile/pack combos) burned through the exact same budget a
  spam script would, one result at a time. `submit` now accepts one or
  more paths (`goesb submit a.json b.json ...`, still fully backward
  compatible with a single path) and the wizard's "Submit a result" step
  submits every chosen file as one call under a single token — cost is now
  1 token per sitting, not 1 per result. Requires the paired
  `oesb-platform` API change (`POST /benchmark/batch`); a locally-invalid
  file never reaches the network, and one result rejected by the API never
  blocks its siblings.

## [0.3.1] - 2026-07-26
### Fixed
- **Auto-fetched pack audio could get permanently stuck empty.** `--param`
  aside, this hit any multi-pack batch run over an auto-fetchable source
  (FLEURS, LibriSpeech): `_resolve_pack_audio()` trusted the shared,
  content-addressed cache directory's mere existence as proof its audio
  was already fetched. An auto-fetch interrupted mid-stream (network
  blip, Ctrl-C) left that directory created-but-empty by
  `_stream_extract()`'s own `mkdir`, and since every sibling pack pointing
  at the same source reuses that exact path, every one of them then
  failed with `PackAudioMissingError` on every subsequent run — the
  directory "existed," so the runner never even tried to fetch again.
  Fixed by checking the cache directory's contents against the pack's own
  `manifest.jsonl` before trusting it, and only short-circuiting the
  fetch when every wanted file is actually present.

## [0.3.0] - 2026-07-26
### Added
- **Parameterized profile configuration (ADR-0009), runner portion.** A
  profile may now declare a bounded, reviewed set of override-eligible
  `model`/`configuration` keys via a new `overridable` block — the
  default stays the value already in the profile body, one source of
  truth. `goesb run <profile> <pack> --param beam_size=8` (repeatable)
  overrides an eligible parameter for that run only; an undeclared key or
  out-of-domain value is a hard error before anything runs (no silent
  fallback, no clamping — ADR-0008's error philosophy). The signed result
  document gains a `parameters` object recording the resolved value of
  every eligible parameter, including untouched defaults
  (`{"beam_size": {"value": 8, "default": 5}}`) — reuses the existing
  signing path unchanged, since it's just another field
  `canonical_asset_sha256` already covers.
- Every adapter now declares, alongside its own registration, the exact
  set of parameters it *genuinely applies* — "no silent knobs" (ADR-0009
  §2/§6): several adapters accept extra kwargs purely for call-shape
  parity and silently ignore them (whisper-cpp: `beam_size`/`vad`/
  `quantization`; vosk: everything), and declaring one of those
  overridable would sign a result asserting a value that had no effect.
  Profile validation now enforces `overridable ⊆ applied` for the
  profile's adapter — a clear error naming both the parameter and the
  adapter, not a schema-shape check alone.
- All 30 faster-whisper batch profiles declare `beam_size`
  (allowed `[1, 2, 4, 5, 8]`), `vad`, `quantization`
  (allowed `[int8, float32]`), and `threads` (range `1-16`) as
  override-eligible; the one streaming profile additionally declares
  `chunk_ms` (allowed `[250, 500, 1000, 2000]`). All 30 whisper-cpp batch
  profiles declare `threads` only — its adapter never applies `beam_size`/
  `vad`/`quantization` despite accepting them. `temperature` is applied by
  every engine but deliberately excluded everywhere: any value > 0
  introduces sampling nondeterminism, conflicting with the repeat-
  tolerance check (FR-5.3). Each profile bumped to `1.1.0` (or `1.1.0`
  from `1.0.1` for the two hand-authored `whisper-medium-{en,nl}-batch`
  examples) with a changelog entry. vosk profiles unchanged — its adapter
  applies no tunable parameters at all.
- The batch wizard gained a parameter step between engine preflight and
  the repeats prompt, grouped **per engine** — an engine-specific
  parameter can never leak onto an engine that lacks it. Enter accepts
  every profile's own default (a full Enter-through is byte-identical to
  pre-0.3.0 behavior — regression-tested); a single value overrides that
  engine's selected cells; a comma-separated list (`1,4,8`) sweeps them,
  cells × values (× values again if more than one parameter is swept for
  the same engine). Values are validated against each affected profile's
  declared domain before run 1, not run 12 of 15.
### Fixed
- The wizard's batch confirmation now enumerates the full expansion (one
  line per profile × pack × parameter values) and states the honest run
  total **including repeats** — previously it counted combos and silently
  ignored `--repeats`, understating the real batch size (e.g. `--repeats
  2` doubled the actual work without saying so). A soft warning now fires
  above ~20 expanded combos.
### Changed
- Result schema bumped to `schema_version: "0.2"` for the new
  `parameters` field (`additionalProperties: false` required the bump).
  ADR-0008's `runtime.backend` field has not landed yet, so it did not
  ride this bump — see the implementation report for details.

## [0.2.7] - 2026-07-25
### Changed
- The wizard's "Submit a result" step now lets you pick multiple result
  files at once (`questionary.checkbox` — press `a` to select/deselect
  all, `i` to invert, space for one), instead of only ever submitting a
  single file per trip through the wizard. Each submission still runs in
  its own isolated subprocess, continue-past-failure, same as the batch
  run loop — one rejected result doesn't block the rest.
### Added
- After submitting, the wizard offers to delete the result files that
  were just successfully submitted (declined by default — not undoable).
  Files that failed to submit are never offered for deletion, so they
  stay around to retry.

## [0.2.6] - 2026-07-25
### Changed
- Batch runs now confirm every distinct engine they need up front, before
  the run loop starts, instead of prompting per-engine inside each
  combo's own `_reexec`'d subprocess as it's encountered. A batch
  spanning several engines could previously stall for however long it
  took someone to notice an unanswered Y/n prompt hours into an
  unattended run. Declining or failing to install one engine now just
  drops the combos that need it (reported), same continue-past-failure
  behavior as the rest of the batch — the other combos still run.

## [0.2.5] - 2026-07-25
### Fixed
- 0.2.4 shipped a real regression: `run`'s audio auto-fetch correctly
  landed files in the new shared cache directory, but `load_pack()` was
  still called with the original (unset) `--audio-dir` value, so it fell
  back to the pack's own empty directory instead — every auto-fetched
  pack failed with `PackAudioMissingError` immediately after printing
  "Fetched N audio files" successfully. Extracted the whole
  resolve-then-fetch sequence into `_resolve_pack_audio()`, which returns
  one path used for both the fetch and the `load_pack()` call, so the two
  can't drift apart again — and added a regression test that invokes the
  real `run` command end-to-end (previous unit tests of the pieces in
  isolation, and even a first draft of this same regression test, all
  passed despite the bug, because none of them exercised `run()`'s actual
  call site).

## [0.2.4] - 2026-07-25
### Changed
- Simplified 0.2.3's shared audio cache: rather than fetching into the
  shared location and then copying into each pack's own directory,
  `goesb run` now points a pack's audio directly at the shared,
  content-addressed folder whenever nothing already exists at the pack's
  own conventional location — `load_pack()` only ever looks up audio by
  the exact filename each manifest.jsonl entry names, never by scanning
  the directory, so every sibling pack pointing at identical audio can
  read the one shared folder directly. No copying, no linking, no
  duplicate bytes on disk at all.

## [0.2.3] - 2026-07-25
### Changed
- Auto-fetched pack audio is now cached once per `(source.type,
  source.params)` under `~/.goesb/cache/audio/<hash>`, shared across every
  pack that points at the same underlying audio — previously each pack
  fetched into its own directory, so all 11 engine/size sibling packs
  generated for one language (identical FLEURS audio, confirmed by their
  matching `manifest_sha256`) independently re-downloaded the same clips
  the first time each was used. Selecting a whole language row in the
  wizard's matrix now triggers the fetch once, not 11 times.

## [0.2.2] - 2026-07-25
### Added
- Full nl-NL (Dutch) profile/pack coverage — the other 10 of 11
  engine/size combos, alongside the pre-existing `whisper-medium-nl-batch`
  example. Adds the Dutch vosk model (`vosk-model-small-nl-0.22`) to the
  vosk adapter's known-models list.
### Fixed
- The on-the-spot engine installer (`_ensure_engine_installed`, triggered
  the first time the wizard hits a profile whose engine isn't installed
  yet) always shelled out to `python -m pip install`, which fails with
  "No module named pip" under pipx's `uv` backend — that backend creates
  venvs with no pip inside them at all, so every uv-backed pipx install
  of goesb-runner hit this on every platform. Now falls back through
  `ensurepip` and then `uv pip install` (against this exact interpreter)
  before giving up.

## [0.2.1] - 2026-07-25
### Fixed
- `goesb version` (and every place `__version__` is used — `--model-override`
  install prompt, the `/answers` runner version check) reported a hardcoded
  `0.0.3` regardless of the actually installed version, because
  `oesb_runner/__init__.py` hardcoded that string instead of deriving it
  from the package's own metadata. Every release since 0.1.0 shipped this
  stale string unchanged. Now reads `importlib.metadata.version("goesb-runner")`
  instead, so it can't drift from `pyproject.toml` again.

## [0.2.0] - 2026-07-25
### Changed
- The batch wizard's flat checkbox picker (row/column "shortcut" entries
  mixed into a scrollable list) is replaced by a real 2D grid — arrow keys
  move a cursor over language x engine/size, space toggles a single cell or
  (from a header) a whole row/column, enter confirms, escape backs out.
  Built directly on `prompt_toolkit` (which `questionary` itself wraps),
  since none of questionary's own prompts support 2D navigation or escape
  handling. The hardware-picker step gained a "back" option returning to
  the grid without losing the run.
- Retired the standalone "Run a benchmark" wizard action — single-cell
  selection in the new grid covers it, so "Run benchmark(s)" (the renamed
  `_wizard_run`) is the wizard's only run flow now.
- Fixed the hardware picker's completion-menu contrast: `prompt_toolkit`'s
  default style leaves entry text color unset, reading poorly against the
  menu's own background in some terminal themes.

## [0.1.0] - 2026-07-25
### Added
- Curated hardware catalog (`hardware/<id>/hardware.yaml`, ~37 seed CPU/GPU
  entries + a `custom` escape hatch) so a result can carry a user-asserted
  `hardware_id` instead of relying solely on the OS-probed CPU/GPU string,
  which is unrecoverably wrong under virtualization (e.g. a real Xeon
  E3-1240 v6 reports to a QEMU/KVM guest as "QEMU Virtual CPU version
  2.5+"). `goesb run --hardware <id>` and a searchable wizard picker
  (`_pick_hardware_id`, backed by `goesb list-hardware`) let a user assert
  it; auto-detection stays as a diagnostic field. Platform-side: ingest
  validates `hardware_id` against the catalog (optional, no backfill of old
  results), `/hardware/catalog` serves it, `/hardware`'s aggregation groups
  catalog-backed and legacy results separately, and `/answers`/
  `/leaderboards` hardware filtering now keys off `hardware_id`.

### Added (earlier, pre-dating versioned changelog entries)
- M0 foundation: monorepo scaffold (runner, api, web, schemas, profiles, packs).
- Documentation: vision, requirements, architecture, roadmap, glossary.
- ADRs: record-decisions, tech stack, open-source strategy, runner security model.
- Specs: metrics, environment capture. JSON Schemas for profile & pack + CI validation.

### Changed
- Made language-agnostic / multilingual design explicit across docs: language is a
  first-class dimension with per-language pluggable normalization rulesets; no
  language is hardcoded or privileged. Added FR-2.6, strengthened NFR-12, and
  added an English example profile + pack alongside the Dutch ones.
- Added ADR-0006 (cloud-API benchmarks as a reference lane): cloud is included as
  a separated, clearly-labelled baseline via an orthogonal `local` | `cloud`
  deployment axis (FR-1.6), with graceful metric degradation, cloud-native
  metrics, and honest weaker-reproducibility snapshot labelling. Added roadmap
  milestone M5b. Edge remains the platform identity.
