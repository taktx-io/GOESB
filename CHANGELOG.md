# Changelog

All notable changes to GOESB are documented here. Format loosely follows
Keep a Changelog; the project uses semantic versioning once it ships releases.

## [Unreleased]

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
