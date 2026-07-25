# Changelog

All notable changes to GOESB are documented here. Format loosely follows
Keep a Changelog; the project uses semantic versioning once it ships releases.

## [Unreleased]

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
