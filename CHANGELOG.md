# Changelog

All notable changes to GOESB are documented here. Format loosely follows
Keep a Changelog; the project uses semantic versioning once it ships releases.

## [Unreleased]

## [0.7.0] - 2026-07-28
### Changed
- **ADR-0011: packs decoupled from profiles, joined on language instead of
  `profile_id`.** `pack.yaml`'s `profile_id` is no longer required (kept,
  informational only) — `goesb run` now checks `pack.metadata.language ==
  profile.language` instead of an exact `profile_id` pin, both a hard error
  before anything runs if they don't match. One pack per (language × audio
  source) is now usable across every profile in that language — no more
  authoring ~11 near-identical packs per language to fake an (engine × size)
  matrix. The wizard's pack-picker checkbox (already built for "more than
  one pack matches") now fires for most languages instead of almost never;
  no UI changes needed for that. `list-packs` gained a LANGUAGE column.
- **Retired the sibling-pack workaround.** The ~58 duplicate packs
  `scripts/generate_bulk_assets.py` generated to fake the old matrix (10 per
  language across the `de`/`es`/`fr`/`pt`/`nl` FLEURS set, 8 for the `en`
  LibriSpeech set) are deleted — every result referencing them is an
  internal benchmark run, reproducible on the same hardware, so nothing
  external is orphaned. The audio-source packs they duplicated
  (`fleurs-*-batch`, `librispeech-en-batch`, and the hand-authored
  `librispeech-en-whispercpp-batch`/`librispeech-en-vosk-batch`) are
  untouched and now cover every batch profile in their language directly.

## [0.6.0] - 2026-07-28
### Added
- **Bulk hardware catalog addition: 80 -> 985 entries.** 846 CPU (Intel
  Core/Xeon + AMD Ryzen/EPYC, 2014-2026) + 59 GPU (AMD Radeon + Intel
  Arc, 2019-2026). No runner/API code changes — the catalog is data,
  already served by the existing `/hardware/catalog` endpoint and
  wizard picker.
- **ADR-0008 implemented: explicit compute backend.** `goesb run` gains
  `--backend` (`cpu`/`cuda`), defaulting to `cpu` and always passed
  explicitly to the underlying library (`device=` for faster-whisper,
  `context_params={"use_gpu": ...}` for whisper.cpp) — never left to
  ctranslate2's own `device="auto"` default, which silently tried CUDA
  whenever a GPU looked present. Root cause of a real Windows install
  failure: cuBLAS/cuDNN often come bundled for free via the Linux pip
  wheel, so `device="auto"` mostly worked by accident there; Windows has
  no equivalent auto-bundling, so "GPU present, CUDA libraries not
  installed" failed deep inside model load with a confusing error.
  Requesting a backend an adapter doesn't declare support for (a new
  per-adapter `backends` registration, defaulting to cpu-only — vosk
  needs no change, it's genuinely CPU-only) is a hard error before the
  engine-install prompt, pack resolution, or any model load, same
  explicit/early/never-silent standard as ADR-0009's `--param`
  validation. `--backend cuda` on a CTranslate2 build without CUDA
  support now fails with a clear, actionable message instead of a bare
  third-party stack trace.
- **`goesb doctor`.** Reports detected accelerators (reusing
  `environment.py`'s existing `nvidia-smi`-based GPU probe) and, per
  installed engine, whether the backends it declares are actually usable
  right now — for faster-whisper, a real (read-only) check via
  `ctranslate2.get_cuda_device_count()`, not just "a GPU exists." Detects
  and suggests; never runs anything or changes any state. `pip install
  "goesb-runner[faster-whisper]"` (macOS/Linux) and the Windows installer
  both nudge toward it as the next step after install.
- **Result schema bumped to `schema_version: "0.3"`** for `runtime.backend`
  (required, `additionalProperties: false`) — `"0.2"` was already claimed
  by `[0.3.0]`'s `parameters` field and backend didn't ride that bump.

## [0.5.0] - 2026-07-28
### Added
- **Per-clip audio integrity check.** `manifest.jsonl` entries may now
  declare an optional `audio_sha256` (content hash of the audio as
  originally captured when the pack was authored). `oesb_runner.pack.
  load_pack` checks fetched/local audio against it when present and raises
  `PackIntegrityError` on mismatch — catches upstream (Common Voice/MDC,
  FLEURS, LibriSpeech) silently serving different bytes behind the same
  filename, which previously went undetected and would silently change
  what a run measured. Optional/backward-compatible: manifest entries
  without it load exactly as before.
- **`min_runner_version` pack field.** A pack can now declare the lowest
  `goesb-runner` version able to run it correctly. Checked against the
  installed runner's own version before anything else — an old install
  gets a clear `pip install --upgrade goesb-runner` message instead of
  either a raw schema error or, worse, silently running without a
  guarantee it doesn't know to check (exactly what would otherwise happen
  with the new `audio_sha256` field above, since `manifest.jsonl` has no
  schema of its own to gate on). `fetch_fleurs_subset.py`,
  `fetch_librispeech_subset.py`, and `build_common_voice_nl_elderly_pack.py`
  now stamp `min_runner_version: 0.5.0` on every pack authored from here
  on. Existing already-published packs get neither field — retrofitting
  either would change `manifest.jsonl`/`pack.yaml`'s bytes, breaking their
  already-published, immutable hashes; a separate, larger migration, not
  done here.

## [0.4.1] - 2026-07-27
### Fixed
- **Clearer error for a pack this runner predates.** A pack whose
  `audio.source.type` isn't in this runner's bundled schema (e.g. a pack
  using `mozilla_data_collective` fetched by a pre-0.4.0 install) used to
  fail with a raw jsonschema enum-mismatch message. Now names the specific
  unrecognized type and points at `pip install --upgrade goesb-runner`.
  Only helps installs made from this version onward — doesn't retroactively
  fix already-installed runners, which still need to upgrade to see it.

## [0.4.0] - 2026-07-27
### Added
- **Gated-pack credential handling (ADR-0010).** Some high-value corpora
  (Mozilla Common Voice via the Mozilla Data Collective platform) are free
  but access-gated behind a personal API key, since the platform's own
  terms forbid GOESB re-hosting the audio. `audio.source.credential` is a
  new optional pack-schema field (`env_var`/`signup_url`/`instructions`);
  existing packs are untouched (additive/optional). The wizard gets a new
  preflight step, `_preflight_pack_credentials`, that dedups by `env_var`
  across a whole batch (prompts once even if several selected packs share
  one), masks the input (`questionary.password`, never echoed), and stores
  it in `~/.goesb/credentials.json` (mode 0600, sibling to the existing
  `~/.goesb/keys/` signing-key convention) so it's asked at most once per
  machine. Declining drops only the combos that needed it — never aborts
  the batch. The credential is never sent to GOESB's own servers, and
  never appears in `capture_environment()`'s output or a signed result
  document. New `mozilla_data_collective` auto-fetch provider
  (`fetch_common_voice_audio`) reuses the credential to fetch audio at run
  time; a rejected/expired key reports a clean stderr message and fails
  just that one combo, never a raw traceback.
- **Wizard pack-picker.** When more than one pack targets the same profile
  (now possible thanks to the above — e.g. an ungated FLEURS pack and a
  gated Common-Voice variant both targeting `whisper-medium-nl-batch`),
  the wizard asks which pack(s) to run for that cell instead of silently
  taking the first match. Checkbox, not single-select — running more than
  one pack for the same profile in one batch is a reasonable thing to
  want. The first ungated pack is pre-checked by default, so Enter with no
  changes reproduces old single-pack-per-profile behavior exactly; every
  profile with only one matching pack (everything except the new pilot
  pack, today) is entirely unaffected, no prompt at all.
- **`common-voice-nl-elderly-batch` pack** — the first (pilot) consumer of
  the above: Dutch, Common Voice's oldest self-reported age buckets
  (seventies+eighties+nineties — the oldest bucket alone, and even the two
  oldest combined, were too small on their own to be a meaningful eval
  set). 40 clips, 6 distinct speakers, 270.5s total. Single-pack pilot,
  gated on manual sign-off before any further Common-Voice packs get
  built — see `packs/common-voice-nl-elderly-batch/README.md` for the full
  breakdown and an explicit caveat that 6 speakers is thin, anecdotal
  signal rather than a statistically robust read on elderly-speaker
  performance.

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
