# Changelog

All notable changes to GOESB are documented here. Format loosely follows
Keep a Changelog; the project uses semantic versioning once it ships releases.

## [Unreleased]
### Added
- **`nemotron`, a fifth runtime adapter — the first non-Kaldi engine with a
  genuinely cache-aware incremental streaming path** (ADR-0013). Backed by
  `nvidia/nemotron-3.5-asr-streaming-0.6b` via `transformers`'
  `AutoModelForRNNT` (no `nemo_toolkit`), implementing `batch`, `streaming`
  and `concurrency`. `run_streaming` feeds `generate()` a generator of
  fixed-geometry mel chunks and carries the encoder's `padding_cache` across
  them — each second of audio is encoded exactly once — instead of the
  bounded-window re-decode the Whisper-family adapters must simulate
  streaming with. Because a streaming RNNT never revises an emitted token,
  every published partial word is final: `partial_stability` is exactly 1.0
  and `first_partial_latency` equals `first_final_latency`, both now
  documented in `docs/specs/metrics.md` as metrics that only discriminate
  among the re-decode engines.
- **`configuration.streaming_latency_ms`**, a new profile parameter,
  deliberately not `chunk_ms`: it is a cache-aware engine's encoder
  right-attention context (which also fixes the exact mel-frame count every
  chunk carries), not a re-decode window an adapter picks. Requesting a mode
  the checkpoint doesn't support is a hard error, never snapped to the
  nearest one. Measured against the real checkpoint: four modes
  (80/320/560/1120 ms), not the five NVIDIA's model card lists.
- **13 `nemotron-3-5-*` profiles** — batch and streaming for en/nl/de/fr/es/pt
  plus one language-less concurrency profile, all GPU-only: `--backend cpu`
  against them is a hard error (ADR-0008), and `metal` is declared on the
  strength of a real Apple Silicon measurement (batch RTF 0.072x, streaming
  0.216x at the 320 ms mode).

### Fixed
- **`parakeet` (and now `nemotron`) were missing from the frozen-build hash
  manifest**, so `hashing._frozen_module_hash` would have raised
  `ValueError: no precomputed hash for ...` inside any standalone binary
  built for them. Latent rather than live — the release matrix builds no
  such binary, and deliberately still doesn't (both bundle PyTorch, which
  would blow past GitHub's per-asset size limit) — but the generator now
  covers all five adapters so adding one later can't crash at runtime. The
  manifest itself is a gitignored build artefact regenerated before every
  freeze, so this is a one-line generator change, not a checked-in file.


## [0.9.26] - 2026-08-15
### Fixed
- **Every pack failed on Windows with `manifest.jsonl hash mismatch`,
  whichever profile you ran.** A pack's `manifest_sha256` is taken over the
  bytes upstream serves — LF-terminated UTF-8 — but the manifest reached
  disk through Python text mode (`Path.write_text` in `remote.fetch_pack`)
  and, for a git checkout, through Git for Windows' default
  `core.autocrlf=true`. Both rewrite every `\n` as `\r\n`, so the bytes on
  disk could never hash to the declared value; text mode also encodes in
  the locale codec (cp1252 on Windows), which mangles or crashes on the
  non-ASCII references in the nl/fr/de/es/pt packs. Fetches now write
  UTF-8 with newline translation off, reads are explicitly UTF-8, a
  `.gitattributes` pins hashed assets to LF, and `load_pack` accepts a
  manifest that differs from its declared hash *only* by CRLF translation
  (a genuine content difference is still rejected) so caches and clones
  already on disk aren't stuck.
- **A Windows source checkout also recorded a different `runtime.sha256`
  than every other platform for identical adapter code** — that field is a
  hash of the adapter's own `.py` source bytes (the "which reviewed code
  produced this result" fingerprint), so a CRLF checkout silently split
  results for the same code into two fingerprints. Same `.gitattributes`
  fix; pip wheels and the frozen binaries were never affected. Results
  produced from a CRLF checkout should be discarded and re-run.

## [0.9.25] - 2026-08-05
### Fixed
- **`goesb` wizard crashed with `KeyError: 'large-v3-turbo'` the instant
  you opened the batch/streaming matrix** — a real user-reported crash on
  0.9.24. `_MATRIX_SIZE_SHORT` (the column-header abbreviation table
  render() indexes by size string) wasn't updated alongside
  `_MATRIX_SIZES`/`_MATRIX_ID_RE` when large-v3-turbo profiles were
  added; the wizard-matrix test added for that same change only checked
  `_build_matrix`'s output shape, not the separate dict render() actually
  looks up. Added `"large-v3-turbo": "LT"`, and a direct test asserting
  every `_MATRIX_COLUMNS` entry resolves in both short-label dicts.

## [0.9.24] - 2026-08-05
### Changed
- **`wer`/`cer` `spread` now pools per-recording ratios, not per-repeat
  values.** Real production feedback (Babbl): the old spread was computed
  over the (typically 2) corpus-aggregate repeat values — degenerate
  (`std` exactly 0) for any deterministic decoder, silently hiding a
  bimodal per-recording failure distribution a corpus-level mean can't
  show. Always attached now (not gated on `repeats > 1`), matching the
  latency-metric convention. See docs/specs/metrics.md "Reporting".
### Added
- **`wer`/`cer` substitution/deletion/insertion breakdown** published as
  their own metric ids (`wer_substitutions`/`wer_deletions`/
  `wer_insertions`, and the `cer_*` equivalents) — a bare WER/CER ratio
  can't distinguish "hears worse" from "runs away," two different
  failure modes with two different fixes.
- **`model.context_reset`/`model.vad` now required on every batch/streaming
  profile** (schema + `validate_assets.py` enforcement) — makes decoder
  context handling an explicit, reviewable fact instead of something a
  reader has to infer from adapter source.
- **`whisper-large-v3-turbo` profiles**, both runtimes (faster-whisper,
  whisper-cpp), all 6 languages + concurrency — verified real support in
  each runtime (not assumed) before adding, and confirmed end-to-end on
  real Dutch audio.
### Fixed
- **The interactive wizard's language x engine/size matrix silently
  dropped any profile whose size wasn't in a fixed, hardcoded list** —
  found while adding the `large-v3-turbo` profiles above, which the
  matrix regex/size list didn't recognize at all (no error, just absent
  from the grid). `large-v3-turbo` added to both.

## [0.9.23] - 2026-08-04
### Added
- **`goesb submit`'s comment/callsign credit is now reachable from the
  interactive wizard**, not just `goesb submit --comment`/`--callsign`
  directly — `_wizard_submit` now prompts for both (comment optional;
  callsign reuses the same 5-case `resolve_identity` flow the standalone
  command already used) right before submitting, rather than being a
  second-class path to submit results.
### Changed
- **Clearer secret-passphrase prompt.** The wizard/`submit --callsign
  <new>`/`set-identity` prompt for a new callsign's secret used to say
  only "(not stored, used only to distinguish identical callsigns from
  different people)". Now spells out the actual mechanism up front: what
  it's for (no real account system — this is what lets two people who
  pick the same callsign show up as distinct leaderboard entries) and
  precisely what happens to the secret (used once in memory to derive a
  discriminator via PBKDF2-SHA256, then discarded — never written to
  disk, never sent over the network; only the callsign + derived
  discriminator are saved).

## [0.9.22] - 2026-08-04
### Added
- **Parakeet `concurrency` benchmark type** (ADR-0012): N independent
  model instances, one per worker, not a shared model — confirmed by
  reading transformers' actual `generate()` override for this model
  (`generation_parakeet.py`): `ParakeetRNNTGenerationMixin.generate()`
  mutates plain instance attributes during decode, unsafe to share
  across threads. Same real constraint whisper.cpp has, same tighter
  `concurrency` ceiling (16, not faster-whisper's 64). New
  `parakeet-tdt-v3-concurrency` profile. Real audio: throughput scales
  10.5x/13.7x/21.1x realtime at concurrency 1/2/4 on an Apple M1 Pro CPU.
- **Parakeet now covers all 6 languages** batch/streaming already have
  for the Whisper-family engines (en/de/es/fr/nl/pt — nl already
  existed). New `scripts/generate_bulk_parakeet_assets.py`, reusing
  already-fetched packs (no new audio fetches). Real per-language batch
  results (Apple M1 Pro CPU): en WER 3.3%/RTF 0.077x, de WER 3.5%/RTF
  0.090x, es WER 0.8%/RTF 0.075x, fr WER 8.7%/RTF 0.105x, pt WER
  1.6%/RTF 0.075x — all genuinely realtime.
### Fixed
- **Silenced Parakeet's cosmetic `max_length` warning.** `model.generate()`
  without an explicit `max_new_tokens` always emitted `UserWarning: Using
  the model-agnostic default max_length=...` on every single utterance —
  confirmed harmless by reading Parakeet's own `generate()` override:
  `max_length` is already a generous, input-proportional output-buffer
  size (`max_symbols_per_step * encoder_output_length`), not the real
  stop condition (encoder exhaustion). Was pure log-line noise.

## [0.9.21] - 2026-08-03
### Added
- **NVIDIA Parakeet-TDT engine** (`parakeet` runtime), batch + streaming.
  Fully open (CC-BY-4.0 weights, Apache-2.0 toolkit), multilingual —
  parakeet-tdt-0.6b-v3 is trained on Granary (25 European languages
  including Dutch), one checkpoint instead of a per-language model swap.
  Via `transformers`' native `AutoModelForTDT`/`AutoProcessor`, not
  `nemo_toolkit` (much heavier: pytorch-lightning, hydra-core, onnx, ...
  none of it needed here). New `parakeet-tdt-v3-nl-batch`/`-streaming`
  profiles. Real Dutch FLEURS results (full 15-clip fleurs-nl pack, Apple
  M1 Pro): batch WER 6.2%, streaming WER 6.57% (bounded-window re-decode,
  same shared `run_windowed_local_agreement_streaming` faster-whisper/
  whisper.cpp already use — Parakeet has no genuine incremental encoder
  path through `transformers`, just like them).
- **`--backend metal` for Parakeet** (real Apple Silicon MPS via plain
  torch — not gated by a compile-time flag the way whisper.cpp's ggml
  Metal is). Real, measured: batch RTF 0.0395x (~4x faster than cpu's
  0.155x), streaming RTF 0.315x (~2.8x faster than cpu's 0.876x), WER
  byte-identical to cpu on both. A one-off ~3.4s MPS JIT/kernel-compile
  cost on first inference is absorbed by a warm-up call before the timed
  loop starts (same "one-off cost excluded from RTF" category model load
  already gets), so it never inflates utterance #1's own timing.
- **whisper.cpp CUDA auto-build.** `pywhispercpp` ships no prebuilt CUDA
  wheel anywhere — `pip install pywhispercpp` is always cpu-only. `goesb
  run --backend cuda` and the interactive wizard now detect a real
  NVIDIA GPU + CUDA Toolkit (`nvcc`) and offer to rebuild it from source
  with `GGML_CUDA=1` automatically, instead of a manual fix repeated on
  every fresh GPU box. Always ends the process after a successful
  rebuild and asks for a re-run — a freshly-built extension can't take
  effect in a process that already imported the old one.
### Fixed
- **`goesb doctor`'s parakeet line silently hid real Metal readiness on
  every Mac.** It sat behind a `gpu is None` early-return shared with
  faster-whisper — correct for faster-whisper (cuda-only, no Metal
  support in ctranslate2 at all) but wrong for parakeet once `metal`
  became a real backend, since Metal readiness has nothing to do with
  the NVIDIA-only `gpu` probe. Moved parakeet's own reporting above that
  early-return; confirmed live before/after on an Apple M1 Pro.

## [0.9.20] - 2026-08-03
### Fixed
- **Removed 5 redundant streaming packs** (`fleurs-{de,es,fr,nl,pt}-streaming`)
  added in 0.9.19 — they duplicated exactly the sibling-pack pattern
  ADR-0011 already retired: a pack is eligible for any profile whose
  `language` matches, regardless of `benchmark_type`. The existing
  `fleurs-<lang>` packs already covered every new streaming profile with
  zero changes. Caught by a real oesb-platform test before anything
  deployed — no user-facing impact.

## [0.9.19] - 2026-08-03
### Added
- **Streaming now covers the same 6 languages x 12 engine/size combos
  batch does** (en/de/es/fr/nl/pt x faster-whisper/whisper-cpp at 5
  sizes + vosk at small/medium — 72 profiles total, up from 4).
  Streaming scores WER same as batch (unlike concurrency, which is
  language-agnostic), so accuracy is genuinely language-dependent and
  streaming shouldn't stay English-only. New
  `scripts/generate_bulk_streaming_assets.py` mirrors the existing
  batch generator. One shared streaming pack per new language, not one
  per engine — confirmed against the real batch packs that non-English
  languages already share a single pack across all 12 combos.

## [0.9.18] - 2026-08-03
### Added
- **whisper.cpp streaming**, third streaming engine, real Metal GPU
  acceleration. `whisper.cpp`'s Metal backend (already wired for batch/
  concurrency) is genuinely GPU-accelerated on Apple Silicon, unlike
  faster-whisper's ctranslate2 backend (no Metal support at all).
  Measured live on an Apple M1 Pro: RTF 0.06-0.09x with the tiny tier —
  the only Whisper-family streaming path on this hardware that's
  actually realtime-capable. New `whispercpp-tiny-en-streaming` profile.
  The bounded-window local-agreement streaming loop built for
  faster-whisper's own fix is now a shared
  `streaming.run_windowed_local_agreement_streaming`, used by both
  engines instead of being duplicated.
### Changed
- **`whisper-medium-en-streaming` is back in the wizard's streaming
  matrix.** It was briefly excluded after measuring RTF 3.19x on one
  Apple M1 Pro CPU — but GOESB is hardware-generic (docs/00-vision.md):
  that's a property of that machine, not of the profile. `--backend
  cuda` on real NVIDIA hardware, or a different CPU, may give a
  completely different answer, and that's exactly what the benchmark
  should measure and show, not hide.

## [0.9.17] - 2026-08-03
### Fixed
- **faster-whisper streaming's re-decode window is now bounded, not the
  whole clip.** The previous version re-decoded the entire clip so far
  every chunk (unbounded), measuring RTF 3.19x on Apple M1 Pro and
  understating its own latency numbers once behind realtime. Now only
  audio since the last committed word gets re-decoded. Three real
  correctness bugs (word-dropping at the trim seam, then duplication
  from the fix for that, then a case-sensitivity gap in the fix for
  that) were found and fixed via real-audio validation before landing —
  see `run_streaming`'s own docstring for the full trail. Net: RTF ~2.5x
  (consistent, real improvement) and WER 0.110 vs the original's 0.078
  (an honest, expected cost of bounded context, not corruption). Still
  not realtime-capable on CPU — an inherent Whisper architecture cost,
  not a bug — so `whisper-medium-en-streaming` stays excluded from the
  wizard's streaming matrix.

## [0.9.16] - 2026-08-01
### Added
- **Streaming matrix in the interactive wizard.** "Run benchmark(s)"
  previously only matched `<engine>-<size>-<lang>-batch` profile ids —
  streaming profiles had no entry point anywhere in the wizard. Added a
  "Run streaming benchmark(s)" menu item with the same language x
  engine/size matrix picker batch already has.

## [0.9.15] - 2026-08-01
### Added
- **vosk streaming adapter** — second engine for the `streaming`
  benchmark type, alongside faster-whisper. vosk's `KaldiRecognizer`
  carries real incremental decoder state across chunks (unlike
  faster-whisper's whole-buffer re-decode): `AcceptWaveform` only ever
  sees new audio, and Kaldi's own endpointing genuinely finalizes text,
  so committed words never need revising. New `vosk-small-en-streaming`
  / `vosk-medium-en-streaming` profiles and a `librispeech-en-vosk-streaming`
  pack.

## [0.9.14] - 2026-08-01
### Fixed
- **`--backend cuda` no longer re-prompts to install `nvidia-cublas-cu12`
  on every run, even right after installing it.** Real report on an RTX
  5090. The check used `cublas_loadable()` alone, which only does a bare
  `dlopen` against the OS loader's *default* search path — a pip-wheel
  install's `nvidia/cublas/lib/` is never on it, so the check kept saying
  "not installed" forever after. Now uses `preload_installed_cublas()`,
  the function that actually accounts for the pip-wheel location too.
- **Profiles and packs no longer get re-fetched over the network
  multiple times within one wizard run.** Real report ("packs and
  profiles ... each time it seems to reinstall"). A prior fix made
  `fetch_profile`/`fetch_pack` always hit the network (correctly, to
  avoid serving a stale cached copy) — but a single wizard run calls
  `_load_profile_for_wizard`/`_load_pack_for_wizard` several times for
  the *same* profile/pack (preflight, parameter resolution, hardware
  picks, ...), and each call re-fetched. Now memoized per process: one
  fresh fetch per id per `goesb` invocation, not one per call — staleness
  across separate runs is unaffected, since each `_reexec`'d run is its
  own fresh process.

## [0.9.13] - 2026-08-01
### Fixed
- **vosk concurrency profiles are no longer split per language.**
  0.9.11 shipped 12 `vosk-{small,medium}-{lang}-concurrency` profiles,
  matching vosk's batch profiles — wrong: concurrency measures hardware
  capacity, not transcription accuracy, so which language's model backs
  it doesn't matter. Replaced with 2 profiles
  (`vosk-{small,medium}-concurrency`), the same naming shape every other
  engine's concurrency profiles already use (no language segment) — each
  still has to pick one concrete model to load (vosk models are
  themselves language-specific), English chosen arbitrarily. See
  ADR-0012's addendum.

## [0.9.12] - 2026-08-01
### Changed
- **Auto-sweep is more robust and better-resolved near the plateau.**
  Requires two consecutive low-gain levels before stopping (a single
  noisy reading no longer ends the sweep early), and switches from
  doubling to a flat +4 step once concurrency reaches 8 — `1, 2, 4, 8,
  12, 16, 20, 24, ...` instead of `1, 2, 4, 8, 16, 32, ...` — so the
  levels near a real knee are close enough together to actually find it,
  not just bracket it loosely. See ADR-0012's addendum.

## [0.9.11] - 2026-08-01
### Added
- **vosk concurrency support** — `whisper-cpp` and `faster-whisper` had it,
  vosk didn't. `run_concurrency` builds one full `vosk.Model` instance per
  worker, not a shared one — alphacep/vosk-api#606 documents a real SIGSEGV
  from racing `KaldiRecognizer` construction against a shared model's C++
  reference counting, and that can't be independently confirmed fixed from
  the Python binding alone. See ADR-0012's addendum. 12 new
  `vosk-{small,medium}-{en,es,fr,de,nl,pt}-concurrency` profiles.
- **vosk's larger per-language model tier**, alongside the existing small
  one — this project had only ever wired up vosk's ~40-50MB models;
  Vosk itself always published a bigger, more accurate model per language
  too (e.g. en-us: 1.8GB, WER 5.69 vs the small tier's 40MB, WER 9.85).
  6 new `vosk-medium-{lang}-batch` profiles; the wizard's batch matrix now
  shows both vosk columns.

## [0.9.10] - 2026-08-01
### Changed
- **`gpu_pct` now reads NVML directly** via `nvidia-ml-py` (optional, the
  `cuda` extra) instead of shelling out to the `nvidia-smi` CLI and parsing
  its CSV output. Same data — `nvidia-smi`'s own utilization.gpu column is
  itself a thin wrapper over the same `nvmlDeviceGetUtilizationRates()`
  call this now makes directly — but in-process, like the CPU/RAM sampler,
  instead of a subprocess spawn per sample. Still NVIDIA-only: CUDA's own
  runtime/driver API has no system-utilization query, NVML is the correct
  (and only) layer for this on NVIDIA hardware, and no vendor-neutral
  equivalent exists for other GPUs.

## [0.9.9] - 2026-08-01
### Added
- **Wizard: auto-detect the useful max concurrency, instead of guessing a
  static sweep.** Blank Enter on the `concurrency` prompt now runs
  concurrency=1, reads back its own throughput, doubles, and keeps doubling
  only while each doubling still buys at least 15% more throughput —
  stopping at the knee (or the profile's own
  `overridable.concurrency.range.max`, whichever comes first) instead of
  the previous static `1,4,8,16` guess. Typing your own comma-separated
  list still works exactly as before and always takes priority — the
  prompt states both options explicitly. See ADR-0012's 2026-08-01
  addendum.

## [0.9.8] - 2026-07-31
### Added
- **whisper-cpp concurrency support** (`metal`/`cuda`/`cpu`) — five new
  `whispercpp-{tiny,base,small,medium,large-v3}-concurrency` profiles.
  Needed a genuinely different `run_concurrency` than faster-whisper's:
  pywhispercpp isn't safe to share one `Model` instance across concurrent
  threads (confirmed against the actual bound C API), so this builds one
  full instance per worker instead — a real N-way memory cost, reflected
  in a tighter `overridable.concurrency.range.max` (16 vs faster-whisper's
  64). See ADR-0012's addendum.
- **More faster-whisper concurrency profiles** —
  `whisper-{tiny,base,small,large-v3}-concurrency` join the existing
  `whisper-medium-concurrency`.
- **Wizard: blank Enter on the `concurrency` prompt now suggests a real
  sweep** (`1,4,8,16`, clamped to the profile's own range) instead of
  silently running once at the default — a single concurrency=1 run
  shows nothing about load behavior, the whole point of this
  benchmark_type.

## [0.9.7] - 2026-07-31
### Added
- **`GOESB_API_URL` environment variable** overrides the default
  `https://www.goesb.com/api` for every subcommand, including the
  interactive wizard — which had no `--api-url` flag of its own to
  point it at a non-production API (e.g. `test.goesb.com`'s API,
  reachable at `http://test.goesb.com:8001`, not under `/api`).
  `export GOESB_API_URL=http://test.goesb.com:8001 && goesb` now Just
  Works instead of needing a manual code edit.

## [0.9.6] - 2026-07-31
### Added
- **New `concurrency` benchmark_type (ADR-0012): does a GPU/CPU stay fast
  under several simultaneous requests, not just one at a time?** Real
  motivation: comparing an RTX A4000 against an RTX 4090 on the
  leaderboard showed near-identical RTF at every model size — because
  every existing benchmark runs one utterance at a time, and small-batch
  Whisper decoding at that concurrency is latency-bound, not
  GPU-compute-bound, so a much more powerful card never got to show a
  difference. `faster-whisper`'s adapter now drives real concurrent
  inference (`WhisperModel(num_workers=N)`, ctranslate2's own
  `inter_threads` mechanism — GOESB never set this before). Concurrency
  levels sweep the same way `--param beam_size=1,4,8` already does; no
  `language` is needed for this benchmark type (it doesn't score
  accuracy), but it still reuses a normal, hash-verified pack for audio
  content rather than bypassing the pack system. New wizard flow: "Run
  concurrency/load benchmark(s)".
- **`gpu_pct` and `throughput` metrics** — both documented in
  `docs/specs/metrics.md` from early on, neither previously implemented.
  `gpu_pct` samples `nvidia-smi` on its own coarser 1s cadence inside the
  existing CPU/RAM/temperature sampler (each sample spawns a subprocess,
  unlike the psutil/hwmon reads next to it). Available on any profile
  that declares them, including existing batch/streaming ones — no
  existing profile is affected until it opts in.

## [0.9.5] - 2026-07-31
### Added
- **`goesb run --backend cuda` now offers to install a missing cuBLAS
  runtime automatically, on Linux.** Real report: a fresh Ubuntu box with
  an NVIDIA driver but no (or a version-mismatched) system CUDA Toolkit
  crashed the first time faster-whisper actually used the GPU —
  ctranslate2 dlopen's `libcublas.so.12` lazily and declares no pip
  dependency on it (confirmed against the actual PyPI wheel: no bundled
  library, no NEEDED entry). NVIDIA also publishes cuBLAS as a standalone
  pip wheel, `nvidia-cublas-cu12` — the same mechanism PyTorch's cu12x
  wheels use — so `goesb run` now offers to install and preload it
  automatically instead of requiring a manual `pip install` after the
  crash. New optional extra: `pip install "goesb-runner[cuda]"`.
### Fixed
- **`--backend cuda` failures other than "not compiled with CUDA support"
  used to surface as a raw, uncaught exception** instead of the same
  clear, actionable message — the missing-cuBLAS crash above is exactly
  this case: ctranslate2's dlopen failure isn't reliably a `ValueError`
  with "CUDA" in it. Broadened to catch the actual shapes this fails in
  and match on cuda/cuBLAS/cuDNN in the message, not one hardcoded phrase.

## [0.9.4] - 2026-07-31
### Fixed
- **Wizard: hardware is now asked once per distinct compute backend, not
  once for the whole batch.** A batch mixing `cuda` for one engine and
  `cpu` for another used to apply one shared `--hardware` answer to
  every combo — wrong for whichever backend didn't match it. Now asks
  "which GPU" and "which CPU" separately, per backend actually in use,
  and applies each answer only to the combos that use that backend.
- **Wizard: pack choice is now asked once per language, not once per
  profile/engine.** A matrix selection spanning several engines that
  share one language (e.g. `whisper` and `vosk`, both `nl-NL`) used to
  prompt the identical "which pack(s)" question once per engine. The
  choice is now scoped to the language and reused for every profile
  sharing it.

## [0.9.3] - 2026-07-31
### Fixed
- **Wizard batches were slow between combos** — each `_reexec`'d `goesb
  run` subprocess was independently paying the `[0.9.1]` outdated-runner
  network check (fresh DNS/TLS per process, no shared cache), so an
  N-combo batch made N redundant round-trips to the same `/health`
  endpoint. The wizard now does that check once, up front, and sets an
  internal env var its `_reexec`'d children inherit to skip their own.

## [0.9.2] - 2026-07-30
### Fixed
- **Wizard: compute backend is now asked before hardware, not after.** The
  hardware picker's auto-guess used to be CPU-only by necessity — the
  backend hadn't been chosen yet, so a machine with both a CPU and an
  NVIDIA GPU always got its CPU catalog entry preselected even when
  `cuda` was picked afterward. Reordering gives the guess real signal:
  choosing a GPU backend now preselects the matching GPU catalog entry
  (via the same `nvidia-smi` probe `goesb doctor` already uses), falling
  back to no preselection rather than a wrong CPU one if nothing matches.
  Dropped the hardware step's "back to matrix" option as part of this —
  it's no longer adjacent to the matrix step, and nothing else in the
  wizard has step-back navigation either.

### Added
- **NVIDIA RTX A4000 hardware catalog entry** (plus RTX A2000/A5000/A40
  and RTX 2000 Ada) — the A4000 specifically is one of the most common
  cloud-rental cards and was missing, causing real GPU results to fall
  back to `hardware_id: "custom"`.

## [0.9.1] - 2026-07-30
### Added
- **`goesb run` fails fast if this install is older than what the platform
  currently accepts**, instead of only finding out at `goesb submit` time
  after a long benchmark has already run. Best-effort and network-optional
  (a short, silently-ignored `/health` check) — `run` still never *requires*
  network access; `--offline` skips the check outright, same as it already
  skips profile/pack fetches. Companion to the `MIN_RUNNER_VERSION` bump in
  `[0.9.0]` — see `_warn_if_runner_outdated` in `cli.py`.

## [0.9.0] - 2026-07-30
### Added
- **`goesb run` prints a formatted results table** (via `rich`) instead of
  flat `metric: value ± std unit` lines — a little service to the user,
  no schema change.
- **Optional submission comment.** `goesb submit --comment "..."` (max 500
  chars) attaches a free-text note to every result in that batch.
- **Optional pseudonymous credit.** `goesb submit --callsign NAME` (or the
  interactive prompt, pre-filled with whatever was last used) credits a
  submission on the public leaderboard as `NAME#<discriminator>`. The
  discriminator is `PBKDF2-HMAC-SHA256(secret, salt=callsign, 200_000
  iterations)`, truncated to 8 hex chars — the secret passphrase is used
  once, in memory, to derive it and is never written to disk or sent over
  the network, only the (callsign, discriminator) pair is. This lets two
  different people who pick the same callsign show up as distinct
  entries without goesb needing real accounts. Persisted locally at
  `~/.goesb/identity.json` (new `set-identity`/`clear-identity` commands
  manage it directly); `--anonymous` opts out for a single submission
  without touching what's saved.
- **Result schema bumped to `schema_version: "0.4"`** for the two optional
  top-level fields above (`comment`, `submitted_by`) — both attached at
  `goesb submit` time, not `goesb run` time, so `payload_sha256` is
  recomputed and the result re-signed with the submission's own ephemeral
  key before it reaches the network; the local file `run` wrote is
  untouched either way.

## [0.8.10] - 2026-07-29
### Fixed
- **`decode_pcm` silently assumed every pack's audio was already at the
  model's target sample rate — it never resampled, and never checked.**
  Real report: a Common Voice pack (48kHz, crowd-sourced personal
  devices) fed straight into whisper.cpp or vosk (both require and
  expect 16kHz, neither resamples internally) produced fluent-sounding
  but 100%+ WER hallucinated garbage — the audio played back 3x too
  fast to the model, not a model-quality or pack-content problem.
  Confirmed by direct reproduction: switching `--backend cuda`/`cpu`/
  `metal` made no difference (ruling out a backend bug), and the exact
  same WER (1.0667) and hallucinated output reproduced on cpu — this
  was never about metal. `faster-whisper` was never affected: its own
  ctranslate2 decode path resamples internally regardless of input
  rate, which is why its historical results on these same packs always
  looked correct. `decode_pcm` now takes a required `target_rate_hz`
  (no default — a future adapter can't reproduce this by omission) and
  resamples via linear interpolation when the native rate differs.
  Re-verified on the real pack after the fix: WER dropped from 1.0667
  to 0.4697 on `common-voice-nl` base model, now in line with
  faster-whisper's own result (0.4303) on the identical pack/model
  size, and the hypothesis text is now a recognizable (if imperfect)
  transcription instead of hallucinated noise.
  **Every `common-voice-*` pack is affected** (all either declare no
  `sample_rate_hz` — mixed native rates — or a non-16kHz one); `fleurs-*`
  and `librispeech-*` packs were never affected (already 16kHz).

## [0.8.9] - 2026-07-29
### Added
- **Wizard prompts for `--backend` per engine, instead of always
  defaulting to cpu.** Previously the only way to run the wizard on a
  non-cpu backend was to skip it and hand-build a `goesb run --backend
  ...` invocation yourself — the wizard's own batch runs always ended up
  cpu-only regardless of what hardware was available. Adds
  `_ready_backends()`, which narrows an engine's declared backend support
  (`get_supported_backends`) down to what's actually verified usable on
  this machine right now — reusing the exact same probes `goesb doctor`
  already reports (whisper-cpp's per-backend build-info check,
  faster-whisper's ctranslate2 CUDA device count) — so the wizard never
  offers a backend certain to fail. One prompt per distinct engine in the
  batch; an engine with only cpu available is never prompted, so a full
  Enter-through-everything session still reproduces today's cpu-only
  behavior byte-for-byte.

## [0.8.8] - 2026-07-29
### Fixed
- **Streaming latency metrics were measured from clip-buffer position 0,
  not real speech onset.** Confirmed by direct measurement: this pack's
  audio carries ~500-600ms of leading/trailing silence on every file, all
  of it previously baked into `first_partial_latency`,
  `first_final_latency`, and `end_of_speech_latency` despite
  `docs/specs/metrics.md` defining these relative to real speech.
  `faster_whisper.run_streaming` now zeroes its virtual clock at
  VAD-detected speech onset (via faster-whisper's own per-chunk segment
  timestamps — no new dependency).
- **`committed_word_count` (the streaming "non-revisable" prefix) was not
  actually monotonic.** The previous "every segment but the last" rule
  had committed text revised in 14 of 30 chunks on real audio, since VAD
  re-segments as more audio is appended — a segment not being the last
  one in a chunk's output doesn't mean it's stable. Replaced with real
  LocalAgreement-2: a word-count prefix commits once it agrees between
  two consecutive hypotheses, tracked as a running max so it can never
  decrease.
- **Speech-offset detection could report an end-of-speech timestamp past
  the final chunk's own buffer end.** Whisper's predicted segment
  timestamps can slightly overshoot the actual audio fed in (~40ms
  observed on real audio) — now clamped to each chunk's own buffer
  length so `audio_duration_s` can never exceed the clip it was derived
  from.

## [0.8.7] - 2026-07-29
### Fixed
- **0.8.6's own CUDA-detection check for whisper-cpp was itself wrong.**
  Found while investigating real Metal-backend support: confirmed by
  reading upstream ggml/whisper.cpp source directly
  (`whisper_print_system_info`), CUDA/Metal/Vulkan don't report through a
  flat `"CUDA = 1"`-style flag the way `OPENVINO`/`COREML` do — they
  register through ggml's dynamic backend registry as their own named
  section, `"CUDA : <features...>"`, present whether or not any features
  are reported. The 0.8.6 regex checked for the flag format, which no
  real CUDA build actually emits — meaning it would very likely have
  reported "no CUDA support" even on a genuinely CUDA-capable build, the
  exact inverse of the bug 0.8.6 was fixing. Corrected to check for the
  backend's actual registration name.
### Added
- **`--backend metal` for whisper-cpp, real and verified — not
  speculative.** Same `use_gpu` mechanism CUDA already used; Apple's
  Metal backend registers as `"MTL"` in ggml's registry (confirmed
  against upstream `ggml-metal.cpp`, and empirically: this is genuinely
  what a Metal-capable Mac build reports). Verified end-to-end on real
  Apple Silicon hardware — a real transcription with `--backend metal`
  produces correct output, and `--backend cuda` on the same (non-CUDA)
  build still correctly raises. `goesb doctor`'s whisper-cpp line now
  reports every backend the runtime declares, not just cuda.

## [0.8.6] - 2026-07-29
### Fixed
- **`--backend cuda` on whisper-cpp is now actually verified, not a
  silent best-effort guess.** `use_gpu` is a single boolean covering
  CUDA/Metal/Vulkan/nothing depending purely on how the installed
  `pywhispercpp` binary was compiled — previously requesting `cuda` on a
  Metal-only or CPU-only build silently ran on CPU with no error and no
  indication anything was wrong. Now checked directly against
  whisper.cpp's own build-info string (`Model.system_info()`, no model
  file needed) before ever loading a model; a build without real CUDA
  support now raises a clear `RuntimeError` pointing at `goesb doctor`,
  matching faster-whisper's existing behavior. `goesb doctor` gives the
  same real answer for this engine now too, instead of "can't be checked
  without running a real transcription."
### Added
- **`goesb doctor` detects non-NVIDIA GPUs too**, advisory-only (Apple
  Metal presence on macOS, `lspci`/`wmic` on Linux/Windows) — previously
  its GPU line was NVIDIA-only (`nvidia-smi`), so it always said "GPU:
  none detected" on every Mac and every AMD/Intel box, misleading
  specifically on the hardware where whisper-cpp's own GPU support would
  matter most. Doesn't touch the signed result document's own environment
  fingerprint, which stays NVIDIA-only and unchanged.
- **`goesb doctor` reports official profiles with no public result yet
  for your detected hardware** — the other ADR-0008-promised half of
  `doctor` that had never shipped. Scoped to profile × hardware, not the
  full profile × backend the ADR describes: the leaderboard API doesn't
  expose which backend a result used, only its runtime, so backend-level
  granularity isn't achievable without a platform-side schema change.
  Degrades silently (prints a one-line skip notice, never fails the rest
  of `doctor`) if hardware can't be confidently detected or the platform
  API is unreachable.

## [0.8.5] - 2026-07-29
### Added
- **The wizard now guesses your hardware from the catalog instead of
  leaving it fully manual.** Typing the right entry out of 985+ catalog
  rows by hand every single run was reported as very error prone. The
  local CPU model (already probed for the environment fingerprint) is
  normalized and fuzzy-matched against the catalog's CPU entries; a
  confident match pre-fills the picker so pressing Enter accepts it —
  it's a suggestion the user still confirms, never a silent auto-assign,
  and if nothing matches well (e.g. under virtualization, where the
  probed string is unrecoverable) the picker is left exactly as blank as
  before. GPU-backed runs aren't guessed yet — hardware is picked once
  per batch before any profile's engine/backend is chosen, so there's no
  reliable signal at that point about which combos will end up
  GPU-backed.
- **Progress logging around a Common Voice audio download.** A real
  fetch of dozens of clips took long enough with zero output between
  "attempting auto-fetch ..." and "Fetched N audio files ..." to read as
  a hang. Now announces the download starting and prints once it's
  done and extraction begins; on a real terminal, `datacollective`'s own
  progress bar is shown too (previously always suppressed) rather than
  us guessing at percent-complete from the outside.

## [0.8.4] - 2026-07-29
### Fixed
- **The actual final piece of the "Missing API key" saga: a credential
  resolved from the on-disk store was never exported to `os.environ`.**
  0.8.2 and 0.8.3 fixed two real, separate bugs in the same area, but a
  third remained: `_preflight_pack_credentials` correctly recognized an
  already-saved credential and skipped re-prompting for it — but
  `load_credential` only *returns* the value, it never touches
  `os.environ`, and nothing filled that gap. Every downstream consumer
  reads `os.environ` directly (`_reexec`'s subprocess inherits it;
  `audio_sources.fetch_common_voice_audio`'s own docstring literally
  assumes the credential is "already in os.environ by the time it runs"),
  so a credential saved on run N was silently unusable on every run after
  — reproduced and confirmed fixed directly against the live production
  API, not just via mocked unit tests, before release this time.
### Fixed
- **Fetched profiles and packs were cached forever with zero revalidation
  — the actual root cause behind 0.8.2's credential-prompt bug.**
  `fetch_profile`/`fetch_pack` (`remote.py`) treated file existence alone
  as "still valid," mirroring how model-weight caching works. Unlike model
  weights or audio (both genuinely immutable once fetched), a profile's or
  pack's own YAML is small, server-mutable metadata — this repo's own
  history has renamed pack ids, added a gated pack's credential
  requirement, and bumped versions, all to already-published packs. A
  real report showed the actual mechanism: a pack cached before it
  required an API key kept serving the credential-less copy forever after,
  so the wizard's own credential preflight (0.8.2) never even saw the
  requirement to prompt for — 0.8.2's fix was correct but unreachable for
  this exact case, since the stale cache meant `credential_by_env_var`
  never got populated in the first place. Network is now always tried
  first; the on-disk cache is used only as a fallback when the network
  request itself fails, not as a substitute for asking again.

## [0.8.2] - 2026-07-29
### Fixed
- **Wizard batches never prompted for a gated pack's API key if the local
  credential store held a blank value for it.** `_preflight_pack_credentials`
  checked `is not None`, so an empty string in `~/.goesb/credentials.json`
  (e.g. left over from before this check existed) looked "already
  resolved" and silently skipped the prompt — every run then failed deep
  in auto-fetch instead, with a "Missing API key" error that gives no hint
  it traces back to this earlier, silently-skipped step. Now checks
  truthiness, same as the environment-variable half of the same lookup
  already did — a blank value re-prompts, same as no value at all.

## [0.8.1] - 2026-07-29
### Added
- **Per-utterance recognition log.** `goesb run` now writes
  `<profile>__<pack>__<timestamp>.utterances.jsonl` alongside every result
  document — one line per utterance per repeat, with the raw reference
  text and what the engine actually produced, captured before
  normalization strips casing/punctuation for WER scoring. The aggregate
  WER alone can't tell you whether it's low because the engine is
  genuinely good or because normalization happens to be hiding garbage
  output. Written as its own file, never merged into the result document
  or covered by its signature — `submit`'s `*.json` glob and
  payload_sha256 check both exclude it automatically.
### Fixed
- **Clearer, install-method-aware fix when a guarded optional dependency
  (e.g. `datacollective`, the Mozilla Data Collective / ADR-0010 auto-fetch
  path) isn't installed.** Previously always suggested a bare `pip install
  <package>` — wrong and silently unfixable for a `pipx install
  goesb-runner` setup, since `pip install` outside that isolated venv is
  invisible to it (confirmed on a real Ubuntu report: `pipx install
  datacollective` appeared to succeed but installed into its own separate,
  throwaway venv). `goesb run` now offers to install it on the spot — same
  UX as a missing STT engine already had — and retries once automatically;
  if declined or non-interactive, the suggested command now matches how
  the runner itself was installed (`pipx inject goesb-runner <package>` vs
  plain `pip install <package>`).

## [0.8.0] - 2026-07-28
### Changed
- **Pack ids drop their `-batch` suffix.** Leftover from when every pack
  was authored against a batch profile specifically (ADR-0011's own
  eligibility rule is language-only, benchmark_type isn't checked) — the
  suffix implied a constraint that no longer holds, confirmed by
  `librispeech-en-streaming`'s audio being byte-identical to
  `librispeech-en`'s. `common-voice-de-batch` -> `common-voice-de`,
  `fleurs-nl-batch` -> `fleurs-nl`, `librispeech-en-batch` -> `librispeech-en`,
  and 13 more — every currently-published pack except
  `librispeech-en-streaming` (not `-batch`-suffixed, untouched). New ids,
  new hashes (id is part of a pack's canonical content) — old ids are
  gone, not aliased.

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
