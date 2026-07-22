# Requirements

This document specifies what GOESB must do (functional requirements, **FR**) and
the qualities it must have (non-functional requirements, **NFR**). Requirements
are numbered so they can be referenced from issues, ADRs, and the roadmap. Each
carries a priority using MoSCoW: **[M]** must, **[S]** should, **[C]** could.

---

## 1. Benchmark types

> **Scope note:** this document covers the open runner/method only. The
> leaderboard product (website, API) that consumes these results moved to
> the separate, private `taktx-io/goesb-platform` repo — see
> [ADR-0007](adr/0007-split-platform-repo.md) — which maintains its own
> requirements for leaderboards/API/accessibility. References elsewhere in
> this repo (ADRs, roadmap history) to FR-7/FR-12/NFR-8/9/11 predate that
> split and are left as historical record, not dangling ids.

Streaming is a fundamental benchmark *type*, not a configuration flag.

- **FR-1.1 [M]** Support three benchmark types: `batch`, `streaming`,
  `conversation`. A benchmark's type is fixed by its profile.
- **FR-1.2 [M] Batch** — process the full audio in one pass. Report WER, CER,
  Real-Time Factor, total processing time, CPU, RAM, energy.
- **FR-1.3 [M] Streaming** — process audio in realtime and additionally report
  First Partial Latency, First Final Latency, End-of-Speech Latency, Update
  Frequency, and Partial Stability, so the platform measures not just speed but
  how *fluently* a transcript forms while speaking.
- **FR-1.4 [S] Conversation** — benchmark the full pipeline
  mic → VAD → ASR → LLM → TTS → speaker, reporting Time-To-First-Response,
  Time-To-First-Audio, End-to-End Latency, Interrupt/Barge-in Latency, and
  Streaming Responsiveness, plus CPU/GPU/NPU/RAM and energy.
- **FR-1.5 [C]** New benchmark types can be added as plugins without core changes.
- **FR-1.6 [S] Deployment target** is an orthogonal axis — `local` (edge) vs
  `cloud` (API) — applying across all benchmark types. Cloud is a **separated
  reference lane**, never merged into edge leaderboards: metrics that cannot apply
  to cloud (energy, hardware, thermals, certification) are marked N/A;
  cloud-native metrics (cost per minute, network round-trip, data-residency,
  "audio leaves device" flag) are added; and cloud results are labelled as
  weaker-reproducibility timestamped snapshots. Edge remains the platform's
  identity. See [ADR-0006](adr/0006-cloud-api-benchmarks.md).

## 2. Benchmark Profiles

- **FR-2.1 [M]** A benchmark runs against an official **Benchmark Profile** that
  defines exactly: benchmark type, runtime, model, configuration, scoring rules,
  normalization, and required metrics.
- **FR-2.2 [M]** Each profile has a unique `id`, a semantic `version`, and a
  `changelog`. Any change to how a benchmark runs or is scored requires a **new
  version** — profiles are never edited in place.
- **FR-2.3 [M]** Public leaderboards accept **official** profiles only. Users may
  create custom profiles, but they appear only locally or in private projects.
- **FR-2.4 [M]** Profiles are validated against
  [`benchmark-profile.schema.json`](../schemas/benchmark-profile.schema.json).
- **FR-2.5 [S]** Example official profiles ship across multiple languages, e.g.
  Whisper Medium (English), Whisper Medium (Dutch), Home Assistant Voice, Meeting
  Transcription, Medical Consultation, Automotive Voice, Call Center. The
  reference profiles must not be limited to a single language.
- **FR-2.6 [M]** **Language is a first-class dimension.** Every profile carries a
  BCP-47 `language`; any language is supported (including low-resource
  languages), with no language hardcoded or privileged in the core. Normalization
  is a **per-language pluggable ruleset** (e.g. `goesb-en-v1`, `goesb-nl-v1`), so
  language-specific text handling (numbers, casing, diacritics, punctuation,
  script) never leaks into the core. Results are only comparable within the same
  profile version, which fixes the language and its normalization.

## 3. Benchmark Packs

- **FR-3.1 [M]** A **pack** bundles audio, transcript, metadata, target profile,
  normalization, scoring rules, documentation, and license.
- **FR-3.2 [M]** Each pack has a unique `id`, `version`, and `sha256`. After
  publication a pack is **immutable**.
- **FR-3.3 [M]** Three visibilities:
  - **Open** (Common Voice, FLEURS, VoxPopuli, LibriSpeech): identical data for
    everyone → directly comparable public results.
  - **Community**: user-published datasets (Dutch elderly, smart home, meetings,
    car, far-field, dialects, background TV, children, noisy factory); may be
    free or commercial.
  - **Private**: an organisation's own data — audio stays local, only metadata
    and results are stored, never shown on public leaderboards.
- **FR-3.4 [M]** Packs are validated against
  [`benchmark-pack.schema.json`](../schemas/benchmark-pack.schema.json).
- **FR-3.5 [M]** Audio and personal data are never committed to the platform's
  public storage; a pack is identified cryptographically, not by shipping audio.

## 4. Metadata

- **FR-4.1 [M]** Every pack carries at minimum: language, dialect, age group,
  recording environment, microphone, sample rate, background noise, number of
  speakers, spontaneous-vs-read speech, duration, transcription style, license.
- **FR-4.2 [S]** Free-form tags are allowed for search/filter but **never**
  affect the benchmark definition or scoring.

## 5. Reproducibility

- **FR-5.1 [M]** Every benchmark must be fully reproducible. The runner
  automatically captures:
  - **Hardware**: CPU, GPU, NPU, RAM, storage, firmware, BIOS, cooling, power.
  - **Software**: OS, kernel, runtime, driver versions, compiler, optional Docker image.
  - **Model**: model, version, quantization, beam size, language, VAD,
    temperature, chunk size, thread settings, streaming settings.
- **FR-5.2 [M]** The captured environment is stored with the result and hashed.
- **FR-5.3 [S]** Two runs with identical profile+pack+environment must produce
  metrics within a documented tolerance; deviations are surfaced, not hidden.
  See [environment capture spec](specs/environment-capture.md).

## 6. Metrics

- **FR-6.1 [M]** Quality: WER, CER (with defined normalization).
- **FR-6.2 [M]** Realtime: First Partial Latency, First Final Latency,
  End-of-Speech Latency, Partial Stability, Streaming Responsiveness.
- **FR-6.3 [M]** Performance: Real-Time Factor, Throughput, CPU, GPU, NPU, RAM,
  temperature, energy.
- **FR-6.4 [M]** Economic: hardware price, watt per realtime stream, euro per
  realtime stream, price/performance index.
- **FR-6.5 [M]** Every metric has a single precise definition; see
  [metrics spec](specs/metrics.md). Metrics are pluggable.

## 7. Leaderboards

Leaderboard/API product requirements (filtering, curated views, what
appears on public leaderboards) now live in `oesb-platform`'s own
requirements doc.

## 8. Benchmark Runner

- **FR-8.1 [M]** All benchmarks run through one official runner that is open,
  digitally signed, reproducible, and platform-independent.
- **FR-8.2 [M]** Supported platforms: Linux, Windows, macOS, ARM, x86.
- **FR-8.3 [M]** The runner emits a signed result document containing metrics and
  the full environment fingerprint.

## 9. Security

- **FR-9.1 [M]** The benchmark software **never executes arbitrary code**. Not
  allowed as inputs: Python scripts, shell scripts, Dockerfiles, executables, or
  plugins with unrestricted rights.
- **FR-9.2 [M]** Users supply only: models, datasets, metadata, and benchmark
  configuration — declarative data, never code.
- **FR-9.3 [M]** Every public benchmark is cryptographically verifiable. Hashes
  are stored for: profile, model, runtime, configuration, dataset, and results.
- **FR-9.4 [M]** Runtime/model adapters are a curated, reviewed set (see
  [ADR-0004](adr/0004-runner-security-model.md)); third parties extend via the
  plugin process, not by shipping arbitrary code into a run.

## 10. Privacy

- **FR-10.1 [M]** Privacy-first by design: audio never has to be made public.
- **FR-10.2 [M]** Support fully local benchmark execution, private packs, public
  metadata, and cryptographic identification of datasets.
- **FR-10.3 [M]** Personal data is never required.

## 11. Architecture & extensibility

- **FR-11.1 [M]** Plugin-based core. New benchmark types, runtimes, models,
  metrics, hardware, and exporters can be added without changing the core.
- **FR-11.2 [M]** Clear, versioned interfaces between core and plugins.

## 12. API

API product requirements now live in `oesb-platform`'s own requirements
doc — this repo's runner only needs to emit a schema-valid, signed result
document (§8/§9 above); it doesn't define the API contract that ingests it.

---

## Non-functional requirements

- **NFR-1 Reproducibility [M]** — the platform's defining quality; every public
  result is recreatable from stored metadata and hashes.
- **NFR-2 Integrity [M]** — results, profiles, packs, and runner are tamper-evident
  via hashing and signatures.
- **NFR-3 Privacy [M]** — no audio or PII leaves the user's machine unless they
  explicitly publish it; GDPR-conscious defaults.
- **NFR-4 Portability [M]** — runner works on Linux/Windows/macOS and ARM/x86.
- **NFR-5 Neutrality [M]** — no vendor bias in defaults, scoring, or presentation;
  sponsorship never influences results.
- **NFR-6 Extensibility [M]** — plugin interfaces are stable and documented.
- **NFR-7 Transparency [S]** — methodology, normalization, and scoring are public.
- **NFR-10 Usability [S]** — a newcomer can run a local benchmark and read a
  leaderboard without reading the source.
- **NFR-12 Internationalisation [M]** — multilingual and language-agnostic by
  design. No language is privileged in the platform's mechanics; every language
  is a first-class dimension with its own pluggable normalization ruleset. The
  platform must work equally for high- and low-resource languages and for
  non-Latin scripts. (Dutch is merely one validation language among many.)

## Traceability

Roadmap milestones in [03-roadmap.md](03-roadmap.md) reference these FR/NFR ids,
so every requirement maps to an iteration and every iteration to requirements.
