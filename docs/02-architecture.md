# Architecture

This document describes the OESB system: its components, how data flows through a
benchmark, the plugin model, and the trust/integrity model that makes public
results believable. It is the technical companion to the
[requirements](01-requirements.md).

## 1. System overview

OESB is a small set of cooperating components around three durable artifacts —
**Profile**, **Pack**, and **Result** — each versioned and hash-identified.

```
                         ┌──────────────────────────────────────┐
                         │              OESB Cloud               │
   local machine         │                                      │
 ┌───────────────┐  signed result   ┌──────────┐   ┌──────────┐ │
 │  Benchmark    │ ───────────────► │   API    │──►│ Postgres │ │
 │  Runner (CLI) │   POST /benchmark│ (FastAPI)│   │ + object │ │
 │               │ ◄─────────────── │          │   │  store   │ │
 └──────┬────────┘  profiles/packs  └────┬─────┘   └──────────┘ │
        │                                │  read-only            │
   captures env,                    ┌────▼─────┐                 │
   runs adapters                    │   Web    │  leaderboards   │
        │                           │ (Next.js)│                 │
 ┌──────▼───────┐                   └──────────┘                 │
 │ runtime      │                         └──────────────────────┘
 │ adapters     │  (faster-whisper, whisper.cpp, vosk, ...)
 └──────────────┘
```

Key idea: **audio and models stay on the local machine**. Only the signed result
document — metrics plus an environment fingerprint plus hashes — is submitted.
For private packs, not even the metadata of the audio is published.

## 2. Components

### 2.1 Benchmark Runner (`runner/`)
The only sanctioned way to produce a result. Responsibilities:
- Load and validate a **profile** and **pack manifest** (schema + hash check).
- Capture the reproducibility **environment fingerprint** (see
  [spec](specs/environment-capture.md)).
- Drive the benchmark through a **runtime adapter** for the chosen type
  (batch / streaming / conversation).
- Compute **metrics** via metric plugins with the profile's normalization/scoring.
- Emit a **signed, hashed result document**.

The runner is open, digitally signed, reproducible, and cross-platform
(Linux/Windows/macOS, ARM/x86). It **never executes user-provided code** — see
§5 and [ADR-0004](adr/0004-runner-security-model.md).

### 2.2 API (`api/`)
An open REST service (FastAPI). Serves read models for the website and accepts
result submissions. Endpoints (see §6). The website never touches the database
directly — it consumes the same public API third parties use.

### 2.3 Web (`web/`)
Next.js/React public site: filterable leaderboards, profile and pack detail
pages, hardware records, and curated views. Read-only against the API.

### 2.4 Storage
Relational database (Postgres) for results, profiles, packs, hardware, and
leaderboard projections; object storage for large immutable artifacts (pack
manifests, result documents). Content is addressed by hash where possible.

### 2.5 Shared schemas (`schemas/`)
JSON Schemas for profiles and packs are the **source of truth**, consumed by the
runner (validation), the API (ingest validation), and CI (asset validation).

## 3. Core domain artifacts

### Profile
Immutable, versioned definition of *how* a benchmark runs and is scored:
type, runtime, model, configuration, normalization, scoring, required metrics,
changelog. Identified by `id` + `version`. Official profiles gate public
leaderboards.

### Pack
Immutable bundle of audio + transcript + metadata + target profile +
normalization + scoring + docs + license. Identified by `id` + `version` +
`sha256`. Open / community / private visibility.

### Result
A signed document: metric values + environment fingerprint + the hashes of the
profile, model, runtime, configuration, and dataset it was produced from. This
tuple of hashes is what makes a result verifiable and comparable.

```
Result ──references──► (profile_hash, pack_hash, model_hash,
                        runtime_hash, config_hash) + env_fingerprint + signature
```

## 4. Plugin model (extensibility)

The core defines stable, versioned interfaces; capabilities are plugins:

| Extension point | Adds | Examples |
|-----------------|------|----------|
| Benchmark type | a new run loop + metric set | batch, streaming, conversation, future |
| Runtime adapter | how a model is driven | faster-whisper, whisper.cpp, vosk, coqui |
| Model descriptor | how a model is identified/loaded | whisper-medium int8 |
| Metric | a measurement + reducer | WER, RTF, First Partial Latency, energy |
| Hardware probe | how a device is fingerprinted | x86 RAPL, ARM sysfs, NVIDIA NVML |
| Exporter | how results leave the system | JSON, CSV, OpenTelemetry, leaderboard |

Plugins are **declarative + reviewed code shipped with the runner**, not
arbitrary code supplied at run time. A new plugin is a pull request that passes
review and CI — never an executable dropped into a benchmark input. This keeps
the "no arbitrary code execution" guarantee intact while staying extensible.

## 5. Trust & integrity model

Public credibility rests on three mechanisms:

1. **Declarative inputs only.** Users supply models, datasets, metadata, and
   configuration — data, never code. The runner refuses scripts, shell,
   Dockerfiles, executables, and unrestricted plugins (FR-9.1/9.2).
2. **Hashing everywhere.** Profile, model, runtime, configuration, dataset, and
   result are hashed (SHA-256). Any tampering is detectable; identical inputs
   produce identical hashes, enabling deduplication and comparison.
3. **Signing.** The runner is digitally signed; result documents are signed so a
   leaderboard entry can be traced to a genuine runner build and an unmodified
   result.

Only results from an **official profile + open pack + verified environment**
appear on public leaderboards. Private results are for internal comparison and
never surface publicly. See [ADR-0004](adr/0004-runner-security-model.md) for the
full threat model.

## 6. API surface (initial)

```
GET  /leaderboards     ?benchmark_type&profile&language&runtime&model&hardware
                        &max_price_eur&max_energy&min_rtf
GET  /profiles         list official profiles
GET  /profiles/{id}    profile incl. version + changelog
GET  /packs            list packs (id, version, sha256, visibility)
GET  /packs/{id}
GET  /hardware         hardware records & fingerprints
POST /benchmark        submit a signed result for verification & ranking
GET  /benchmark/{id}   fetch a result
GET  /health
```

An OpenAPI document is generated from the API and published/versioned so the
whole platform is automatable (FR-12).

## 7. Data flow: one benchmark run

1. User selects an official **profile** and a **pack** (or a private pack).
2. Runner validates both against schemas and verifies pack `sha256`.
3. Runner captures the **environment fingerprint**.
4. Runner executes the type-specific run loop through the **runtime adapter**,
   streaming audio for streaming/conversation types.
5. Metric plugins compute values; normalization + scoring from the profile apply.
6. Runner assembles a **result**, hashes inputs, and **signs** it.
7. For public packs: `POST /benchmark`; the API re-verifies hashes/signature and
   projects the result onto leaderboards. For private packs: stored locally or in
   the org's private space; never public.

## 8. Deployment topology

- **Web + API**: containerised, deployed behind a CDN; API is stateless and
  horizontally scalable; leaderboard reads are cached (NFR-8).
- **Database + object store**: managed Postgres + S3-compatible bucket.
- **Runner**: distributed as signed binaries/wheels per platform; runs entirely
  on the user's machine.

See the [roadmap](03-roadmap.md) for how these come online iteration by iteration.

## 9. Technology choices

Python for the runner and API (aligns with the ASR/ML ecosystem —
faster-whisper, whisper.cpp bindings, torch, psutil), Next.js/TypeScript for the
website. Rationale and alternatives in [ADR-0002](adr/0002-tech-stack.md).
