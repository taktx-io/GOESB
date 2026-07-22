# Open Edge Speech Benchmark (GOESB)

**Objective, reproducible benchmarks for local (edge) Automatic Speech
Recognition and realtime voice pipelines.**

GOESB helps developers, researchers, hardware vendors, and companies pick the
optimal combination of **hardware × runtime × model × configuration × dataset ×
language** for running speech AI *locally* — no cloud required. It is designed to
become the de-facto standard for reproducible benchmarks of on-device speech
processing, with first-class support for privacy-first use cases such as smart
home, assistive technology, robotics, healthcare, and products like Babbl.

**Language-agnostic by design.** Language is just another selectable dimension —
like hardware or runtime — not a built-in assumption. Any language works: a
profile carries a BCP-47 `language` field, normalization is a per-language
pluggable ruleset, and leaderboards filter by language. Dutch appears in the
examples because it is one strong privacy-first use case, not because the
platform is Dutch-specific.

> Status: **early scaffold**. This repository currently contains the project
> structure, specifications, and documentation. Implementation follows the
> [roadmap](docs/03-roadmap.md).

## Design principles

Open source (Apache-2.0 — see Licensing below) · objective · reproducible ·
privacy-first · secure · extensible · vendor-neutral · scientifically sound ·
practically useful.

## What it answers

- Which hardware runs Whisper Medium in realtime?
- Which runtime is fastest? Which model is best for a given language (e.g. Dutch, English, German, Spanish, ...)?
- How much power does a realtime voice assistant draw?
- Which combination gives the best price/performance?
- Which hardware is suitable for fully local AI?

## Three benchmark types

Streaming is not a setting — it is a fundamental benchmark *type*.

| Type | Processes | Example uses | Signature metrics |
|------|-----------|--------------|-------------------|
| **Batch** | whole audio at once | transcription, subtitles, minutes | WER, CER, Real-Time Factor, CPU/RAM, energy |
| **Streaming** | audio in realtime | voice assistants, live captions | + First Partial/Final Latency, End-of-Speech Latency, Partial Stability |
| **Conversation** | the full mic→VAD→ASR→LLM→TTS→speaker pipeline | complete voice assistants | Time-To-First-Response/Audio, End-to-End & Barge-in Latency |

## Repository layout

```
runner/     Python benchmark runner (CLI, environment capture, hashing)
schemas/    JSON Schemas for profiles and packs (source of truth)
profiles/   Official benchmark profiles (per language, e.g. whisper-medium-en-batch, whisper-medium-nl-batch)
packs/      Pack manifests & metadata (never audio)
docs/       Vision, requirements, architecture, roadmap, ADRs, specs
scripts/    Repo tooling (schema validation, etc.)
```

The leaderboard product — the API and public website that consume this
method — lives in the separate, private `taktx-io/goesb-platform` repo, not
here. See [ADR-0007](docs/adr/0007-split-platform-repo.md) for why.

## Documentation

- [Vision](docs/00-vision.md)
- [Requirements](docs/01-requirements.md)
- [Architecture](docs/02-architecture.md)
- [Roadmap to a live website](docs/03-roadmap.md)
- [Glossary](docs/04-glossary.md)
- [Architecture Decision Records](docs/adr/)
- Specs: [metrics](docs/specs/metrics.md), [environment capture](docs/specs/environment-capture.md)

## Quick start (developers)

```bash
make setup        # install the runner
python scripts/validate_assets.py
```

For the leaderboard/API product (requires this repo cloned as a sibling —
see its README), go to `taktx-io/goesb-platform`.

## Licensing

**Apache-2.0** for `runner/`, `schemas/`, `profiles/`, `packs/`, and
`scripts/` — the parts that are built and that the trust/reproducibility
claim depends on. See [ADR-0003](docs/adr/0003-open-source-strategy.md) for
the reasoning, [ADR-0007](docs/adr/0007-split-platform-repo.md) for why
`api/`/`web/` now live in a separate private repo rather than an unlicensed
directory here, and [governance.md](docs/governance.md) for what actually
keeps results neutral. Datasets/packs carry their own independent content
licenses (e.g. CC0-1.0, CC-BY-4.0 — see each pack's `pack.yaml`). Commercial
modules (Enterprise Edition, Hosted Service, Hardware Certification, Pack
Marketplace) are separately licensed and not part of this repository.

## Non-goals

GOESB is **not** a transcription service and **not** an AI chat platform. It
benchmarks local speech processing objectively and reproducibly — nothing more.
