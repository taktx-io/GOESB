# Open Edge Speech Benchmark (OESB)

**Objective, reproducible benchmarks for local (edge) Automatic Speech
Recognition and realtime voice pipelines.**

OESB helps developers, researchers, hardware vendors, and companies pick the
optimal combination of **hardware × runtime × model × configuration × dataset ×
language** for running speech AI *locally* — no cloud required. It is designed to
become the de-facto standard for reproducible benchmarks of on-device speech
processing, with first-class support for privacy-first use cases such as smart
home, assistive technology, robotics, healthcare, and products like Babbl.

> Status: **early scaffold**. This repository currently contains the project
> structure, specifications, and documentation. Implementation follows the
> [roadmap](docs/03-roadmap.md).

## Design principles

Open source (intended — see note below) · objective · reproducible ·
privacy-first · secure · extensible · vendor-neutral · scientifically sound ·
practically useful.

## What it answers

- Which hardware runs Whisper Medium in realtime?
- Which runtime is fastest? Which model is best for Dutch?
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
api/        FastAPI service: /leaderboards /profiles /packs /benchmark
web/        Next.js public website & leaderboards
schemas/    JSON Schemas for profiles and packs (source of truth)
profiles/   Official benchmark profiles (e.g. whisper-medium-nl-batch)
packs/      Pack manifests & metadata (never audio)
docs/       Vision, requirements, architecture, roadmap, ADRs, specs
scripts/    Repo tooling (schema validation, etc.)
```

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
make setup        # install runner, api, web
make dev-api      # http://127.0.0.1:8000/docs
make dev-web      # http://localhost:3000
python scripts/validate_assets.py
```

## Licensing

This repository is currently **All Rights Reserved** (placeholder). The intended
long-term model is an **open-source core** with optional commercial extensions;
the specific license is an open decision documented in
[ADR-0003](docs/adr/0003-open-source-strategy.md). Nothing is locked in yet.

## Non-goals

OESB is **not** a transcription service and **not** an AI chat platform. It
benchmarks local speech processing objectively and reproducibly — nothing more.
