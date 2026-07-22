# Vision

## Mission

Design and build an open platform for **objectively benchmarking local (edge)
Automatic Speech Recognition and realtime voice pipelines**, so that anyone
building on-device speech AI can choose the optimal combination of hardware,
runtime, model, configuration, dataset, and language with confidence.

GOESB is fully generic — **language-agnostic and multilingual by design**. No
language is privileged in the platform's mechanics: language is a first-class,
selectable dimension (a field on every profile and pack, filterable on every
leaderboard), and normalization is a per-language pluggable ruleset. The platform
must be equally usable for English, German, Spanish, Arabic, Mandarin, or any
other language, including low-resource ones. It must also be excellent for
privacy-first applications: smart home, assistive technology, robotics,
healthcare, and products such as Babbl. Dutch recurs in examples only because it
is one concrete privacy-first use case — never because the platform is
Dutch-specific.

## The end state

When a developer asks *"which combination of hardware, model, runtime, and
benchmark type best fits my application?"*, GOESB should be the first answer.

Success means GOESB is the worldwide reference for benchmarking edge speech AI —
trusted because it combines objectivity, reproducibility, security, privacy, and
practical relevance, so hobbyists and enterprises alike can make reliable
choices when building local speech applications.

## Design principles

- **Open source** (intended core — see [ADR-0003](adr/0003-open-source-strategy.md))
- **Objective** — the platform measures; it does not favour a vendor.
- **Reproducible** — any result can be recreated from captured metadata.
- **Privacy-first** — audio never has to leave the machine.
- **Secure** — the runner never executes arbitrary code.
- **Extensible** — new benchmark types, runtimes, models, metrics, hardware, and
  exporters are added via plugins without touching the core.
- **Vendor-neutral** — no built-in bias toward any chip, runtime, or model.
- **Language-agnostic** — any language is a first-class citizen; no language is
  hardcoded or privileged, and per-language normalization is pluggable.
- **Scientifically sound** — defined normalization, scoring, and statistics.
- **Practically useful** — answers real purchasing and deployment questions.

## Why this needs to exist

Today, "which edge device runs Whisper in realtime?" is answered by scattered
blog posts with incompatible methodologies: different audio, different
normalization, different quantization, unreported thermal throttling, no power
numbers. None of it composes. GOESB fixes this by making the *method* — profile,
pack, environment, scoring — a first-class, versioned, hash-identified artifact,
so numbers from different people, machines, and dates are directly comparable.

## Non-goals

- Not a transcription service.
- Not an AI chat platform.
- Not a model zoo or a runtime — it benchmarks them.
- Not a general edge-AI benchmark. LLM benchmarking exists only as the
  reasoning-step component inside the `conversation` benchmark type
  (mic→VAD→ASR→LLM→TTS→speaker, FR-1.4) — never a standalone LLM
  quality/accuracy leaderboard. Vision models and image generation are out
  of scope entirely. Expanding there would need a different name and a
  deliberate, separate bet — not a natural extension of this project.

GOESB benchmarks local speech processing objectively and reproducibly. That is
the whole job.
