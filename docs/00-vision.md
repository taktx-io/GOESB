# Vision

## Mission

Design and build an open platform for **objectively benchmarking local (edge)
Automatic Speech Recognition and realtime voice pipelines**, so that anyone
building on-device speech AI can choose the optimal combination of hardware,
runtime, model, configuration, dataset, and language with confidence.

OESB is fully generic, but must be excellent for privacy-first applications:
smart home, assistive technology, robotics, healthcare, and products such as
Babbl.

## The end state

When a developer asks *"which combination of hardware, model, runtime, and
benchmark type best fits my application?"*, OESB should be the first answer.

Success means OESB is the worldwide reference for benchmarking edge speech AI —
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
- **Scientifically sound** — defined normalization, scoring, and statistics.
- **Practically useful** — answers real purchasing and deployment questions.

## Why this needs to exist

Today, "which edge device runs Whisper in realtime?" is answered by scattered
blog posts with incompatible methodologies: different audio, different
normalization, different quantization, unreported thermal throttling, no power
numbers. None of it composes. OESB fixes this by making the *method* — profile,
pack, environment, scoring — a first-class, versioned, hash-identified artifact,
so numbers from different people, machines, and dates are directly comparable.

## Non-goals

- Not a transcription service.
- Not an AI chat platform.
- Not a model zoo or a runtime — it benchmarks them.

OESB benchmarks local speech processing objectively and reproducibly. That is
the whole job.
