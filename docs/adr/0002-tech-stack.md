# ADR-0002 — Technology stack

- **Status:** Accepted (revisit at M2 if the runner's performance overhead is
  material)
- **Date:** 2026-07-21

## Context

OESB has three code surfaces with different demands: a **runner** that must hook
into the ASR/ML ecosystem and measure resource use precisely; an **API** serving
leaderboards; and a **website**. The stack must be approachable for
open-source contributors and portable across Linux/Windows/macOS and ARM/x86.

## Decision

- **Runner: Python 3.11+.** The edge-ASR ecosystem is Python-native
  (faster-whisper, whisper.cpp bindings, vosk, torch, onnxruntime). `psutil`,
  NVML, and platform sysfs give resource/thermal/power probes. CLI via Typer,
  models via Pydantic v2, hashing via stdlib.
- **API: Python + FastAPI.** Shares Pydantic models and schema code with the
  runner, auto-generates OpenAPI (FR-12.3), and is fast enough for cached
  leaderboard reads.
- **Web: Next.js (React) + TypeScript.** Mature ecosystem for filterable,
  SEO-friendly public leaderboards; static/ISR rendering keeps hosting cheap.
- **Data: Postgres + S3-compatible object storage.**

## Alternatives considered

- **Rust runner.** Faster, single signed binary, lower measurement overhead —
  but higher contributor friction and weaker direct access to Python-only ASR
  runtimes. We keep this open as a future optimisation for the *measurement
  harness* while adapters stay in Python. Revisit if runner overhead measurably
  distorts metrics (tracked against ADR-0002).
- **Go API.** Excellent, but splitting API off from the runner's language
  duplicates the shared domain/schema models. Rejected for now.
- **Single-language JS/TS everywhere.** Poor fit for the ASR/ML runtime layer.

## Consequences

- Runner and API share one language and one set of schema-derived models —
  less drift, easier validation parity.
- Measurement overhead of Python must be characterised (M2) so it does not bias
  RTF/latency numbers; if it does, isolate hot paths (native ext / subprocess) or
  reconsider a Rust harness.
