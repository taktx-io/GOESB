# ADR-0008 — Compute backend is explicit, captured, and never auto-selected

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

Modern edge hardware offers several ways to execute the same model on the same
board: an RK3588 can run Whisper on its Cortex CPU cores or on its 6-TOPS NPU
(via RKNN); an AMD Ryzen laptop can use CPU or its iGPU (realistically via
whisper.cpp's Vulkan backend — ROCm on iGPUs is patchy); Apple silicon has
Metal; Intel has OpenVINO. The question is whether the runner should
**auto-select the fastest available accelerator** when it detects one.

Auto-selection is attractive product behaviour but is an anti-pattern for a
measurement instrument. If the runner silently picks the NPU when a driver
happens to be present, then two results under the same profile on the *same
board* stop being the same experiment: numbers diverge based on driver
installation state, and a leaderboard row silently mixes CPU runs with NPU
runs. That is precisely the "benchmarks measured under different conditions"
mess GOESB exists to eliminate (NFR-1; docs/00-vision.md "Why this needs to
exist"). At the same time, "is the NPU worth it?" is one of the platform's
founding questions (docs README "What it answers") — the *platform* must be
able to answer it.

## Decision

Put the intelligence in three different places, none of them silent:

### 1. Backend is an explicit, first-class dimension of a result

- A run's **compute backend** (e.g. `cpu`, `rknn-npu`, `vulkan`, `cuda`,
  `metal`, `openvino`) is passed explicitly to the runner (`--backend`),
  **defaults to `cpu`** — the one backend every machine has, so the default is
  deterministic — and is recorded in the signed result. The runner never
  infers or upgrades it.
- Schema change: the result document's `runtime` object gains a required
  `backend` field (string enum owned by the adapter registry). Existing
  results without the field are migrated as `backend: "cpu"`, which is what
  all current adapters (faster-whisper, vosk, whisper.cpp) did in fact use.
- Environment capture continues to record what accelerators are *present*
  (NPU/GPU model + driver, per docs/specs/environment-capture.md); the result
  records what was *used*. Presence ≠ use, and both are visible.
- Consistent with ADR-0004: each backend is **reviewed adapter code shipped
  with the runner** (an RKNN Whisper port is a distinct adapter, not a flag on
  faster-whisper; whisper.cpp's Vulkan build is a build/backend variant of its
  adapter). Adapters declare which backends they support; requesting an
  unsupported backend is a hard error, never a silent fallback. Profiles may
  optionally constrain `runtime.backends_allowed`; by default any registered
  backend is permitted, since backend — like hardware — is an axis results are
  *compared across*, not part of the workload definition.

### 2. The runner detects and suggests; the user decides

A `goesb doctor` command reports, without running anything: detected
accelerators and driver versions, which registered adapters/backends this
machine can execute, and which (profile × backend) combinations have **no
verified public result yet** for this hardware — feeding the contribution
flywheel ("only RK3588 owners can fill these cells"). Detection informs the
human; it never changes the experiment.

### 3. "Best config for this hardware" is a platform query, not runner magic

With backend an explicit axis, the leaderboard/decision tool answers "on
RK3588: NPU via rknn — RTF 0.4 at 2.1 W vs CPU — RTF 1.3 at 6 W" as a
transparent query over verified results (guardrail G1 in oesb-platform's
product direction). Same-board backend comparisons (CPU vs NPU vs iGPU) are
first-class, shareable views.

### Thread/core affinity rides along

`configuration.threads: 4` is ambiguous on big.LITTLE parts (RK3588:
4×A76 + 4×A55 — *which* four?). Core affinity joins backend in the
explicit-and-captured bucket: the runner records the actual affinity/core
classes used; profiles keep pinning thread *count*, and a later profile schema
revision may add an affinity expression (e.g. `performance-cores`). Until
then, ambiguity is at least visible in the result, not hidden.

## Consequences

- **+** Same profile + same backend + same hardware = same experiment; results
  remain honestly comparable, and driver state can no longer silently change
  what was measured.
- **+** The NPU/iGPU value question becomes answerable with signed data, per
  board — a differentiating, killer-chart-grade capability.
- **+** `goesb doctor` turns heterogeneous hardware from a comparability
  hazard into a targeted contribution engine.
- **−** More friction: an RK3588 owner must ask for the NPU explicitly. This is
  deliberate; `doctor`'s suggestions are the ergonomic escape hatch.
- **−** Schema/migration work: result schema gains `runtime.backend` (with
  `additionalProperties: false`, this is a versioned schema change), adapters
  gain a backend registry, and the platform's ingest/leaderboards gain a
  backend axis (filter + column).
- New accelerator support arrives at review pace (a vetted adapter per
  backend), not drop-in pace — the standing ADR-0004 trade-off, accepted.

## Relationships

Builds on ADR-0004 (reviewed adapters, no arbitrary code) and the environment
capture spec (presence vs use); extends the result schema
(`schemas/benchmark-result.schema.json`); feeds oesb-platform's decision tool
(G1) and most-wanted list (G2). Touches FR-5 (reproducibility), FR-6.3
(NPU/GPU metrics), FR-11 (plugin extension points).
