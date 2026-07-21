# Roadmap — from scaffold to a live website

This is an **iterative** plan. Each milestone is a thin, end-to-end slice that
produces something demonstrable, reduces risk, and builds toward the same north
star: a public, trustworthy leaderboard. We ship the *method* first (so numbers
are meaningful), then breadth (more runtimes, packs, hardware), then scale and
commercial edges.

Milestones are sized in "iterations" (roughly 2-week increments), not fixed
dates, so the plan stays honest as scope is learned. Each milestone lists its
goal, scope, the requirements it satisfies, and its exit criteria.

---

## Guiding sequencing principles

1. **Correctness before coverage.** One profile + one open pack, computed
   correctly and reproducibly, beats ten half-defined benchmarks.
2. **Vertical slices.** Every milestone touches runner → result → (from M3) API →
   web, so integration risk surfaces early.
3. **Trust is a feature, gated before launch.** Hashing, signing, and
   verification must exist *before* any public leaderboard (M3/M4 gate).
4. **Batch first, then streaming, then conversation** — increasing measurement
   complexity in that order.

---

## M0 — Foundation *(done / this deliverable)*

**Goal:** a navigable repo and an agreed method on paper.

- Monorepo scaffold: `runner/`, `api/`, `web/`, `schemas/`, `profiles/`,
  `packs/`, `docs/`, CI skeleton.
- Documentation: vision, requirements, architecture, this roadmap, glossary,
  ADRs (stack, open-source, security), metrics + environment specs.
- JSON Schemas for profile & pack, with a validating example of each and a CI
  validation script.

**Satisfies:** FR-2.4, FR-3.4, FR-11.2 (interfaces on paper), documentation for
FR-5/6/9/10.
**Exit:** repo builds/validates in CI; Eric decides open-source direction
([ADR-0003](adr/0003-open-source-strategy.md)).

## M1 — Reproducible batch runner (local, offline)

**Goal:** run a real **batch** benchmark end-to-end on one machine and get
correct, reproducible WER/CER/RTF plus a full environment fingerprint — no
server yet.

- One runtime adapter (**faster-whisper**), one official profile
  (`whisper-medium-nl-batch`), one open pack wired to real audio locally.
- Metric plugins: WER, CER, RTF, CPU, RAM; normalization ruleset `oesb-nl-v1`.
- Full **environment capture** (hardware/software/model) per the spec.
- `oesb run` produces a signed, hashed **result document** on disk.

**Satisfies:** FR-1.1/1.2, FR-2.1/2.2, FR-3.1/3.2, FR-5.1/5.2, FR-6.1/6.3,
FR-8.1/8.3, FR-9.3.
**Exit:** two runs on the same machine agree within tolerance; result validates
and its hashes verify.

## M2 — Cross-platform + energy + more runtimes

**Goal:** make batch numbers *portable and complete*.

- Runner runs on Linux/Windows/macOS and ARM/x86; signed release artifacts
  (signed PyPI package/git tag — see distribution model in
  [02-architecture.md §2.1](02-architecture.md)) — not per-platform native
  binaries.
- Energy/thermal probes (RAPL, battery delta, hwmon; external-meter hook).
- Second and third runtime adapters (e.g. `whisper.cpp`, `vosk`) to prove the
  adapter interface.
- Characterise runner overhead so it does not bias RTF/latency
  ([ADR-0002](adr/0002-tech-stack.md)).

**Satisfies:** FR-8.2, FR-6.3 (energy/temp), FR-11.1 (runtime plugins), NFR-4.
**Exit:** same profile+pack runs on ≥3 OS/arch combos; energy reported; adapters
swap without core changes.

## M3 — API + result ingestion + verification

**Goal:** stand up the backend and the **trust gate**.

- FastAPI service with `GET /profiles`, `GET /packs`, `GET /hardware`,
  `POST /benchmark`, `GET /benchmark/{id}`, `GET /leaderboards` (unfiltered).
- Postgres + object storage; ingest **re-verifies** signatures and hashes before
  accepting a result (the [ADR-0004](adr/0004-runner-security-model.md) gate).
- Official-profile / open-pack / verified-environment enforcement for public
  results; private results never ingested publicly.
- Published, versioned OpenAPI.

**Satisfies:** FR-12.1/12.2/12.3, FR-7.3, FR-9.3, NFR-2.
**Exit:** a locally produced result is submitted, verified, and retrievable via
the API; tampered results are rejected.
**Decision needed:** `POST /benchmark` is anonymous/unauthenticated (per
[ADR-0004](adr/0004-runner-security-model.md) — trust comes from the
runner-embedded signature, not submitter identity). Abuse mitigation
(rate-limiting, submission review) is explicitly deferred to M8, not a blocker
here — but must be revisited before traffic makes anonymous submission a
liability.

## M4 — Public website & leaderboards (first live release)

**Goal:** the **live website** — the milestone Eric asked to reach.

- Next.js site: filterable leaderboards (benchmark type, profile, language,
  runtime, model, hardware, price, energy, RTF) reading only from the API.
- Profile and pack detail pages (incl. version + changelog); hardware records.
- 2–3 curated views ("Best Dutch under €300", "Lowest energy", "Best CPU-only").
- Deploy: CDN-fronted web, stateless API, managed DB; leaderboard reads cached.
- **Open-source flip** (if ADR-0003 accepted): apply chosen license, publish repo,
  add governance/neutrality statement.

**Satisfies:** FR-7.1/7.2, NFR-8/10, and the public face of FR-12.
**Exit:** a public URL shows real, verified batch results that anyone can
reproduce with the open runner + open pack.

## M5 — Streaming benchmark type

**Goal:** add the streaming *type* and its fluency metrics.

- Streaming run loop + adapters; metrics: First Partial/Final Latency,
  End-of-Speech Latency, Update Frequency, Partial Stability, Streaming
  Responsiveness (p50/p95).
- Streaming leaderboards and profiles (e.g. Home Assistant Voice).

**Satisfies:** FR-1.3, FR-6.2.
**Exit:** streaming results live on the site with tail-latency reporting.

## M6 — Community & private packs

**Goal:** open the ecosystem safely.

- Pack contribution flow (validation, hashing, immutability, licensing) for
  **community** packs; curated review.
- **Private packs**: audio never leaves the machine; only metadata + results
  stored; guaranteed absent from public leaderboards.
- Search/filter by metadata + free tags.

**Satisfies:** FR-3.3, FR-4.1/4.2, FR-10.1/10.2/10.3.
**Exit:** a third party publishes a community pack; an org runs a private pack
with audio provably staying local.

## M7 — Conversation benchmark type

**Goal:** benchmark full voice assistants.

- Conversation pipeline harness (mic→VAD→ASR→LLM→TTS→speaker) with
  Time-To-First-Response/Audio, End-to-End & Barge-in Latency, GPU/NPU/energy.

**Satisfies:** FR-1.4.
**Exit:** at least one end-to-end assistant profile with published results.

## M8 — Scale, economics & commercial edges

**Goal:** durability and the business model.

- Economic metrics surfaced and ranked (watt/€ per stream, price/perf index).
- Performance/scale hardening (caching, pagination) for growing result volumes.
- Commercial modules per [ADR-0003](adr/0003-open-source-strategy.md): Enterprise
  Edition, Hosted Service, Hardware Certification, Pack Marketplace — built
  around the open core without compromising neutrality.

**Satisfies:** FR-6.4, NFR-8/9, and the commercial-extension vision.
**Exit:** economic leaderboards live; at least one commercial module in pilot.

---

## Cross-cutting tracks (run continuously)

- **Trust & governance:** signing-key management, plugin review process,
  neutrality/sponsorship policy (must precede M4 launch).
- **Testing:** unit + integration per component; a reproducibility regression
  suite that re-runs reference benchmarks and flags drift.
- **Docs:** keep specs/ADRs current; publish methodology alongside the site.

## Definition of "live website" (the M4 bar)

A public URL where anyone can (a) browse filterable leaderboards of **verified**
results, (b) open a profile/pack and see exactly how a number was produced, and
(c) reproduce that number themselves with the open runner and an open pack. That
is the minimum that makes OESB a *standard* rather than a demo.
