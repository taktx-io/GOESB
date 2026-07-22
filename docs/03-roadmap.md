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

## M2 — Cross-platform + energy + more runtimes *(done)*

**Goal:** make batch numbers *portable and complete*.

- Runner runs on Linux/Windows/macOS and ARM/x86 — CI matrix across
  `ubuntu-latest`(x86_64)/`windows-latest`(x86_64)/`macos-14`(arm64), plus a
  best-effort `ubuntu-24.04-arm` leg (`.github/workflows/ci.yml`). Proves
  install + the fast unit suite on every leg, not full real-model inference
  per leg (extras/model downloads stay CI-excluded by design, same as
  before M2). Intel macOS deliberately not targeted — 6+ year old hardware,
  and x86_64/macOS are each independently covered elsewhere in the matrix,
  so this isn't a gap against FR-8.2/NFR-4, just a bounded target set.
  Signed release artifacts: `.github/workflows/release.yml` builds
  sdist+wheel and attaches a keyless GitHub-OIDC/Sigstore build-provenance
  attestation to a GitHub Release on a `runner-v*` tag (see distribution
  model in [02-architecture.md §2.1](02-architecture.md)) — not
  per-platform native binaries. **Gap, deliberately not closed yet:** no
  tag has been pushed / no real release cut (a visible, semi-irreversible
  action held for explicit go-ahead), and PyPI publish itself needs the
  account owner's own trusted-publisher setup — not attempted, no
  credentials held.
- Energy/thermal probes: RAPL (`runner/src/oesb_runner/energy.py`) and hwmon
  peak-temperature sampling, wired into `oesb run` as `energy_wh` /
  `temperature_c`, plus an `--external-energy-wh` override for a manually
  read power meter (declarative input, ADR-0004). **Gap, deliberately not
  closed yet:** RAPL/hwmon are Linux-only by construction and unit-tested
  against synthetic sysfs fixtures only — not yet proven against real Linux
  hardware (this session's dev machine is macOS/arm64; the external-meter
  override path *is* proven end-to-end). Battery-delta sampling remains
  unimplemented (only instantaneous battery percent/source is captured, via
  M1's `environment.py`).
- Second and third runtime adapters — `vosk` and `whisper.cpp` (via
  `pywhispercpp`) — proving the adapter interface: both registered and
  proven end-to-end (real LibriSpeech audio, schema-valid + signed results)
  with zero changes to `cli.py`'s dispatch logic.
- Runner overhead characterised
  ([ADR-0002](adr/0002-tech-stack.md), detail in
  [runner-overhead.md](specs/runner-overhead.md)): not material at
  production sampling settings on the one machine measured so far.

**Satisfies:** FR-8.2, FR-6.3 (energy/temp), FR-11.1 (runtime plugins), NFR-4.
**Exit:** same profile+pack runs on ≥3 OS/arch combos (✅ CI matrix); energy
reported (✅ via RAPL where present, or `--external-energy-wh`); adapters swap
without core changes (✅ vosk + whisper.cpp, no `cli.py` dispatch changes).

## M3 — API + result ingestion + verification *(done)*

**Goal:** stand up the backend and the **trust gate**.

- FastAPI service with `GET /profiles[/{id}]`, `GET /packs[/{id}]`,
  `GET /hardware`, `POST /benchmark`, `GET /benchmark/{id}`,
  `GET /leaderboards` (unfiltered — real filtering is M4). All served with
  typed Pydantic response models (list/summary endpoints) or the
  schema-validated document itself (detail endpoints), so `/openapi.json` is
  real, not stub shapes (FR-12.3).
- Postgres (real, via `docker-compose.yml`, migrated with Alembic —
  `api/src/oesb_api/{db,models}.py`); ingest **re-verifies** hashes and
  signatures before accepting a result, reusing the runner's own
  `schema_validation`/`signing` primitives rather than reimplementing them
  (`api/src/oesb_api/ingest.py`, the [ADR-0004](adr/0004-runner-security-model.md)
  gate). `benchmark_id` = `payload_sha256` — content-addressed, naturally
  idempotent on resubmission.
- Official-profile / open-pack enforcement: every `profile.yaml`/`pack.yaml`
  in the monorepo is loaded + schema-validated at startup
  (`api/src/oesb_api/assets.py`) and a submission's declared id/version/hash
  is checked against that set — not a separate DB table, since these are
  static, git-reviewed data. "Verified environment" (FR-7.3) is read as "the
  environment fingerprint is present and covered by the already-verified
  signature" — there's no separate environment-attestation mechanism
  documented anywhere, so none was invented.
- **Object storage: deliberately not wired.** Architecture doc's "Postgres +
  object storage" bullet is for large immutable artifacts; a result document
  is a few KB, stored directly in a `JSONB` column. Revisit when something
  actually needs it (e.g. M6 pack audio bundling).
- `GET /hardware` is *derived* from ingested results' environment
  fingerprints (distinct CPU/OS/arch combos + counts), not a separately
  curated table yet — no write path needed for this thin slice.

**Satisfies:** FR-12.1/12.2/12.3, FR-7.3, FR-9.3, NFR-2.
**Exit:** a locally produced result is submitted, verified, and retrievable via
the API (✅ proven on a real signed result, byte-identical round-trip);
tampered results are rejected (✅ proven — a post-signing metric mutation and
a corrupted signature both rejected with `400`, never stored).
**Decision needed:** `POST /benchmark` is anonymous/unauthenticated (per
[ADR-0004](adr/0004-runner-security-model.md) — trust comes from the
runner-embedded signature, not submitter identity). Abuse mitigation
(rate-limiting, submission review) is explicitly deferred to M8, not a blocker
here — but must be revisited before traffic makes anonymous submission a
liability.
**Known gap:** signing-key distribution across machines/deployments is still
the M1-era "local keypair, read from `~/.oesb/keys`" model (see
`runner/src/oesb_runner/signing.py`'s own docstring) — ingest verifies
against whatever key produced the result, correct for this milestone's
"locally produced, locally ingested" exit criterion. What that mechanism
actually proves (and doesn't) turned out to need real correction, not just a
distribution fix — see [ADR-0005](adr/0005-signing-token-distribution-and-trust-limits.md)
for the honest trust-claim and the designed (not yet implemented) call-home
ephemeral-token replacement.

## M4 — Public website & leaderboards (first live release)

**Goal:** the **live website** — the milestone Eric asked to reach.

- Next.js site: filterable leaderboards (benchmark type, profile, language,
  runtime, model, hardware, energy, RTF) reading only from the API. **Not
  price** — `hardware_price_eur` (docs/specs/metrics.md) is defined but
  nothing captures it yet; it's a manually-sourced/dated reference value
  with no probe, scoped to M8's economics work, not M3's pipeline.
- Profile and pack detail pages (incl. version + changelog); hardware records.
- 2–3 curated views achievable from what M1-M3 actually measure, e.g.
  "Lowest energy", "Best CPU-only", "Fastest Dutch STT" — **not**
  price-based views ("Best X under €N") until hardware pricing data exists
  (M8, or a small pulled-forward price-lookup addition before then).
- Deploy: CDN-fronted web, stateless API, managed DB; leaderboard reads cached.
- **Open-source flip**: publish repo under the Apache-2.0 scope decided in
  [ADR-0003](adr/0003-open-source-strategy.md) (runner/schemas/profiles/packs;
  `api/`/`web/` licensing revisited here at M3/M4), with the
  [governance/neutrality statement](governance.md) already in place.

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

### Design note (unscheduled) — TTS as a first-class benchmark type

LLM stays scoped to the `conversation` type above only — never a standalone
LLM leaderboard (see [00-vision.md](00-vision.md) non-goals). TTS, however,
is a reasonable **first-class** addition once there's a concrete need (e.g.
Babbl ships TTS today): a new `task` field on profiles (orthogonal to
`benchmark_type`), a pack shape that inverts (input text, not input audio),
and metrics that mostly reuse M1's existing plugins unchanged
(`real_time_factor`, `cpu_pct`, `ram_mb`, `energy_wh`) plus one new
ASR-based intelligibility metric (`tts_intelligibility_wer`/`_cer` — resynthesize,
re-transcribe with a fixed *versioned* reference ASR profile, score against
the original text with the existing WER/CER plugins). Not scheduled against
a specific milestone number yet — pull forward whenever a concrete TTS
hardware-selection need arises, same reasoning as Dutch-streaming-STT
possibly preceding M2 in strict order.

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
