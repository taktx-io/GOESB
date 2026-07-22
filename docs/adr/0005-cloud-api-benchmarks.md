# ADR-0005 — Cloud-API benchmarks as a reference lane

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

GOESB is edge-first: its identity, differentiator, and reason to exist are
*local, on-device, privacy-first, reproducible* speech benchmarking. But the
first decision most users actually face is one level up: **run locally, or call
a cloud API** (Deepgram, OpenAI Whisper API, Google STT, AWS Transcribe, Azure
Speech, ...). To answer *"is local good enough versus cloud, and at what cost and
privacy trade-off?"* the platform needs comparable cloud numbers as a baseline.
Without a cloud yardstick, "good enough" has nothing to be measured against.

However, cloud APIs violate several core GOESB assumptions:
- **Unmeasurable resources.** You cannot measure a provider's datacenter energy,
  CPU/GPU/NPU, or thermals; hardware price and hardware certification are N/A.
- **Weak reproducibility.** A cloud endpoint silently changes its model behind
  the same URL — no pinned version, no weights to hash, no frozen environment.
  This directly conflicts with GOESB's strongest promise (NFR-1, FR-5).
- **Network & geography variability.** Latency is dominated by round-trip and
  provider load, not local compute; results vary by region and time of day.
- **Secrets & connectivity.** Cloud runs need API keys and network access, unlike
  the clean, offline, signed edge runner path (ADR-0004).
- **Privacy.** Cloud sends audio off-device — the opposite of the privacy-first
  principle (FR-10).

## Decision

Include cloud-API benchmarks, but strictly as a **separated reference lane** that
makes the edge story legible — never as a co-equal focus and never merged into
the edge leaderboards.

1. **Deployment target is an orthogonal axis:** `local` (edge) vs `cloud` (API).
   It is **not** a new benchmark type — batch / streaming / conversation still
   apply to both. It cross-cuts every type.
2. **Separate leaderboards.** Cloud results are filterable and shown in their own
   sections, never silently ranked against edge results (apples-to-oranges).
3. **Metrics degrade gracefully.** Quality (WER/CER), observed latency, and RTF
   carry over. Energy, CPU/GPU/NPU, temperature, watt-per-stream, hardware price,
   and hardware certification are **N/A** for cloud and shown as such. Add
   cloud-native metrics: **cost per 1000 minutes (or per hour)**, network
   round-trip contribution, region / data-residency, availability, and a
   prominent **"audio leaves device"** privacy flag.
4. **Reproducibility honesty.** Cloud results are explicitly labelled as
   **timestamped snapshots** with weaker reproducibility. Capture the endpoint,
   provider-reported model version (if any), date, region, and client location.
   They are never presented as equivalent to hash-verified edge results.
5. **Identity preserved.** Edge remains the headline and the name stays *Open
   Edge Speech Benchmark*. Cloud exists as a yardstick, not a second product.
6. **Security firewall.** The cloud path is isolated from the offline edge runner.
   API keys are user-supplied secrets, never stored in results or on leaderboards.
   The declarative-inputs / no-arbitrary-code rule still holds: a cloud adapter is
   reviewed code shipped with the runner; the user supplies only endpoint,
   credentials, and configuration.
7. **Timing.** Not in M1–M4. Cloud lands as its own milestone once the edge core
   (methodology + leaderboards) is proven (see roadmap M5b).

## Consequences

- **+** Answers the edge-vs-cloud question buyers — and Babbl — actually face;
  enables a strong TCO and privacy narrative ("this €200 board matches Deepgram's
  accuracy at a fraction of the 3-year cost, with zero data egress").
- **+** The "audio leaves device" flag reinforces, rather than dilutes, the
  edge/privacy positioning.
- **+** Compelling for hardware-vendor sponsors: "edge is now competitive with
  cloud" is exactly their narrative.
- **−** Adds network, secrets, and geography complexity that must be firewalled
  from the clean offline runner.
- **−** Cloud comparability is inherently weaker (non-reproducible, drifting
  models); mitigated by transparent snapshot labelling and separate lanes.
- **−** Slight tension with the "Edge" in the name; mitigated by keeping cloud
  strictly a reference and never the headline.
- Cross-run cloud comparisons need care (provider load, time-of-day, region);
  always report date/region and consider periodic re-runs.

## Relationships

Extends FR-1 (benchmark types) with a deployment-target axis (**FR-1.6**);
interacts with reproducibility (FR-5, NFR-1), privacy (FR-10), economic metrics
(FR-6.4), and the runner security model (ADR-0004).
