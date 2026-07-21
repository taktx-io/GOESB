# ADR-0003 — Open-source strategy

- **Status:** Proposed — **decision pending** (owner: Eric)
- **Date:** 2026-07-21

## Context

The project vision is for OESB to become the *de-facto, vendor-neutral,
reproducible standard* for edge speech benchmarks. Separately, the plan
envisions commercial extensions: Enterprise Edition, a Hosted Benchmark Service,
Hardware Certification, a Pack Marketplace, consultancy, and sponsorship.

There is a real, deliberate tension here: a benchmark's authority comes from
being inspectable ("why should I trust your numbers?"), which pulls toward open
source; while sustainable funding pulls toward capturing some value commercially.
Eric has not yet decided whether OESB should be open source at all. This ADR
frames the choice; it does **not** lock anything in. The repository stays
**private and All Rights Reserved** until this ADR is accepted.

## Options

1. **Fully proprietary.** Maximum control and easiest monetisation, but
   undermines the core value proposition — an "objective" benchmark whose method
   is a black box is a hard sell to researchers, hardware vendors, and the
   privacy-first community. High risk to adoption and neutrality perception.
2. **Fully open (permissive, e.g. Apache-2.0 or MIT).** Maximum adoption and
   trust; anyone can run and verify. Risk: a well-funded actor could offer a
   competing hosted service with no obligation to contribute back.
3. **Open core + commercial extensions (recommended).** The runner, schemas,
   scoring/normalization rules, reference profiles, open packs, API, and website
   are open source; commercial modules (enterprise management, hosted service,
   certification, marketplace) are separately licensed. Trust and neutrality live
   in the open core; revenue lives at the edges.
4. **Copyleft core (AGPL-3.0).** Open and verifiable, but anyone offering it as a
   network service must publish modifications — discourages proprietary hosted
   clones while keeping the method open. Compatible with a separate commercial
   license for those who want to avoid AGPL obligations (a common "open core via
   dual licensing" pattern).

## Recommendation

Adopt **open core**. For the core license, choose between:
- **Apache-2.0** — best for maximum adoption and hardware-vendor participation
  (patent grant matters to them); rely on brand, certification, and hosted
  service for monetisation.
- **AGPL-3.0 (+ optional commercial dual license)** — best if protecting against
  closed hosted clones is a priority; keeps the method open while steering
  service providers toward a commercial agreement.

Recommended default: **Apache-2.0 core** for the runner/schemas/API/web (to win
neutral-standard status and vendor buy-in), with commercial modules under a
separate proprietary license. Reconsider AGPL if a hosted-clone threat becomes
concrete.

## Consequences

- If accepted, replace the placeholder `LICENSE` with the chosen SPDX license,
  add per-module licensing notes, and flip the repo to public at the milestone
  agreed in the roadmap (default: end of M1/M2, once the method is presentable).
- Neutrality commitments (sponsorship never influences results) should be written
  into a governance doc before going public.
- Until accepted, treat everything as All Rights Reserved and keep the repo private.
