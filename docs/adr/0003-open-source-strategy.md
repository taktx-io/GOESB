# ADR-0003 — Open-source strategy

- **Status:** Accepted — **the deferred `api/`/`web/` licensing question is
  resolved by [ADR-0006](0006-split-platform-repo.md)**: private, and in a
  separate repository (`taktx-io/oesb-platform`), not a same-repo license
  split. Everything else below (Apache-2.0 scope for
  runner/schemas/profiles/packs, no AGPL, no CLA) stands as originally
  decided. Per [ADR-0001](0001-record-architecture-decisions.md), this
  record is left intact rather than rewritten.
- **Date:** 2026-07-21

## Context

The project vision is for OESB to become the *de-facto, vendor-neutral,
reproducible standard* for edge speech benchmarks. Separately, the plan
envisions commercial extensions: Enterprise Edition, a Hosted Benchmark Service,
Hardware Certification, a Pack Marketplace, consultancy, and sponsorship.

There is a real, deliberate tension here: a benchmark's authority comes from
being inspectable ("why should I trust your numbers?"), which pulls toward open
source; while sustainable funding pulls toward capturing some value commercially.
This ADR resolves that tension (see Decision below).

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

## Decision

**Open core, licensed Apache-2.0** — scoped to what's actually built and
proven: `runner/`, `schemas/`, scoring/normalization rules, reference
`profiles/`, and open `packs/`. No AGPL, no dual licensing, no CLA.

**`api/` and `web/` are explicitly out of scope for this decision — license
TBD, not yet Apache-2.0, not yet anything.** They're unbuilt scaffolding
(M3/M4), and this is where the roadmap's own "Hosted Service" commercial
line (M8) will actually live. Committing that unbuilt product to a
permissive license today, before it exists and before any commercial
pressure is real, would repeat the same mistake as adopting AGPL pre-
emptively against a hypothetical threat — just in the opposite direction.
Revisit at M3, once `api/` is actually built and there's a real read on
whether a self-hostable-competitor risk is concrete.

Rationale for Apache-2.0 (runner/schemas/profiles/packs) over AGPL-3.0(+dual license):
- OESB's value is becoming *the* neutral standard, which needs hardware
  vendors (chip makers) and runtime maintainers comfortable contributing
  adapters/profiles. Apache-2.0's explicit patent grant is what corporate
  legal teams look for before their engineers can contribute; AGPL's
  copyleft reputation is friction most standards bodies (MLPerf, SPEC, TPC)
  deliberately avoid for exactly this reason.
- The AGPL "hosted-clone" defense is a real pattern (Mongo, Elastic) but
  solves a problem OESB doesn't have yet — there is no adoption to clone.
  Dual licensing also requires a CLA from day one, which itself repels some
  contributors, as a permanent cost against a hypothetical threat.
- This is reversible in the direction that matters: if a concrete
  hosted-clone threat emerges later, OESB can relicense *new* versions (as
  Mongo/Elastic did), at the cost of a CLA at that point — a cost worth
  paying once the threat is real, not before.
- What actually protects neutrality long-term is governance, not license
  terms — see the neutrality/governance doc this ADR requires below.

## Consequences

- `LICENSE` is now Apache-2.0, scoped to `runner/`, `schemas/`, `profiles/`,
  `packs/`, and repo tooling (`scripts/`) — replacing the All Rights Reserved
  placeholder for those. Datasets/packs keep their own independent licenses
  (e.g. CC0-1.0, CC-BY-4.0, already tracked per-pack in `pack.yaml`) —
  unaffected by the repository's code license.
- `api/` and `web/` remain unlicensed (default: all rights reserved) until
  this ADR is revisited at M3 — a placeholder note in each directory says so
  explicitly, so no one assumes the root `LICENSE` covers them by proximity.
- No CLA; contributions to the licensed scope are accepted under Apache-2.0
  terms as-is.
- Neutrality commitments (sponsorship never influences results) and the
  plugin review process are written into `docs/governance.md`, a
  precondition for going public, not an afterthought.
- Commercial modules (Enterprise Edition, Hosted Service, Hardware
  Certification, Pack Marketplace — see roadmap M8) are separately licensed
  and live outside this open core; they are not yet built.
