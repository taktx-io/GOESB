# ADR-0001 — Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

OESB aims to be a long-lived, community-trusted standard. Decisions about the
stack, licensing, security model, and scoring will be questioned and revisited.
Undocumented decisions get re-litigated and erode trust.

## Decision

We record every significant architectural or strategic decision as an
Architecture Decision Record (ADR) in `docs/adr/`, numbered sequentially, using
the lightweight format: Context, Decision, Consequences, Status. ADRs are
immutable once accepted; a reversal is a new ADR that supersedes the old one.

## Consequences

- New contributors can read *why*, not just *what*.
- Changes to scoring/normalization/security are auditable — essential for a
  benchmark whose value is its credibility.
- Small ongoing overhead: notable PRs should include or reference an ADR.
