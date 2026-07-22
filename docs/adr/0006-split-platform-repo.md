# ADR-0006 — Split the leaderboard product into a separate private repo

- **Status:** Accepted
- **Date:** 2026-07-22
- **Amends:** [ADR-0003](0003-open-source-strategy.md)'s deferred `api/`/`web/`
  licensing decision ("TBD, revisit at M3/M4"). ADR-0003's actual decision —
  Apache-2.0 for `runner/`/`schemas/`/`profiles/`/`packs/`, no AGPL, no
  CLA — is unchanged. Only the api/web question, left open there, is
  resolved here. Per [ADR-0001](0001-record-architecture-decisions.md),
  ADR-0003's body is left intact; it gets a status-line pointer to this ADR.

## Context

M3 built `api/` for real (FastAPI, Postgres, the ADR-0004/0005 trust gate).
That's the point ADR-0003 named as when this decision would have a real
answer instead of a hypothetical one: is `api/`/`web/` (the leaderboard
product) open, or does GOESB face a self-hostable-clone risk worth guarding
against?

The answer decided here: keep the leaderboard/database product **private**,
and — the actual decision this ADR adds beyond that — **in a separate
repository**, not a different license inside the same repo. A same-repo
split (open `runner/`, unlicensed/proprietary `api/`+`web/`) was the default
shape ADR-0003 left the door open to, but it has a real, avoidable risk: it
puts private product code, and eventually private product secrets
(deployment config, customer data handling, commercial-module code), in the
same git history as a repo that's meant to go fully public (roadmap M4's
"open-source flip"). One misjudged commit or an incomplete
history-scrub before that flip is a real, embarrassing, and avoidable
failure mode. A separate repo makes that structurally impossible instead of
relying on care.

## Decision

- **`taktx-io/GOESB`** (this repo) stays scoped exactly as ADR-0003 decided:
  `runner/`, `schemas/`, `profiles/`, `packs/`, `scripts/`, Apache-2.0. It no
  longer contains `api/`, `web/`, or `docker-compose.yml` at all.
- **`taktx-io/goesb-platform`** (new, private) holds `api/` and `web/` — the
  leaderboard, the database, the public website. All rights reserved, not
  part of the open core, not part of this repo's history going forward.
- `api/` keeps depending directly on `goesb-runner`'s trust primitives
  (`schema_validation`, `signing`, `hashing` — ADR-0004/0005) rather than
  reimplementing them, exactly as before the split — now via a sibling
  editable install (`pip install -e ../GOESB/runner`) instead of a
  same-repo relative path. Verified working end-to-end from the new
  location before this split was made permanent.
- History: `goesb-platform` starts fresh (one initial commit, noting
  extraction from `GOESB` at a specific commit for provenance) rather than a
  `git filter-repo`/subtree split — low value this early against real
  complexity/risk, and it's exactly the "avoid mixed history" goal this ADR
  is about.
- `GOESB` staying **private** for now is unaffected by this decision — the
  "open-source flip" (making it publicly visible) is roadmap M4's own
  distinct, separate action, not bundled into this repo split.

## Consequences

- `goesb-platform`'s CI needs read access to `GOESB` while it's still
  private — a fine-grained PAT (`contents: read` scoped to `GOESB` only),
  stored as `OESB_REPO_TOKEN` in `goesb-platform`'s repo secrets. **Not yet
  configured** — a one-time setup step for the account owner, same category
  as M2's still-open PyPI-publish gap. `goesb-platform`'s `api` CI job fails
  until this exists (fails fast with a clear "token not supplied" error, not
  silently). Goes away once `GOESB` is public — a plain checkout needs no
  credentials for a public repo.
- Local dev for `goesb-platform` requires cloning `GOESB` as a sibling
  directory — documented in `goesb-platform/README.md`.
- `docs/03-roadmap.md`'s M4 onward (website, curated views, and M8's
  commercial modules) now execute in `goesb-platform`, which has its own
  roadmap starting there; this repo's own forward scope is the method
  itself (languages, runtimes, M6 community/private packs, M7 conversation
  type).
- `governance.md` gets a line noting the open-method/private-product
  separation is now a repo boundary, not just a documented intent —
  strengthens, not weakens, the neutrality argument (anyone can audit the
  method without needing access to, or trust in claims about, the product).
- `docs/02-architecture.md` §2.2/§2.3 point at `goesb-platform` rather than
  describing `api/`/`web/` as living here.
