# Governance & neutrality

GOESB's authority rests on being trustworthy, not just open ([00-vision.md](00-vision.md),
NFR-5). Apache-2.0 ([ADR-0003](adr/0003-open-source-strategy.md)) makes the code
inspectable; this document makes the *process* around it inspectable too —
what actually stops a sponsor, vendor, or maintainer from tilting a result.

The scoring/normalization/measurement method (this repo) and the commercial
leaderboard product built on it (`taktx-io/goesb-platform`, private) are
separate repositories, not just a license boundary within one —
[ADR-0006](adr/0006-split-platform-repo.md). Anyone can inspect exactly how
a number is produced without needing access to, or trusting claims about,
the product built around it.

## Sponsorship never influences results

If GOESB accepts funding, hardware donations, or sponsorship at any point
(hosting costs, hardware for cross-platform testing, etc.), that funding:

- **Never** affects which profile is "official," which pack is "open," or
  which runtime/model appears more favourably.
- **Never** affects scoring, normalization, or leaderboard ranking logic.
- **Never** buys placement on a curated view ("Best Dutch under €300", etc.) —
  those are defined by their stated selection criteria only, applied
  mechanically to verified results.
- Any sponsorship is disclosed publicly (a `SPONSORS.md` once any exist), and
  a sponsor's own hardware/runtime is benchmarked under the exact same
  profile+pack+verification path as anyone else's — no separate lane.

A sponsor may fund *infrastructure* (compute, hosting). A sponsor never funds
*a number*.

## Plugin & contribution review

New runtime adapters, metrics, hardware probes, or normalization rulesets are
**reviewed, in-tree code merged via pull request** — never a runtime plugin
supplied by a user ([ADR-0004](adr/0004-runner-security-model.md)). Review
checks, in order:

1. **Security** — no arbitrary code execution path introduced (FR-9.1).
2. **Neutrality** — the contribution doesn't hardcode an assumption that
   favours its author's own hardware/runtime/model (e.g. a metric defined in
   a way that only one vendor's chip can score well on).
3. **Correctness** — matches its documented definition
   ([metrics.md](specs/metrics.md), [environment-capture.md](specs/environment-capture.md)),
   has tests.
4. **Reproducibility** — doesn't silently change how an *existing* official
   profile scores (that requires a new profile version, never an in-place edit).

## Conflict of interest

A reviewer or maintainer employed by, or with a financial stake in, a
hardware vendor or runtime project **discloses that when reviewing a
contribution touching that vendor/runtime**, and does not merge it solo —
a second, unaffiliated reviewer signs off.

## Profile/pack integrity (mechanical, not discretionary)

These aren't governance judgment calls — they're enforced by the schemas and
runner already:

- A profile is immutable per version; any scoring change is a new version
  with a changelog entry (FR-2.2).
- A pack is immutable and hash-identified after publication (FR-3.2).
- Public leaderboards accept only official-profile + open-pack + verified-
  environment results (FR-7.3) — no manual curation of which results appear.

## Decision-making (current stage)

GOESB is pre-community: architectural and roadmap decisions are made by Eric
Hendriks, recorded as ADRs for traceability. As external contributors and
users join, this section will be revisited toward a more distributed model
(e.g. a maintainers group, an RFC process for profile/schema changes) — not
pre-built speculatively before there's a community to govern.

## Reporting a neutrality concern

If a result, profile, ranking, or piece of documentation looks like it favours
a vendor, runtime, or sponsor, open a GitHub issue tagged `neutrality` (or use
the contact in [SECURITY.md](../SECURITY.md) for sensitive reports). These are
treated as high priority — neutrality is the product.
