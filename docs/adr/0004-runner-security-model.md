# ADR-0004 — Runner security model

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

OESB's results must be trustworthy and safe to produce. Two threats dominate:
(1) a malicious pack/profile/config coercing the runner into executing arbitrary
code on a contributor's machine; (2) forged or tampered results polluting public
leaderboards. The vision is explicit: the runner must **never execute arbitrary
code**, and every public benchmark must be **cryptographically verifiable**.

## Decision

**Declarative inputs only.** Users provide models, datasets, metadata, and
benchmark configuration — data, never code. Explicitly disallowed as inputs:
Python scripts, shell scripts, Dockerfiles, executables, and plugins with
unrestricted rights. The runner has no "run this hook" mechanism.

**Capabilities ship as reviewed plugins, not runtime code.** Runtime adapters,
metrics, and hardware probes are code that lives in the runner's repository,
passes review + CI, and is released as part of a signed runner build. Extending
OESB means opening a pull request, not injecting an executable into a run.

**Hash everything.** SHA-256 over profile, model, runtime, configuration,
dataset, and result. Hashes are stored with the result, enabling tamper
detection, deduplication, and exact comparison.

**Sign the runner and results.** Official runner builds are digitally signed;
result documents are signed. Public leaderboard ingestion verifies the signature
chains and re-checks hashes before ranking.

**Least privilege at run time.** The runner reads its declared inputs and writes
its result; it does not require network access to produce a result (submission is
a separate, explicit step). Private audio never leaves the machine.

## Consequences

- Strong safety and integrity guarantees that back the "objective + reproducible"
  claim.
- Less flexible than "bring your own script": novel runtimes/metrics require a
  reviewed contribution rather than a drop-in. This is an accepted, deliberate
  trade-off — it is the mechanism that keeps results trustworthy.
- Requires signing-key management and a plugin review process (governance).
- Verification logic (hash + signature re-check) must exist on the ingest path
  before any public leaderboard launches (roadmap M3/M4 gate).
