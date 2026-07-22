# ADR-0004 — Runner security model

- **Status:** Accepted — **amended by [ADR-0005](0005-signing-token-distribution-and-trust-limits.md)**,
  which corrects this ADR's claim about what a result signature proves and
  replaces the "local keypair generated on first use" key model. Everything
  else below (declarative inputs only, hash everything, reviewed-plugin
  model, no arbitrary code execution) stands as originally decided. Per
  [ADR-0001](0001-record-architecture-decisions.md), this record is left
  intact rather than rewritten — read ADR-0005 for the current signing story.
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

**Signing key is runner-embedded, not per-user identity.** The key that signs a
result document belongs to the official runner build/installation, not to a
personal or account-linked identity. A signature therefore proves "this result
was produced by a genuine, unmodified official runner" — not "this specific
person submitted it." This deliberately avoids requiring user accounts/identity
infrastructure before M3, and matches the trust claim OESB actually needs
(genuine measurement, not attributed authorship). `POST /benchmark` submission
itself may remain anonymous/unauthenticated as a result; abuse mitigation
(rate-limiting, submission review) is deferred to scale-hardening (M8) rather
than gating M3/M4.

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
