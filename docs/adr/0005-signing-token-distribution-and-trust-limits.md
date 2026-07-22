# ADR-0005 — Signing token distribution, and the honest limits of what signing proves

- **Status:** Accepted (design only — not yet implemented; see Consequences)
- **Date:** 2026-07-22
- **Supersedes/amends:** [ADR-0004](0004-runner-security-model.md)'s claims about
  what a result signature proves, and its "local keypair, generated on first
  use" key model. ADR-0004's other decisions (declarative inputs only,
  hash everything, reviewed-plugin model, no arbitrary code execution) stand
  unchanged — this ADR narrows one specific claim and replaces one specific
  mechanism.

## Context

ADR-0004 states a signature proves "a genuine, unmodified official runner
produced this" and deliberately avoids requiring identity infrastructure.
Working through the actual distribution model exposes two problems with that:

1. **The "runner-embedded key" was never actually implemented as embedded.**
   `load_or_create_keypair()` (`runner/src/oesb_runner/signing.py`) generates
   a *new* keypair locally on first use, per install. There is no single
   official signing key — every `pip install oesb-runner` mints its own.
2. **Even if a single key were embedded, it couldn't stay secret.** The
   runner's primary distribution channel is a `pip install`-able Python
   package — plain source in a wheel/sdist (docs/02-architecture.md §2.1).
   Any secret embedded in it is in plaintext for every user who installs it;
   there is no meaningful extraction barrier. Bundled binaries are explicitly
   "optional convenience, never required," so this isn't fixable by assuming
   a compiled distribution.

A further question raised in discussion: could the runner also prove *its own
code hasn't been tampered with*? Reporting a hash of the adapter's source file
(`runtime.sha256`, already implemented via `sha256_module_source`) can be
checked against a known-good hash from an official CI build. But that check
only proves the file *on disk* matches — it says nothing about whether
something else in the same unsandboxed process (a monkey-patch, a wrapper,
a separate script) altered what actually got signed. Genuinely proving "this
exact code executed unmodified and produced this output" is a remote/hardware
attestation problem (TPM, SGX/TrustZone, secure boot chains) — real
technology, but platform-fragmented, and what it attests (boot-chain
integrity) is far short of "this Python call stack wasn't tampered with at
runtime." Disproportionate for this project's stage; notably, MLPerf and SPEC
don't attempt it either, relying instead on submission review and
reproducibility challenges.

## Decision

**1. Correct the claim.** A result signature proves the document has not been
altered since it was signed (tamper-evidence) and, with the mechanism below,
which server-issued credential authorized the signing (submitter
accountability). It does **not** prove the measurement itself is genuine, and
never will without disproportionate attestation infrastructure. Docs should
say this plainly rather than imply more.

**2. Replace per-install static keys with server-issued, ephemeral,
single-use signing tokens — required only for public submission, not for
producing a result.** Preserves ADR-0004's "the runner does not require
network access to produce a result" — network is needed only at the
submission step, same as today's `POST /benchmark` being separate from
`oesb run`.

Flow:
- Before submitting, the runner generates an ephemeral ed25519 keypair
  **locally, in memory** — the private key never leaves the machine that
  generates it (this is the meaningful improvement over a transmitted
  private key: nothing to intercept in transit or leak via logs).
- Runner calls `POST /runner-tokens` with the ephemeral **public** key.
  API generates a `token_id`, stores `(token_id, public_key, issued_at,
  expires_at, used_at nullable, requester_ip)` in a new `runner_tokens`
  table, returns `token_id`. Rate-limited per IP (e.g. N/hour); short TTL
  (e.g. 24h); single-use (marked `used_at` on first successful `/benchmark`
  referencing it — a second submission with the same `token_id` is
  rejected, closing "mint one token, sign a thousand fake results with it").
- Runner signs the result locally with its ephemeral private key, exactly
  as today, and puts `token_id` in the existing `signature.key_id` field
  (already a free-form string in the schema — no schema change needed).
- Ingest verification changes from "read a public key off local disk" to
  "look up `signature.key_id` in `runner_tokens`, reject if missing /
  expired / already used, then verify the signature against *that* public
  key." `oesb_runner.signing` stays DB-agnostic (a pure, reusable module,
  used by both runner and API) — it gains a `public_key_resolver:
  Callable[[key_id], bytes | None]` parameter instead of hardcoding a
  `key_dir` file read; the API supplies a DB-backed resolver, the CLI's
  local/offline path keeps the current file-based resolver as the default.
- Local/private runs (no submission intended) are entirely unaffected —
  they keep signing with a local keypair as today; call-home is opt-in,
  triggered only by the submission path.

**3. Add a non-blocking runtime-integrity signal, not a gate.** Ingest
compares `runtime.sha256` against a small allowlist of hashes from official,
Sigstore-attested release builds (the attestation pipeline already exists —
`.github/workflows/release.yml`, M2) and records whether it matched — surfaced
on the result (e.g. `runtime_attested: bool`), not a rejection. Legitimate
forks/dev builds still get ingested; they're just visibly not
attested-official, same "surfaced, not hidden" pattern as every other
unavailable/unverified field in this codebase. This does **not** address the
deeper live-tampering problem in Context — it only catches "ran an obviously
modified adapter file," which is still worth catching cheaply.

**4. What this still doesn't solve, and what covers it instead.** No
combination of the above proves a determined, technically capable user didn't
fabricate a result. The actual backstop, unchanged from ADR-0004 and already
partly built, is layered: the reviewed-plugin/no-arbitrary-code model bounds
what a genuine run can even report; FR-5.3's repeat-run tolerance surfaces
suspicious variance; and an "official/reviewed" leaderboard tier (not yet
built) would require the same human/reputational review SPEC and MLPerf rely
on — cryptography narrows the cheap-fraud window, it doesn't close it.

## Consequences

- New `runner_tokens` table + `POST /runner-tokens` endpoint (api/).
- `oesb_runner.signing` gains a pluggable key-resolution seam; CLI behavior
  for local/private runs is unchanged.
- `benchmark-result.schema.json` needs no changes — `signature.key_id` is
  repurposed, not restructured.
- Meaningfully improves M8's deferred abuse-mitigation story (rate-limit,
  revoke, correlate submissions) as a side effect of fixing the key-secrecy
  problem — not a separate build.
- **Design only.** Nothing here is implemented yet — tracked as a follow-up
  once prioritized against the milestone roadmap (naturally fits alongside
  M8's abuse-mitigation scope, or can be pulled forward if abuse becomes a
  concern before M8).
- Docs referencing ADR-0004's original signing-trust claim (roadmap M3 entry,
  governance.md) should point readers to this ADR for the corrected version.
