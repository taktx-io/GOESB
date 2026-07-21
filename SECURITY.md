# Security Policy

OESB's credibility depends on the integrity of its benchmark results and the
safety of the runner. Security reports are taken seriously.

## Reporting a vulnerability
Please report privately to the maintainers (do not open a public issue for
security-sensitive matters). Provide a description, reproduction, and impact.

## Scope of particular interest
- The **Benchmark Runner** never executes arbitrary user code. Any path that
  could lead to code execution from a pack, profile, model, or config is a
  critical vulnerability.
- Tampering with result integrity, hashes, or signatures.
- Leakage of private/local audio or personally identifiable data.

See docs/adr/0004-runner-security-model.md for the threat model.
