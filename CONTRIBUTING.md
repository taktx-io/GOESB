# Contributing to GOESB

Thanks for helping build an objective, reproducible benchmark for edge speech AI.

## Repository layout
- `runner/` — Python benchmark runner (CLI, environment capture, hashing).
- `schemas/` — JSON Schemas for profiles and packs (the source of truth).
- `profiles/` — official benchmark profiles.
- `packs/` — pack manifests/metadata (never audio).
- `docs/` — vision, requirements, architecture, roadmap, ADRs, specs.
- `scripts/` — repo tooling (schema validation, etc.).

The leaderboard/API product (`api/`, `web/`) lives in the separate, private
`taktx-io/goesb-platform` repo — see
[ADR-0006](docs/adr/0006-split-platform-repo.md).

## Ground rules
1. **Reproducibility first.** Any change to how a benchmark runs or is scored
   requires a new profile *version* and a changelog entry — never edit in place.
2. **No arbitrary code execution.** The runner must never run user-provided
   scripts, shell, Dockerfiles, or plugins with unrestricted rights.
3. **Privacy-first.** Never commit audio or personal data. Private data stays local.
4. **Vendor-neutral.** No favouring of a specific chip, runtime, or model.

## Development
```bash
make setup     # install runner, api, web
make lint
make test
python scripts/validate_assets.py   # validate profiles/packs against schemas
```

## Commit / PR
- Small, focused PRs. Fill in the PR template.
- Add/adjust tests and docs. New profiles/packs must validate and record hashes.
