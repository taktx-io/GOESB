# OESB API

Open REST API that serves leaderboards, benchmark profiles, packs, hardware
records, and accepts benchmark result submissions. Everything the website does
is automatable through this API.

> **Licensing:** unlike `runner/`/`schemas/`/`profiles/`/`packs/` (Apache-2.0,
> see the root `LICENSE`), this directory's license is **not yet decided** —
> treat it as All Rights Reserved until [ADR-0003](../docs/adr/0003-open-source-strategy.md)
> is revisited at milestone M3.

See docs/specs and docs/02-architecture.md for the endpoint contract.

## Quick start (placeholder)
```bash
pip install -e ".[dev]"
uvicorn oesb_api.main:app --reload
# open http://127.0.0.1:8000/docs
```
