# OESB API

Open REST API that serves leaderboards, benchmark profiles, packs, hardware
records, and accepts benchmark result submissions. Everything the website does
is automatable through this API.

> **Licensing:** unlike `runner/`/`schemas/`/`profiles/`/`packs/` (Apache-2.0,
> see the root `LICENSE`), this directory's license is **not yet decided** —
> treat it as All Rights Reserved until [ADR-0003](../docs/adr/0003-open-source-strategy.md)
> is revisited at milestone M3.

See docs/specs and docs/02-architecture.md for the endpoint contract.

## Quick start

Ingest re-verifies every submitted result's hash and signature (ADR-0004)
using the runner's own primitives, so `oesb-runner` is installed alongside
this package — not a separate reimplementation.

```bash
# from the repo root
docker compose up -d postgres
pip install -e ./runner -e "./api[dev]"
cd api
alembic upgrade head
uvicorn oesb_api.main:app --reload
# open http://127.0.0.1:8000/docs
```

Run the tests the same way (`DATABASE_URL` defaults to the docker-compose
Postgres above): `cd api && pytest -q`.
