# OESB API

Open REST API that serves leaderboards, benchmark profiles, packs, hardware
records, and accepts benchmark result submissions. Everything the website does
is automatable through this API.

See docs/specs and docs/02-architecture.md for the endpoint contract.

## Quick start (placeholder)
```bash
pip install -e ".[dev]"
uvicorn oesb_api.main:app --reload
# open http://127.0.0.1:8000/docs
```
