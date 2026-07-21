# OESB Web

Public website for OESB: browse and filter leaderboards, inspect profiles and
packs, and explore hardware records. Talks to the OESB API only (no direct DB
access), so anything the site shows is also available programmatically.

> **Licensing:** unlike `runner/`/`schemas/`/`profiles/`/`packs/` (Apache-2.0,
> see the root `LICENSE`), this directory's license is **not yet decided** —
> treat it as All Rights Reserved until [ADR-0003](../docs/adr/0003-open-source-strategy.md)
> is revisited at milestone M4.

## Quick start (placeholder)
```bash
npm install
OESB_API_URL=http://127.0.0.1:8000 npm run dev
# open http://localhost:3000
```
