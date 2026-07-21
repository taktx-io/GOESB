# Benchmark Packs

A pack bundles audio, transcripts, metadata, a target profile, normalization and
scoring rules, docs, and a license. Every published pack is **immutable** and
identified by an `id`, `version`, and `sha256`.

- **Open packs** (e.g. Common Voice, FLEURS, VoxPopuli, LibriSpeech) power the
  public leaderboards — everyone uses identical data, so results compare directly.
- **Community packs** are contributed datasets (Dutch elderly, smart home,
  meetings, far-field, dialects, children, noisy factory, ...).
- **Private packs** stay local: audio never leaves the machine, only metadata
  and results are stored, and they never appear on public leaderboards.

Audio files are never committed to this repository (privacy-first). A pack in git
contains only the manifest and metadata; audio is referenced by hashed manifest.
