# Benchmark Profiles

A profile defines *exactly* how a benchmark is run and scored: benchmark type,
runtime, model, configuration, normalization, scoring rules, and required
metrics. Each profile has a unique `id`, a semantic `version`, and a `changelog`.

Public leaderboards accept **official** profiles only. Users may create their own
profiles, but those appear only locally or in private projects.

**Language is a first-class field.** Every profile sets a BCP-47 `language` and
selects a per-language normalization ruleset (e.g. `goesb-en-v1`, `goesb-nl-v1`).
Any language is supported — adding one means adding a profile (and its ruleset),
not changing the platform. Naming convention: `<model>-<language>-<type>`.

Worked examples (any language, not just these):
- `whisper-medium-en-batch/` — Whisper Medium, English, batch.
- `whisper-medium-nl-batch/` — Whisper Medium, Dutch, batch.

See `schemas/benchmark-profile.schema.json` for the contract.
