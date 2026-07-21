# Benchmark Profiles

A profile defines *exactly* how a benchmark is run and scored: benchmark type,
runtime, model, configuration, normalization, scoring rules, and required
metrics. Each profile has a unique `id`, a semantic `version`, and a `changelog`.

Public leaderboards accept **official** profiles only. Users may create their own
profiles, but those appear only locally or in private projects.

See `schemas/benchmark-profile.schema.json` for the contract and
`whisper-medium-nl-batch/` for a worked example.
