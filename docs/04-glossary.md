# Glossary

**ASR** — Automatic Speech Recognition; converting speech audio to text.

**Batch** — benchmark type that processes complete audio in one pass.

**Benchmark Pack (Pack)** — immutable, hash-identified bundle of audio,
transcript, metadata, target profile, normalization, scoring, docs, and license.
Open, community, or private.

**Benchmark Profile (Profile)** — official, versioned definition of exactly how a
benchmark runs and is scored (type, runtime, model, config, normalization,
scoring, metrics). Gates public leaderboards.

**Benchmark type** — one of `batch`, `streaming`, `conversation`; a fundamental
category, not a setting.

**CER** — Character Error Rate.

**Conversation** — benchmark type covering the full voice pipeline
mic→VAD→ASR→LLM→TTS→speaker.

**Edge / local** — runs entirely on the user's own device, no cloud.

**Environment fingerprint** — structured capture of hardware, software, and model
settings recorded with every result for reproducibility.

**First Partial / First Final Latency** — time to the first partial / first
finalized transcript in streaming.

**LLM** — Large Language Model (the reasoning step in a conversation pipeline).

**Normalization** — text-cleanup rules applied identically to reference and
hypothesis before WER/CER; fixed by the profile.

**NPU** — Neural Processing Unit.

**Partial Stability** — how much of a streaming partial hypothesis survives
unchanged into the final transcript (anti-flicker measure).

**Runner** — the official, signed, cross-platform program that runs benchmarks
and emits signed results. Never executes arbitrary code.

**Runtime / Runtime adapter** — the engine that drives a model (e.g.
faster-whisper, whisper.cpp) and the plugin that integrates it.

**RTF (Real-Time Factor)** — processing time ÷ audio duration; < 1.0 is faster
than realtime.

**TTS** — Text-To-Speech.

**VAD** — Voice Activity Detection.

**Visibility** — a pack's scope: `open` (public leaderboards), `community`
(shared datasets), `private` (local only, never public).

**WER** — Word Error Rate.
