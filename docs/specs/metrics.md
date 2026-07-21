# Metrics specification

Every metric has exactly one precise, reproducible definition. Profiles declare
which metric ids they require; metric plugins compute them. Units are explicit.

> Convention: lower is better unless marked ↑. Timing metrics are wall-clock
> measured by the runner harness; the harness's own overhead is characterised and
> subtracted where it would bias a metric (see [ADR-0002](../adr/0002-tech-stack.md)).

## Quality

| id | Name | Unit | Definition |
|----|------|------|-----------|
| `wer` | Word Error Rate | ratio | `(S + D + I) / N_ref` after profile normalization, where S/D/I are substitutions/deletions/insertions from reference-hypothesis alignment and `N_ref` is reference word count. |
| `cer` | Character Error Rate | ratio | Same as WER at character granularity after normalization. |

Normalization (lowercasing, punctuation, number expansion, ruleset id) is fixed
by the profile and applied identically to reference and hypothesis **before**
alignment, so WER/CER are comparable only within the same profile version.

Normalization is **per-language and pluggable**: each language has its own
versioned ruleset (e.g. `oesb-en-v1`, `oesb-nl-v1`, `oesb-de-v1`) handling
language-specific number expansion, casing, diacritics, punctuation, and script.
The metric implementations (WER/CER alignment) are language-agnostic; only the
ruleset is language-aware. This keeps the core free of any language assumption
while allowing correct scoring for high- and low-resource languages and
non-Latin scripts alike.

## Realtime (streaming)

| id | Name | Unit | Definition |
|----|------|------|-----------|
| `first_partial_latency` | First Partial Latency | ms | Time from first speech audio input to the first partial hypothesis emitted. |
| `first_final_latency` | First Final Latency | ms | Time from first speech audio to the first finalized (non-revisable) token/segment. |
| `end_of_speech_latency` | End-of-Speech Latency | ms | Time from the true end of speech to the final transcript being emitted. |
| `update_frequency` ↑ | Update Frequency | Hz | Rate of partial hypothesis updates during continuous speech. |
| `partial_stability` ↑ | Partial Stability | ratio 0–1 | Fraction of partial-hypothesis tokens that survive unchanged into the final transcript (measures "flicker"). 1.0 = partials never rewritten. |
| `streaming_responsiveness` ↑ | Streaming Responsiveness | index | Composite of update frequency and stability against latency; defined per profile. OESB's default (used unless a profile overrides it): `(update_frequency_hz * partial_stability) / first_partial_latency_p50_s`. |

## Performance

| id | Name | Unit | Definition |
|----|------|------|-----------|
| `real_time_factor` | Real-Time Factor (RTF) | ratio | `processing_time / audio_duration`. < 1.0 means faster than realtime. |
| `throughput` ↑ | Throughput | audio-s/s | Seconds of audio processed per wall-clock second (≈ 1/RTF for batch). |
| `cpu_pct` | CPU utilisation | % | Mean CPU across the run (all cores normalised). |
| `gpu_pct` | GPU utilisation | % | Mean GPU utilisation, where applicable. |
| `npu_pct` | NPU utilisation | % | Mean NPU utilisation, where a probe exists. |
| `ram_mb` | Peak RAM | MB | Peak resident memory of the benchmark process tree. |
| `temperature_c` | Temperature | °C | Peak package/SoC temperature during the run (throttling indicator). |
| `energy_wh` | Energy | Wh | Total energy consumed for the run (RAPL / battery delta / external meter). |

## Economic

| id | Name | Unit | Definition |
|----|------|------|-----------|
| `hardware_price_eur` | Hardware price | € | Reference price of the device under test (sourced, dated). |
| `watt_per_stream` | Watt per realtime stream | W | Sustained power to keep one realtime stream at RTF < 1.0. |
| `eur_per_stream` | Euro per realtime stream | € | Amortised hardware+energy cost per concurrent realtime stream. |
| `price_perf_index` ↑ | Price/performance index | index | Composite ranking quality+speed against price; formula fixed per leaderboard view. |

## Reporting

Each metric is reported with: value, unit, and — where meaningful — a spread
(e.g. mean ± std over repeats, or p50/p95 for latencies). Latency metrics for
streaming/conversation should always report p50 and p95, never mean alone, since
tail latency is what users feel.

## Conversation (pipeline) metrics

Defined by the `conversation` benchmark type; measured across
mic→VAD→ASR→LLM→TTS→speaker: `time_to_first_response`, `time_to_first_audio`,
`end_to_end_latency`, `barge_in_latency`, plus the performance/energy metrics
above aggregated over the pipeline.
