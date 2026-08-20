# Metrics specification

Every metric has exactly one precise, reproducible definition. Profiles declare
which metric ids they require; metric plugins compute them. Units are explicit.

> Convention: lower is better unless marked ↑. Timing metrics are wall-clock
> measured by the runner harness; the harness's own overhead is characterised and
> subtracted where it would bias a metric (see [ADR-0002](../adr/0002-tech-stack.md)).

## Quality

| id | Name | Unit | Definition |
|----|------|------|-----------|
| `wer` | Word Error Rate | ratio | `(S + D + I) / N_ref` after profile normalization, where S/D/I are substitutions/deletions/insertions from reference-hypothesis alignment and `N_ref` is reference word count. `value` is corpus-level (sum of edits / sum of ref words across the whole pack, not the mean of per-utterance ratios, which would bias short utterances); `spread` (p50/p95/std/min/max) is computed **per recording**, not per repeat — see Reporting below. |
| `wer_substitutions` / `wer_deletions` / `wer_insertions` | WER error composition | count | Corpus-level totals from the same alignment that produces `wer` — a WER ratio alone can't distinguish "hears worse" (substitutions/deletions rising) from "runs away" (insertions rising), two different failure modes with two different fixes. |
| `cer` | Character Error Rate | ratio | Same as WER at character granularity after normalization. Same `value`/`spread` convention as `wer`. |
| `cer_substitutions` / `cer_deletions` / `cer_insertions` | CER error composition | count | Same as the `wer_*` breakdown, at character granularity. |

Normalization (lowercasing, punctuation, number expansion, ruleset id) is fixed
by the profile and applied identically to reference and hypothesis **before**
alignment, so WER/CER are comparable only within the same profile version.

Normalization is **per-language and pluggable**: each language has its own
versioned ruleset (e.g. `goesb-en-v1`, `goesb-nl-v1`, `goesb-de-v1`) handling
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
| `streaming_responsiveness` ↑ | Streaming Responsiveness | index | Composite of update frequency and stability against latency; defined per profile. GOESB's default (used unless a profile overrides it): `(update_frequency_hz * partial_stability) / first_partial_latency_p50_s`. |

**Two structurally different kinds of streaming engine sit behind these
metrics, and three of them only discriminate among one kind.** GOESB's
streaming adapters split in two (ADR-0013):

- **Genuinely incremental** — `vosk` (Kaldi decoder state carried across
  `AcceptWaveform`) and `nemotron` (cache-aware `chunked_limited` streaming;
  each second of audio is encoded exactly once). Emission is left-to-right
  and never revised.
- **Bounded-window re-decode** — `faster-whisper`, `whisper-cpp` and
  `parakeet`, which have no incremental path and re-transcribe a bounded
  window every chunk (`streaming.run_windowed_local_agreement_streaming`),
  reaching "final" only by a local-agreement approximation.

For the incremental engines, `partial_stability` is **trivially ~1.0** —
there is no revision mechanism for it to measure. `nemotron` reports exactly
1.0 by construction (measured, not asserted: every published partial is a
strict word-wise extension of the previous one). For the same reason
`first_partial_latency` and `first_final_latency` **converge**, and for
`nemotron` they are byte-identical: the first non-empty partial IS the first
finalized word.

Those are true statements about the engines, not flattering artefacts — but
a leaderboard column showing `partial_stability` 1.00 next to
faster-whisper's 0.8x invites exactly the wrong reading, because the two
numbers are not answers to the same question. **`partial_stability` only
discriminates among the re-decode engines**, and a `first_final_latency`
column ranked across all five is comparing a measured revision delay against
a quantity that is structurally zero. Anything ranking these across both
kinds in one column must say which kind each row is.

The nominal latency axes are not comparable either: the re-decode engines
declare `chunk_ms` (a re-decode window an adapter picks) and `nemotron`
declares `streaming_latency_ms` (an encoder right-attention context the
checkpoint fixes). Deliberately different parameter names, because they are
different physical quantities (ADR-0013 §3).

**No backpressure/queueing model.** The runner simulates streaming with a
virtual real-time clock: chunk *k*'s audio is always modeled as "arriving"
at its fixed nominal offset, regardless of how long the previous chunk
actually took to decode. A genuinely slow backend (RTF > 1) in a live
deployment would fall behind and see audio queue up — more of it bundled
into each subsequent decode call, not the fixed nominal chunk this
simulation always feeds it. These streaming metrics are therefore an
accurate measure of per-chunk decode latency, but **not** a faithful
simulation of what a live deployment would feel like once a backend is
too slow to keep up with real-time audio — worth keeping in mind when
comparing streaming numbers across backends of very different speeds
(exactly the CPU/GPU/NPU edge-hardware comparisons GOESB exists for).

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

Each metric is reported with: value, unit, and — where meaningful — a spread.
The axis a spread is computed over depends on the metric:

- **Most scalar metrics** (`real_time_factor`, `cpu_pct`, `ram_mb`, ...): mean ±
  std **over repeats** — how stable is this number if you run the exact same
  benchmark again.
- **Latency metrics** (`first_partial_latency`, ...): p50/p95 pooled **over
  per-utterance samples**, always reported (never mean alone, since tail
  latency is what users feel), independent of repeat count.
- **`wer`/`cer`**: p50/p95/std/min/max pooled **over per-recording ratios**
  (one WER/CER value per utterance in the pack), always reported — not over
  repeats. A corpus-aggregate WER can look unremarkable while a handful of
  individual recordings sit at 60-90%+; that bimodal failure is invisible in
  a mean and obvious in a p95-over-recordings (real report: this was
  originally computed over the 1-2 repeats instead, which for a deterministic
  decode — no adapter here samples with `temperature > 0` — produced a
  spread of exactly zero and hid this signal entirely).

## Conversation (pipeline) metrics

Defined by the `conversation` benchmark type; measured across
mic→VAD→ASR→LLM→TTS→speaker: `time_to_first_response`, `time_to_first_audio`,
`end_to_end_latency`, `barge_in_latency`, plus the performance/energy metrics
above aggregated over the pipeline.
