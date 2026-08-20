# ADR-0013 — `nemotron` runtime adapter: genuinely cache-aware streaming, GPU-only

- **Status:** Accepted (drafted and accepted 2026-08-20, by Eric). Two
  passages in §Decision were corrected in place on acceptance, against what
  implementation actually measured — see the addendum for the numbers behind
  each. Everything else is the draft as written.
- **Date:** 2026-08-20
- **Builds on / relates to:** [ADR-0004](0004-runner-security-model.md)
  (curated, in-tree adapters — a new runtime is a reviewed addition, not a
  plugin), [ADR-0008](0008-explicit-compute-backend.md) (explicit
  `--backend`, never auto-selected — this adapter is the first that would
  be *tempted* to violate it, see §4), [ADR-0009](0009-parameterized-profile-configuration.md)
  (`overridable` must be a subset of what the adapter genuinely applies —
  the whole reason §3 adds a new parameter instead of reusing `chunk_ms`),
  [ADR-0011](0011-decouple-packs-from-profiles.md) and
  [ADR-0012](0012-concurrency-benchmark-type.md) (the language-less
  concurrency profile shape this reuses verbatim).

## Context

### The streaming numbers on the board are mostly a simulation

GOESB ships four runtime adapters. Of their `streaming` implementations,
exactly one is genuinely incremental:

| runtime | streaming mechanism | incremental? |
|---|---|---|
| `vosk` | `KaldiRecognizer.AcceptWaveform`, real decoder state carried across calls, Kaldi endpointing decides finality | **yes** |
| `faster-whisper` | `streaming.run_windowed_local_agreement_streaming` | no — bounded-window re-decode |
| `whisper-cpp` | same shared function | no — bounded-window re-decode |
| `parakeet` | same shared function | no — bounded-window re-decode |

The re-decode path is honest about what it is (each adapter's docstring says
so at length, and the local-agreement commit rule exists precisely because
those engines have no real notion of "final"), but it is a *simulation of*
streaming, not streaming. `first_partial_latency` — the primary scoring
metric of every `*-streaming` profile — is therefore, for three of four
engines, the latency of re-transcribing a growing buffer, not the latency of
a model that consumes audio once.

`adapters/parakeet.py`'s `run_streaming` docstring names the exact missing
capability:

> there is no documented way to feed the Fast Conformer *encoder* new audio
> incrementally through the high-level `generate()` API this adapter uses —
> NeMo's real cache-aware-streaming Parakeet needs its own purpose-trained
> streaming checkpoints and NeMo-side buffer/context-size plumbing this
> `transformers` port doesn't carry.

### That capability now exists, in a dependency the runner already pins

Verified directly against the installed library (`transformers` 5.14.1, in
`runner/.venv` — the `parakeet` extra already pins `transformers>=5.9`), not
inferred from a changelog:

- `transformers/models/nemotron3_5_asr/` and
  `transformers/models/nemotron_asr_streaming/` are present, and both
  register into `MODEL_FOR_RNNT_MAPPING_NAMES` (`AutoModelForRNNT`;
  Parakeet's TDT variant sits in the separate `MODEL_FOR_TDT_MAPPING_NAMES`
  behind `AutoModelForTDT`).
- `NemotronAsrStreamingGenerationMixin.generate` accepts `input_features` as
  **a generator of mel chunks**, "encoded incrementally and appended to the
  encoder frame buffer as the decoder consumes it", carrying a
  `padding_cache` across chunks — NeMo's `chunked_limited` cache-aware
  streaming, reached through plain `transformers`. `_required_stream_chunk_frames`
  and `_validate_stream_chunk` enforce the exact fixed chunk geometry that
  mode requires (first chunk `1 + subsampling_factor * right`, subsequent
  `subsampling_factor * (right + 1)` mel frames).
- `processing_nemotron3_5_asr.py` exposes `supported_streaming_latencies_ms`
  as `{right: (right + 1) * encoder_frame_ms}` derived from the checkpoint's
  own `supported_num_lookahead_tokens` — **not a hardcoded list in the
  library**, which is why §3 below reads it at runtime rather than baking
  values into a profile schema.

### The checkpoint

`nvidia/nemotron-3.5-asr-streaming-0.6b` (released June 2026): 600M
parameters, cache-aware streaming FastConformer/RNNT, 40 language-locales in
three tiers. All six languages GOESB currently has packs for — en, es, fr,
pt, nl, de — are in the top "transcription-ready" tier (nl-NL explicitly
listed). One checkpoint covers every language, the same property that made
Parakeet attractive and that vosk/Whisper-family adapters need a per-language
model swap to fake.

License: **OpenMDW-1.1** (Linux Foundation), a permissive license granting
unrestricted, royalty-free use, modification and redistribution with no
field-of-use restriction; obligations are limited to retaining the LICENSE
file and origin notices, plus a litigation-termination clause. No copyleft,
no attribution requirement on model *outputs*. Compatible with how this
project already treats Parakeet's CC-BY-4.0 checkpoint, and less restrictive
(CC-BY requires attribution). Caveat worth recording: OpenMDW-1.1 was not yet
an official SPDX License ID as of May 2026 — an SBOM/compliance nit, not a
usage barrier.

### Why this wasn't already scoped

The 2026-08-01 streaming decision (`docs/handoffs/2026-08-01-streaming-benchmark-type.md`)
picked Vosk first and whisper.cpp second and deferred faster-whisper, on the
correct reading of the landscape *at that time*. Nemotron 3.5 ASR shipped in
June 2026 and reached `transformers` in 5.13; it simply wasn't on the list.
This ADR is not a reversal of that decision — it adds the engine that makes
the honest version of the metric available on a second, non-Kaldi engine.

## Decision

**Add a fifth curated runtime adapter, `nemotron`, backed by the single
checkpoint `nvidia/nemotron-3.5-asr-streaming-0.6b`, implementing `batch`,
`streaming` and `concurrency`, GPU-only, with a new `streaming_latency_ms`
configuration parameter that is deliberately *not* `chunk_ms`.**

### 1. One checkpoint, one runtime id

- Runtime adapter id: `nemotron`. New module
  `runner/src/oesb_runner/adapters/nemotron.py`, registered in
  `adapters/__init__.py`'s import tail.
- `model.name` in profiles: `nemotron-3.5-asr-streaming-0.6b`;
  `_resolve_model_id` prefixes `nvidia/`, the same runtime-agnostic-name →
  HF-Hub-id translation `parakeet.py` and the Whisper-family adapters
  already do.
- The older EN-only `nvidia/nemotron-speech-streaming-en-0.6b` (Jan 2026) is
  **out of scope** — superseded by 3.5, and an EN-only second checkpoint
  would double the profile count for one language GOESB already covers three
  other ways.
- New optional extra: `nemotron = ["transformers>=5.13", "torch>=2.2",
  "soundfile>=0.12", "librosa>=0.10"]`. `>=5.13` is the release line that
  actually ships these model classes (Parakeet's own extra stays at `>=5.9`;
  the two extras overlap but pin what each genuinely needs). **No
  `nemo_toolkit`** — same reasoning as `parakeet.py`'s module docstring.

### 2. All three benchmark types

`streaming` is the point, but ships alongside the other two:

- **`batch`** — the checkpoint supports offline transcription (the
  `pipeline("automatic-speech-recognition", ...)` path), and it is the
  *compute* baseline Nemotron's streaming numbers have to be read against:
  measured, chunked streaming costs ~3x the compute of the same decode done
  in one pass (0.216x vs 0.072x RTF). It is **not** an accuracy baseline.
  There is no full-right-context offline mode on this checkpoint — the
  processor stamps its own `default_num_lookahead_tokens` onto offline calls
  too, so "batch" is the same cache-aware, limited-right-context decode run
  in one shot, and at a given mode it produces byte-identical text to
  streaming. Measured, not assumed; see addendum §4. The batch profiles say
  this in their own header comments so no leaderboard reader infers two
  independent accuracy numbers where there is one.
- **`concurrency`** — a fifth engine under ADR-0012. See §5 for the one thing
  that must be verified rather than assumed.

### 3. `streaming_latency_ms`, a new parameter — not `chunk_ms`

Cache-aware streaming does not have a re-decode window. It has a fixed
**right attention context** (lookahead), declared by the checkpoint, which
determines both the algorithmic latency and the exact mel-frame count every
chunk must carry. Published modes are right-context `{0, 1, 3, 6, 13}` →
**80 / 160 / 320 / 560 / 1120 ms**. None of these coincide with the
250/500/1000/2000 ms `chunk_ms` values every existing `*-streaming` profile
declares, and the mismatch is not cosmetic: they are different physical
quantities.

- `benchmark-profile.schema.json`'s `configuration` gains
  `streaming_latency_ms` (integer, minimum 1), declared explicitly per that
  file's own convention (the block is not `additionalProperties: false`, so
  this is documentation, not a gate).
- Nemotron `*-streaming` profiles declare `configuration.streaming_latency_ms`
  and list it under `overridable` with an `allowed` enum. They **do not
  declare `chunk_ms` at all** — per ADR-0009 §2, declaring a parameter the
  adapter does not apply would sign a result asserting a value that had no
  effect.
- The `allowed` enum in each profile must be the set the *checkpoint's own
  processor* reports via `supported_streaming_latencies_ms`, read once during
  implementation and pinned into the profile — not copied from the model
  card. **Open discrepancy to resolve empirically:** the model card lists five
  modes (80/160/320/560/1120), while the `transformers` doc example prints
  four (`{0: 80, 3: 320, 6: 560, 13: 1120}`). Whichever the installed
  processor actually reports for this checkpoint is the answer; the profile
  enum matches it exactly, and the adapter validates against the processor at
  run time rather than trusting the profile.
- Requesting a latency the checkpoint doesn't support is a **hard error**,
  never a silent snap to the nearest supported mode — the same stance ADR-0008
  takes on backends, for the same reason.

Rejected alternative — *reuse `chunk_ms` and widen its allowed list*: keeps
one parameter name across engines at the cost of one name meaning two
unrelated things (re-decode window vs. encoder right-context). A leaderboard
column that silently compares those is worse than a leaderboard that shows
two axes. Rejected alternative — *snap to the nearest existing `chunk_ms`*
(320→250 etc.): the reported value would not be the value run. This project
does not do that.

### 4. GPU-only, and explicitly so

- `backends=frozenset({"cuda"})` on all three registrations, unless MPS is
  verified to actually work (see §6) — in which case `"metal"` is added on
  the strength of that measurement, not on the strength of torch shipping MPS
  support.
- `--backend cpu` against a `nemotron` profile is a **hard error** from the
  existing `get_supported_backends` gate. NVIDIA's model card lists GPU
  architectures only (Turing→Blackwell, Jetson), and GOESB's direction is
  GPU-based STT. This is a **scoping decision about what this project
  publishes for this engine, not a performance cliff**: torch CPU measures
  0.227x RTF here — faster than realtime, and in the same range as
  `parakeet`'s own shipped CPU path (see addendum §2). Adding a `cpu`
  backend later is therefore a live option, not a thing the hardware rules
  out; it is deliberately not this ADR's decision.
- **`device_map="auto"` is not used.** The `transformers` documentation
  example for this model uses it; ADR-0008 forbids exactly that, and
  `parakeet.py` already refuses it for the same reason. The device string
  comes from `--backend` through a `_DEVICE_BY_BACKEND` map, as in
  `parakeet.py`.
- This is the project's first profile set that a CPU-only contributor cannot
  run at all. Accepted deliberately: GOESB's direction is GPU-based STT, and
  the `goesb doctor` flywheel (ADR-0008 §2) already exists to surface
  "hardware you own can fill this empty cell."

### 5. RNNT finality is real — and that changes two metrics' meaning

`StreamTrace.PartialUpdate.committed_word_count` documents "committed" as
words that agreed across two consecutive hypotheses — a local-agreement
*approximation* of finality, needed because the Whisper-family engines revise
their own output. A streaming RNNT does not: tokens are emitted left-to-right
and are never revised. So, as `vosk.py`'s docstring already establishes for
the one other genuinely incremental engine, **every word emitted is committed
at the moment it is emitted**, and `run_windowed_local_agreement_streaming`
must not be reached from this adapter at all.

Consequences, to be documented rather than hidden:

- `partial_stability` becomes trivially ~1.0 for `nemotron` (and is already
  structurally so for `vosk`). That is a true statement about the engine, not
  a flattering artefact — but a leaderboard column showing 1.00 next to
  faster-whisper's 0.8x invites the wrong reading, so it needs a note in
  `docs/specs/metrics.md` stating that this metric only discriminates among
  re-decode engines.
- `first_final_latency` and `first_partial_latency` collapse toward each other
  for the same reason. Also true, also worth saying out loud.

### 6. Concurrency: verify thread-safety, don't assume it

ADR-0012 landed on three *different* answers for three engines
(faster-whisper: safe to share via `num_workers`; whisper-cpp: confirmed
unsafe, N full instances; vosk: unconfirmable, N full instances). Nemotron
gets the same treatment — no pattern is assumed to transfer. Additional
GPU-specific wrinkle this ADR flags but does not resolve: N independent model
instances is an N× **VRAM** cost, not an N× RAM cost, and a GPU that runs out
of VRAM fails harder and less legibly than a host that swaps. ADR-0012's
deliberately-deferred pre-run OOM check becomes materially more valuable here.

### 7. Profile ids and the wizard matrix

- Streaming/batch ids: `nemotron-3-5-<lang>-streaming` /
  `nemotron-3-5-<lang>-batch`, for the six languages with packs (12
  profiles). The dot-free `3-5` size segment is forced by the profile
  schema's own `id` pattern (`^[a-z0-9][a-z0-9-]*$`); the exact checkpoint
  identity lives in `model.name`, which permits dots — the same id/model-name
  split `parakeet-tdt-v3` (id) vs `parakeet-tdt-0.6b-v3` (`model.name`)
  already uses.
- Concurrency id: `nemotron-3-5-concurrency`, no language segment, following
  ADR-0012's corrected shape (one profile, not one per language) and its
  filler-pack convention.
- `cli.py`'s `_MATRIX_ID_RE` engine alternation gains `nemotron`, its size
  alternation gains `3-5`, and `_MATRIX_COLUMNS` gains `("nemotron", "3-5")`.

## Non-goals / explicit constraints

- **Not CPU, and not NeMo-Speech.cpp.** NVIDIA ships a C++ runtime for local
  inference; integrating it is a separate adapter and a separate ADR, tracked
  as future work, not smuggled in as a "fallback" here.
- **Not `nemotron-speech-streaming-en-0.6b`.**
- **Not `nemo_toolkit`.**
- **Not a re-litigation of Parakeet's bounded-window streaming.** That path
  stays as-is and stays honest about itself. Whether a future `transformers`
  exposes a genuine incremental encoder path for Parakeet is worth
  re-checking periodically; it is not this ADR's business.
- **Not a leaderboard UI change.** Presenting two structurally different
  streaming latency axes (`chunk_ms` re-decode window vs.
  `streaming_latency_ms` right-context) side by side without misleading a
  reader is a real design problem — see Consequences — tracked separately,
  the same way ADR-0012 deferred its own concurrency chart.

## Consequences

- **Streaming results split into two kinds that must not be silently
  compared.** `vosk` + `nemotron` measure a genuinely incremental engine;
  `faster-whisper` + `whisper-cpp` + `parakeet` measure bounded-window
  re-decode. At equal nominal latency these are not the same experiment.
  Anything that ranks `first_partial_latency` across all five engines in one
  column is producing a misleading ordering unless it says which kind each row
  is. This is the single largest downstream consequence of this ADR.
- **First GPU-required profile set.** Results can only come from GPU
  contributors, which makes the RunPod/GitHub-Actions multi-GPU automation
  (currently deferred, pending the manual Vast.ai T4 verification pass) the
  practical route to populating these cells rather than a nice-to-have.
- **`web/components/ProfileBuilder.tsx`'s `RUNTIMES` list is already stale** —
  it reads `["faster-whisper", "vosk", "whisper-cpp"]` and has been missing
  `parakeet` since that adapter landed. Same for
  `web/app/docs/create/profiles/page.tsx` ("faster-whisper, whisper.cpp, or
  vosk today") and two other docs pages. Adding `nemotron` means fixing a
  pre-existing omission at the same time, not just appending one entry.
- **No API/database migration.** `runtime.name` and `runtime.backend` are
  free strings owned by the adapter registry (the result schema says so
  explicitly), `configuration` is a generic passthrough, and metrics are
  JSONB. The platform ingests a `nemotron` result with no change.
- Establishes the precedent that **a genuinely different execution model gets
  its own parameter name**, rather than being folded into an existing one for
  UI tidiness — the counterpart to ADR-0012's "reuse the data path, add a
  narrow named exception."

## To verify during implementation (not assumed here)

Every item below is stated in this ADR as an expectation, not a finding. The
implementing change should confirm each against the real checkpoint on real
hardware and correct this ADR by addendum where reality differs — the same
discipline ADR-0012's addenda applied.

1. The checkpoint's actual `supported_streaming_latencies_ms` (four modes or
   five — see §3).
2. Whether `--backend metal` (torch MPS) runs this checkpoint at all, and at
   what RTF. If yes, `"metal"` joins the declared backend set; if no, it is
   documented as unsupported rather than left ambiguous.
3. VRAM footprint of one instance, and whether one instance is safe to share
   across concurrent threads (§6).
4. Whether the offline/`batch` path and the streaming path produce
   meaningfully different WER on the same audio — for a cache-aware model
   they should be close but not identical, and the size of that gap is itself
   a publishable result.
5. That the first-chunk vs subsequent-chunk mel-frame geometry
   (`_required_stream_chunk_frames`) can be satisfied from GOESB's existing
   16 kHz PCM decode path without resampling surprises, and how the final
   short chunk is padded (the library raises on any chunk of the wrong
   length).

## Addendum (2026-08-20): what implementation actually measured

Every item in "To verify during implementation" above was resolved against
the real checkpoint (`nvidia/nemotron-3.5-asr-streaming-0.6b`, HF revision
`1c8deae`, 638M parameters, 2552 MB of fp32 safetensors) on real hardware —
Apple Silicon MPS, `torch` 2.13.0 / `transformers` 5.14.1, `packs/fleurs-nl`
(15 real Dutch FLEURS clips, 119.5s). **Four of the five expectations were
confirmed. Two claims in the original draft were contradicted by
measurement; both were corrected in §Decision above on acceptance, and this
addendum holds the numbers behind those corrections.**

**Hardware caveat, stated up front:** no CUDA device was available to the
implementing session. Everything below is a real GPU measurement on Apple
Silicon MPS, not a simulation — but the CUDA numbers (and specifically the
CUDA VRAM figure the concurrency ceiling is derived from) remain an
extrapolation from the fp32 parameter/activation footprint, and should be
re-measured on the first NVIDIA box that runs these profiles.

### 1. Four latency modes, not five — the model card is wrong for this checkpoint

`processor.supported_streaming_latencies_ms` reports exactly:

```
{0: 80, 3: 320, 6: 560, 13: 1120}
```

Four modes. The `transformers` doc example was right and NVIDIA's model
card's fifth mode (160 ms, right context 1) **does not exist on this
checkpoint** — `set_num_lookahead_tokens(1)` raises. `default_num_lookahead_tokens`
is 3, i.e. the checkpoint's own default is the 320 ms mode. Every streaming
profile's `overridable.streaming_latency_ms.allowed` is exactly
`[80, 320, 560, 1120]`, and the adapter re-validates against the processor
at run time rather than trusting that list.

Incidental confirmation of the chunk arithmetic: each streaming chunk
advances the audio stream by `subsampling_factor * (right + 1)` mel frames,
which is exactly `(right + 1) * encoder_frame_ms` — so a mode's chunk audio
advance and its declared `streaming_latency_ms` are the same number by
construction (320 ms of audio per chunk at the 320 ms mode).

### 2. `metal` works and is declared — but "GPU-only" is a scoping call, not a performance cliff

`--backend metal` (torch MPS) runs this checkpoint correctly and quickly.
Full 15-clip `fleurs-nl` pack, through the real adapter:

| path | backend | RTF | WER |
|---|---|---|---|
| batch | metal | **0.072x** | 0.1314 |
| batch | cpu | 0.227x | 0.1314 |
| streaming, 80 ms | metal | 0.539x | 0.1204 |
| streaming, 320 ms | metal | **0.216x** | 0.1314 |
| streaming, 560 ms | metal | 0.160x | 0.1095 |
| streaming, 1120 ms | metal | 0.123x | 0.1095 |

So `"metal"` joins `"cuda"` in the declared backend set on the strength of
that number, as §4 said it would.

The `cpu` row is why §4 now reads as a scoping decision rather than a
performance one. The draft justified GPU-only partly on "a silent, unusably
slow CPU path"; measured, torch CPU is 0.227x RTF — faster than realtime,
~3x slower than metal, and in the same range as `parakeet`'s own measured
CPU path (~0.155x), an engine GOESB does ship a `cpu` backend for. GPU-only
is implemented as specified and stands on its other grounds; the number is
recorded here so a future reader deciding whether to add a `cpu` backend has
it rather than the assumption.

### 3. VRAM per instance, and a fourth thread-safety answer

- **2552 MB** of fp32 weights; **~3.35 GB** peak device allocation for one
  loaded, warmed-up instance (MPS driver allocation). No fp16/bf16 path is
  wired up yet — `quantization` remains an accepted-and-unused knob.
- **Not safe to share one instance across threads.** ADR-0012's discipline
  of reading the library rather than assuming a pattern transfers found
  *two* independent races here, one more than Parakeet has:
  1. `NemotronAsrStreamingGenerationMixin.generate` sets `self._streaming`,
     `self._stream_exhausted`, `self._streaming_num_lookahead_tokens` as
     plain instance attributes and `delattr`s them in its `finally`.
  2. `Nemotron3_5AsrGenerationMixin.generate` **rebinds
     `self.get_audio_features`** to a closure over `self._prompt_ids` and
     `del`s the attribute in `finally`. Two concurrent calls don't merely
     race on decode state — they cross-contaminate language conditioning,
     and whichever returns first leaves the other calling a deleted
     attribute.

  Both sit on top of the inherited `ParakeetRNNTGenerationMixin` state
  Parakeet already documents. So: N full instances, one per worker —
  whisper-cpp's and vosk's shape, not faster-whisper's.
- `profiles/nemotron-3-5-concurrency` therefore caps `concurrency` at **8**
  (≈27 GB), derived from the per-instance figure above rather than copied
  from another engine. A 24 GB consumer card realistically tops out around
  4; the ceiling is a cap, not a claim every GPU reaches it.
- **Pre-run OOM check: still deferred, deliberately, and not silently.** A
  `torch.cuda.mem_get_info()` precheck would cover CUDA only (MPS has no
  equivalent free-VRAM query), would still race any other process on the
  card, and buys less than §6 assumed: model construction is sequential and
  happens *before* the timed window, so an over-ambitious `concurrency`
  today already fails during load with a torch OOM naming the allocation,
  not midway through a signed run. Worth doing properly; not worth doing
  halfway here.

### 4. Batch and streaming are the *same* decode

Expectation: "close but not identical … the size of that gap is itself a
publishable result." Measured at the 320 ms mode on all 15 clips: the gap is
**zero**. Batch and streaming produce normalize-identical text on every
utterance, WER 0.1314 both ways, streaming-vs-batch divergence WER 0.0000.

The mechanism, confirmed by direct measurement rather than inferred: the
processor stamps its own `default_num_lookahead_tokens` onto **offline**
calls too, so this checkpoint's "batch" path is the same cache-aware,
limited-right-context encoder run in one shot. Batch WER tracks the mode
exactly:

| right context | latency | batch WER | streaming WER |
|---|---|---|---|
| 0 | 80 ms | 0.1204 | 0.1204 |
| 3 | 320 ms | 0.1314 | 0.1314 |
| 6 | 560 ms | 0.1058 | 0.1095 |
| 13 | 1120 ms | 0.1095 | 0.1095 |

There is no full-right-context offline mode to be a baseline. The draft's §2
justified shipping `batch` as the "same-model offline RTF/WER baseline"; the
RTF half is real and large (0.072x vs 0.216x for byte-identical output), the
WER half does not exist. §2 above now says so, and the batch profiles repeat
it in their own header comments so no leaderboard reader infers two
independent accuracy numbers where there is one.

For context on the WER itself: `parakeet-tdt-0.6b-v3` measures 6.2% batch /
6.57% streaming on this same pack. Nemotron 3.5 is roughly **twice the WER
on Dutch here** while being the only one of the two with a genuine
incremental path. That tradeoff is exactly what these profiles exist to put
on the board; it is not a wiring bug (the same code produces correct Dutch,
and both engines' numbers are stable across backends).

### 5. Chunk geometry: transformers' own helpers are off by one

The library's `processor.num_samples_first_audio_chunk` /
`num_samples_per_audio_chunk` document themselves as "the number of raw
audio samples to feed the processor so it returns exactly N frames." **They
return one frame too many at every one of the four modes**, and
`_validate_stream_chunk` then rejects the result outright:

| right | required frames (first/sub) | library's samples | frames it actually yields |
|---|---|---|---|
| 0 | 1 / 8 | 200 / 1680 | 2 / 9 |
| 3 | 25 / 32 | 4040 / 5520 | 26 / 33 |
| 6 | 49 / 56 | 7880 / 9360 | 50 / 57 |
| 13 | 105 / 112 | 16840 / 18320 | 106 / 113 |

Cause: the properties assume the extractor windows at `win_length` (400)
when it actually windows at `n_fft` (512). Worked around in
`adapters/nemotron.py::_mel_chunks` rather than pinned to a hypothetical
transformers patch: feed each chunk enough samples to cover its frames' full
STFT windows, then slice the mel to the exact required frame count.

The rest of §5's question resolves cleanly:

- **No resampling surprises.** The extractor is 16 kHz native, exactly what
  GOESB's `decode_pcm` already produces. Nothing resamples.
- **The geometry is verified, not merely valid.** Concatenating every
  chunk's mel output reproduces the offline full-utterance pass at all four
  modes on real Dutch audio — max absolute difference 3.8e-3 across ~1240
  interior frames, mean 5.6e-7, i.e. float32 STFT noise (a wrong chunk
  offset differs by order 10, not 1e-3). The only materially differing frame
  is the clip's very last, a zero-padding tail artefact the offline path has
  too. This
  matters because a chunk layout that merely satisfies
  `_validate_stream_chunk` can still feed the encoder subtly wrong features
  at every seam.
- **Tail padding fabricates nothing.** Zero-padding the final short chunk
  produced no trailing words on any of the 15 clips — consistent with
  `center=True`'s own right-edge zero padding in the offline path, which the
  model was trained through.

### 6. Two things the ADR didn't anticipate

- **`language` is genuinely applied here**, unlike `parakeet` where it is
  documented as accepted-and-unused. `Nemotron3_5AsrProcessor.__call__`
  resolves `language` through a `prompt_dictionary` into a real `prompt_ids`
  model input that conditions the decode. This is a per-utterance prompt,
  not the per-language *model swap* §1 correctly says is unnecessary — both
  statements are true at once.
- **GOESB's `es-419` is not in that dictionary** and raises
  `ValueError: Unknown language='es-419'`. `en-US`, `nl-NL`, `de-DE`,
  `fr-FR` and `pt-BR` all are. `_resolve_language` falls back full tag →
  base subtag (`es-419` → `es`) → `auto`, so a regional subtag can never
  kill a run while still getting language conditioning.

### 7. §5's metric consequences, confirmed exactly

Both hold, and more strongly than "converge":

- `partial_stability` is **exactly 1.0**, not approximately. Every published
  partial is a strict word-wise extension of the previous one. This required
  one deliberate design choice: the adapter holds a trailing *incomplete*
  word back until the next word's first BPE token lands, so a half-emitted
  word never counts as "rewritten" when the rest of it arrives. Without
  that, the metric would have measured the tokenizer rather than the engine.
- `first_partial_latency` and `first_final_latency` are **byte-identical**,
  not merely close — the first non-empty partial *is* the first finalized
  word.

`docs/specs/metrics.md` now records both, plus the broader point that these
metrics only discriminate among the bounded-window re-decode engines.

### 8. Frozen binaries: no torch engine ships one

Checked before committing to shipping one, as the brief asked. `torch` alone
is 501 MB installed for the CPU-only macOS arm64 wheel and multiple GB for a
Linux CUDA wheel with its bundled CUDA runtime, against GitHub's 2 GB
per-release-asset limit — and a CPU-torch binary would be useless for an
engine that declares no CPU backend. So `.github/workflows/release-binaries.yml`
gets **no** `parakeet` or `nemotron` matrix entry, now with a comment saying
that is deliberate.

The latent gap the brief flagged is closed anyway:
`scripts/generate_frozen_adapter_hashes.py`'s `ADAPTER_MODULES` now lists
**all five** adapters (it previously listed three), so
`hashing._frozen_module_hash`'s `ValueError: no precomputed hash for ...`
can no longer be triggered by someone adding a matrix entry later.

One correction to the brief's framing of that gap: `runner/src/oesb_runner/
_frozen_adapter_hashes.json` is **gitignored** — a build artefact the release
workflow regenerates from source immediately before invoking PyInstaller, not
a checked-in manifest that can drift. So the "matching three entries" it
described was a local build leftover, and the entire fix lives in the
generator script. (Which is also why this change touching four other
adapters' source — they all gained the `streaming_latency_ms` call-shape
kwarg — needs no manifest commit: the next build recomputes them.)

## Addendum (2026-08-20, second): CUDA verification on real NVIDIA hardware

The first addendum closed with a hardware caveat: no CUDA device was
available, so the `cuda` backend and the VRAM-derived concurrency ceiling
were extrapolations. Both have now been measured on a rented **NVIDIA RTX
A6000 (47.7 GB, driver 555.58.02, compute capability 8.6)**, torch
2.13.0+cu126 / transformers 5.15.1 / Python 3.12, against the same
`packs/fleurs-nl` pack. Nine signed result documents were produced,
schema-validated and signature-verified.

**The extrapolation was wrong in the safe direction on memory, and the
concurrency ceiling turned out to be limited by something else entirely.**

### 1. `cuda` works, and the four modes hold on a newer transformers

Batch: **RTF 0.0349x, WER 0.1314** — the same WER as metal and as cpu, on a
third backend and a different transformers minor version. Streaming, all
four modes:

| mode | RTF (cuda) | RTF (metal) | WER | first_partial = first_final (p50) | partial_stability |
|---|---|---|---|---|---|
| 80 ms | 0.4354x | 0.539x | 0.1204 | 359 ms | 1.0000 |
| 320 ms | 0.1297x | 0.216x | 0.1314 | 680 ms | 1.0000 |
| 560 ms | 0.1040x | 0.160x | 0.1095 | 733 ms | 1.0000 |
| 1120 ms | 0.0661x | 0.123x | 0.1095 | 995 ms | 1.0000 |

`supported_streaming_latencies_ms` reports the same four modes on
transformers 5.15.1 as on 5.14.1, and the chunk-geometry test still passes
there — so both the "four not five" finding and the off-by-one workaround in
`_mel_chunks` are not artefacts of one library version.

Both ADR-0008 gates behaved on real hardware: `--backend cpu` exited 1 with
*"--backend 'cpu' is not supported by 'nemotron' ('batch') — this runtime
supports: cuda, metal"*, and `--param streaming_latency_ms=160` exited 1
with *"not in allowed values [80, 320, 560, 1120]"*. Neither was snapped or
silently accepted.

### 2. VRAM: the estimate was ~25% too high

| measure | estimated (from MPS) | measured (cuda) |
|---|---|---|
| fp32 weights | 2.55 GB | 2.56 GB |
| device memory, one warmed instance | ~3.35 GB | **~2.67 GB** |
| peak VRAM at concurrency 8 | ~27 GB | **20.3 GiB** |

The 3.35 GB figure came from MPS `driver_allocated_memory`, which counts
more than CUDA's actual device consumption. **Eight workers fit on a 24 GB
card, not the 32 GB+ the profile originally claimed.** Corrected in
`profiles/nemotron-3-5-concurrency/profile.yaml` and in
`run_concurrency`'s docstring.

### 3. The real concurrency limit is throughput, and it collapses after 2

Measured twice, independently, on the same box:

| concurrency | throughput run 1 | throughput run 2 | RTF per call (run 2) | peak VRAM |
|---|---|---|---|---|
| 1 | 41.5 audio-s/s | 36.3 | 0.0285x | 3.2 GiB |
| 2 | **50.6** | **50.8** | 0.0406x | 5.6 GiB |
| 3 | — | 20.9 | 0.1485x | — |
| 4 | 21.4 | 19.9 | 0.2105x | 10.5 GiB |
| 8 | 18.9 | 17.7 | 0.4954x | 20.3 GiB |

Throughput peaks at **2** and falls off a cliff between 2 and 3, reproducibly.

**It is not CPU-thread contention**, which was the obvious suspect given that
mel extraction is `torch.stft` on CPU inside every timed call: re-running
concurrency 2 / 3 / 8 at `threads=1` and `threads=4` moved throughput by less
than the ~11-15% run-to-run noise (50.1 vs 50.8, 21.7 vs 20.4, 17.8 vs 21.0),
and `cpu_pct` sat at ~150-190% either way. The cause is GPU-side: N
independent model instances contend for the same SMs, and nothing batches
across requests — the direct cost of the thread-safety finding in the first
addendum, which forces N instances rather than one shared model.

This is a real result for Babbl's actual question, not just a benchmark
curiosity: **on one A6000 this engine serves about two concurrent streams
well and gains nothing past that.** The profile keeps its ceiling of 8 so a
`--param concurrency=1,2,4,8` sweep shows both the peak and the collapse;
that ceiling is now documented as "where the curve is flat", not "where the
memory runs out".

Two side notes worth recording rather than dropping:

- `threads` genuinely is applied (`torch.set_num_threads`, and the
  `overridable` declaration depends on that being true), but its measured
  effect on this workload and this hardware is inside the noise. Applied ≠
  impactful; the declaration stays honest either way.
- Every concurrency run tripped the FR-5.3 reproducibility warning
  (`real_time_factor` relative std 9.7-14.8% vs the 5% tolerance). That is
  the runner surfacing genuine per-call variance under concurrent GPU load,
  working as designed — not a defect, but a reason not to read a single
  concurrency number too precisely.

### 4. Two defects the CUDA box found in this change itself

- **`tests/test_adapter_nemotron.py` was not portable.** Its fake-based
  tests hardcoded `backend="metal"`, so `_load`'s real availability probe
  failed on any machine without Apple Silicon: 8 of 28 tests failed on first
  contact with CUDA hardware while passing locally. CI never caught it —
  CI installs no torch, so the whole module skips. Fixed by forcing the probe
  for a `FAKE_BACKEND` constant, and by routing the slow real-audio tests
  through whichever GPU the machine actually has. Same file now passes 28/28
  on both metal and cuda.
- **`goesb doctor` gave nemotron no readiness line.** It printed only
  "installed, supports ['cuda', 'metal']" while giving parakeet a full
  per-backend readiness breakdown on the same box — worst possible place for
  that gap, since nemotron is the one engine that cannot fall back to cpu.
  `_doctor_engine_line`'s parakeet branch now covers both engines, and for a
  cpu-less engine it says so explicitly rather than claiming "cpu ready".

Total cost of this verification: **$0.98** of vast.ai credit, ~2.2 hours on
one A6000. The instance was destroyed afterwards.
