# Addendum to the ADR-0009 runner handoff (2026-07-26, later the same day)

For the team implementing blocks 1–3 of
`docs/handoffs/2026-07-26-adr-0009-implementation.md`. The parameter catalog has now
been worked out against the actual adapter code (spec §6), and it **corrects block 1's
guidance**. Three changes:

## 1. Correction: whisper-cpp must NOT declare beam_size or vad

Block 1 said to add `beam_size`/`vad` overridable blocks to "the whisper engines".
That's right for **faster-whisper only**. The whisper-cpp adapter accepts `beam_size`,
`vad`, and `quantization` purely for call-shape parity and does not apply them (its own
docstring: beam search is not wired through pywhispercpp — decoding stays greedy; there
is no VAD; quantization is a ggml model-file choice). Declaring them would produce
signed results asserting values that had no effect. whisper-cpp profiles declare only
`threads` for now. Full per-engine catalog, including domains and the deliberate
exclusion of `temperature` (nondeterminism vs FR-5.3): spec §6.

## 2. New requirement: applied-parameters registry ("no silent knobs", spec §2)

Each adapter declares the set of parameters it genuinely applies (a reviewed registry
constant next to the adapter — e.g. alongside the existing `@register` metadata):

- faster-whisper (batch): quantization, beam_size, temperature, vad, threads
  (+ chunk_ms for streaming)
- whisper-cpp (batch): threads, temperature
- vosk (batch): —

Validation uses it twice: profile validation enforces `overridable ⊆ applied` for the
profile's runtime, and `--param` targeting an unapplied parameter is a hard error even
if a profile mis-declares it. Add tests for both.

## 3. Additional acceptance criterion

7. A profile that declares an overridable parameter its adapter does not apply fails
   validation with a clear error naming the parameter and the adapter; and
   `goesb run <whispercpp-profile> <pack> --param beam_size=8` fails before model load
   for the same reason.

Everything else in the original handoff stands unchanged.
