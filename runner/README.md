# GOESB Runner

The official, reproducible benchmark runner. It loads a signed **Benchmark
Profile** and **Benchmark Pack**, captures the full hardware/software/model
environment, runs the benchmark (batch / streaming / conversation) against a
pluggable **runtime adapter**, computes metrics, and emits a signed,
hash-verifiable result.

The runner **never executes arbitrary user code** (see the security model).

## Quick start
```bash
pip install -e ".[dev,faster-whisper]"
goesb --help
goesb env          # print captured environment fingerprint
goesb validate profiles/whisper-medium-en-batch/profile.yaml
goesb validate packs/example-librispeech-en-batch/pack.yaml

# Fetch the example pack's audio (not committed to git — FR-3.5) before
# running against it. Reruns are safe with --skip-download.
python ../scripts/fetch_librispeech_subset.py

# Run a batch benchmark end-to-end: environment capture, transcription,
# normalization + WER/CER/RTF/CPU/RAM, a signed+hashed result on disk.
goesb run whisper-medium-en-batch example-librispeech-en-batch --repeats 2
```

Pass `--model-override tiny` (or `base`) to `goesb run` for a fast local smoke
test instead of downloading the full `whisper-medium` the profile specifies —
results from an override are for pipeline verification only, never comparable
to the official profile's numbers.

Schema validation currently requires running from within this monorepo
checkout (see `oesb_runner.schema_validation`); standalone packaging of the
schemas is tracked for M2.
