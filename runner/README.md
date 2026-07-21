# OESB Runner

The official, reproducible benchmark runner. It loads a signed **Benchmark
Profile** and **Benchmark Pack**, captures the full hardware/software/model
environment, runs the benchmark (batch / streaming / conversation) against a
pluggable **runtime adapter**, computes metrics, and emits a signed,
hash-verifiable result.

The runner **never executes arbitrary user code** (see the security model).

## Quick start (placeholder)
```bash
pip install -e ".[dev]"
oesb --help
oesb env          # print captured environment
oesb validate profiles/whisper-medium-nl-batch/profile.yaml
```
