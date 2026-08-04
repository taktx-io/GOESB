#!/usr/bin/env python3
"""Extend the Parakeet-TDT engine (single multilingual checkpoint) to the
same 5 additional languages batch/streaming already cover for the
Whisper-family engines: en, de, es, fr, pt (nl already exists, hand-authored
earlier).

Unlike scripts/generate_bulk_assets.py / generate_bulk_streaming_assets.py,
this needs no size-tier loop (parakeet-tdt-0.6b-v3 is one checkpoint, no
per-size variants) and no new pack fetching at all: librispeech-en and
fleurs-{de,es,fr,pt} already exist with real fetched audio from the earlier
Whisper-family bulk generation, and ADR-0011 makes any of them eligible for
these new profiles purely by matching `language` -- no per-engine pack
duplication needed or wanted.

Idempotent: skips any profile that already exists.

Usage:
    python scripts/generate_bulk_parakeet_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

LANGUAGES: list[dict[str, Any]] = [
    {"code": "en", "bcp47": "en-US"},
    {"code": "de", "bcp47": "de-DE"},
    {"code": "es", "bcp47": "es-419"},
    {"code": "fr", "bcp47": "fr-FR"},
    {"code": "pt", "bcp47": "pt-BR"},
]

_MODEL_NAME = "parakeet-tdt-0.6b-v3"
_MIN_VERSION = "5.9.0"


def _base_doc(lang: dict, benchmark_type: str) -> dict[str, Any]:
    lang_title = lang["bcp47"].split("-")[0].upper()
    doc: dict[str, Any] = {
        "id": f"parakeet-tdt-v3-{lang['code']}-{benchmark_type}",
        "version": "1.0.0",
        "title": f"Parakeet TDT 0.6b v3 {lang_title} ({benchmark_type.title()})",
        "benchmark_type": benchmark_type,
        "language": lang["bcp47"],
        "runtime": {"name": "parakeet", "min_version": _MIN_VERSION},
        "model": {"name": _MODEL_NAME},
    }
    return doc


def write_batch_profile(lang: dict) -> None:
    profile_id = f"parakeet-tdt-v3-{lang['code']}-batch"
    path = ROOT / "profiles" / profile_id / "profile.yaml"
    if path.exists():
        print(f"skip (exists): {path}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = _base_doc(lang, "batch")
    doc["configuration"] = {"threads": 4}
    doc["normalization"] = {
        "lowercase": True,
        "remove_punctuation": True,
        "expand_numbers": True,
        "ruleset_id": f"goesb-{lang['code']}-v1",
    }
    doc["scoring"] = {"primary_metric": "wer", "tie_breakers": ["real_time_factor", "energy_wh"]}
    doc["metrics"] = ["wer", "cer", "real_time_factor", "cpu_pct", "ram_mb", "energy_wh", "temperature_c"]
    doc["overridable"] = {"threads": {"range": {"min": 1, "max": 16}}}
    doc["changelog"] = [{"version": "1.0.0", "notes": "Initial profile."}]

    header = (
        f"# NVIDIA Parakeet-TDT-0.6b-v3 (via transformers, no nemo_toolkit) --\n"
        f"# {lang['bcp47'].split('-')[0].upper()} batch. One multilingual checkpoint\n"
        "# (Granary-trained, 25 European languages) rather than a per-language\n"
        "# model swap -- same shape as parakeet-tdt-v3-nl-batch. Validated\n"
        "# against schemas/benchmark-profile.schema.json.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"wrote {path}", file=sys.stderr)


def write_streaming_profile(lang: dict) -> None:
    profile_id = f"parakeet-tdt-v3-{lang['code']}-streaming"
    path = ROOT / "profiles" / profile_id / "profile.yaml"
    if path.exists():
        print(f"skip (exists): {path}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = _base_doc(lang, "streaming")
    doc["configuration"] = {"threads": 4, "chunk_ms": 1000}
    doc["normalization"] = {
        "lowercase": True,
        "remove_punctuation": True,
        "expand_numbers": True,
        "ruleset_id": f"goesb-{lang['code']}-v1",
    }
    doc["scoring"] = {
        "primary_metric": "first_partial_latency",
        "tie_breakers": ["streaming_responsiveness", "real_time_factor"],
    }
    doc["metrics"] = [
        "wer", "real_time_factor", "cpu_pct", "ram_mb",
        "first_partial_latency", "first_final_latency", "end_of_speech_latency",
        "update_frequency", "partial_stability", "streaming_responsiveness",
    ]
    doc["overridable"] = {
        "threads": {"range": {"min": 1, "max": 16}},
        "chunk_ms": {"allowed": [250, 500, 1000, 2000]},
    }
    doc["changelog"] = [{"version": "1.0.0", "notes": "Initial profile."}]

    header = (
        f"# NVIDIA Parakeet-TDT-0.6b-v3 (via transformers, no nemo_toolkit) --\n"
        f"# {lang['bcp47'].split('-')[0].upper()} streaming. Same bounded-window\n"
        "# re-decode approach as the Whisper-family streaming profiles (see\n"
        "# runner/src/oesb_runner/adapters/parakeet.py's run_streaming docstring\n"
        "# for why this engine has no genuine incremental path through\n"
        "# transformers). Validated against schemas/benchmark-profile.schema.json.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"wrote {path}", file=sys.stderr)


def main() -> int:
    for lang in LANGUAGES:
        write_batch_profile(lang)
        write_streaming_profile(lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
