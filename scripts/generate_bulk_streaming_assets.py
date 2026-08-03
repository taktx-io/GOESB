#!/usr/bin/env python3
"""Bulk-generate streaming profiles (+ one shared pack per new language) to
match batch's existing language/engine/size coverage.

Why: GOESB is hardware-generic (docs/00-vision.md) -- a profile scoring
badly on one CPU isn't a universal property, so faster-whisper streaming
stays a full wizard citizen regardless of speed on any one machine (see
cli.py's _build_matrix docstring). The same principle applies to
language: streaming scores WER, same as batch and unlike concurrency, so
accuracy is genuinely language-dependent (vosk literally has different
per-language models) and streaming should cover the same languages batch
does, not stay English-only.

Scope: same 6 languages (en, de, es, fr, nl, pt) x 12 combos/language
(faster-whisper + whisper-cpp at 5 sizes each, vosk at small+medium)
batch already has. Reuses generate_bulk_assets.py's own
LANGUAGES/SIZES/ID_PREFIX/MIN_VERSIONS/TITLE_ENGINE/model_name_for
directly rather than duplicating that config.

Packs: ONE streaming pack per NEW language (fleurs-<lang>-streaming), not
one per engine/size -- confirmed against the real batch packs (fleurs-de
etc.) that all 12 batch combos for a non-English language already share
a single pack, so the earlier per-engine convention
(librispeech-en-streaming / -vosk-streaming / -whispercpp-streaming) was
never actually load-bearing, just how English happened to be built
first (by hand, one engine at a time, across separate sessions). Those
3 existing English packs are reused as-is for its 8 missing combos -- no
new English pack.

Vosk's medium-tier models aren't in generate_bulk_assets.py's LANGUAGES
dict (that script predates the vosk-medium tier, added later) -- listed
here directly instead, confirmed against each language's own real
vosk-medium-<lang>-batch profile.yaml.

Idempotent: skips any profile/pack that already exists.

Usage:
    python scripts/generate_bulk_streaming_assets.py
    python scripts/generate_bulk_streaming_assets.py --languages de es
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from generate_bulk_assets import (
    ID_PREFIX,
    LANGUAGES,
    MIN_VERSIONS,
    SIZES,
    TITLE_ENGINE,
    model_name_for,
)

from oesb_runner.hashing import canonical_asset_sha256

# Confirmed against each language's own real vosk-medium-<lang>-batch
# profile.yaml -- not derivable from generate_bulk_assets.py's LANGUAGES
# dict, which only ever recorded the small-tier model name.
VOSK_MEDIUM_MODEL = {
    "en": "vosk-model-en-us-0.22",
    "de": "vosk-model-de-0.21",
    "es": "vosk-model-es-0.42",
    "fr": "vosk-model-fr-0.22",
    "nl": "vosk-model-nl-spraakherkenning-0.6",
    "pt": "vosk-model-pt-fb-v0.1.1-20220516_2113",
}

# English already has 3 hand-authored streaming packs, one per engine
# (built incrementally, earlier sessions) -- reused as-is rather than
# consolidated (write_streaming_pack is simply never called for "en"),
# so already-live profiles referencing them are untouched.

STREAMING_METRICS = [
    "wer", "real_time_factor", "cpu_pct", "ram_mb",
    "first_partial_latency", "first_final_latency", "end_of_speech_latency",
    "update_frequency", "partial_stability", "streaming_responsiveness",
]


def profile_id_for(engine: str, size: str, lang_code: str) -> str:
    return f"{ID_PREFIX[engine]}-{size}-{lang_code}-streaming"


def streaming_pack_id_for(lang: dict) -> str:
    return f"{lang['primary_pack_id']}-streaming"


def overridable_block_for(engine: str) -> dict[str, Any]:
    """Only chunk_ms is genuinely new here vs each engine's batch
    overridable block -- see each adapter's own applied_parameters for
    streaming (adapters/*.py) for what's actually wired, matching the
    hand-authored English streaming profiles this mirrors."""
    chunk_ms = {"allowed": [250, 500, 1000, 2000]}
    if engine == "faster-whisper":
        return {
            "beam_size": {"allowed": [1, 2, 4, 5, 8]},
            "vad": {},
            "quantization": {"allowed": ["int8", "float32"]},
            "threads": {"range": {"min": 1, "max": 16}},
            "chunk_ms": chunk_ms,
        }
    if engine == "whisper-cpp":
        return {"threads": {"range": {"min": 1, "max": 16}}, "chunk_ms": chunk_ms}
    return {"chunk_ms": chunk_ms}  # vosk -- Kaldi has no other real tunable, see run_batch's own docstring


def write_profile(profile_id: str, engine: str, size: str, model_name: str, lang: dict) -> None:
    path = ROOT / "profiles" / profile_id / "profile.yaml"
    if path.exists():
        print(f"skip (exists): {path}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    title_size = size.replace("-", " ").title()
    title_lang = lang["bcp47"].split("-")[0].upper()

    model_block: dict[str, Any] = {"name": model_name}
    if engine in ("faster-whisper", "whisper-cpp"):
        model_block["beam_size"] = 5
        model_block["temperature"] = 0.0
    if engine == "faster-whisper":
        model_block["quantization"] = "int8"
        model_block["vad"] = True

    doc = {
        "id": profile_id,
        "version": "1.0.0",
        "title": f"{TITLE_ENGINE[engine]} {title_size} {title_lang} (Streaming)",
        "benchmark_type": "streaming",
        "language": lang["bcp47"],
        "runtime": {"name": engine, "min_version": MIN_VERSIONS[engine]},
        "model": model_block,
        "configuration": {"threads": 4, "chunk_ms": 1000},
        "normalization": {
            "lowercase": True,
            "remove_punctuation": True,
            "expand_numbers": True,
            "ruleset_id": f"goesb-{lang['code']}-v1",
        },
        "scoring": {
            "primary_metric": "first_partial_latency",
            "tie_breakers": ["streaming_responsiveness", "real_time_factor"],
        },
        "metrics": STREAMING_METRICS,
        "overridable": overridable_block_for(engine),
        "changelog": [{
            "version": "1.0.0",
            "notes": "Initial profile (bulk-generated streaming set, matching batch's language/engine/size coverage).",
        }],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"wrote {path}", file=sys.stderr)


def write_streaming_pack(lang: dict) -> None:
    """One shared streaming pack per (new) language, cloning the existing
    batch primary pack's audio -- same audio, different id/profile_id/
    tags, matching the librispeech-en-vosk-streaming precedent."""
    primary_pack_dir = ROOT / "packs" / lang["primary_pack_id"]
    pack_id = streaming_pack_id_for(lang)
    pack_dir = ROOT / "packs" / pack_id
    if (pack_dir / "pack.yaml").exists():
        print(f"skip (exists): {pack_dir}", file=sys.stderr)
        return

    primary_pack = yaml.safe_load((primary_pack_dir / "pack.yaml").read_text())
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "manifest.jsonl").write_text((primary_pack_dir / "manifest.jsonl").read_text())
    shutil.copytree(primary_pack_dir / "audio", pack_dir / "audio", dirs_exist_ok=True)

    fetch_cmd = primary_pack["audio"]["source"]["fetch_instructions"]
    source = dict(primary_pack["audio"]["source"])
    source["fetch_instructions"] = (
        f"Identical audio to {lang['primary_pack_id']} — auto-fetched the same way. "
        f"Manual fallback: run `{fetch_cmd}`, then pass "
        f"`--audio-dir packs/{lang['primary_pack_id']}/audio` to `goesb run`, "
        "or symlink/copy that pack's audio/ directory here."
    )

    metadata = dict(primary_pack.get("metadata", {}))
    tags = [*metadata.pop("tags", []), "streaming"]

    doc = {
        "id": pack_id,
        "version": "1.0.0",
        "sha256": "0" * 64,  # placeholder, overwritten below in place to keep its conventional early position
        "profile_id": profile_id_for("faster-whisper", "medium", lang["code"]),
        "visibility": "open",
        "license": primary_pack.get("license", "CC-BY-4.0"),
        "audio": {
            "count": primary_pack["audio"]["count"],
            "total_duration_s": primary_pack["audio"]["total_duration_s"],
            "sample_rate_hz": primary_pack["audio"]["sample_rate_hz"],
            "manifest_sha256": primary_pack["audio"]["manifest_sha256"],
            "source": source,
        },
        "metadata": {**metadata, "tags": tags},
    }
    doc["sha256"] = canonical_asset_sha256(doc)
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"wrote {pack_dir / 'pack.yaml'}", file=sys.stderr)


def generate_language(lang: dict) -> None:
    is_en = lang["code"] == "en"
    if not is_en:
        write_streaming_pack(lang)

    for engine in ("faster-whisper", "whisper-cpp"):
        for size in SIZES:
            model_name = model_name_for(engine, size, lang["code"], lang["vosk_model"])
            profile_id = profile_id_for(engine, size, lang["code"])
            write_profile(profile_id, engine, size, model_name, lang)

    for size, model_name in (("small", lang["vosk_model"]), ("medium", VOSK_MEDIUM_MODEL[lang["code"]])):
        profile_id = profile_id_for("vosk", size, lang["code"])
        write_profile(profile_id, "vosk", size, model_name, lang)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=None,
                         help="Language codes to generate (default: all in LANGUAGES).")
    args = parser.parse_args()

    wanted = set(args.languages) if args.languages else None
    for lang in LANGUAGES:
        if wanted is not None and lang["code"] not in wanted:
            continue
        print(f"=== {lang['code']} ===", file=sys.stderr)
        generate_language(lang)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
