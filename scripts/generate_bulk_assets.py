#!/usr/bin/env python3
"""Bulk-generate official profiles + packs for the top European languages.

Scope (see docs/handoffs or the session that introduced this): English
(existing) plus Spanish, French, Portuguese, German — 5 languages, 2
Whisper-family engines (faster-whisper, whisper-cpp) at 5 sizes each
(tiny/base/small/medium/large-v3), plus vosk (1 model, no sizes). 11 combos
per language.

For each new language, exactly one real network fetch happens (the
"primary" pack's FLEURS audio, via the existing, unmodified
scripts/fetch_fleurs_subset.py) — every other combo for that language is a
"sibling" pack that reuses the same audio via the same
audio.source.type/params auto-fetch mechanism `goesb run` already
implements (runner/src/oesb_runner/audio_sources.py), exactly matching the
existing librispeech-en-vosk-batch/librispeech-en-whispercpp-batch pattern.
English itself needs no new audio fetch at all — its siblings reuse the
already-committed librispeech-en-batch manifest.

Idempotent: skips any file/pack that already exists, so a partial run (e.g.
a dropped connection mid-language) can be safely re-run.

Usage:
    python scripts/generate_bulk_assets.py                  # all languages
    python scripts/generate_bulk_assets.py --languages es    # just Spanish
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner" / "src"))
from oesb_runner.hashing import canonical_asset_sha256

SIZES = ["tiny", "base", "small", "medium", "large-v3"]
ENGINES = ["faster-whisper", "whisper-cpp", "vosk"]

MIN_VERSIONS = {"faster-whisper": "1.0.0", "whisper-cpp": "1.5.0", "vosk": "0.3.44"}
ID_PREFIX = {"faster-whisper": "whisper", "whisper-cpp": "whispercpp", "vosk": "vosk"}
TITLE_ENGINE = {"faster-whisper": "Whisper", "whisper-cpp": "Whisper.cpp", "vosk": "Vosk"}

# New languages: fetched fresh via FLEURS, no existing combos.
# English: no new audio fetch (reuses librispeech-en-batch), and 3 of its
# 11 combos already exist as hand-authored profiles/packs from before this
# bulk effort — those are left untouched, not regenerated.
LANGUAGES: list[dict[str, Any]] = [
    {
        "code": "es", "bcp47": "es-419", "fleurs": "es_419",
        "vosk_model": "vosk-model-small-es-0.42",
        "primary_pack_id": "fleurs-es-batch", "source_type": "fleurs",
        "existing_combos": set(),
    },
    {
        "code": "fr", "bcp47": "fr-FR", "fleurs": "fr_fr",
        "vosk_model": "vosk-model-small-fr-0.22",
        "primary_pack_id": "fleurs-fr-batch", "source_type": "fleurs",
        "existing_combos": set(),
    },
    {
        "code": "pt", "bcp47": "pt-BR", "fleurs": "pt_br",
        "vosk_model": "vosk-model-small-pt-0.3",
        "primary_pack_id": "fleurs-pt-batch", "source_type": "fleurs",
        "existing_combos": set(),
    },
    {
        "code": "de", "bcp47": "de-DE", "fleurs": "de_de",
        "vosk_model": "vosk-model-small-de-0.15",
        "primary_pack_id": "fleurs-de-batch", "source_type": "fleurs",
        "existing_combos": set(),
    },
    {
        "code": "en", "bcp47": "en-US", "fleurs": None,
        "vosk_model": "vosk-model-small-en-us-0.15",
        "primary_pack_id": "librispeech-en-batch", "source_type": "librispeech",
        "existing_combos": {("faster-whisper", "medium"), ("whisper-cpp", "base"), ("vosk", None)},
    },
    {
        "code": "nl", "bcp47": "nl-NL", "fleurs": "nl_nl",
        "vosk_model": "vosk-model-small-nl-0.22",
        "primary_pack_id": "fleurs-nl-batch", "source_type": "fleurs",
        "existing_combos": {("faster-whisper", "medium")},
    },
]


def profile_id_for(engine: str, size: str | None, lang_code: str) -> str:
    size_slug = "small" if engine == "vosk" else size
    return f"{ID_PREFIX[engine]}-{size_slug}-{lang_code}-batch"


def pack_id_for(lang: dict, engine: str, size: str | None, is_primary: bool) -> str:
    if is_primary:
        return lang["primary_pack_id"]
    size_slug = "small" if engine == "vosk" else size
    base = "librispeech-en" if lang["code"] == "en" else f"fleurs-{lang['code']}"
    return f"{base}-{ID_PREFIX[engine]}-{size_slug}-batch"


def model_name_for(engine: str, size: str | None, lang_code: str, vosk_model: str) -> str:
    if engine == "vosk":
        return vosk_model
    if engine == "whisper-cpp" and lang_code == "en" and size != "large-v3":
        return f"whisper-{size}.en"
    return f"whisper-{size}"


def write_profile(profile_id: str, engine: str, size: str | None, lang: dict) -> None:
    path = ROOT / "profiles" / profile_id / "profile.yaml"
    if path.exists():
        print(f"skip (exists): {path}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    model_name = model_name_for(engine, size, lang["code"], lang["vosk_model"])
    title_size = "Small" if engine == "vosk" else size.replace("-", " ").title()
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
        "title": f"{TITLE_ENGINE[engine]} {title_size} {title_lang} (Batch)",
        "benchmark_type": "batch",
        "language": lang["bcp47"],
        "runtime": {"name": engine, "min_version": MIN_VERSIONS[engine]},
        "model": model_block,
        "configuration": {"threads": 4},
        "normalization": {
            "lowercase": True,
            "remove_punctuation": True,
            "expand_numbers": True,
            "ruleset_id": f"goesb-{lang['code']}-v1",
        },
        "scoring": {"primary_metric": "wer", "tie_breakers": ["real_time_factor", "energy_wh"]},
        "metrics": ["wer", "cer", "real_time_factor", "cpu_pct", "ram_mb", "energy_wh", "temperature_c"],
        "changelog": [{"version": "1.0.0", "notes": "Initial profile (bulk-generated official set)."}],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"wrote {path}", file=sys.stderr)


def fetch_primary_pack(lang: dict, primary_profile_id: str) -> Path:
    pack_dir = ROOT / "packs" / lang["primary_pack_id"]
    if lang["code"] == "en":
        return pack_dir  # librispeech-en-batch already exists, nothing to fetch

    pack_path = pack_dir / "pack.yaml"
    if pack_path.exists() and (pack_dir / "manifest.jsonl").exists() and (pack_dir / "audio").exists():
        print(f"skip (already fetched): {pack_dir}", file=sys.stderr)
        return pack_dir

    pack_dir.mkdir(parents=True, exist_ok=True)
    if not pack_path.exists():
        skeleton = {
            "id": lang["primary_pack_id"],
            "version": "1.0.0",
            "sha256": "0" * 64,  # placeholder - fetch_fleurs_subset.py overwrites this key
            # in place (Python dict assignment to an existing key preserves
            # position), keeping sha256 in its conventional early position
            # instead of appended at the end.
            "profile_id": primary_profile_id,
            "visibility": "open",
            "license": "CC-BY-4.0",
            "metadata": {
                "language": lang["bcp47"],
                "recording_environment": "studio",
                "speech_style": "read",
                "transcription_style": "verbatim",
                "tags": ["fleurs", lang["code"]],
            },
        }
        pack_path.write_text(yaml.safe_dump(skeleton, sort_keys=False, allow_unicode=True))

    print(f"Fetching real FLEURS audio for {lang['code']} ...", file=sys.stderr)
    # Relative --pack-dir (not the absolute `pack_dir` Path) so the
    # fetch_instructions this writes into pack.yaml stay portable
    # (matching e.g. fleurs-nl-batch's "packs/fleurs-nl-batch", not this
    # machine's absolute filesystem path).
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fetch_fleurs_subset.py"),
         "--language", lang["fleurs"], "--pack-dir", f"packs/{lang['primary_pack_id']}"],
        check=True, cwd=ROOT,
    )
    return pack_dir


def write_sibling_pack(pack_id: str, profile_id: str, primary_pack_dir: Path, lang: dict) -> None:
    pack_dir = ROOT / "packs" / pack_id
    if (pack_dir / "pack.yaml").exists():
        print(f"skip (exists): {pack_dir}", file=sys.stderr)
        return
    pack_dir.mkdir(parents=True, exist_ok=True)

    primary_pack = yaml.safe_load((primary_pack_dir / "pack.yaml").read_text())
    (pack_dir / "manifest.jsonl").write_text((primary_pack_dir / "manifest.jsonl").read_text())

    fetch_cmd = primary_pack["audio"]["source"]["fetch_instructions"]
    source = dict(primary_pack["audio"]["source"])
    source["fetch_instructions"] = (
        f"Identical audio to {primary_pack_dir.name} — auto-fetched the same way. "
        f"Manual fallback: run `{fetch_cmd}`, then pass "
        f"`--audio-dir packs/{primary_pack_dir.name}/audio` to `goesb run`."
    )

    doc = {
        "id": pack_id,
        "version": "1.0.0",
        "sha256": "0" * 64,  # placeholder, overwritten below in place to keep its position early
        "profile_id": profile_id,
        "visibility": "open",
        "license": "CC-BY-4.0",
        "audio": {
            "count": primary_pack["audio"]["count"],
            "total_duration_s": primary_pack["audio"]["total_duration_s"],
            "sample_rate_hz": primary_pack["audio"]["sample_rate_hz"],
            "manifest_sha256": primary_pack["audio"]["manifest_sha256"],
            "source": source,
        },
        "metadata": {
            "language": lang["bcp47"],
            "recording_environment": "studio",
            "speech_style": "read",
            "transcription_style": "verbatim",
            "tags": [lang["source_type"], lang["code"]],
        },
    }
    doc["sha256"] = canonical_asset_sha256(doc)
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"wrote {pack_dir / 'pack.yaml'}", file=sys.stderr)


def generate_language(lang: dict) -> None:
    combos: list[tuple[str, str | None]] = []
    for engine in ("faster-whisper", "whisper-cpp"):
        for size in SIZES:
            combos.append((engine, size))
    combos.append(("vosk", None))

    # The primary pack targets the profile that already has real hand-picked
    # settings elsewhere in the codebase (medium, matching the existing
    # English/Dutch convention) - pick faster-whisper/medium as primary.
    primary_profile_id = profile_id_for("faster-whisper", "medium", lang["code"])
    primary_pack_dir = fetch_primary_pack(lang, primary_profile_id)

    for engine, size in combos:
        if (engine, size) in lang["existing_combos"]:
            continue
        profile_id = profile_id_for(engine, size, lang["code"])
        write_profile(profile_id, engine, size, lang)

        is_primary = engine == "faster-whisper" and size == "medium" and lang["code"] != "en"
        pack_id = pack_id_for(lang, engine, size, is_primary)
        if is_primary:
            continue  # already written by fetch_primary_pack
        write_sibling_pack(pack_id, profile_id, primary_pack_dir, lang)


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
