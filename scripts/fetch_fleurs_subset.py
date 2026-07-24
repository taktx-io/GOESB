#!/usr/bin/env python3
"""Fetch a small FLEURS subset for a new-language pack.

FLEURS (google/fleurs on Hugging Face) is genuinely ungated: no account, no
consent screen, no API token, 102 languages, plain HTTPS downloads. That's
why it's the preferred source for a *new* language pack over Common Voice
specifically — Common Voice's own consent/download-session flow can't be
scripted like this one can (see docs/create/packs "Sourcing audio" in the
oesb-platform repo for the full comparison). Generic across language, unlike
fetch_librispeech_subset.py, which is intentionally one script per corpus.

What it does:
1. Downloads <split>.tsv (the transcript index) for the requested language
   and takes the first --count rows.
2. Streams <split>.tar.gz and extracts only those clips, stopping as soon as
   every requested file has been found.
3. Writes manifest.jsonl and fills in an EXISTING pack.yaml's audio fields +
   top-level sha256 — create pack.yaml first (id/profile_id/license/
   visibility/metadata), same division of labour as the LibriSpeech script.

Usage:
    python scripts/fetch_fleurs_subset.py --language nl_nl \
      --pack-dir packs/example-fleurs-nl-batch
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FLEURS_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"

sys.path.insert(0, str(ROOT / "runner" / "src"))
from oesb_runner.audio import wav_duration_s  # noqa: E402
from oesb_runner.hashing import canonical_asset_sha256, sha256_file  # noqa: E402


def fetch_tsv_rows(language: str, split: str, count: int) -> list[dict]:
    url = f"{FLEURS_BASE_URL}/{language}/{split}.tsv"
    print(f"Fetching {url} ...", file=sys.stderr)
    with urllib.request.urlopen(url) as resp:  # nosec B310 - fixed HF Hub URL
        text = resp.read().decode("utf-8")

    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        # id, filename, raw_transcription, normalized_transcription,
        # phonemes, num_samples, gender - FLEURS' own fixed tsv shape.
        _id, filename, raw_text, _norm_text, _phonemes, _num_samples, _gender = line.split("\t")
        rows.append({"filename": filename, "reference_text": raw_text})
        if len(rows) >= count:
            break
    if not rows:
        raise SystemExit(f"{language}/{split}.tsv had no rows")
    return rows


def fetch_audio(language: str, split: str, wanted_filenames: set[str], audio_dir: Path) -> None:
    url = f"{FLEURS_BASE_URL}/{language}/audio/{split}.tar.gz"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {url} looking for {len(wanted_filenames)} clips ...", file=sys.stderr)
    collected: set[str] = set()
    with urllib.request.urlopen(url) as resp:  # nosec B310 - fixed HF Hub URL
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if name not in wanted_filenames or not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                (audio_dir / name).write_bytes(extracted.read())
                collected.add(name)
                if collected == wanted_filenames:
                    break  # every requested clip found - stop reading the archive

    missing = wanted_filenames - collected
    if missing:
        raise SystemExit(f"never found {len(missing)} requested clip(s) in the archive: {sorted(missing)}")
    print(f"Fetched {len(collected)} files into {audio_dir}", file=sys.stderr)


def build_manifest(rows: list[dict], audio_dir: Path) -> list[dict]:
    entries = []
    for row in rows:
        path = audio_dir / row["filename"]
        entries.append({
            "utterance_id": Path(row["filename"]).stem,
            "relative_path": row["filename"],
            "reference_text": row["reference_text"],
            "duration_s": round(wav_duration_s(path), 3),
        })
    return entries


def write_manifest(pack_dir: Path, entries: list[dict]) -> Path:
    manifest_path = pack_dir / "manifest.jsonl"
    lines = [json.dumps(e, sort_keys=True) for e in entries]
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def update_pack_yaml(pack_dir: Path, manifest_path: Path, entries: list[dict]) -> None:
    pack_path = pack_dir / "pack.yaml"
    if not pack_path.exists():
        raise SystemExit(
            f"{pack_path} doesn't exist yet - create it first (the in-browser pack "
            "builder, or by hand) with id/profile_id/license/visibility/metadata "
            "filled in. This script only fills in the audio fields."
        )
    pack = yaml.safe_load(pack_path.read_text())

    manifest_sha256 = sha256_file(manifest_path)
    total_duration_s = round(sum(e["duration_s"] for e in entries), 3)

    pack.setdefault("audio", {})
    pack["audio"]["count"] = len(entries)
    pack["audio"]["total_duration_s"] = total_duration_s
    pack["audio"]["sample_rate_hz"] = 16000  # FLEURS is published at 16kHz throughout
    pack["audio"]["manifest_sha256"] = manifest_sha256
    pack["sha256"] = canonical_asset_sha256(pack)

    pack_path.write_text(yaml.safe_dump(pack, sort_keys=False, allow_unicode=True))
    print(f"Updated {pack_path}: count={len(entries)} "
          f"total_duration_s={total_duration_s} manifest_sha256={manifest_sha256}",
          file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--language", required=True,
                         help="FLEURS language/locale code, e.g. nl_nl, de_de, ja_jp.")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"],
                         help="dev is the smallest split - the right default for an example pack.")
    parser.add_argument("--count", type=int, default=15,
                         help="How many clips to fetch, taken in tsv order.")
    parser.add_argument("--pack-dir", type=Path, required=True,
                         help="Existing pack directory, e.g. packs/example-fleurs-nl-batch.")
    parser.add_argument("--audio-dir", type=Path, default=None,
                         help="Defaults to <pack-dir>/audio.")
    parser.add_argument("--skip-download", action="store_true",
                         help="Rebuild the manifest from an already-fetched --audio-dir only.")
    args = parser.parse_args()

    audio_dir = args.audio_dir or (args.pack_dir / "audio")
    rows = fetch_tsv_rows(args.language, args.split, args.count)

    if not args.skip_download:
        fetch_audio(args.language, args.split, {r["filename"] for r in rows}, audio_dir)

    entries = build_manifest(rows, audio_dir)
    manifest_path = write_manifest(args.pack_dir, entries)
    update_pack_yaml(args.pack_dir, manifest_path, entries)
    print(f"Wrote {manifest_path} ({len(entries)} utterances).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
