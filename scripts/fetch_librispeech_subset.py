#!/usr/bin/env python3
"""Fetch a small LibriSpeech subset for `example-librispeech-en-batch`.

LibriSpeech (OpenSLR-12) is a plain, ungated HTTPS download — unlike Common
Voice, no consent/session step is required, so this can run unattended.

What it does:
1. Streams `dev-clean.tar.gz` from OpenSLR and extracts only the one
   speaker/chapter directory requested (stops reading the stream as soon as
   that directory's entries are collected).
2. Parses the chapter's official `*.trans.txt` for reference transcripts and
   reads each FLAC's duration (see oesb_runner.audio.flac_duration_s).
3. Writes `packs/example-librispeech-en-batch/manifest.jsonl` — the
   deterministic, committed list of {utterance_id, relative_path,
   reference_text, duration_s}. No audio bytes are committed (FR-3.5); this
   manifest is text-only metadata.
4. Writes the actual FLAC files to `--audio-dir` (default: a gitignored
   `audio/` folder next to the pack), and updates the pack.yaml's
   audio.count / audio.total_duration_s / audio.manifest_sha256 / sha256 to
   match reality.

Usage:
    python scripts/fetch_librispeech_subset.py --speaker 1272 --chapter 128104
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
PACK_DIR = ROOT / "packs" / "example-librispeech-en-batch"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

sys.path.insert(0, str(ROOT / "runner" / "src"))
from oesb_runner.audio import flac_duration_s  # noqa: E402
from oesb_runner.hashing import canonical_asset_sha256, sha256_file  # noqa: E402


def _chapter_prefix(speaker: str, chapter: str) -> str:
    return f"LibriSpeech/dev-clean/{speaker}/{chapter}/"


def fetch_chapter(speaker: str, chapter: str, audio_dir: Path) -> None:
    prefix = _chapter_prefix(speaker, chapter)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {LIBRISPEECH_URL} looking for {prefix} ...", file=sys.stderr)
    collected: set[str] = set()
    seen_prefix = False
    with urllib.request.urlopen(LIBRISPEECH_URL) as resp:  # nosec B310 - fixed OpenSLR URL
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.name.startswith(prefix):
                    if seen_prefix:
                        # We've read past the target chapter's entries; stop
                        # pulling more bytes off the network.
                        break
                    continue
                seen_prefix = True
                if not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                dest = audio_dir / Path(member.name).name
                dest.write_bytes(extracted.read())
                collected.add(dest.name)
    if not collected:
        raise SystemExit(f"no files found under {prefix} — check --speaker/--chapter")
    print(f"Fetched {len(collected)} files into {audio_dir}", file=sys.stderr)


def build_manifest(speaker: str, chapter: str, audio_dir: Path) -> list[dict]:
    trans_path = audio_dir / f"{speaker}-{chapter}.trans.txt"
    if not trans_path.exists():
        raise SystemExit(f"missing transcript file: {trans_path}")

    references: dict[str, str] = {}
    for line in trans_path.read_text().splitlines():
        utt_id, _, text = line.partition(" ")
        references[utt_id] = text

    entries = []
    for flac_path in sorted(audio_dir.glob(f"{speaker}-{chapter}-*.flac")):
        utt_id = flac_path.stem
        if utt_id not in references:
            raise SystemExit(f"no reference transcript for {utt_id}")
        entries.append({
            "utterance_id": utt_id,
            "relative_path": flac_path.name,
            "reference_text": references[utt_id],
            "duration_s": round(flac_duration_s(flac_path), 3),
        })
    return entries


def write_manifest(entries: list[dict]) -> Path:
    manifest_path = PACK_DIR / "manifest.jsonl"
    lines = [json.dumps(e, sort_keys=True) for e in entries]
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def update_pack_yaml(manifest_path: Path, entries: list[dict], speaker: str, chapter: str) -> None:
    pack_path = PACK_DIR / "pack.yaml"
    pack = yaml.safe_load(pack_path.read_text())

    manifest_sha256 = sha256_file(manifest_path)
    total_duration_s = round(sum(e["duration_s"] for e in entries), 3)

    pack["audio"]["count"] = len(entries)
    pack["audio"]["total_duration_s"] = total_duration_s
    pack["audio"]["manifest_sha256"] = manifest_sha256
    pack["audio"]["source"] = {
        "type": "librispeech",
        "params": {"speaker": speaker, "chapter": chapter, "split": "dev-clean"},
        "fetch_instructions": (
            f"python scripts/fetch_librispeech_subset.py --speaker {speaker} --chapter {chapter}"
        ),
    }
    pack["sha256"] = canonical_asset_sha256(pack)

    pack_path.write_text(
        "# Example open pack manifest — English (LibriSpeech dev-clean subset).\n"
        "# Audio bytes are NOT committed (FR-3.5); fetch them locally with\n"
        "# scripts/fetch_librispeech_subset.py, verified against manifest.jsonl.\n"
        + yaml.safe_dump(pack, sort_keys=False, allow_unicode=True)
    )
    print(f"Updated {pack_path}: count={len(entries)} "
          f"total_duration_s={total_duration_s} manifest_sha256={manifest_sha256}",
          file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", default="1272")
    parser.add_argument("--chapter", default="128104")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=PACK_DIR / "audio",
        help="Local (gitignored) directory to store fetched FLAC files.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Rebuild the manifest from an already-fetched --audio-dir only.",
    )
    args = parser.parse_args()

    if not args.skip_download:
        fetch_chapter(args.speaker, args.chapter, args.audio_dir)

    entries = build_manifest(args.speaker, args.chapter, args.audio_dir)
    manifest_path = write_manifest(entries)
    update_pack_yaml(manifest_path, entries, args.speaker, args.chapter)
    print(f"Wrote {manifest_path} ({len(entries)} utterances).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
