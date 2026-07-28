#!/usr/bin/env python3
"""General-purpose authoring script for a Common Voice pack in any locale
(ADR-0010), not restricted to an age/demographic slice.

Sibling to build_common_voice_nl_elderly_pack.py, which stays as-is for
that one already-published, age-restricted pilot. This script is the
"no --age-buckets, just take the first N validated rows" variant the
pilot's own docstring anticipated — same access-gated, one-time-authoring
relationship to oesb_runner.audio_sources.fetch_common_voice_audio: this
script picks and freezes a subset; the runtime provider just replays that
choice for every downstream user of the published pack.

What it does:
1. Downloads the Common Voice dataset archive via `datacollective`
   (`download_dataset`), extracts it, and reads `validated.tsv` (Common
   Voice's own standard release layout: a TSV with, among others,
   `client_id`, `path`, `sentence`, `age`, `gender`, `locale` columns —
   one row per validated clip).
2. Filters to --locale, takes the first --count validated rows (no
   demographic filtering — that's what makes this "general" rather than
   a slice like the elderly pilot), and writes manifest.jsonl + fills in
   an EXISTING pack.yaml's audio fields + top-level sha256. Same division
   of labour as fetch_fleurs_subset.py: create pack.yaml first
   (id/profile_id/license/visibility/metadata), this script only fills
   in the audio fields.

NOTE: written against the documented, real `datacollective` 0.5.x API
(`download_dataset(dataset_id) -> Path`) but not run against a live
dataset — run with --dry-run first (stops after reporting the row count)
to sanity-check the archive layout matches the standard Common Voice
validated.tsv shape before trusting the manifest it produces.

Usage:
    export MDC_API_KEY=...
    python scripts/build_common_voice_pack.py \
      --dataset-id <id> --locale de \
      --pack-dir packs/common-voice-de-batch \
      --dry-run   # inspect the row count first

    python scripts/build_common_voice_pack.py \
      --dataset-id <id> --locale de \
      --pack-dir packs/common-voice-de-batch --count 40
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# Manifests built by this script always include audio_sha256 (per-clip
# content hash) — a runner older than this doesn't check it and would
# silently skip that guarantee, so packs authored from here on require it.
# (mozilla_data_collective already implies >=0.4.0; this supersedes it.)
MIN_RUNNER_VERSION = "0.5.0"

sys.path.insert(0, str(ROOT / "runner" / "src"))
from oesb_runner.hashing import canonical_asset_sha256, sha256_file


def download_and_extract(dataset_id: str) -> Path:
    try:
        from datacollective import download_dataset
    except ImportError as exc:
        raise SystemExit(
            "datacollective is not installed; run `pip install datacollective`"
        ) from exc

    print(f"Downloading Common Voice dataset {dataset_id!r} via Mozilla Data Collective ...", file=sys.stderr)
    archive_path = download_dataset(dataset_id, show_progress=True)

    extract_dir = archive_path.parent / archive_path.name.split(".")[0]
    if extract_dir.exists():
        print(f"Already extracted at {extract_dir}, reusing.", file=sys.stderr)
        return extract_dir
    extract_dir.mkdir(parents=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:
        with tarfile.open(archive_path, mode="r:*") as tar:
            tar.extractall(extract_dir, filter="data")
    return extract_dir


def find_validated_tsv(extract_dir: Path) -> Path:
    candidates = list(extract_dir.rglob("validated.tsv"))
    if not candidates:
        raise SystemExit(
            f"no validated.tsv found under {extract_dir} — the MDC-hosted archive "
            "layout may differ from the standard Common Voice release; inspect "
            f"{extract_dir} by hand and adjust this script."
        )
    return candidates[0]


def read_validated_rows(tsv_path: Path) -> list[dict]:
    # Common Voice's validated.tsv is plain tab-separated text with no
    # quote-escaping convention of its own — QUOTE_NONE tells the csv
    # module to treat `"` as a literal character rather than a field
    # delimiter. Without it, a sentence containing an unmatched `"` (real
    # example: Portuguese) makes the default QUOTE_MINIMAL dialect swallow
    # the rest of the file into one field until it finds a closing quote,
    # blowing past csv's 128KB field-size limit with a confusing error far
    # from the actual cause.
    with tsv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def clip_path_for(extract_dir: Path, row: dict, clip_suffix: str) -> Path:
    # Standard Common Voice layout: clips/<row["path"]>, already carrying
    # its own extension. clip_suffix lets you override if the MDC archive
    # ships a different container (e.g. re-encoded to .wav).
    clip_dir = next(extract_dir.rglob("clips"), extract_dir)
    name = row["path"]
    if clip_suffix and not name.endswith(clip_suffix):
        name = Path(name).stem + clip_suffix
    return clip_dir / name


def _clip_info(path: Path) -> tuple[float | None, int | None]:
    """(duration_s, sample_rate_hz). soundfile (already a transitive dep via
    the vosk/whisper-cpp extras) reads MP3 duration/rate directly via
    libsndfile >=1.1 — no separate mp3-specific library needed. Falls back
    to (None, None) rather than raising, so one unreadable clip doesn't
    kill the whole authoring run."""
    try:
        import soundfile as sf
    except ImportError:
        return None, None
    try:
        info = sf.info(str(path))
    except Exception:
        return None, None
    return round(info.frames / info.samplerate, 3), info.samplerate


def build_manifest(
    rows: list[dict], extract_dir: Path, audio_dir: Path, clip_suffix: str
) -> tuple[list[dict], int, set[int]]:
    """Returns (entries, speaker_count, sample_rates_seen). Common Voice's
    own client_id (an already-anonymized per-contributor hash) is used to
    *count* distinct speakers but deliberately not persisted into
    manifest.jsonl."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    speakers: set[str] = set()
    sample_rates: set[int] = set()
    for row in rows:
        src = clip_path_for(extract_dir, row, clip_suffix)
        if not src.exists():
            raise SystemExit(f"expected clip not found: {src} (row: {row.get('path')})")
        dest = audio_dir / src.name
        shutil.copyfile(src, dest)
        client_id = row.get("client_id", "").strip()
        if client_id:
            speakers.add(client_id)
        duration_s, sample_rate_hz = _clip_info(dest)
        if sample_rate_hz is not None:
            sample_rates.add(sample_rate_hz)
        entries.append({
            "utterance_id": Path(row["path"]).stem,
            "relative_path": dest.name,
            "reference_text": row["sentence"],
            "duration_s": duration_s,
            "audio_sha256": sha256_file(dest),
            "speaker_age_bucket": row.get("age", "").strip() or None,
            "speaker_gender": row.get("gender", "").strip() or None,
        })
    return entries, len(speakers), sample_rates


def write_manifest(pack_dir: Path, entries: list[dict]) -> Path:
    manifest_path = pack_dir / "manifest.jsonl"
    lines = [json.dumps(e, sort_keys=True) for e in entries]
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def update_pack_yaml(
    pack_dir: Path, manifest_path: Path, entries: list[dict], speaker_count: int,
    sample_rates: set[int], dataset_id: str,
) -> None:
    pack_path = pack_dir / "pack.yaml"
    if not pack_path.exists():
        raise SystemExit(
            f"{pack_path} doesn't exist yet - create it first with "
            "id/profile_id/license/visibility/metadata filled in. This script "
            "only fills in the audio fields."
        )
    pack = yaml.safe_load(pack_path.read_text())

    manifest_sha256 = sha256_file(manifest_path)
    known_durations = [e["duration_s"] for e in entries if e["duration_s"] is not None]
    total_duration_s = round(sum(known_durations), 3) if known_durations else None
    missing_durations = len(entries) - len(known_durations)
    if missing_durations:
        print(
            f"WARNING: {missing_durations}/{len(entries)} clips have no readable "
            "duration (soundfile missing or a clip it couldn't parse) — "
            "total_duration_s is an undercount. Fix before publishing.",
            file=sys.stderr,
        )

    pack.setdefault("audio", {})
    pack["audio"]["count"] = len(entries)
    if total_duration_s is not None:
        pack["audio"]["total_duration_s"] = total_duration_s
    if len(sample_rates) == 1:
        pack["audio"]["sample_rate_hz"] = next(iter(sample_rates))
    elif sample_rates:
        print(f"WARNING: mixed sample rates across clips: {sorted(sample_rates)} — leaving audio.sample_rate_hz unset.", file=sys.stderr)
    pack["audio"]["manifest_sha256"] = manifest_sha256
    pack["audio"]["source"] = {
        "type": "mozilla_data_collective",
        "params": {"dataset_id": dataset_id},
        "credential": {
            "env_var": "MDC_API_KEY",
            "signup_url": "https://mozilladatacollective.com",
            "instructions": (
                "Create a free Mozilla Data Collective account at "
                "https://mozilladatacollective.com, agree to the Common Voice "
                "dataset's terms on its dataset page, then generate an API key "
                "from the dashboard (https://mozilladatacollective.com/api-reference). "
                "Install the client with `pip install datacollective`, then set "
                "MDC_API_KEY to your key."
            ),
        },
    }
    pack.setdefault("metadata", {})
    pack["metadata"]["num_speakers"] = speaker_count
    pack["min_runner_version"] = MIN_RUNNER_VERSION
    pack["sha256"] = canonical_asset_sha256(pack)

    pack_path.write_text(yaml.safe_dump(pack, sort_keys=False, allow_unicode=True))
    print(
        f"Updated {pack_path}: count={len(entries)} "
        f"num_speakers={speaker_count} manifest_sha256={manifest_sha256}",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", required=True, help="MDC dataset id, from its dataset page URL.")
    parser.add_argument("--locale", required=True, help="Common Voice locale code to filter validated.tsv rows by, e.g. de, es, fr, pt.")
    parser.add_argument("--pack-dir", type=Path, required=True, help="Existing pack directory, e.g. packs/common-voice-de-batch.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Defaults to <pack-dir>/audio.")
    parser.add_argument("--count", type=int, default=40, help="Number of validated clips to take (first N in validated.tsv order).")
    parser.add_argument("--clip-suffix", type=str, default="", help="Override clip file extension if the archive re-encodes clips (e.g. .wav).")
    parser.add_argument("--dry-run", action="store_true", help="Stop after reporting the row count — no manifest/pack.yaml written.")
    args = parser.parse_args()

    extract_dir = download_and_extract(args.dataset_id)
    tsv_path = find_validated_tsv(extract_dir)
    rows = read_validated_rows(tsv_path)
    locale_rows = [r for r in rows if r.get("locale", args.locale) == args.locale]
    print(f"{len(locale_rows)} validated {args.locale!r} clips available.", file=sys.stderr)

    if args.dry_run:
        print("--dry-run: stopping before writing anything.", file=sys.stderr)
        return 0

    if len(locale_rows) < args.count:
        print(
            f"Not viable: only {len(locale_rows)} validated {args.locale!r} clips, "
            f"fewer than --count={args.count}.",
            file=sys.stderr,
        )
        return 1

    selected_rows = locale_rows[: args.count]
    print(f"Selected the first {len(selected_rows)} validated clips.", file=sys.stderr)

    audio_dir = args.audio_dir or (args.pack_dir / "audio")
    entries, speaker_count, sample_rates = build_manifest(selected_rows, extract_dir, audio_dir, args.clip_suffix)
    manifest_path = write_manifest(args.pack_dir, entries)
    update_pack_yaml(args.pack_dir, manifest_path, entries, speaker_count, sample_rates, args.dataset_id)
    print(f"Wrote {manifest_path} ({len(entries)} utterances).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
