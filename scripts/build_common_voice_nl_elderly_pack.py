#!/usr/bin/env python3
"""One-time authoring script for the Common Voice NL "elderly" pack (ADR-0010 §5).

Unlike scripts/fetch_fleurs_subset.py and fetch_librispeech_subset.py, the
source here is access-gated: this script needs *your own* `MDC_API_KEY`
(https://mozilladatacollective.com) to run at all, and is a one-time
authoring step, separate from — and in addition to — the runtime fetch
provider (`oesb_runner.audio_sources.fetch_common_voice_audio`) that later
re-fetches these exact already-known filenames for every downstream user of
the published pack. This script picks and freezes the subset; the runtime
provider just replays that choice.

What it does:
1. Downloads the Common Voice NL dataset archive via `datacollective`
   (`download_dataset`), extracts it, and reads `validated.tsv` (Common
   Voice's own standard release layout: a TSV with, among others,
   `client_id`, `path`, `sentence`, `up_votes`, `down_votes`, `age`,
   `gender`, `locale` columns — one row per validated clip).
2. Prints the full age-bucket distribution for the validated split so the
   viability of the oldest bucket(s) can be judged before committing to a
   pack (per ADR-0010 §5: if the oldest bucket(s) are too small to be a
   meaningful eval set, report that rather than quietly padding with
   younger speakers).
3. Takes up to --count clips from the oldest non-empty bucket(s) (or
   --age-buckets, to force a specific set once you've looked at the
   distribution) and writes manifest.jsonl + fills in an EXISTING
   pack.yaml's audio fields + top-level sha256 — same division of labour as
   fetch_fleurs_subset.py: create pack.yaml first (id/profile_id/license/
   visibility/metadata), this script only fills the audio fields.

NOTE: this script is written against the documented, real `datacollective`
0.5.x API (`download_dataset(dataset_id) -> Path` returning a downloaded
archive; `PermissionError`/`ValueError`/`FileNotFoundError`/`RuntimeError`
on auth/rate-limit/not-found failures) but has not been run against the
live Common Voice NL dataset — the author of this script did not have an
MDC_API_KEY. Run it with `--dry-run` first (stops after the bucket
distribution report) to sanity-check the archive layout matches the
standard Common Voice `validated.tsv` shape before trusting the manifest it
produces; adjust `read_validated_tsv`/`clip_path_for` if the MDC-hosted
layout differs (e.g. a subdirectory prefix, or `.wav` vs `.mp3` clips —
`--clip-suffix` covers the latter without a code change).

Usage:
    export MDC_API_KEY=...
    python scripts/build_common_voice_nl_elderly_pack.py \
      --dataset-id cmn2g7nu901fmo107a1ydn0n5 \
      --pack-dir packs/common-voice-nl-elderly \
      --dry-run   # inspect the age-bucket distribution first

    python scripts/build_common_voice_nl_elderly_pack.py \
      --dataset-id cmn2g7nu901fmo107a1ydn0n5 \
      --pack-dir packs/common-voice-nl-elderly \
      --age-buckets eighties,nineties --count 40
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# Manifests built by this script always include audio_sha256 (per-clip
# content hash) — a runner older than this doesn't check it and would
# silently skip that guarantee, so packs authored from here on require it.
# (mozilla_data_collective already implies >=0.4.0; this supersedes it.)
MIN_RUNNER_VERSION = "0.5.0"
# Common Voice's own bucket order, oldest last — used to pick "the oldest
# available bucket(s)" without assuming which ones actually have data for
# any given language. Note "fourties", not "forties" — that's Common
# Voice's own (mis-)spelling in validated.tsv; get this wrong and that
# whole bucket silently falls out of AGE_BUCKET_ORDER matching.
AGE_BUCKET_ORDER = [
    "teens", "twenties", "thirties", "fourties", "fifties",
    "sixties", "seventies", "eighties", "nineties",
]

sys.path.insert(0, str(ROOT / "runner" / "src"))
from oesb_runner.hashing import canonical_asset_sha256, sha256_file  # noqa: E402


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
    # delimiter. Without it, a sentence containing an unmatched `"` makes
    # the default QUOTE_MINIMAL dialect swallow the rest of the file into
    # one field until it finds a closing quote, blowing past csv's 128KB
    # field-size limit with a confusing error far from the actual cause
    # (hit in practice building common-voice-pt).
    with tsv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def report_age_distribution(rows: list[dict]) -> Counter:
    counts = Counter(row.get("age", "").strip() or "(unlabeled)" for row in rows)
    print(f"Age-bucket distribution across {len(rows)} validated nl clips:", file=sys.stderr)
    for bucket in [*AGE_BUCKET_ORDER, "(unlabeled)"]:
        if counts.get(bucket):
            print(f"  {bucket:>12}: {counts[bucket]}", file=sys.stderr)
    unknown_buckets = set(counts) - set(AGE_BUCKET_ORDER) - {"(unlabeled)"}
    for bucket in sorted(unknown_buckets):
        print(f"  {bucket:>12}: {counts[bucket]}  (unrecognized bucket label)", file=sys.stderr)
    return counts


def oldest_viable_buckets(counts: Counter, min_clips: int) -> list[str]:
    """Walk AGE_BUCKET_ORDER from the oldest end, accumulating buckets
    until --count worth of clips is available. Returns [] if even every
    bucket combined can't reach min_clips — the caller should treat that as
    "not viable" and say so, per ADR-0010 §5, rather than silently
    widening into younger speakers on its own."""
    chosen: list[str] = []
    total = 0
    for bucket in reversed(AGE_BUCKET_ORDER):
        if counts.get(bucket):
            chosen.append(bucket)
            total += counts[bucket]
            if total >= min_clips:
                return chosen
    return chosen if total >= min_clips else []


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
    the vosk/whisper-cpp extras, and present in this authoring venv) reads
    MP3 duration/rate directly via libsndfile >=1.1 — no separate
    mp3-specific library needed. Falls back to (None, None) rather than
    raising, so one unreadable clip doesn't kill the whole authoring run —
    a null in the output is visible and worth checking by hand, not
    silently dropped."""
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
    own client_id (an already-anonymized per-contributor hash — using it
    to *count* distinct speakers is exactly what Common Voice's own
    datasheets do, not an attempt to identify anyone, which the dataset's
    terms forbid) is used for the count but deliberately not persisted
    into manifest.jsonl."""
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
    sample_rates: set[int], dataset_id: str, age_buckets: list[str],
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
    pack["metadata"]["age_group"] = "+".join(age_buckets)
    pack["metadata"]["num_speakers"] = speaker_count
    pack["min_runner_version"] = MIN_RUNNER_VERSION
    pack["sha256"] = canonical_asset_sha256(pack)

    pack_path.write_text(yaml.safe_dump(pack, sort_keys=False, allow_unicode=True))
    print(
        f"Updated {pack_path}: count={len(entries)} age_group={pack['metadata']['age_group']} "
        f"num_speakers={speaker_count} manifest_sha256={manifest_sha256}",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", required=True, help="MDC dataset id for Common Voice NL, from its dataset page URL.")
    parser.add_argument("--pack-dir", type=Path, required=True, help="Existing pack directory, e.g. packs/common-voice-nl-elderly.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Defaults to <pack-dir>/audio.")
    parser.add_argument("--count", type=int, default=40, help="Target clip count from the chosen bucket(s).")
    parser.add_argument("--age-buckets", type=str, default=None, help="Comma-separated bucket names to force (e.g. eighties,nineties) instead of auto-picking the oldest viable set.")
    parser.add_argument("--clip-suffix", type=str, default="", help="Override clip file extension if the archive re-encodes clips (e.g. .wav).")
    parser.add_argument("--dry-run", action="store_true", help="Stop after printing the age-bucket distribution — no manifest/pack.yaml written.")
    args = parser.parse_args()

    extract_dir = download_and_extract(args.dataset_id)
    tsv_path = find_validated_tsv(extract_dir)
    rows = read_validated_rows(tsv_path)
    nl_rows = [r for r in rows if r.get("locale", "nl") == "nl"]
    counts = report_age_distribution(nl_rows)

    if args.dry_run:
        print("--dry-run: stopping before writing anything.", file=sys.stderr)
        return 0

    if args.age_buckets:
        chosen_buckets = [b.strip() for b in args.age_buckets.split(",") if b.strip()]
    else:
        chosen_buckets = oldest_viable_buckets(counts, args.count)
        if not chosen_buckets:
            print(
                f"Not viable: even every age bucket combined has fewer than "
                f"--count={args.count} clips for nl. Reporting per ADR-0010 §5 "
                "rather than padding with younger speakers.",
                file=sys.stderr,
            )
            return 1

    selected_rows = [r for r in nl_rows if (r.get("age", "").strip() or "(unlabeled)") in chosen_buckets][: args.count]
    print(f"Selected {len(selected_rows)} clips from bucket(s) {chosen_buckets}.", file=sys.stderr)

    audio_dir = args.audio_dir or (args.pack_dir / "audio")
    entries, speaker_count, sample_rates = build_manifest(selected_rows, extract_dir, audio_dir, args.clip_suffix)
    manifest_path = write_manifest(args.pack_dir, entries)
    update_pack_yaml(args.pack_dir, manifest_path, entries, speaker_count, sample_rates, args.dataset_id, chosen_buckets)
    print(f"Wrote {manifest_path} ({len(entries)} utterances).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
