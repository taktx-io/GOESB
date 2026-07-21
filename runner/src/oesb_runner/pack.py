"""Load and integrity-check a benchmark pack (FR-3.1/3.2, ADR-0004).

A pack in git is manifest + metadata only (FR-3.5); audio is fetched
separately (see scripts/fetch_librispeech_subset.py for the shipped example)
and verified here against the committed manifest before a run proceeds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .hashing import canonical_asset_sha256, sha256_file


class PackIntegrityError(Exception):
    """Raised when a pack's manifest or content hash doesn't match pack.yaml."""


class PackAudioMissingError(Exception):
    """Raised when the manifest references audio that isn't present locally."""


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    audio_path: Path
    reference_text: str
    duration_s: float


@dataclass(frozen=True)
class Pack:
    id: str
    version: str
    profile_id: str
    visibility: str
    utterances: list[Utterance]

    @property
    def total_duration_s(self) -> float:
        return sum(u.duration_s for u in self.utterances)


def load_pack(pack_dir: Path, audio_dir: Path | None = None) -> Pack:
    """Load `pack_dir`'s pack.yaml + manifest.jsonl, verifying both hashes and
    that every referenced audio file actually exists in `audio_dir`."""
    pack_dir = Path(pack_dir)
    audio_dir = Path(audio_dir) if audio_dir is not None else pack_dir / "audio"

    pack_yaml = yaml.safe_load((pack_dir / "pack.yaml").read_text())

    declared_sha256 = pack_yaml["sha256"]
    actual_sha256 = canonical_asset_sha256(pack_yaml)
    if actual_sha256 != declared_sha256:
        raise PackIntegrityError(
            f"pack.yaml content hash mismatch for {pack_yaml['id']}: "
            f"declared {declared_sha256}, computed {actual_sha256}"
        )

    manifest_path = pack_dir / "manifest.jsonl"
    declared_manifest_sha256 = pack_yaml["audio"]["manifest_sha256"]
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != declared_manifest_sha256:
        raise PackIntegrityError(
            f"manifest.jsonl hash mismatch for {pack_yaml['id']}: "
            f"declared {declared_manifest_sha256}, computed {actual_manifest_sha256}"
        )

    utterances: list[Utterance] = []
    missing: list[str] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        audio_path = audio_dir / entry["relative_path"]
        if not audio_path.exists():
            missing.append(entry["relative_path"])
            continue
        utterances.append(Utterance(
            utterance_id=entry["utterance_id"],
            audio_path=audio_path,
            reference_text=entry["reference_text"],
            duration_s=entry["duration_s"],
        ))

    if missing:
        raise PackAudioMissingError(
            f"{len(missing)} audio file(s) missing from {audio_dir} "
            f"(e.g. {missing[0]}). Fetch the pack's audio first — see "
            f"{pack_dir / 'README.md'} or scripts/fetch_librispeech_subset.py."
        )

    return Pack(
        id=pack_yaml["id"],
        version=pack_yaml["version"],
        profile_id=pack_yaml["profile_id"],
        visibility=pack_yaml["visibility"],
        utterances=utterances,
    )
