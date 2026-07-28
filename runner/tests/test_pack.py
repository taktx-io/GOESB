import json
from pathlib import Path

import pytest
import yaml

from oesb_runner.hashing import canonical_asset_sha256, sha256_bytes, sha256_file
from oesb_runner.pack import PackAudioMissingError, PackIntegrityError, load_pack

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "packs" / "librispeech-en-batch"

requires_fetched_audio = pytest.mark.skipif(
    not (PACK_DIR / "audio").exists(),
    reason="requires fetched audio: run scripts/fetch_librispeech_subset.py first",
)


def _write_pack(pack_dir: Path, audio_dir: Path, manifest_entry: dict, audio_bytes: bytes):
    """A minimal, self-contained pack (no real audio corpus needed) for
    exercising load_pack's own hash checks in isolation."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / manifest_entry["relative_path"]).write_bytes(audio_bytes)

    manifest_path = pack_dir / "manifest.jsonl"
    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_entry, sort_keys=True) + "\n")

    pack = {
        "id": "fixture-pack", "version": "1.0.0", "profile_id": "whisper-medium-en-batch",
        "visibility": "private",
        "audio": {"manifest_sha256": sha256_file(manifest_path)},
    }
    pack["sha256"] = canonical_asset_sha256(pack)
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(pack, sort_keys=False))


@requires_fetched_audio
def test_load_pack_verifies_and_returns_utterances():
    pack = load_pack(PACK_DIR)
    assert pack.id == "librispeech-en-batch"
    assert pack.profile_id == "whisper-medium-en-batch"
    assert len(pack.utterances) == 15
    assert all(u.audio_path.exists() for u in pack.utterances)
    assert pack.total_duration_s > 0


@requires_fetched_audio
def test_load_pack_missing_audio_raises(tmp_path):
    with pytest.raises(PackAudioMissingError):
        load_pack(PACK_DIR, audio_dir=tmp_path)


def test_load_pack_accepts_audio_matching_its_declared_hash(tmp_path):
    audio_bytes = b"fake but consistent audio content"
    entry = {
        "utterance_id": "u1", "relative_path": "u1.flac", "reference_text": "hello",
        "duration_s": 1.0, "audio_sha256": sha256_bytes(audio_bytes),
    }
    _write_pack(tmp_path / "pack", tmp_path / "pack" / "audio", entry, audio_bytes)

    pack = load_pack(tmp_path / "pack")
    assert len(pack.utterances) == 1
    assert pack.utterances[0].audio_path.read_bytes() == audio_bytes


def test_load_pack_rejects_audio_that_does_not_match_its_declared_hash(tmp_path):
    # Simulates the real-world case this whole check exists for: upstream
    # (Common Voice/MDC, FLEURS, LibriSpeech) serves different bytes behind
    # the same filename than what the pack was originally built from.
    entry = {
        "utterance_id": "u1", "relative_path": "u1.flac", "reference_text": "hello",
        "duration_s": 1.0, "audio_sha256": sha256_bytes(b"original audio at authoring time"),
    }
    _write_pack(tmp_path / "pack", tmp_path / "pack" / "audio", entry, b"different audio served now")

    with pytest.raises(PackIntegrityError, match="audio content hash mismatch"):
        load_pack(tmp_path / "pack")


def test_load_pack_skips_hash_check_when_manifest_entry_has_no_audio_sha256(tmp_path):
    # Backward compat: every pack published before this field existed has
    # no per-clip hash — must keep loading exactly as before, not start
    # rejecting packs that never declared one.
    entry = {
        "utterance_id": "u1", "relative_path": "u1.flac",
        "reference_text": "hello", "duration_s": 1.0,
    }
    _write_pack(tmp_path / "pack", tmp_path / "pack" / "audio", entry, b"whatever bytes")

    pack = load_pack(tmp_path / "pack")
    assert len(pack.utterances) == 1
