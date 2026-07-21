from pathlib import Path

import pytest

from oesb_runner.pack import PackAudioMissingError, load_pack

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "packs" / "example-librispeech-en-batch"

pytestmark = pytest.mark.skipif(
    not (PACK_DIR / "audio").exists(),
    reason="requires fetched audio: run scripts/fetch_librispeech_subset.py first",
)


def test_load_pack_verifies_and_returns_utterances():
    pack = load_pack(PACK_DIR)
    assert pack.id == "example-librispeech-en-batch"
    assert pack.profile_id == "whisper-medium-en-batch"
    assert len(pack.utterances) == 15
    assert all(u.audio_path.exists() for u in pack.utterances)
    assert pack.total_duration_s > 0


def test_load_pack_missing_audio_raises(tmp_path):
    with pytest.raises(PackAudioMissingError):
        load_pack(PACK_DIR, audio_dir=tmp_path)
