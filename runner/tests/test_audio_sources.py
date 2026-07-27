import io
import sys
import tarfile
import types
import zipfile

import pytest

from oesb_runner import audio_sources
from oesb_runner.audio_sources import (
    GatedFetchAuthError,
    auto_fetch_audio,
    shared_audio_dir,
)


def _fake_tar_gz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_auto_fetch_audio_fleurs_extracts_only_wanted_files(tmp_path, monkeypatch):
    archive = _fake_tar_gz({
        "nl_nl/audio/dev/wanted.wav": b"real audio bytes",
        "nl_nl/audio/dev/unwanted.wav": b"should not be extracted",
    })
    monkeypatch.setattr(
        audio_sources.urllib.request, "urlopen", lambda url, **kw: _FakeResponse(archive)
    )

    fetched = auto_fetch_audio(
        {"type": "fleurs", "params": {"language": "nl_nl", "split": "dev"}},
        {"wanted.wav"},
        tmp_path,
    )

    assert fetched == {"wanted.wav"}
    assert (tmp_path / "wanted.wav").read_bytes() == b"real audio bytes"
    assert not (tmp_path / "unwanted.wav").exists()


def test_auto_fetch_audio_returns_none_for_unknown_source_type(tmp_path):
    assert auto_fetch_audio({"type": "common-voice"}, {"a.wav"}, tmp_path) is None


def test_auto_fetch_audio_returns_none_for_manual_source(tmp_path):
    assert auto_fetch_audio({"type": "manual"}, {"a.wav"}, tmp_path) is None


def test_auto_fetch_audio_returns_none_when_no_source_declared(tmp_path):
    assert auto_fetch_audio({}, {"a.wav"}, tmp_path) is None


def test_shared_audio_dir_is_stable_for_identical_source(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    source_a = {"type": "fleurs", "params": {"language": "nl_nl", "split": "dev"}}
    source_b = {"type": "fleurs", "params": {"language": "nl_nl", "split": "dev"}}

    assert shared_audio_dir(source_a) == shared_audio_dir(source_b)
    assert shared_audio_dir(source_a).parent == tmp_path / "cache" / "audio"


def test_shared_audio_dir_differs_for_different_params(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    nl = {"type": "fleurs", "params": {"language": "nl_nl", "split": "dev"}}
    de = {"type": "fleurs", "params": {"language": "de_de", "split": "dev"}}

    assert shared_audio_dir(nl) != shared_audio_dir(de)


def test_sibling_packs_share_one_fetch_via_shared_audio_dir(tmp_path, monkeypatch):
    """Two "packs" (distinct audio_dir targets) with identical audio.source
    end up reading from the exact same shared directory — the real
    scenario this whole design exists for: 11 engine/size sibling packs
    for one language, all pointing at the same FLEURS split."""
    monkeypatch.setattr(audio_sources, "CACHE_ROOT", tmp_path / "cache")
    archive = _fake_tar_gz({"nl_nl/audio/dev/wanted.wav": b"real audio bytes"})
    call_count = 0

    def _counting_urlopen(url, **kw):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(archive)

    monkeypatch.setattr(audio_sources.urllib.request, "urlopen", _counting_urlopen)
    source = {"type": "fleurs", "params": {"language": "nl_nl", "split": "dev"}}

    pack_a_dir = shared_audio_dir(source)
    fetched_a = auto_fetch_audio(source, {"wanted.wav"}, pack_a_dir)
    assert fetched_a == {"wanted.wav"}
    assert call_count == 1

    # A second, different pack with the same source resolves to the same
    # directory and finds the file already there — no second fetch.
    pack_b_dir = shared_audio_dir(source)
    assert pack_b_dir == pack_a_dir
    assert (pack_b_dir / "wanted.wav").read_bytes() == b"real audio bytes"
    assert call_count == 1


def _fake_datacollective_module(*, download_dataset):
    module = types.ModuleType("datacollective")
    module.download_dataset = download_dataset
    return module


def test_fetch_common_voice_audio_extracts_only_wanted_files(tmp_path, monkeypatch):
    archive_path = tmp_path / "cv-nl.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for name, content in {
            "cv-nl/wanted.wav": b"real audio bytes",
            "cv-nl/unwanted.wav": b"should not be extracted",
        }.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

    monkeypatch.setitem(
        sys.modules, "datacollective",
        _fake_datacollective_module(download_dataset=lambda dataset_id, **kw: archive_path),
    )
    audio_dir = tmp_path / "audio"

    fetched = auto_fetch_audio(
        {"type": "mozilla_data_collective", "params": {"dataset_id": "abc123"}},
        {"wanted.wav"},
        audio_dir,
    )

    assert fetched == {"wanted.wav"}
    assert (audio_dir / "wanted.wav").read_bytes() == b"real audio bytes"
    assert not (audio_dir / "unwanted.wav").exists()


def test_fetch_common_voice_audio_extracts_from_zip_archive(tmp_path, monkeypatch):
    archive_path = tmp_path / "cv-nl.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("cv-nl/wanted.wav", b"real audio bytes")
        zf.writestr("cv-nl/unwanted.wav", b"should not be extracted")

    monkeypatch.setitem(
        sys.modules, "datacollective",
        _fake_datacollective_module(download_dataset=lambda dataset_id, **kw: archive_path),
    )
    audio_dir = tmp_path / "audio"

    fetched = audio_sources.fetch_common_voice_audio(
        {"dataset_id": "abc123"}, {"wanted.wav"}, audio_dir
    )

    assert fetched == {"wanted.wav"}
    assert (audio_dir / "wanted.wav").read_bytes() == b"real audio bytes"
    assert not (audio_dir / "unwanted.wav").exists()


def test_fetch_common_voice_audio_wraps_permission_error_as_gated_fetch_auth_error(tmp_path, monkeypatch):
    def _rejected(dataset_id, **kw):
        raise PermissionError("Access denied.")

    monkeypatch.setitem(
        sys.modules, "datacollective", _fake_datacollective_module(download_dataset=_rejected)
    )

    with pytest.raises(GatedFetchAuthError):
        audio_sources.fetch_common_voice_audio({"dataset_id": "abc123"}, {"a.wav"}, tmp_path)


def test_fetch_common_voice_audio_wraps_missing_key_value_error_as_gated_fetch_auth_error(tmp_path, monkeypatch):
    def _missing_key(dataset_id, **kw):
        raise ValueError("MDC_API_KEY is not set")

    monkeypatch.setitem(
        sys.modules, "datacollective", _fake_datacollective_module(download_dataset=_missing_key)
    )

    with pytest.raises(GatedFetchAuthError):
        audio_sources.fetch_common_voice_audio({"dataset_id": "abc123"}, {"a.wav"}, tmp_path)


def test_fetch_common_voice_audio_does_not_mask_non_auth_errors(tmp_path, monkeypatch):
    def _rate_limited(dataset_id, **kw):
        raise RuntimeError("Rate limit exceeded. Please try again later.")

    monkeypatch.setitem(
        sys.modules, "datacollective", _fake_datacollective_module(download_dataset=_rate_limited)
    )

    with pytest.raises(RuntimeError) as exc_info:
        audio_sources.fetch_common_voice_audio({"dataset_id": "abc123"}, {"a.wav"}, tmp_path)
    assert not isinstance(exc_info.value, GatedFetchAuthError)


def test_fetch_common_voice_audio_raises_clear_error_when_datacollective_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "datacollective", None)  # forces ImportError on import

    with pytest.raises(RuntimeError, match="datacollective is not installed"):
        audio_sources.fetch_common_voice_audio({"dataset_id": "abc123"}, {"a.wav"}, tmp_path)
