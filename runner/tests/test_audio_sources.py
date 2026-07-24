import io
import tarfile

from oesb_runner import audio_sources
from oesb_runner.audio_sources import auto_fetch_audio


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
