import io
import json
import urllib.error

import pytest
import yaml

from oesb_runner import remote


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(payloads):
    """payloads: list of bytes, one per call, consumed in order — lets a
    test express "first call returns X, second call returns Y" without a
    real server."""
    calls = []

    def _urlopen(url, **kw):
        calls.append(url)
        return _FakeResponse(payloads[len(calls) - 1])

    return _urlopen, calls


def test_fetch_profile_prefers_network_over_an_existing_cache(tmp_path, monkeypatch):
    """Real bug: a profile/pack cached once was served forever after, even
    once the server-side document changed — this is the regression test
    for the fix. A stale cache on disk must never win over a reachable,
    successful network fetch."""
    monkeypatch.setattr(remote, "CACHE_ROOT", tmp_path)
    cache_path = tmp_path / "profiles" / "whisper-tiny-nl-batch.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"id": "whisper-tiny-nl-batch", "version": "old-stale"}))

    fresh = {"id": "whisper-tiny-nl-batch", "version": "fresh-from-network"}
    urlopen, calls = _fake_urlopen([json.dumps(fresh).encode()])
    monkeypatch.setattr(remote.urllib.request, "urlopen", urlopen)

    result = remote.fetch_profile("whisper-tiny-nl-batch", "https://api.example")

    assert result == fresh
    assert len(calls) == 1  # network was actually hit, not skipped for the cache
    assert json.loads(cache_path.read_text()) == fresh  # cache refreshed too


def test_fetch_profile_falls_back_to_cache_when_network_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CACHE_ROOT", tmp_path)
    cache_path = tmp_path / "profiles" / "whisper-tiny-nl-batch.json"
    cache_path.parent.mkdir(parents=True)
    cached = {"id": "whisper-tiny-nl-batch", "version": "cached-fallback"}
    cache_path.write_text(json.dumps(cached))

    def _fail(url, **kw):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _fail)

    result = remote.fetch_profile("whisper-tiny-nl-batch", "https://api.example")

    assert result == cached


def test_fetch_profile_raises_when_network_fails_and_nothing_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CACHE_ROOT", tmp_path)

    def _fail(url, **kw):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _fail)

    with pytest.raises(urllib.error.URLError):
        remote.fetch_profile("whisper-tiny-nl-batch", "https://api.example")


def test_fetch_pack_prefers_network_over_an_existing_cache(tmp_path, monkeypatch):
    """Same regression as fetch_profile, for the field that actually broke
    a real user: a pack cached before it required a credential kept
    serving the credential-less copy forever, so the wizard's own
    credential preflight never saw the requirement to prompt for it."""
    monkeypatch.setattr(remote, "CACHE_ROOT", tmp_path)
    cache_dir = tmp_path / "packs" / "common-voice-nl"
    cache_dir.mkdir(parents=True)
    (cache_dir / "pack.yaml").write_text(yaml.safe_dump({"id": "common-voice-nl", "audio": {"source": {"type": "mozilla_data_collective"}}}))
    (cache_dir / "manifest.jsonl").write_text('{"relative_path": "old.wav"}\n')

    fresh_pack = {
        "id": "common-voice-nl",
        "audio": {"source": {"type": "mozilla_data_collective", "credential": {"env_var": "MDC_API_KEY"}}},
    }
    urlopen, calls = _fake_urlopen([
        json.dumps(fresh_pack).encode(),
        b'{"relative_path": "fresh.wav"}\n',
    ])
    monkeypatch.setattr(remote.urllib.request, "urlopen", urlopen)

    result_dir = remote.fetch_pack("common-voice-nl", "https://api.example")

    assert len(calls) == 2  # both the pack API and the manifest were actually hit
    written_pack = yaml.safe_load((result_dir / "pack.yaml").read_text())
    assert written_pack["audio"]["source"]["credential"]["env_var"] == "MDC_API_KEY"
    assert (result_dir / "manifest.jsonl").read_text() == '{"relative_path": "fresh.wav"}\n'


def test_fetch_pack_falls_back_to_cache_when_network_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CACHE_ROOT", tmp_path)
    cache_dir = tmp_path / "packs" / "common-voice-nl"
    cache_dir.mkdir(parents=True)
    (cache_dir / "pack.yaml").write_text(yaml.safe_dump({"id": "common-voice-nl"}))
    (cache_dir / "manifest.jsonl").write_text('{"relative_path": "cached.wav"}\n')

    def _fail(url, **kw):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _fail)

    result_dir = remote.fetch_pack("common-voice-nl", "https://api.example")

    assert result_dir == cache_dir
    assert (result_dir / "manifest.jsonl").read_text() == '{"relative_path": "cached.wav"}\n'


def test_fetch_pack_raises_when_network_fails_and_nothing_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CACHE_ROOT", tmp_path)

    def _fail(url, **kw):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _fail)

    with pytest.raises(urllib.error.URLError):
        remote.fetch_pack("common-voice-nl", "https://api.example")


def test_fetch_pack_does_not_write_a_half_updated_cache_when_manifest_fetch_fails(tmp_path, monkeypatch):
    """The pack.yaml API call and the manifest.jsonl call are two separate
    requests — if the pack.yaml succeeds but the manifest fetch then fails,
    the old cache (if any) must be left exactly as it was, not partially
    overwritten with a fresh pack.yaml paired with a stale manifest."""
    monkeypatch.setattr(remote, "CACHE_ROOT", tmp_path)
    cache_dir = tmp_path / "packs" / "common-voice-nl"
    cache_dir.mkdir(parents=True)
    (cache_dir / "pack.yaml").write_text(yaml.safe_dump({"id": "common-voice-nl", "version": "original"}))
    (cache_dir / "manifest.jsonl").write_text('{"relative_path": "original.wav"}\n')

    calls = []

    def _urlopen(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResponse(json.dumps({"id": "common-voice-nl", "version": "partial-fetch"}).encode())
        raise urllib.error.URLError("manifest host unreachable")

    monkeypatch.setattr(remote.urllib.request, "urlopen", _urlopen)

    result_dir = remote.fetch_pack("common-voice-nl", "https://api.example")

    assert result_dir == cache_dir
    assert yaml.safe_load((result_dir / "pack.yaml").read_text())["version"] == "original"
    assert (result_dir / "manifest.jsonl").read_text() == '{"relative_path": "original.wav"}\n'
