import re

import pytest

from oesb_runner.adapters import (
    get_adapter,
    get_applied_parameters,
    log_progress,
    register,
)
from oesb_runner.adapters.vosk import _MODEL_URLS


def test_get_adapter_dispatches_on_benchmark_type():
    batch = get_adapter("faster-whisper", benchmark_type="batch")
    streaming = get_adapter("faster-whisper", benchmark_type="streaming")
    assert batch is not streaming
    assert batch.__name__ == "run_batch"
    assert streaming.__name__ == "run_streaming"


def test_get_adapter_defaults_to_batch():
    assert get_adapter("faster-whisper") is get_adapter("faster-whisper", benchmark_type="batch")


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError):
        get_adapter("no-such-runtime")
    with pytest.raises(ValueError):
        get_adapter("faster-whisper", benchmark_type="conversation")


def test_register_rejects_duplicate_key():
    register("test-only-runtime", benchmark_type="batch")(lambda: None)
    with pytest.raises(ValueError):
        register("test-only-runtime", benchmark_type="batch")(lambda: None)


# --- ADR-0009 §2 "no silent knobs": applied-parameters registry ---


def test_get_applied_parameters_faster_whisper_batch():
    assert get_applied_parameters("faster-whisper", "batch") == {
        "quantization", "beam_size", "temperature", "vad", "threads",
    }


def test_get_applied_parameters_faster_whisper_streaming_adds_chunk_ms():
    streaming = get_applied_parameters("faster-whisper", "streaming")
    assert streaming == get_applied_parameters("faster-whisper", "batch") | {"chunk_ms"}


def test_get_applied_parameters_whisper_cpp_excludes_beam_size_and_vad():
    """The exact "no silent knobs" case: whisper-cpp's adapter accepts
    beam_size/vad/quantization for call-shape parity but never applies
    them — confirmed against the adapter's own code/docstring."""
    applied = get_applied_parameters("whisper-cpp", "batch")
    assert applied == {"threads", "temperature"}
    assert "beam_size" not in applied
    assert "vad" not in applied
    assert "quantization" not in applied


def test_get_applied_parameters_vosk_applies_nothing():
    assert get_applied_parameters("vosk", "batch") == frozenset()


def test_get_applied_parameters_unknown_pair_defaults_to_empty_fail_closed():
    """A future adapter that forgets to declare applied_parameters applies
    nothing by default, not everything — fail closed, not open."""
    register("test-only-unregistered-params-runtime", benchmark_type="batch")(lambda: None)
    assert get_applied_parameters("test-only-unregistered-params-runtime", "batch") == frozenset()


def test_log_progress_writes_to_stderr(capsys):
    log_progress(3, 15, "1272-128104-0002", 1.234)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[3/15]" in captured.err
    assert "1272-128104-0002" in captured.err
    assert "1.23s" in captured.err


_VOSK_URL_RE = re.compile(r"^https://alphacephei\.com/vosk/models/(.+)\.zip$")


def test_vosk_model_urls_match_the_expected_pattern():
    """Regression test for the _MODEL_URLS dict itself — every entry's URL
    must be exactly alphacephei's fixed pattern with the dict's own key as
    the filename stem. Catches copy-paste drift (wrong version suffix,
    mismatched key/filename) without needing vosk installed or any network
    call — the dict is importable standalone since vosk itself is only
    lazy-imported inside run_batch."""
    assert len(_MODEL_URLS) >= 5  # en (original) + es, fr, de, pt (this session)
    for model_name, url in _MODEL_URLS.items():
        match = _VOSK_URL_RE.match(url)
        assert match is not None, f"{model_name}: url {url!r} doesn't match the expected pattern"
        assert match.group(1) == model_name, f"{model_name}: url filename stem doesn't match the dict key"
