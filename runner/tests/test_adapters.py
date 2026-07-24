import pytest

from oesb_runner.adapters import get_adapter, log_progress, register


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


def test_log_progress_writes_to_stderr(capsys):
    log_progress(3, 15, "1272-128104-0002", 1.234)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[3/15]" in captured.err
    assert "1272-128104-0002" in captured.err
    assert "1.23s" in captured.err
