import pytest

from oesb_runner.adapters import get_adapter, register


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
