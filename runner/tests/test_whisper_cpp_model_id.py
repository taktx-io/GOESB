"""Kept in its own file, same rationale as test_faster_whisper_model_id.py:
importing `_resolve_model_id` doesn't require `pywhispercpp` (only
`run_batch` lazy-imports it)."""
import pytest

from oesb_runner.adapters.whisper_cpp import _resolve_model_id


@pytest.mark.parametrize(
    "profile_name,expected",
    [
        ("whisper-base.en", "base.en"),
        ("whisper-tiny", "tiny"),
        ("whisper-large-v3", "large-v3"),
        ("base.en", "base.en"),  # already-bare override names pass through unchanged
    ],
)
def test_resolve_model_id(profile_name, expected):
    assert _resolve_model_id(profile_name) == expected
