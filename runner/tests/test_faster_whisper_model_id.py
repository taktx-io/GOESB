"""Kept in its own file: importing `_resolve_model_id` doesn't require the
`faster_whisper` package (only `run_batch` lazy-imports it), so this must not
be collected in a module gated by `pytest.importorskip("faster_whisper")` —
that would wrongly skip a pure-string-logic test whenever the optional
faster-whisper extra isn't installed.

This is the exact translation that was missing when the CLI first called
run_batch("whisper-medium", ...) with the profile's own model name and
faster-whisper rejected it: "Invalid model size 'whisper-medium'".
"""
import pytest

from oesb_runner.adapters.faster_whisper import _resolve_model_id


@pytest.mark.parametrize(
    "profile_name,expected",
    [
        ("whisper-medium", "medium"),
        ("whisper-tiny", "tiny"),
        ("whisper-large-v3", "large-v3"),
        ("tiny", "tiny"),  # already-bare override names pass through unchanged
    ],
)
def test_resolve_model_id(profile_name, expected):
    assert _resolve_model_id(profile_name) == expected
