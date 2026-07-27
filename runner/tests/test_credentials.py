import stat
import sys

import pytest

from oesb_runner import credentials


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.delenv("MDC_API_KEY", raising=False)
    path = tmp_path / "credentials.json"

    credentials.save_credential("MDC_API_KEY", "secret-value", path=path)

    assert credentials.load_credential("MDC_API_KEY", path=path) == "secret-value"


def test_environment_variable_takes_priority_over_stored_value(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    credentials.save_credential("MDC_API_KEY", "stored-value", path=path)
    monkeypatch.setenv("MDC_API_KEY", "env-value")

    assert credentials.load_credential("MDC_API_KEY", path=path) == "env-value"


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file-permission bits (chmod 0600) don't apply on Windows"
)
def test_save_sets_file_mode_0600(tmp_path):
    path = tmp_path / "nested" / "credentials.json"

    credentials.save_credential("MDC_API_KEY", "secret-value", path=path)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_save_preserves_other_keys(tmp_path):
    path = tmp_path / "credentials.json"
    credentials.save_credential("FIRST_KEY", "first-value", path=path)

    credentials.save_credential("SECOND_KEY", "second-value", path=path)

    assert credentials.load_credential("FIRST_KEY", path=path) == "first-value"
    assert credentials.load_credential("SECOND_KEY", path=path) == "second-value"


def test_load_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("MDC_API_KEY", raising=False)
    assert credentials.load_credential("MDC_API_KEY", path=tmp_path / "nope.json") is None


def test_load_returns_none_on_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.delenv("MDC_API_KEY", raising=False)
    path = tmp_path / "credentials.json"
    path.write_text("{not valid json")

    assert credentials.load_credential("MDC_API_KEY", path=path) is None


def test_load_returns_none_when_json_is_not_an_object(tmp_path, monkeypatch):
    monkeypatch.delenv("MDC_API_KEY", raising=False)
    path = tmp_path / "credentials.json"
    path.write_text("[1, 2, 3]")

    assert credentials.load_credential("MDC_API_KEY", path=path) is None


def test_load_returns_none_for_unknown_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_OTHER_KEY", raising=False)
    path = tmp_path / "credentials.json"
    credentials.save_credential("MDC_API_KEY", "secret-value", path=path)

    assert credentials.load_credential("SOME_OTHER_KEY", path=path) is None
