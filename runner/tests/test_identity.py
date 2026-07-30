import stat
import sys

import pytest

from oesb_runner import identity


def test_discriminator_is_deterministic():
    a = identity.compute_discriminator("callsign", "secret")
    b = identity.compute_discriminator("callsign", "secret")
    assert a == b


def test_discriminator_differs_for_different_secrets_same_callsign():
    a = identity.compute_discriminator("callsign", "secret-one")
    b = identity.compute_discriminator("callsign", "secret-two")
    assert a != b


def test_discriminator_differs_for_same_secret_different_callsigns():
    """Callsign is the KDF salt -- a leaked/guessed secret for one callsign
    must not let an attacker correlate or impersonate under another."""
    a = identity.compute_discriminator("alice", "shared-secret")
    b = identity.compute_discriminator("bob", "shared-secret")
    assert a != b


def test_discriminator_is_8_hex_chars():
    d = identity.compute_discriminator("callsign", "secret")
    assert len(d) == 8
    int(d, 16)  # raises if not valid hex


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "identity.json"
    saved = identity.Identity("anon", "a1b2c3d4")

    identity.save_identity(saved, path=path)

    assert identity.load_identity(path=path) == saved


def test_save_overwrites_previous_identity(tmp_path):
    path = tmp_path / "identity.json"
    identity.save_identity(identity.Identity("first", "11111111"), path=path)

    identity.save_identity(identity.Identity("second", "22222222"), path=path)

    assert identity.load_identity(path=path) == identity.Identity("second", "22222222")


def test_clear_removes_saved_identity(tmp_path):
    path = tmp_path / "identity.json"
    identity.save_identity(identity.Identity("anon", "a1b2c3d4"), path=path)

    identity.clear_identity(path=path)

    assert identity.load_identity(path=path) is None


def test_clear_is_a_no_op_when_nothing_saved(tmp_path):
    path = tmp_path / "identity.json"
    identity.clear_identity(path=path)  # must not raise
    assert identity.load_identity(path=path) is None


def test_load_returns_none_when_file_missing(tmp_path):
    assert identity.load_identity(path=tmp_path / "nope.json") is None


def test_load_returns_none_on_corrupt_json(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text("{not valid json")

    assert identity.load_identity(path=path) is None


def test_load_returns_none_when_json_is_not_an_object(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text("[1, 2, 3]")

    assert identity.load_identity(path=path) is None


def test_load_returns_none_when_fields_missing(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text('{"callsign": "anon"}')  # no discriminator

    assert identity.load_identity(path=path) is None


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file-permission bits (chmod 0600) don't apply on Windows"
)
def test_save_sets_file_mode_0600(tmp_path):
    path = tmp_path / "nested" / "identity.json"

    identity.save_identity(identity.Identity("anon", "a1b2c3d4"), path=path)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
