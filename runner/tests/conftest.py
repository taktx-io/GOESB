import pytest


@pytest.fixture(autouse=True)
def _no_network_version_check(monkeypatch):
    """`goesb run` does a best-effort network check for whether the
    platform currently requires a newer runner
    (`cli._warn_if_runner_outdated`) before starting a benchmark. Real
    network reachability must never affect whether -- or how fast -- this
    test suite passes, so it's silenced for every test here, the same way
    individual tests already fake `_get_json`/`_post_json` wherever they
    specifically want to exercise submit's own network behavior.
    `test_cli.py` imports the real function directly (before this fixture
    ever runs) for the tests that exercise its actual logic."""
    from oesb_runner import cli as cli_module

    monkeypatch.setattr(cli_module, "_warn_if_runner_outdated", lambda *a, **k: None)
