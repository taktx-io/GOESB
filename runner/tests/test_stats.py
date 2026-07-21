import pytest

from oesb_runner.stats import relative_std, summarize


def test_summarize_single_value():
    s = summarize([1.0])
    assert s["value"] == 1.0
    assert s["std"] == 0.0
    assert s["min"] == s["max"] == s["p50"] == s["p95"] == 1.0


def test_summarize_two_values():
    s = summarize([1.0, 3.0])
    assert s["value"] == 2.0
    assert s["min"] == 1.0
    assert s["max"] == 3.0
    assert s["std"] == pytest.approx(1.0)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_relative_std_zero_mean_zero_std_is_zero():
    assert relative_std({"value": 0.0, "std": 0.0}) == 0.0


def test_relative_std_zero_mean_nonzero_std_is_infinite():
    assert relative_std({"value": 0.0, "std": 0.1}) == float("inf")


def test_relative_std_basic():
    assert relative_std({"value": 2.0, "std": 0.2}) == pytest.approx(0.1)
