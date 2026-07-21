import os

import psutil
import pytest

from oesb_runner.metrics import cer, cpu_ram, rtf, wer
from oesb_runner.metrics._align import edit_distance


def test_edit_distance_known_values():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], []) == 0
    assert edit_distance(["a", "b"], ["a", "b"]) == 0
    assert edit_distance(["a", "b"], []) == 2


def test_wer_single_utterance_one_substitution():
    # "de kat zit" vs "de mat zit" -> 1 substitution / 3 ref words
    pairs = [("de kat zit", "de mat zit")]
    assert wer.compute(pairs) == pytest.approx(1 / 3)


def test_wer_perfect_match_is_zero():
    assert wer.compute([("hallo wereld", "hallo wereld")]) == 0.0


def test_wer_is_corpus_level_not_mean_of_ratios():
    # utterance 1: 1 word, 1 error (ratio 1.0); utterance 2: 9 words, 0 errors (ratio 0.0)
    # mean-of-ratios would give 0.5; corpus-level gives 1/10.
    pairs = [
        ("fout", "anders"),
        ("een twee drie vier vijf zes zeven acht negen", "een twee drie vier vijf zes zeven acht negen"),
    ]
    assert wer.compute(pairs) == pytest.approx(1 / 10)


def test_wer_requires_non_empty_reference():
    with pytest.raises(ValueError):
        wer.compute([("", "iets")])


def test_cer_single_utterance():
    # "kat" vs "kot": 1 char substitution / 3 ref chars
    assert cer.compute([("kat", "kot")]) == pytest.approx(1 / 3)


def test_cer_requires_non_empty_reference():
    with pytest.raises(ValueError):
        cer.compute([("", "iets")])


def test_rtf_basic():
    assert rtf.compute(5.0, 10.0) == pytest.approx(0.5)


def test_rtf_faster_than_realtime():
    assert rtf.compute(2.0, 10.0) < 1.0


def test_rtf_rejects_zero_duration():
    with pytest.raises(ValueError):
        rtf.compute(1.0, 0.0)


def test_cpu_ram_sample_and_reduce():
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)  # prime the baseline, per psutil convention
    samples = [cpu_ram.sample_process_tree(proc) for _ in range(3)]
    assert all(s["rss_mb"] > 0 for s in samples)
    assert cpu_ram.reduce_cpu_pct(samples) >= 0.0
    assert cpu_ram.reduce_peak_ram_mb(samples) == max(s["rss_mb"] for s in samples)


def test_cpu_ram_reducers_reject_empty():
    with pytest.raises(ValueError):
        cpu_ram.reduce_cpu_pct([])
    with pytest.raises(ValueError):
        cpu_ram.reduce_peak_ram_mb([])
