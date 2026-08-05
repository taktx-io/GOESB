import os
import types

import psutil
import pytest

from oesb_runner.metrics import (
    cer,
    cpu_ram,
    energy,
    gpu_pct,
    rtf,
    temperature,
    throughput,
    wer,
)
from oesb_runner.metrics._align import edit_distance


def test_edit_distance_known_values():
    assert edit_distance("kitten", "sitting").total == 3
    assert edit_distance([], []).total == 0
    assert edit_distance(["a", "b"], ["a", "b"]).total == 0
    assert edit_distance(["a", "b"], []).total == 2


def test_edit_distance_breaks_down_a_pure_substitution():
    counts = edit_distance(["a", "b", "c"], ["a", "x", "c"])
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 0)
    assert counts.total == 1


def test_edit_distance_breaks_down_a_pure_deletion():
    # "b" is in the reference but missing from the hypothesis.
    counts = edit_distance(["a", "b", "c"], ["a", "c"])
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 0)
    assert counts.total == 1


def test_edit_distance_breaks_down_a_pure_insertion():
    # "b" is extra in the hypothesis, not present in the reference.
    counts = edit_distance(["a", "c"], ["a", "b", "c"])
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)
    assert counts.total == 1


def test_wer_single_utterance_one_substitution():
    # "de kat zit" vs "de mat zit" -> 1 substitution / 3 ref words
    pairs = [("de kat zit", "de mat zit")]
    assert wer.compute(pairs) == pytest.approx(1 / 3)


def test_wer_compute_detailed_matches_compute_and_reports_breakdown():
    pairs = [("de kat zit", "de mat zit")]
    detailed = wer.compute_detailed(pairs)
    assert detailed.value == pytest.approx(wer.compute(pairs))
    assert (detailed.substitutions, detailed.deletions, detailed.insertions) == (1, 0, 0)
    assert detailed.ref_words == 3


def test_wer_compute_detailed_sums_breakdown_across_the_whole_corpus():
    pairs = [
        ("een fout woord", "een verkeerd woord"),  # 1 substitution
        ("nog een woord hier", "nog woord hier"),  # 1 deletion ("een" missing)
        ("laatste zin", "laatste extra zin"),  # 1 insertion ("extra")
    ]
    detailed = wer.compute_detailed(pairs)
    assert (detailed.substitutions, detailed.deletions, detailed.insertions) == (1, 1, 1)
    assert detailed.value == pytest.approx(3 / detailed.ref_words)


def test_wer_compute_detailed_reports_per_utterance_ratios_not_averaged_into_value():
    pairs = [
        ("fout", "anders"),  # 1/1 = 1.0
        ("een twee drie vier vijf zes zeven acht negen", "een twee drie vier vijf zes zeven acht negen"),  # 0/9 = 0.0
    ]
    detailed = wer.compute_detailed(pairs)
    assert detailed.per_utterance == (pytest.approx(1.0), pytest.approx(0.0))
    assert detailed.value == pytest.approx(1 / 10)  # corpus-level, not mean(per_utterance)


def test_wer_compute_detailed_excludes_empty_reference_from_per_utterance():
    pairs = [("", "iets"), ("hallo wereld", "hallo wereld")]
    detailed = wer.compute_detailed(pairs)
    assert detailed.per_utterance == (0.0,)


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


def test_cer_compute_detailed_matches_compute_and_reports_breakdown():
    pairs = [("kat", "kot")]
    detailed = cer.compute_detailed(pairs)
    assert detailed.value == pytest.approx(cer.compute(pairs))
    assert (detailed.substitutions, detailed.deletions, detailed.insertions) == (1, 0, 0)
    assert detailed.ref_chars == 3


def test_cer_compute_detailed_reports_per_utterance_ratios():
    pairs = [("kat", "kot"), ("hond", "hond")]
    detailed = cer.compute_detailed(pairs)
    assert detailed.per_utterance == (pytest.approx(1 / 3), pytest.approx(0.0))


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


def test_energy_compute_converts_uj_to_wh():
    # 3600 * 1e6 uJ == 1 Wh, by construction
    assert energy.compute(3_600_000_000.0) == pytest.approx(1.0)


def test_energy_compute_rejects_negative_delta():
    with pytest.raises(ValueError):
        energy.compute(-1.0)


def test_temperature_reduce_peak_returns_max():
    assert temperature.reduce_peak_temp_c([42.0, 55.5, 30.0]) == pytest.approx(55.5)


def test_temperature_reduce_peak_rejects_empty():
    with pytest.raises(ValueError):
        temperature.reduce_peak_temp_c([])


def test_throughput_basic():
    # 10s of audio processed in 5s wall-clock -> 2.0 audio-s/s
    assert throughput.compute(10.0, 5.0) == pytest.approx(2.0)


def test_throughput_rejects_zero_wall_time():
    with pytest.raises(ValueError):
        throughput.compute(10.0, 0.0)


def test_gpu_pct_reduce_mean_returns_average():
    assert gpu_pct.reduce_mean_gpu_pct([40.0, 60.0]) == pytest.approx(50.0)


def test_gpu_pct_reduce_mean_rejects_empty():
    with pytest.raises(ValueError):
        gpu_pct.reduce_mean_gpu_pct([])


class _FakeNVMLError(Exception):
    pass


def _fake_pynvml(*, init_raises=False, handle_raises=False, util_raises=False, gpu_value=0.0, on_init=None):
    """A minimal stand-in for the real `pynvml` module -- just the three
    calls gpu_pct.py actually uses, plus the NVMLError type its except
    clauses catch by name."""
    rates = types.SimpleNamespace(gpu=gpu_value)

    def nvmlInit():
        if on_init:
            on_init()
        if init_raises:
            raise _FakeNVMLError("no NVIDIA driver")

    def nvmlDeviceGetHandleByIndex(index):
        if handle_raises:
            raise _FakeNVMLError("no device 0")
        return f"handle-{index}"

    def nvmlDeviceGetUtilizationRates(handle):
        if util_raises:
            raise _FakeNVMLError("transient read error")
        return rates

    return types.SimpleNamespace(
        NVMLError=_FakeNVMLError,
        nvmlInit=nvmlInit,
        nvmlDeviceGetHandleByIndex=nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates=nvmlDeviceGetUtilizationRates,
    )


def _reset_gpu_pct(monkeypatch, fake_pynvml):
    # gpu_pct caches its NVML handle at module scope on purpose (nvmlInit()
    # once per process, not once per sample) -- tests must reset that cache
    # explicitly or an earlier test's state leaks into the next one.
    monkeypatch.setattr(gpu_pct, "pynvml", fake_pynvml)
    monkeypatch.setattr(gpu_pct, "_handle", None)
    monkeypatch.setattr(gpu_pct, "_unavailable", fake_pynvml is None)


def test_gpu_pct_sample_returns_none_without_nvml_installed(monkeypatch):
    _reset_gpu_pct(monkeypatch, None)
    assert gpu_pct.sample_gpu_pct() is None


def test_gpu_pct_sample_returns_none_when_nvml_init_fails(monkeypatch):
    _reset_gpu_pct(monkeypatch, _fake_pynvml(init_raises=True))
    assert gpu_pct.sample_gpu_pct() is None


def test_gpu_pct_sample_returns_none_when_no_device_0(monkeypatch):
    _reset_gpu_pct(monkeypatch, _fake_pynvml(handle_raises=True))
    assert gpu_pct.sample_gpu_pct() is None


def test_gpu_pct_sample_reads_utilization_gpu(monkeypatch):
    _reset_gpu_pct(monkeypatch, _fake_pynvml(gpu_value=37))
    assert gpu_pct.sample_gpu_pct() == pytest.approx(37.0)


def test_gpu_pct_sample_returns_none_on_a_transient_read_error(monkeypatch):
    _reset_gpu_pct(monkeypatch, _fake_pynvml(util_raises=True))
    assert gpu_pct.sample_gpu_pct() is None


def test_gpu_pct_sample_only_calls_nvml_init_once_across_many_samples(monkeypatch):
    init_calls = []
    _reset_gpu_pct(monkeypatch, _fake_pynvml(gpu_value=10, on_init=lambda: init_calls.append(1)))
    gpu_pct.sample_gpu_pct()
    gpu_pct.sample_gpu_pct()
    gpu_pct.sample_gpu_pct()
    assert len(init_calls) == 1
