from pathlib import Path
from typing import ClassVar

import pytest

from oesb_runner.adapters import parakeet
from oesb_runner.adapters.parakeet import (
    _merge_tokens_to_words,
    _resolve_model_id,
    run_batch,
)
from oesb_runner.normalization import normalize
from oesb_runner.pack import Utterance, load_pack

transformers = pytest.importorskip(
    "transformers", reason="requires `pip install goesb-runner[parakeet]`"
)
torch = pytest.importorskip(
    "torch", reason="requires `pip install goesb-runner[parakeet]`"
)

from oesb_runner.metrics import rtf, wer

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "packs" / "fleurs-nl"

pytestmark = pytest.mark.skipif(
    not (PACK_DIR / "audio").exists(),
    reason="requires fetched audio: run scripts/fetch_fleurs_subset.py --language nl_nl first",
)


@pytest.mark.slow
def test_run_batch_transcribes_real_dutch_audio_within_wer_tolerance(tmp_path):
    """End-to-end proof against real Dutch FLEURS audio — this adapter's
    actual reason for existing (Babbl's realtime Dutch STT hardware
    question), not just "it ran"."""
    pack = load_pack(PACK_DIR)
    transcriptions = run_batch(
        "parakeet-tdt-0.6b-v3", pack.utterances, download_root=tmp_path / "models"
    )
    by_id = {t.utterance_id: t for t in transcriptions}

    pairs = []
    for utterance in pack.utterances:
        hypothesis = by_id[utterance.utterance_id].hypothesis_text
        pairs.append((
            normalize("goesb-nl-v1", utterance.reference_text),
            normalize("goesb-nl-v1", hypothesis),
        ))

    result_wer = wer.compute(pairs)
    total_processing_s = sum(t.processing_time_s for t in transcriptions)
    result_rtf = rtf.compute(total_processing_s, pack.total_duration_s)

    # Loose bound: proving the wiring produces real Dutch transcriptions,
    # not pinning an exact accuracy number.
    assert result_wer < 0.5
    assert result_rtf > 0


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Silicon MPS")
def test_run_batch_metal_backend_transcribes_real_dutch_audio_correctly(tmp_path):
    """Real, measured proof MPS isn't just "doesn't crash": correct text
    AND genuinely faster steady-state than cpu (~0.06-0.09x RTF measured
    across 5 real clips vs cpu's ~0.15x), once `_warm_up`'s one-off
    JIT/kernel-compile cost (measured ~3.4s) is paid outside the timed
    loop -- this asserts the steady-state number, not the first-call one,
    is what actually lands in the result."""
    pack = load_pack(PACK_DIR)
    transcriptions = run_batch(
        "parakeet-tdt-0.6b-v3", pack.utterances, backend="metal", download_root=tmp_path / "models"
    )
    by_id = {t.utterance_id: t for t in transcriptions}

    pairs = []
    for utterance in pack.utterances:
        hypothesis = by_id[utterance.utterance_id].hypothesis_text
        pairs.append((
            normalize("goesb-nl-v1", utterance.reference_text),
            normalize("goesb-nl-v1", hypothesis),
        ))

    result_wer = wer.compute(pairs)
    # None of the individually-timed utterances should carry the one-off
    # warm-up cost -- every one should already be in steady-state MPS
    # territory (measured ~0.5-0.9s per clip), nowhere near the ~3.4s
    # warm-up call this same audio took before _warm_up existed.
    assert all(t.processing_time_s < 3.0 for t in transcriptions)
    assert result_wer < 0.5


def test_resolve_model_id_prefixes_nvidia_org():
    assert _resolve_model_id("parakeet-tdt-0.6b-v3") == "nvidia/parakeet-tdt-0.6b-v3"


def test_run_batch_cuda_backend_hard_fails_when_torch_reports_unavailable(monkeypatch, tmp_path):
    """ADR-0008: --backend cuda must fail loudly, before any model weights
    load, when torch itself reports no usable CUDA device — never a silent
    fallback to cpu."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        run_batch(
            "parakeet-tdt-0.6b-v3",
            [_fake_utterance(tmp_path)],
            backend="cuda",
            download_root=tmp_path / "models",
        )


def test_run_batch_metal_backend_hard_fails_when_torch_reports_unavailable(monkeypatch, tmp_path):
    """Same ADR-0008 contract as cuda, for Apple Silicon's MPS backend --
    must fail loudly before any model weights load, never a silent
    fallback to cpu."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.backends.mps.is_available"):
        run_batch(
            "parakeet-tdt-0.6b-v3",
            [_fake_utterance(tmp_path)],
            backend="metal",
            download_root=tmp_path / "models",
        )


class _FakeBatchFeature(dict):
    """Stands in for the BatchFeature `ParakeetProcessor.__call__` returns
    — real code calls `.to(device, dtype=...)` on it before unpacking as
    `**inputs` into `model.generate`."""

    def to(self, *_args, **_kwargs):
        return self


class _FakeGenerateOutput:
    sequences: ClassVar[list] = [[1, 2, 3]]
    durations: ClassVar[list] = [[1, 1, 1]]


class _FakeProcessor:
    last_call_kwargs: ClassVar[dict] = {}
    decode_return: ClassVar[list] = ["fake dutch hypothesis"]
    # Real tokens captured against real Dutch FLEURS audio (see
    # _merge_tokens_to_words's own tests) -- reused here so the streaming
    # integration test exercises the same real word-boundary shape, not a
    # hand-simplified one.
    decode_timestamps_return: ClassVar[list] = [[
        {"token": "hal", "start": 0.0, "end": 0.2},
        {"token": "lo", "start": 0.2, "end": 0.4},
        {"token": " wereld", "start": 0.5, "end": 0.9},
    ]]

    def __init__(self):
        self.feature_extractor = _FakeFeatureExtractor()

    @classmethod
    def from_pretrained(cls, model_id, cache_dir=None):
        return cls()

    def __call__(self, samples, sampling_rate=None, return_tensors=None):
        _FakeProcessor.last_call_kwargs = {
            "sampling_rate": sampling_rate, "return_tensors": return_tensors,
        }
        return _FakeBatchFeature()

    def decode(self, sequences, skip_special_tokens=True, durations=None):
        if durations is not None:
            return _FakeProcessor.decode_return, _FakeProcessor.decode_timestamps_return
        return _FakeProcessor.decode_return


class _FakeFeatureExtractor:
    sampling_rate = 16000


class _FakeModel:
    last_set_num_threads: ClassVar[int | None] = None
    dtype = None  # set from a real torch dtype in from_pretrained

    def __init__(self):
        self.dtype = torch.float32

    @classmethod
    def from_pretrained(cls, model_id, cache_dir=None):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **_kwargs):
        return _FakeGenerateOutput()


def _fake_utterance(tmp_path: Path) -> Utterance:
    return Utterance(
        utterance_id="u1", audio_path=tmp_path / "fake.wav",
        reference_text="hallo wereld", duration_s=1.0,
    )


def test_run_batch_reads_sample_rate_from_processor_and_returns_decoded_text(monkeypatch, tmp_path):
    monkeypatch.setattr("transformers.AutoProcessor", _FakeProcessor)
    monkeypatch.setattr("transformers.AutoModelForTDT", _FakeModel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # never consulted for backend="cpu"
    monkeypatch.setattr(parakeet, "decode_pcm", lambda path, rate, dtype: [0.0])

    results = run_batch(
        "parakeet-tdt-0.6b-v3", [_fake_utterance(tmp_path)], download_root=tmp_path / "models"
    )

    assert len(results) == 1
    assert results[0].hypothesis_text == "fake dutch hypothesis"
    assert results[0].utterance_id == "u1"
    assert _FakeProcessor.last_call_kwargs["sampling_rate"] == 16000


def test_run_batch_sets_torch_threads_for_cpu_backend(monkeypatch, tmp_path):
    monkeypatch.setattr("transformers.AutoProcessor", _FakeProcessor)
    monkeypatch.setattr("transformers.AutoModelForTDT", _FakeModel)
    monkeypatch.setattr(parakeet, "decode_pcm", lambda path, rate, dtype: [0.0])

    captured = {}
    monkeypatch.setattr(torch, "set_num_threads", lambda n: captured.setdefault("threads", n))

    run_batch(
        "parakeet-tdt-0.6b-v3", [_fake_utterance(tmp_path)],
        threads=7, download_root=tmp_path / "models",
    )

    assert captured["threads"] == 7


def test_load_runs_a_warm_up_call_before_the_real_utterance(monkeypatch, tmp_path):
    """`_warm_up` (real, measured ~3.4s one-off MPS JIT cost on Apple
    Silicon) must run once, before the timed loop, not leak into
    utterance #1's own processing_time_s. Verified here via call count:
    one extra `generate()` call beyond the single real utterance."""
    monkeypatch.setattr("transformers.AutoProcessor", _FakeProcessor)
    monkeypatch.setattr("transformers.AutoModelForTDT", _FakeModel)
    monkeypatch.setattr(parakeet, "decode_pcm", lambda path, rate, dtype: [0.0])

    generate_calls = []
    original_generate = _FakeModel.generate

    def _counting_generate(self, **kwargs):
        generate_calls.append(kwargs)
        return original_generate(self, **kwargs)

    monkeypatch.setattr(_FakeModel, "generate", _counting_generate)

    run_batch("parakeet-tdt-0.6b-v3", [_fake_utterance(tmp_path)], download_root=tmp_path / "models")

    # One warm-up call (silent dummy audio) + one real call for the single utterance.
    assert len(generate_calls) == 2


# --- _merge_tokens_to_words (pure function; the only genuinely new logic
# run_streaming needs beyond what run_batch already does) ---


def test_merge_tokens_to_words_matches_real_dutch_audio_capture():
    """Real tokens captured directly against real Dutch FLEURS audio
    (nvidia/parakeet-tdt-0.6b-v3, fleurs-nl utterance 165998319534607478,
    'Eerst moesten alle staten unaniem instemmen met de artikelen, ...') —
    not a hand-simplified example. Confirms the leading-space convention
    holds for real multi-subword Dutch words, not just the two-token
    English demo in the transformers docs."""
    tokens = [
        {"token": "E", "start": 0.24, "end": 0.56},
        {"token": "erst", "start": 0.56, "end": 0.88},
        {"token": " mo", "start": 1.04, "end": 1.28},
        {"token": "est", "start": 1.28, "end": 1.52},
        {"token": "en", "start": 1.52, "end": 1.76},
        {"token": " alle", "start": 1.76, "end": 2.08},
        {"token": " st", "start": 2.08, "end": 2.32},
        {"token": "aten", "start": 2.32, "end": 2.56},
        {"token": " un", "start": 2.56, "end": 2.72},
        {"token": "an", "start": 2.72, "end": 2.88},
        {"token": "iem", "start": 2.88, "end": 3.04},
        {"token": " inst", "start": 3.04, "end": 3.2},
        {"token": "em", "start": 3.28, "end": 3.44},
        {"token": "men", "start": 3.44, "end": 3.52},
        {"token": " met", "start": 3.52, "end": 3.6},
        {"token": " de", "start": 3.6, "end": 3.76},
        {"token": " ar", "start": 3.76, "end": 3.84},
        {"token": "ti", "start": 3.84, "end": 4.0},
        {"token": "kel", "start": 4.0, "end": 4.08},
        {"token": "en", "start": 4.08, "end": 4.24},
        {"token": ",", "start": 4.24, "end": 4.24},
    ]

    words = _merge_tokens_to_words(tokens)

    assert [w["word"] for w in words] == [
        "Eerst", "moesten", "alle", "staten", "unaniem", "instemmen", "met", "de", "artikelen,",
    ]
    assert words[0]["start"] == 0.24
    assert words[0]["end"] == 0.88
    assert words[-1]["end"] == 4.24  # trailing comma's own end, correctly merged in


def test_merge_tokens_to_words_first_token_starts_a_word_even_without_leading_space():
    words = _merge_tokens_to_words([{"token": "hi", "start": 0.0, "end": 0.5}])
    assert words == [{"word": "hi", "start": 0.0, "end": 0.5}]


def test_merge_tokens_to_words_filters_tokens_that_are_only_whitespace():
    tokens = [{"token": " ", "start": 0.0, "end": 0.0}, {"token": " hi", "start": 0.1, "end": 0.3}]
    words = _merge_tokens_to_words(tokens)
    assert words == [{"word": "hi", "start": 0.1, "end": 0.3}]


def test_merge_tokens_to_words_empty_input_returns_empty():
    assert _merge_tokens_to_words([]) == []


# --- run_streaming (shared bounded-window engine, streaming.py; this
# adapter's own job is just the decode call + token->word merge) ---


def _fake_streaming_deps(monkeypatch):
    monkeypatch.setattr("transformers.AutoProcessor", _FakeProcessor)
    monkeypatch.setattr("transformers.AutoModelForTDT", _FakeModel)
    monkeypatch.setattr(parakeet, "decode_pcm", lambda *a, **k: [0.0] * 32000)  # 2s at 16kHz


def test_run_streaming_produces_merged_words_from_real_captured_tokens(monkeypatch, tmp_path):
    from oesb_runner.adapters.parakeet import run_streaming

    _fake_streaming_deps(monkeypatch)

    traces = run_streaming("parakeet-tdt-0.6b-v3", [_fake_utterance(tmp_path)], chunk_ms=1000)

    assert len(traces) == 1
    assert "hallo" in traces[0].final_text
    assert "wereld" in traces[0].final_text


def test_run_streaming_cuda_backend_hard_fails_when_torch_reports_unavailable(monkeypatch, tmp_path):
    from oesb_runner.adapters.parakeet import run_streaming

    _fake_streaming_deps(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        run_streaming("parakeet-tdt-0.6b-v3", [_fake_utterance(tmp_path)], backend="cuda")


@pytest.mark.slow
def test_run_streaming_transcribes_real_dutch_audio_within_wer_tolerance():
    from oesb_runner.adapters.parakeet import run_streaming

    pack = load_pack(PACK_DIR)
    traces = run_streaming("parakeet-tdt-0.6b-v3", pack.utterances[:3], chunk_ms=1000)

    pairs = []
    for utterance, trace in zip(pack.utterances[:3], traces, strict=True):
        pairs.append((
            normalize("goesb-nl-v1", utterance.reference_text),
            normalize("goesb-nl-v1", trace.final_text),
        ))

    result_wer = wer.compute(pairs)
    total_processing_s = sum(t.processing_time_s for t in traces)
    total_audio_s = sum(u.duration_s for u in pack.utterances[:3])
    result_rtf = rtf.compute(total_processing_s, total_audio_s)

    # Loose bound, same spirit as the batch test: proving the wiring
    # produces real Dutch streaming transcriptions, not pinning an exact
    # accuracy number (bounded-window re-decode is expected to cost some
    # WER relative to batch, same honest tradeoff faster-whisper/whisper.cpp
    # streaming already document).
    assert result_wer < 0.6
    assert result_rtf > 0


# --- run_concurrency (ADR-0012) ---


class _FakeConcurrentModel(_FakeModel):
    """Tracks how many distinct instances actually receive a `generate()`
    call -- proves the harness builds one full model per worker rather
    than (incorrectly) sharing one, which Parakeet's real `generate()` is
    not safe for (confirmed by reading transformers' own generate()
    override for this model -- see run_concurrency's own docstring)."""

    instances_generated_from: ClassVar[set] = set()

    def generate(self, **kwargs):
        _FakeConcurrentModel.instances_generated_from.add(id(self))
        return super().generate(**kwargs)


def _fake_utterances(tmp_path: Path, n: int = 3) -> list[Utterance]:
    return [
        Utterance(
            utterance_id=f"u{i}", audio_path=tmp_path / f"fake{i}.wav",
            reference_text="hallo", duration_s=1.0,
        )
        for i in range(n)
    ]


def _patch_load_with_fake_instances(monkeypatch) -> list:
    """Replaces `_load` itself rather than `transformers.AutoProcessor`/
    `AutoModelForTDT` directly. `run_concurrency` calls `_load` once per
    worker, which is exactly the seam these tests need (does it build N
    genuinely distinct instances, does each worker only ever touch its
    own?) without also re-exercising `_load`'s own AutoProcessor/
    AutoModelForTDT wiring — already covered by the batch/streaming
    tests, and (real, confirmed finding, not assumed) unreliable to
    monkeypatch here: `transformers`' own lazy-module machinery replaces
    `sys.modules["transformers"]` with a distinct module object at some
    point during this test file's session, after this file's own
    module-level `transformers` reference was already captured -- so
    patching either that stale reference *or* a freshly re-fetched
    `sys.modules["transformers"]` at test time was observed to silently
    miss the object `_load()` actually reads its attributes from later.
    Patching `_load` sidesteps that instability entirely."""
    created: list = []

    def _fake_load(model_name, backend, threads, download_root):
        instance = _FakeConcurrentModel()
        created.append(instance)
        return _FakeProcessor(), instance

    monkeypatch.setattr(parakeet, "_load", _fake_load)
    monkeypatch.setattr(parakeet, "decode_pcm", lambda *a, **k: [0.0])
    return created


def test_run_concurrency_builds_one_model_instance_per_worker(monkeypatch, tmp_path):
    from oesb_runner.adapters.parakeet import run_concurrency

    created = _patch_load_with_fake_instances(monkeypatch)

    run_concurrency("parakeet-tdt-0.6b-v3", _fake_utterances(tmp_path), concurrency=4, duration_s=0.02)

    assert len(created) == 4
    # Every instance genuinely distinct -- not the same object constructed
    # once and appended 4 times.
    assert len({id(m) for m in created}) == 4


def test_run_concurrency_each_worker_only_calls_its_own_instance(monkeypatch, tmp_path):
    from oesb_runner.adapters.parakeet import run_concurrency

    _FakeConcurrentModel.instances_generated_from = set()
    created = _patch_load_with_fake_instances(monkeypatch)

    run_concurrency("parakeet-tdt-0.6b-v3", _fake_utterances(tmp_path), concurrency=3, duration_s=0.05)

    created_ids = {id(m) for m in created}
    # Every constructed instance was actually used, and nothing outside
    # that set was ever called -- proves the worker->instance mapping is
    # exactly one-to-one, no accidental sharing or leftover unused models.
    assert _FakeConcurrentModel.instances_generated_from <= created_ids
    assert _FakeConcurrentModel.instances_generated_from


def test_run_concurrency_respects_the_duration_s_deadline(monkeypatch, tmp_path):
    import time

    from oesb_runner.adapters.parakeet import run_concurrency

    _patch_load_with_fake_instances(monkeypatch)

    start = time.perf_counter()
    run_concurrency("parakeet-tdt-0.6b-v3", _fake_utterances(tmp_path), concurrency=2, duration_s=0.1)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0


def test_run_concurrency_returns_calls_with_the_utterances_own_duration(monkeypatch, tmp_path):
    from oesb_runner.adapters.parakeet import run_concurrency

    _patch_load_with_fake_instances(monkeypatch)

    calls = run_concurrency(
        "parakeet-tdt-0.6b-v3", _fake_utterances(tmp_path, n=1), concurrency=1, duration_s=0.02
    )

    assert calls
    assert all(c.audio_duration_s == 1.0 for c in calls)
    assert all(c.processing_time_s >= 0.0 for c in calls)


def test_run_concurrency_cuda_backend_hard_fails_when_torch_reports_unavailable(monkeypatch, tmp_path):
    from oesb_runner.adapters.parakeet import run_concurrency

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        run_concurrency(
            "parakeet-tdt-0.6b-v3", _fake_utterances(tmp_path),
            concurrency=2, duration_s=0.02, backend="cuda",
        )


@pytest.mark.slow
def test_run_concurrency_transcribes_real_dutch_audio(tmp_path):
    """Real end-to-end proof, not just faked instance bookkeeping: real
    audio, real concurrent generate() calls, on this adapter's actual
    reason for existing."""
    from oesb_runner.adapters.parakeet import run_concurrency

    pack = load_pack(PACK_DIR)
    calls = run_concurrency(
        "parakeet-tdt-0.6b-v3", pack.utterances, concurrency=2, duration_s=5,
        download_root=tmp_path / "models",
    )

    assert calls
    assert all(c.processing_time_s > 0.0 for c in calls)
    assert all(c.audio_duration_s > 0.0 for c in calls)
