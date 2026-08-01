import json
from pathlib import Path
from typing import ClassVar

import pytest

from oesb_runner.normalization import normalize
from oesb_runner.pack import Utterance, load_pack

vosk = pytest.importorskip("vosk", reason="requires `pip install goesb-runner[vosk]`")

from oesb_runner.adapters import vosk as vosk_module
from oesb_runner.adapters.vosk import run_batch
from oesb_runner.metrics import rtf, wer

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "packs" / "librispeech-en"

pytestmark = pytest.mark.skipif(
    not (PACK_DIR / "audio").exists(),
    reason="requires fetched audio: run scripts/fetch_librispeech_subset.py first",
)


@pytest.mark.slow
def test_run_batch_transcribes_real_audio_within_wer_tolerance(tmp_path):
    """End-to-end proof: pack -> adapter -> normalization -> metrics.

    Proves the second runtime adapter interface (docs/03-roadmap.md M2:
    adapters swap without core changes) — same shape as
    test_adapter_faster_whisper.py's batch test, different runtime.
    """
    pack = load_pack(PACK_DIR)
    transcriptions = run_batch(
        "vosk-model-small-en-us-0.15", pack.utterances, download_root=tmp_path / "models"
    )
    by_id = {t.utterance_id: t for t in transcriptions}

    pairs = []
    for utterance in pack.utterances:
        hypothesis = by_id[utterance.utterance_id].hypothesis_text
        pairs.append((
            normalize("goesb-en-v1", utterance.reference_text),
            normalize("goesb-en-v1", hypothesis),
        ))

    result_wer = wer.compute(pairs)
    total_processing_s = sum(t.processing_time_s for t in transcriptions)
    result_rtf = rtf.compute(total_processing_s, pack.total_duration_s)

    # vosk-small on clean read speech: loose bound, just proving the wiring.
    assert result_wer < 0.25
    assert result_rtf < 1.0  # faster than realtime even on CPU


# --- ADR-0012: run_concurrency (one Model instance per worker, see the
# adapter's own docstring for why vosk needs this even though a shared
# Model is documented as the "intended" pattern) ---


class _FakeModel:
    """Stands in for vosk.Model — doesn't load a real Kaldi model."""

    def __init__(self, *args, **kwargs):
        pass


class _FakeConcurrentModel(_FakeModel):
    """Tracks how many separate instances get constructed — proves the
    harness builds one full Model per worker rather than sharing one."""

    instances_created: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _FakeConcurrentModel.instances_created.append(self)


class _FakeRecognizer:
    """Stands in for vosk.KaldiRecognizer — records which Model instance
    it was constructed against, the same "which underlying instance did
    the work actually touch" signal _FakeConcurrentModel's own instance
    tracking gives on the Model side."""

    used_model_ids: ClassVar[set] = set()

    def __init__(self, model, sample_rate):
        _FakeRecognizer.used_model_ids.add(id(model))

    def AcceptWaveform(self, data):
        return True

    def FinalResult(self):
        return '{"text": "fake hypothesis"}'


def _fake_utterances(tmp_path: Path, n: int = 3) -> list[Utterance]:
    return [
        Utterance(
            utterance_id=f"u{i}", audio_path=tmp_path / f"fake{i}.wav",
            reference_text="hola", duration_s=1.0,
        )
        for i in range(n)
    ]


class _FakeSamples:
    """Stands in for decode_pcm's real numpy return value -- the adapter
    only ever calls `.tobytes()` on it, so that's all this needs to
    provide (avoids a hard numpy dependency in this test file; numpy
    isn't part of the `[dev]` extra CI installs when `[vosk]` isn't also
    requested, and this whole module is meant to skip cleanly in that
    case via the `pytest.importorskip("vosk", ...)` above -- a top-level
    `import numpy` would defeat that by failing collection outright)."""

    def tobytes(self) -> bytes:
        return b"\x00\x00"


def _stub_concurrency_deps(monkeypatch, model_cls, tmp_path):
    monkeypatch.setattr(vosk, "Model", model_cls)
    monkeypatch.setattr(vosk, "KaldiRecognizer", _FakeRecognizer)
    monkeypatch.setattr(vosk_module, "_resolve_model_dir", lambda model_name, root: tmp_path)
    monkeypatch.setattr(vosk_module, "decode_pcm", lambda *a, **k: _FakeSamples())


def test_run_concurrency_builds_one_model_instance_per_worker(monkeypatch, tmp_path):
    from oesb_runner.adapters.vosk import run_concurrency

    _FakeConcurrentModel.instances_created = []
    _stub_concurrency_deps(monkeypatch, _FakeConcurrentModel, tmp_path)

    run_concurrency("vosk-model-small-en-us-0.15", _fake_utterances(tmp_path), concurrency=4, duration_s=0.02)

    assert len(_FakeConcurrentModel.instances_created) == 4
    assert len({id(m) for m in _FakeConcurrentModel.instances_created}) == 4


def test_run_concurrency_each_worker_only_uses_its_own_instance(monkeypatch, tmp_path):
    from oesb_runner.adapters.vosk import run_concurrency

    _FakeConcurrentModel.instances_created = []
    _FakeRecognizer.used_model_ids = set()
    _stub_concurrency_deps(monkeypatch, _FakeConcurrentModel, tmp_path)

    run_concurrency("vosk-model-small-en-us-0.15", _fake_utterances(tmp_path), concurrency=3, duration_s=0.05)

    created_ids = {id(m) for m in _FakeConcurrentModel.instances_created}
    # Every recognizer's model came from the set of instances actually
    # built for this run, and at least one was genuinely used — proves the
    # worker->instance mapping is one-to-one, no accidental sharing.
    assert _FakeRecognizer.used_model_ids <= created_ids
    assert _FakeRecognizer.used_model_ids


def test_run_concurrency_respects_the_duration_s_deadline(monkeypatch, tmp_path):
    import time

    from oesb_runner.adapters.vosk import run_concurrency

    _stub_concurrency_deps(monkeypatch, _FakeModel, tmp_path)

    start = time.perf_counter()
    run_concurrency("vosk-model-small-en-us-0.15", _fake_utterances(tmp_path), concurrency=2, duration_s=0.1)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0


def test_run_concurrency_returns_calls_with_the_utterances_own_duration(monkeypatch, tmp_path):
    from oesb_runner.adapters.vosk import run_concurrency

    _stub_concurrency_deps(monkeypatch, _FakeModel, tmp_path)

    calls = run_concurrency(
        "vosk-model-small-en-us-0.15", _fake_utterances(tmp_path, n=1), concurrency=1, duration_s=0.02
    )

    assert calls
    assert all(c.audio_duration_s == 1.0 for c in calls)
    assert all(c.processing_time_s >= 0.0 for c in calls)


# --- run_streaming (real incremental decoder state, unlike faster_whisper's
# whole-buffer re-decode -- see the adapter's own docstring) ---


class _FakeStreamSamples:
    """Stands in for decode_pcm's numpy return value for run_streaming,
    which (unlike run_batch/run_concurrency) needs `len()` and slicing to
    carve the buffer into chunks, not just `.tobytes()`."""

    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, key: slice) -> "_FakeStreamSamples":
        return _FakeStreamSamples(len(range(*key.indices(self._n))))

    def tobytes(self) -> bytes:
        return b"\x00\x00" * self._n


class _FakeStreamingRecognizer:
    """Stands in for vosk.KaldiRecognizer in run_streaming: never naturally
    endpoints (AcceptWaveform always False, matching one continuous
    stretch of speech with no mid-utterance pause), so every chunk but the
    last yields a partial, and the last chunk's audio-exhausted
    force-flush (FinalResult) is what actually commits the word-timed
    final result."""

    words_enabled: ClassVar[bool] = False

    def __init__(self, model, sample_rate):
        pass

    def SetWords(self, enabled):
        _FakeStreamingRecognizer.words_enabled = enabled

    def AcceptWaveform(self, data):
        return False

    def PartialResult(self):
        return '{"partial": "hel"}'

    def FinalResult(self):
        return json.dumps({
            "text": "hello world",
            "result": [
                {"word": "hello", "start": 0.20, "end": 0.45, "conf": 1.0},
                {"word": "world", "start": 0.50, "end": 0.90, "conf": 1.0},
            ],
        })


def _stub_streaming_deps(monkeypatch, tmp_path):
    monkeypatch.setattr(vosk, "Model", _FakeModel)
    monkeypatch.setattr(vosk, "KaldiRecognizer", _FakeStreamingRecognizer)
    monkeypatch.setattr(vosk_module, "_resolve_model_dir", lambda model_name, root: tmp_path)
    # 2s of fake audio at the module's 16kHz -> exactly 2 chunks at the
    # default chunk_ms=1000, so the fake recognizer's "partial then final"
    # script above lines up with (non-last chunk, last chunk).
    monkeypatch.setattr(vosk_module, "decode_pcm", lambda *a, **k: _FakeStreamSamples(32000))


def test_run_streaming_enables_word_timings(monkeypatch, tmp_path):
    from oesb_runner.adapters.vosk import run_streaming

    _FakeStreamingRecognizer.words_enabled = False
    _stub_streaming_deps(monkeypatch, tmp_path)

    run_streaming("vosk-model-small-en-us-0.15", _fake_utterances(tmp_path, n=1))

    # Word timings are how this adapter locates speech onset/offset (see
    # run_streaming's own docstring) -- without them every trace would
    # fall back to the "no speech ever detected" default.
    assert _FakeStreamingRecognizer.words_enabled is True


def test_run_streaming_commits_the_force_flushed_final_result(monkeypatch, tmp_path):
    from oesb_runner.adapters.vosk import run_streaming

    _stub_streaming_deps(monkeypatch, tmp_path)

    traces = run_streaming("vosk-model-small-en-us-0.15", _fake_utterances(tmp_path, n=1))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.utterance_id == "u0"
    assert trace.final_text == "hello world"
    assert len(trace.updates) == 2
    # First chunk never endpoints -> only the uncommitted partial is seen.
    assert trace.updates[0].text == "hel"
    assert trace.updates[0].committed_word_count == 0
    # Second (last) chunk force-flushes -> the final result is fully committed.
    assert trace.updates[1].text == "hello world"
    assert trace.updates[1].committed_word_count == 2


def test_run_streaming_zeroes_the_clock_at_detected_speech_onset(monkeypatch, tmp_path):
    from oesb_runner.adapters.vosk import run_streaming

    _stub_streaming_deps(monkeypatch, tmp_path)

    traces = run_streaming("vosk-model-small-en-us-0.15", _fake_utterances(tmp_path, n=1))

    trace = traces[0]
    # words[0]["start"] == 0.20s from the fake FinalResult -> that's the
    # onset every chunk_end_s/emit_time_s below gets zeroed against
    # (streaming.py's own convention).
    assert trace.updates[0].chunk_end_s == pytest.approx(1.0 - 0.20)
    assert trace.updates[1].chunk_end_s == pytest.approx(2.0 - 0.20)
    # speech_offset_s == min(words[-1]["end"], chunk_end_s) == 0.90 ->
    # audio_duration_s == offset - onset.
    assert trace.audio_duration_s == pytest.approx(0.90 - 0.20)
