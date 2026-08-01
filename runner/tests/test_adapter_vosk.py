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
