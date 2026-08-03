from pathlib import Path
from typing import ClassVar

import pytest

from oesb_runner.normalization import normalize
from oesb_runner.pack import Utterance, load_pack

pywhispercpp = pytest.importorskip(
    "pywhispercpp", reason="requires `pip install goesb-runner[whisper-cpp]`"
)

from oesb_runner.adapters import whisper_cpp
from oesb_runner.adapters.whisper_cpp import run_batch
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

    Proves the third runtime adapter interface (docs/03-roadmap.md M2:
    adapters swap without core changes) — same shape as
    test_adapter_faster_whisper.py's batch test, different runtime.
    """
    pack = load_pack(PACK_DIR)
    transcriptions = run_batch("base.en", pack.utterances, download_root=tmp_path / "models")
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

    # whisper.cpp base.en on clean read speech: loose bound, just proving the wiring.
    assert result_wer < 0.25
    assert result_rtf < 1.0  # faster than realtime even on CPU


class _FakeSegment:
    text = "fake hypothesis"


class _FakeModel:
    """Stands in for pywhispercpp.model.Model — captures the kwargs it was
    constructed with instead of loading a real ggml model. system_info_return
    defaults to a CUDA-capable build's string so every existing test that
    doesn't care about the CUDA-support check keeps passing unchanged;
    tests exercising that check override it via monkeypatch."""

    last_init_kwargs: ClassVar[dict] = {}
    # Realistic ggml dynamic-backend-registry format (confirmed against
    # upstream ggml-cuda.cu/whisper.cpp source) — CUDA appears as its own
    # "CUDA : ..." section, never a flat "CUDA = 1" pair.
    system_info_return: ClassVar[str] = "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : ARCHS = 89 | "

    def __init__(self, *_args, **kwargs):
        _FakeModel.last_init_kwargs = kwargs

    def transcribe(self, *_args, **_kwargs):
        return [_FakeSegment()]

    @staticmethod
    def system_info():
        return _FakeModel.system_info_return


def _fake_utterance(tmp_path: Path) -> Utterance:
    return Utterance(
        utterance_id="u1", audio_path=tmp_path / "fake.wav",
        reference_text="hola", duration_s=1.0,
    )


def test_run_batch_passes_language_to_the_model(monkeypatch, tmp_path):
    """Regression test: whisper.cpp's own default `language` is a hardcoded
    "en" (not auto-detect, confirmed by reading pywhispercpp's own
    docstring) — a profile's declared language must actually reach the
    model, or non-English audio gets decoded as if it were English and
    produces fluent-but-wrong hallucinated English text instead of a real
    transcription (this is exactly what happened before this fix)."""
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    run_batch("tiny", [_fake_utterance(tmp_path)], language="es")

    assert _FakeModel.last_init_kwargs.get("language") == "es"
    assert "detect_language" not in _FakeModel.last_init_kwargs


def test_run_batch_detects_language_when_none_given(monkeypatch, tmp_path):
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    run_batch("tiny", [_fake_utterance(tmp_path)], language=None)

    assert _FakeModel.last_init_kwargs.get("detect_language") is True
    assert "language" not in _FakeModel.last_init_kwargs


def test_run_batch_english_only_model_never_tries_to_detect(monkeypatch, tmp_path):
    """Regression test: an English-only (`.en`) ggml model has no other
    language representation to detect against — asking it to `detect_language`
    anyway (confirmed for real: `whisper-base.en` with no language given)
    doesn't fail loudly, it silently "detects" unrelated languages (az, nn,
    be, ...) at ~1% confidence, garbage no one would notice without looking.
    An `.en` model must always just get language="en", never detection."""
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    run_batch("whisper-base.en", [_fake_utterance(tmp_path)], language=None)

    assert _FakeModel.last_init_kwargs.get("language") == "en"
    assert "detect_language" not in _FakeModel.last_init_kwargs


def test_run_batch_passes_use_gpu_false_for_default_cpu_backend(monkeypatch, tmp_path):
    """ADR-0008: leaving context_params unset lets pywhispercpp fall back to
    whisper.cpp's own compiled-in default rather than an explicit choice —
    must always be passed explicitly, even for the default backend."""
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    run_batch("tiny", [_fake_utterance(tmp_path)])

    assert _FakeModel.last_init_kwargs.get("context_params") == {"use_gpu": False}


def test_run_batch_passes_use_gpu_true_for_cuda_backend(monkeypatch, tmp_path):
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    run_batch("tiny", [_fake_utterance(tmp_path)], backend="cuda")

    assert _FakeModel.last_init_kwargs.get("context_params") == {"use_gpu": True}


def test_run_batch_cuda_backend_raises_when_build_has_no_cuda_support(monkeypatch, tmp_path):
    """Real correctness gap this closes: use_gpu is a single boolean
    covering CUDA/Metal/Vulkan/nothing depending purely on how this exact
    binary was compiled — without this check, --backend cuda on a
    Metal-only or CPU-only build would silently run on CPU with no error
    and no indication anything was wrong."""
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])
    monkeypatch.setattr(
        _FakeModel, "system_info_return",
        "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ",
    )
    _FakeModel.last_init_kwargs = {}  # reset — a prior test's leftover state otherwise persists

    with pytest.raises(RuntimeError, match="no cuda support"):
        run_batch("tiny", [_fake_utterance(tmp_path)], backend="cuda")

    # Must fail before ever constructing the model — no partial/wasted work.
    assert _FakeModel.last_init_kwargs == {}


def test_run_batch_passes_use_gpu_true_for_metal_backend(monkeypatch, tmp_path):
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])
    monkeypatch.setattr(
        _FakeModel, "system_info_return",
        "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | ",
    )

    run_batch("tiny", [_fake_utterance(tmp_path)], backend="metal")

    assert _FakeModel.last_init_kwargs.get("context_params") == {"use_gpu": True}


def test_run_batch_metal_backend_raises_when_build_has_no_metal_support(monkeypatch, tmp_path):
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])
    monkeypatch.setattr(
        _FakeModel, "system_info_return",
        "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : ARCHS = 89 | ",
    )
    _FakeModel.last_init_kwargs = {}

    with pytest.raises(RuntimeError, match="no metal support"):
        run_batch("tiny", [_fake_utterance(tmp_path)], backend="metal")

    assert _FakeModel.last_init_kwargs == {}


def test_run_batch_cpu_backend_never_calls_system_info(monkeypatch, tmp_path):
    """The CUDA-support check only makes sense (and only has a real cost —
    system_info() has observed side effects like initializing the Metal
    backend) when --backend cuda was actually requested."""
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])
    calls = []
    monkeypatch.setattr(_FakeModel, "system_info", staticmethod(lambda: calls.append(1) or "irrelevant"))

    run_batch("tiny", [_fake_utterance(tmp_path)])  # default backend="cpu"

    assert calls == []


# --- ADR-0012: run_concurrency (one Model instance per worker) ---


class _FakeConcurrentModel(_FakeModel):
    """Tracks how many separate instances get constructed and how many
    distinct instances actually receive a transcribe() call -- proves the
    harness builds one full Model per worker rather than (incorrectly)
    sharing one, which pywhispercpp's real Model is not safe for (see
    run_concurrency's own docstring)."""

    instances_created: ClassVar[list] = []
    instances_transcribed_from: ClassVar[set] = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _FakeConcurrentModel.instances_created.append(self)

    def transcribe(self, *args, **kwargs):
        _FakeConcurrentModel.instances_transcribed_from.add(id(self))
        return super().transcribe(*args, **kwargs)


def _fake_utterances(tmp_path: Path, n: int = 3) -> list[Utterance]:
    return [
        Utterance(
            utterance_id=f"u{i}", audio_path=tmp_path / f"fake{i}.wav",
            reference_text="hola", duration_s=1.0,
        )
        for i in range(n)
    ]


def test_run_concurrency_builds_one_model_instance_per_worker(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_concurrency

    _FakeConcurrentModel.instances_created = []
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeConcurrentModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    run_concurrency("tiny", _fake_utterances(tmp_path), concurrency=4, duration_s=0.02)

    assert len(_FakeConcurrentModel.instances_created) == 4
    # Every instance genuinely distinct -- not the same object constructed
    # once and appended 4 times.
    assert len({id(m) for m in _FakeConcurrentModel.instances_created}) == 4


def test_run_concurrency_each_worker_only_calls_its_own_instance(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_concurrency

    _FakeConcurrentModel.instances_created = []
    _FakeConcurrentModel.instances_transcribed_from = set()
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeConcurrentModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    run_concurrency("tiny", _fake_utterances(tmp_path), concurrency=3, duration_s=0.05)

    created_ids = {id(m) for m in _FakeConcurrentModel.instances_created}
    # Every constructed instance was actually used, and nothing outside
    # that set was ever called -- proves the worker->instance mapping is
    # exactly one-to-one, no accidental sharing or leftover unused models.
    assert _FakeConcurrentModel.instances_transcribed_from <= created_ids
    assert _FakeConcurrentModel.instances_transcribed_from


def test_run_concurrency_respects_the_duration_s_deadline(monkeypatch, tmp_path):
    import time

    from oesb_runner.adapters.whisper_cpp import run_concurrency

    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    start = time.perf_counter()
    run_concurrency("tiny", _fake_utterances(tmp_path), concurrency=2, duration_s=0.1)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0


def test_run_concurrency_returns_calls_with_the_utterances_own_duration(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_concurrency

    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    calls = run_concurrency("tiny", _fake_utterances(tmp_path, n=1), concurrency=1, duration_s=0.02)

    assert calls
    assert all(c.audio_duration_s == 1.0 for c in calls)
    assert all(c.processing_time_s >= 0.0 for c in calls)


def test_run_concurrency_cuda_backend_raises_when_build_has_no_cuda_support(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_concurrency

    monkeypatch.setattr("pywhispercpp.model.Model", _FakeModel)
    monkeypatch.setattr(_FakeModel, "system_info_return", "WHISPER : COREML = 0 | OPENVINO = 0 | ")
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0])

    with pytest.raises(RuntimeError, match="no cuda support"):
        run_concurrency("tiny", _fake_utterances(tmp_path), concurrency=2, duration_s=0.02, backend="cuda")


def test_cuda_available_true_when_system_info_reports_the_cuda_backend_section():
    """Real report, second bug in the same area: CUDA/Metal/Vulkan register
    through ggml's dynamic backend registry as their own named section
    ("CUDA : <features...>"), not a flat "CUDA = 1" pair the way OPENVINO
    and COREML do — confirmed by reading upstream ggml-cuda.cu/whisper.cpp
    source directly. A "CUDA = 1" check (what an earlier version of this
    function used) matches no real CUDA build's actual output."""
    class _Model:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : ARCHS = 89 | "

    assert whisper_cpp.cuda_available(_Model) is True


def test_cuda_available_true_even_with_no_reported_features():
    """The backend section header appears whether or not the backend
    reports any features at all (e.g. no GGML_CUDA_FORCE_MMQ-style macros
    defined at compile time) — presence of "CUDA :" alone is the signal."""
    class _Model:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : "

    assert whisper_cpp.cuda_available(_Model) is True


def test_cuda_available_false_when_system_info_omits_cuda():
    class _Model:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | "

    assert whisper_cpp.cuda_available(_Model) is False


def test_metal_available_true_when_system_info_reports_the_mtl_backend_section():
    """Confirmed against upstream ggml-metal.cpp: Metal registers under
    the name "MTL", not "METAL" — matches this exact Mac's real output."""
    class _Model:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | "

    assert whisper_cpp.metal_available(_Model) is True


def test_metal_available_false_when_system_info_omits_mtl():
    class _Model:
        @staticmethod
        def system_info():
            return "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : ARCHS = 89 | "

    assert whisper_cpp.metal_available(_Model) is False


# --- run_streaming (shared bounded-window engine, streaming.py; this
# adapter's own job is just the decode call + language resolution) ---


class _FakeStreamWord:
    def __init__(self, text: str, t0: int, t1: int):
        self.text = text
        self.t0 = t0
        self.t1 = t1


class _FakeStreamingModel:
    """Stands in for pywhispercpp.model.Model for run_streaming tests --
    unlike every other adapter call shape, language is resolved once per
    utterance and passed to `transcribe()` per CALL, not baked in at
    construction (see run_streaming's own docstring for the real bug --
    `detect_language=True` at construction time silently returned zero
    segments once combined with word-timestamp params -- this fixed)."""

    last_init_kwargs: ClassVar[dict] = {}
    transcribe_calls: ClassVar[list] = []
    auto_detect_calls: ClassVar[list] = []
    detected_language: ClassVar[tuple] = (("en", 0.99), {"en": 0.99})
    words: ClassVar[list] = [_FakeStreamWord("hello", 0, 50)]

    def __init__(self, *_args, **kwargs):
        _FakeStreamingModel.last_init_kwargs = kwargs

    def transcribe(self, _samples_slice, **kwargs):
        _FakeStreamingModel.transcribe_calls.append(kwargs)
        return list(_FakeStreamingModel.words)

    def auto_detect_language(self, media, **_kwargs):
        _FakeStreamingModel.auto_detect_calls.append(media)
        return _FakeStreamingModel.detected_language

    @staticmethod
    def system_info():
        return "WHISPER : COREML = 0 | OPENVINO = 0 | MTL : EMBED_LIBRARY = 1 | "


def _reset_fake_streaming_model():
    _FakeStreamingModel.last_init_kwargs = {}
    _FakeStreamingModel.transcribe_calls = []
    _FakeStreamingModel.auto_detect_calls = []
    _FakeStreamingModel.words = [_FakeStreamWord("hello", 0, 50)]


def _fake_streaming_deps(monkeypatch):
    monkeypatch.setattr("pywhispercpp.model.Model", _FakeStreamingModel)
    monkeypatch.setattr(whisper_cpp, "decode_pcm", lambda *a, **k: [0.0] * 32000)  # 2s at 16kHz


def test_run_streaming_passes_explicit_language_to_every_decode_call(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_streaming

    _reset_fake_streaming_model()
    _fake_streaming_deps(monkeypatch)

    run_streaming("tiny", [_fake_utterance(tmp_path)], language="es", chunk_ms=1000)

    assert _FakeStreamingModel.transcribe_calls
    assert all(c.get("language") == "es" for c in _FakeStreamingModel.transcribe_calls)
    assert _FakeStreamingModel.auto_detect_calls == []


def test_run_streaming_english_only_model_never_detects(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_streaming

    _reset_fake_streaming_model()
    _fake_streaming_deps(monkeypatch)

    run_streaming("whisper-base.en", [_fake_utterance(tmp_path)], language=None, chunk_ms=1000)

    assert all(c.get("language") == "en" for c in _FakeStreamingModel.transcribe_calls)
    assert _FakeStreamingModel.auto_detect_calls == []


def test_run_streaming_detects_language_once_per_utterance_not_once_per_chunk(monkeypatch, tmp_path):
    """Real bug this guards against: detecting language fresh every chunk
    (a sub-second window) would be unreliable even setting aside the
    zero-segment bug -- must resolve once per utterance, from its own
    full audio, and reuse that for every chunk's decode call."""
    from oesb_runner.adapters.whisper_cpp import run_streaming

    _reset_fake_streaming_model()
    _fake_streaming_deps(monkeypatch)

    # 2s of fake audio at chunk_ms=500 -> 4 chunks, so a once-per-chunk
    # bug would show up as 4 auto_detect_language calls, not 1.
    run_streaming("tiny", [_fake_utterance(tmp_path)], language=None, chunk_ms=500)

    assert len(_FakeStreamingModel.auto_detect_calls) == 1
    assert all(c.get("language") == "en" for c in _FakeStreamingModel.transcribe_calls)


def test_run_streaming_filters_empty_text_segments(monkeypatch, tmp_path):
    """whisper.cpp occasionally emits an empty leading Segment
    (t0=0, t1=0, text='') alongside real word segments -- confirmed on
    real audio, not assumed -- must never surface as a "word" with
    duration 0."""
    from oesb_runner.adapters.whisper_cpp import run_streaming

    _reset_fake_streaming_model()
    _fake_streaming_deps(monkeypatch)
    _FakeStreamingModel.words = [_FakeStreamWord("", 0, 0), _FakeStreamWord("hello", 0, 50)]

    traces = run_streaming("tiny", [_fake_utterance(tmp_path)], language="en", chunk_ms=1000)

    assert "hello" in traces[0].final_text
    assert traces[0].final_text.strip() != ""


def test_run_streaming_converts_centiseconds_to_seconds(monkeypatch, tmp_path):
    """whisper.cpp's own native timestamp unit is centiseconds (t1=50 ->
    0.5s), NOT the same unit faster-whisper's Word.start/.end use
    (seconds) -- confirmed by direct measurement on real audio (a
    single word's t1 came back as a small integer like 53, not 0.53).
    Getting this wrong would make every streaming latency metric off by
    100x."""
    from oesb_runner.adapters.whisper_cpp import run_streaming

    _reset_fake_streaming_model()
    _fake_streaming_deps(monkeypatch)
    _FakeStreamingModel.words = [_FakeStreamWord("hi", 100, 150)]  # 1.0s -> 1.5s

    traces = run_streaming("tiny", [_fake_utterance(tmp_path)], language="en", chunk_ms=1000)

    # audio_duration_s is derived from detected word start/end (see
    # streaming.py) -- 100/150 centiseconds misread as raw seconds would
    # blow this metric up by 100x instead of landing at a plausible
    # sub-clip-length value.
    assert 0.0 <= traces[0].audio_duration_s < 2.0


def test_run_streaming_passes_use_gpu_true_for_metal_backend(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_streaming

    _reset_fake_streaming_model()
    _fake_streaming_deps(monkeypatch)

    run_streaming("tiny", [_fake_utterance(tmp_path)], language="en", backend="metal")

    assert _FakeStreamingModel.last_init_kwargs.get("context_params") == {"use_gpu": True}


def test_run_streaming_passes_use_gpu_false_for_default_cpu_backend(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_streaming

    _reset_fake_streaming_model()
    _fake_streaming_deps(monkeypatch)

    run_streaming("tiny", [_fake_utterance(tmp_path)], language="en")

    assert _FakeStreamingModel.last_init_kwargs.get("context_params") == {"use_gpu": False}


def test_run_streaming_metal_backend_raises_when_build_has_no_metal_support(monkeypatch, tmp_path):
    from oesb_runner.adapters.whisper_cpp import run_streaming

    _reset_fake_streaming_model()
    _fake_streaming_deps(monkeypatch)
    monkeypatch.setattr(
        _FakeStreamingModel, "system_info",
        staticmethod(lambda: "WHISPER : COREML = 0 | OPENVINO = 0 | CUDA : ARCHS = 89 | "),
    )

    with pytest.raises(RuntimeError, match="no metal support"):
        run_streaming("tiny", [_fake_utterance(tmp_path)], language="en", backend="metal")

    assert _FakeStreamingModel.last_init_kwargs == {}


@pytest.mark.slow
def test_run_streaming_transcribes_real_audio_within_wer_tolerance():
    """End-to-end proof, real audio, default (cpu) backend for CI
    portability -- same loose-bound convention run_batch's own real-audio
    test uses ("just proving the wiring"), not a performance assertion."""
    from oesb_runner.adapters.whisper_cpp import run_streaming
    from oesb_runner.metrics import rtf, wer

    pack = load_pack(PACK_DIR)
    traces = run_streaming("tiny.en", pack.utterances, chunk_ms=1000)
    by_id = {t.utterance_id: t for t in traces}

    pairs = []
    for utterance in pack.utterances:
        hypothesis = by_id[utterance.utterance_id].final_text
        pairs.append((
            normalize("goesb-en-v1", utterance.reference_text),
            normalize("goesb-en-v1", hypothesis),
        ))

    result_wer = wer.compute(pairs)
    total_processing_s = sum(t.processing_time_s for t in traces)
    result_rtf = rtf.compute(total_processing_s, pack.total_duration_s)

    assert result_wer < 0.3
    assert result_rtf < 5.0
