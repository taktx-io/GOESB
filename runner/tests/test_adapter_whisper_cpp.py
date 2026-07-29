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
