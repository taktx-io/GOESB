import itertools
from pathlib import Path
from typing import ClassVar

import pytest

from oesb_runner.adapters import get_supported_backends, nemotron
from oesb_runner.adapters.nemotron import (
    _mel_chunks,
    _resolve_language,
    _resolve_model_id,
    _select_streaming_latency,
    _words_with_completion_times,
    run_batch,
    run_streaming,
)
from oesb_runner.normalization import normalize
from oesb_runner.pack import Utterance, load_pack

transformers = pytest.importorskip(
    "transformers", reason="requires `pip install goesb-runner[nemotron]`"
)
torch = pytest.importorskip(
    "torch", reason="requires `pip install goesb-runner[nemotron]`"
)

from oesb_runner.metrics import (
    first_final_latency,
    first_partial_latency,
    partial_stability,
    rtf,
    wer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "packs" / "fleurs-nl"
MODEL = "nemotron-3.5-asr-streaming-0.6b"

# The backend the fake-based tests drive. Arbitrary — the fakes replace the
# model, not the device — but it must be one `_load` will accept, and
# `_patch_transformers` forces its availability probe True so these tests run
# anywhere. Real report: they originally hardcoded "metal" and passed only on
# Apple Silicon; 8 of them failed on the first real CUDA box with
# `--backend metal failed: torch.backends.mps.is_available() is False`.
FAKE_BACKEND = "cuda"


def _real_gpu_backend() -> str | None:
    """Whichever GPU backend this machine genuinely has, for the slow
    real-audio tests — "cuda" on an NVIDIA box, "metal" on Apple Silicon,
    None otherwise. Deliberately not hardcoded to either: this engine is
    GPU-only and both of its backends are real, so a test pinned to one of
    them silently stops covering the other's hardware."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "metal"
    return None


requires_gpu = pytest.mark.skipif(
    _real_gpu_backend() is None,
    reason="nemotron is GPU-only (ADR-0013 §4): requires CUDA or Apple Silicon MPS",
)

requires_pack = pytest.mark.skipif(
    not (PACK_DIR / "audio").exists(),
    reason="requires fetched audio: run scripts/fetch_fleurs_subset.py --language nl_nl first",
)


def _fake_utterance(tmp_path: Path, index: int = 1) -> Utterance:
    return Utterance(
        utterance_id=f"u{index}", audio_path=tmp_path / f"fake{index}.wav",
        reference_text="hallo wereld", duration_s=1.0,
    )


def _fake_utterances(tmp_path: Path, n: int = 3) -> list[Utterance]:
    return [_fake_utterance(tmp_path, i) for i in range(n)]


# --- pure helpers ---


def test_resolve_model_id_prefixes_nvidia_org():
    assert _resolve_model_id(MODEL) == f"nvidia/{MODEL}"


class _LanguageOnlyProcessor:
    prompt_dictionary: ClassVar[dict] = {"auto": 101, "nl": 16, "nl-NL": 16, "es": 2, "pt-BR": 12}


def test_resolve_language_prefers_the_exact_bcp47_tag():
    assert _resolve_language(_LanguageOnlyProcessor(), "nl-NL") == "nl-NL"
    assert _resolve_language(_LanguageOnlyProcessor(), "pt-BR") == "pt-BR"


def test_resolve_language_falls_back_to_the_base_subtag():
    """GOESB's Spanish profiles declare `es-419`, which this checkpoint's
    prompt dictionary does not carry — confirmed by direct lookup against
    the real checkpoint. Without the fallback the processor raises
    `Unknown language=` and the whole run dies over a regional subtag."""
    assert _resolve_language(_LanguageOnlyProcessor(), "es-419") == "es"


def test_resolve_language_falls_back_to_auto_for_an_unknown_language():
    assert _resolve_language(_LanguageOnlyProcessor(), "xx-YY") == "auto"
    assert _resolve_language(_LanguageOnlyProcessor(), None) == "auto"


def test_words_with_completion_times_merges_subword_tokens_into_words():
    """Real token shape captured against real Dutch FLEURS audio
    (fleurs-nl utterance 165998319534607478) — a word is built from
    leading-space-delimited BPE subwords, exactly as
    `parakeet._merge_tokens_to_words` confirmed for the sibling engine."""
    tokens = [
        {"token": "E", "start": 1.04, "end": 1.12},
        {"token": "er", "start": 1.12, "end": 1.2},
        {"token": "st", "start": 1.52, "end": 1.6},
        {"token": " mu", "start": 2.08, "end": 2.16},
        {"token": "st", "start": 2.32, "end": 2.4},
        {"token": "en", "start": 2.4, "end": 2.48},
        {"token": " alle", "start": 2.56, "end": 2.64},
    ]

    words = _words_with_completion_times(tokens, clip_duration_s=12.36)

    assert [w for w, _ in words] == ["Eerst", "musten", "alle"]
    # "Eerst" is only known-finished when "musten"'s first token lands;
    # the last word is only finished when the clip's audio runs out.
    assert [t for _, t in words] == [2.08, 2.56, 12.36]


def test_words_with_completion_times_handles_no_tokens():
    assert _words_with_completion_times([], clip_duration_s=1.0) == []


# --- streaming latency validation (ADR-0013 §3) ---


class _LatencyOnlyProcessor:
    def __init__(self):
        self.selected = None

    @property
    def supported_streaming_latencies_ms(self):
        return {3: 320, 0: 80, 6: 560, 13: 1120}

    def set_num_lookahead_tokens(self, right):
        self.selected = right


def test_select_streaming_latency_maps_latency_to_right_attention_context():
    processor = _LatencyOnlyProcessor()
    _select_streaming_latency(processor, 560)
    assert processor.selected == 6


def test_select_streaming_latency_rejects_an_unsupported_mode_and_never_snaps():
    """The whole point of ADR-0013 §3's "hard error, never snap": 160 ms
    is on NVIDIA's model card but is NOT one of the four modes this
    checkpoint's own processor reports, and 300 ms sits right next to the
    supported 320. Both must raise rather than quietly run a different
    latency than the one the result will be signed with."""
    for unsupported in (160, 300, 1000):
        processor = _LatencyOnlyProcessor()
        with pytest.raises(ValueError, match="not supported by this checkpoint"):
            _select_streaming_latency(processor, unsupported)
        assert processor.selected is None


# --- ADR-0008 backend contract ---


def test_nemotron_declares_no_cpu_backend_for_any_benchmark_type():
    """ADR-0013 §4's guarantee, at the layer `cli.run` actually enforces
    it: `--backend cpu` against a nemotron profile is rejected before any
    weights load, rather than being a slow success that produces real
    signed results NVIDIA never claims support for."""
    for benchmark_type in ("batch", "streaming", "concurrency"):
        backends = get_supported_backends("nemotron", benchmark_type)
        assert "cpu" not in backends
        assert backends == frozenset({"cuda", "metal"})


def test_run_batch_cpu_backend_hard_fails_before_loading_anything(tmp_path):
    """Belt and braces for the same guarantee: even called directly, past
    the CLI gate, the adapter refuses cpu instead of picking a device."""
    with pytest.raises(RuntimeError, match="GPU-only"):
        run_batch(MODEL, [_fake_utterance(tmp_path)], backend="cpu", download_root=tmp_path / "models")


def test_run_batch_cuda_backend_hard_fails_when_torch_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        run_batch(MODEL, [_fake_utterance(tmp_path)], backend="cuda", download_root=tmp_path / "models")


def test_run_batch_metal_backend_hard_fails_when_torch_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.backends.mps.is_available"):
        run_batch(MODEL, [_fake_utterance(tmp_path)], backend="metal", download_root=tmp_path / "models")


# --- fakes for the wiring tests ---


class _FakeBatchFeature(dict):
    def to(self, *_args, **_kwargs):
        return self


class _FakeDeviceTensor:
    """`run_streaming` calls `.to(device)` on the processor's real
    `prompt_ids` tensor before any faking can intercept it — which would
    need FAKE_BACKEND's device to genuinely exist. Stand in for it so the
    wiring tests stay hardware-independent; nothing in them inspects its
    value."""

    def to(self, *_args, **_kwargs):
        return self


class _FakeFeatureExtractor:
    sampling_rate = 16000
    hop_length = 160
    n_fft = 512
    win_length = 400


class _FakeProcessor:
    """Mirrors the real `Nemotron3_5AsrProcessor` surface this adapter
    touches — including the chunk-geometry properties, so `_mel_chunks`
    runs for real against it."""

    last_call_kwargs: ClassVar[dict] = {}
    decode_return: ClassVar[list] = ["hallo wereld"]
    decode_timestamps_return: ClassVar[list] = [[
        {"token": "hal", "start": 0.0, "end": 0.08},
        {"token": "lo", "start": 0.08, "end": 0.16},
        {"token": " wereld", "start": 0.48, "end": 0.56},
    ]]
    prompt_dictionary: ClassVar[dict] = {"auto": 101, "nl": 16, "nl-NL": 16}

    def __init__(self):
        self.feature_extractor = _FakeFeatureExtractor()
        self.default_num_lookahead_tokens = 3

    @classmethod
    def from_pretrained(cls, model_id, cache_dir=None):
        return cls()

    @property
    def supported_streaming_latencies_ms(self):
        return {3: 320, 0: 80, 6: 560, 13: 1120}

    def set_num_lookahead_tokens(self, right):
        self.default_num_lookahead_tokens = right

    @property
    def num_mel_frames_first_audio_chunk(self):
        return 1 + 8 * self.default_num_lookahead_tokens

    @property
    def num_mel_frames_per_audio_chunk(self):
        return 8 * (self.default_num_lookahead_tokens + 1)

    def __call__(self, samples, sampling_rate=None, language=None, is_streaming=False,
                 is_first_audio_chunk=True, return_tensors=None):
        _FakeProcessor.last_call_kwargs = {
            "sampling_rate": sampling_rate, "language": language,
            "is_streaming": is_streaming, "is_first_audio_chunk": is_first_audio_chunk,
        }
        frames = (
            self.num_mel_frames_first_audio_chunk if is_first_audio_chunk
            else self.num_mel_frames_per_audio_chunk
        )
        return _FakeBatchFeature(
            input_features=torch.zeros(1, frames + 1, 128),
            prompt_ids=_FakeDeviceTensor(),
        )

    def decode(self, sequences, skip_special_tokens=True, durations=None):
        if durations is not None:
            return _FakeProcessor.decode_return, _FakeProcessor.decode_timestamps_return
        return _FakeProcessor.decode_return


class _FakeGenerateOutput:
    sequences: ClassVar[list] = [[1, 2, 3]]
    durations: ClassVar[list] = [[1, 1, 1]]


class _FakeModel:
    def __init__(self):
        self.dtype = torch.float32

    @classmethod
    def from_pretrained(cls, model_id, cache_dir=None):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        features = kwargs.get("input_features")
        if hasattr(features, "__next__"):
            # The real streaming mixin pulls the generator lazily as the
            # decoder consumes each chunk; draining it here is what makes
            # the adapter's per-chunk marks (and `_mel_chunks` itself) run.
            for _chunk in features:
                pass
        return _FakeGenerateOutput()


def _patch_transformers(monkeypatch, samples_len: int = 16000):
    monkeypatch.setattr("transformers.AutoProcessor", _FakeProcessor)
    monkeypatch.setattr("transformers.AutoModelForRNNT", _FakeModel, raising=False)
    # Faking the model doesn't fake the device: `_load` still runs the real
    # torch availability probe for FAKE_BACKEND, which is False on any
    # machine without that particular accelerator. The genuine probe is
    # covered by the dedicated hard-fail tests above; here it just has to
    # not gate wiring tests on what hardware happens to be present.
    monkeypatch.setitem(nemotron._BACKEND_AVAILABLE_CHECK, FAKE_BACKEND, lambda _torch: True)
    monkeypatch.setattr(nemotron, "decode_pcm", lambda *a, **k: __import__("numpy").zeros(samples_len, dtype="float32"))


# --- _mel_chunks geometry ---


def test_mel_chunks_slices_to_exactly_the_required_frame_counts():
    """The library's `_validate_stream_chunk` raises on any chunk of the
    wrong length, and its own `num_samples_*_audio_chunk` helpers are off
    by one frame for this checkpoint (see `_mel_chunks`' docstring) — so
    this asserts the frame counts the adapter actually hands over, not the
    sample counts it asks for."""
    import numpy as np

    processor = _FakeProcessor()
    chunks = list(_mel_chunks(processor, np.zeros(16000, dtype="float32")))

    assert chunks[0][0].shape[1] == processor.num_mel_frames_first_audio_chunk
    assert all(c.shape[1] == processor.num_mel_frames_per_audio_chunk for c, _ in chunks[1:])


def test_mel_chunks_advance_matches_the_declared_streaming_latency():
    """Each chunk advances the stream by `subsampling * (right + 1)` mel
    frames, i.e. exactly `(right + 1) * encoder_frame_ms` of audio — the
    same number as the mode's `streaming_latency_ms`. 320 ms per chunk at
    the 320 ms mode is a property worth pinning: it's what makes
    `chunk_end_s` in the trace mean what the profile says it means."""
    import numpy as np

    processor = _FakeProcessor()
    chunks = list(_mel_chunks(processor, np.zeros(16000 * 3, dtype="float32")))
    advances = [round(b - a, 6) for (_, a), (_, b) in itertools.pairwise(chunks) if b < 3.0]

    assert advances
    assert all(advance == 0.32 for advance in advances)


@pytest.mark.slow
def test_mel_chunks_reproduce_the_offline_mel_pass_on_real_audio():
    """The chunk geometry doesn't merely validate — it must produce the
    same features a single full-utterance pass would, or the encoder is
    quietly fed wrong input at every chunk seam. Checked at every one of
    the checkpoint's four supported modes. Only the clip's very last frame
    differs materially, a zero-padding tail artefact the offline path has
    too; the bound below is float32 STFT noise (measured: max 3.8e-3 across
    ~1240 interior frames, mean 5.6e-7), not slack for a real mismatch — a
    wrong chunk offset shows up as a difference of order 10, not 1e-3."""
    import numpy as np
    import soundfile as sf
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(_resolve_model_id(MODEL))
    pack = load_pack(PACK_DIR)
    samples, _rate = sf.read(pack.utterances[0].audio_path, dtype="float32")
    offline = processor(np.asarray(samples), sampling_rate=16000, return_tensors="pt")["input_features"]

    for latency_ms in (80, 320, 560, 1120):
        _select_streaming_latency(processor, latency_ms)
        stitched = torch.cat([c for c, _ in _mel_chunks(processor, np.asarray(samples))], dim=1)
        overlap = offline.shape[1] - 1  # exclude the padded tail frame
        difference = (stitched[:, :overlap] - offline[:, :overlap]).abs().max().item()
        assert difference < 1e-2, f"{latency_ms}ms mode diverges from the offline pass"


# --- run_batch wiring ---


def test_run_batch_passes_the_resolved_language_into_the_processor(monkeypatch, tmp_path):
    """Unlike parakeet, `language` is a real model input here (`prompt_ids`)
    — so it has to actually reach the processor, and as a key the
    checkpoint carries."""
    _patch_transformers(monkeypatch)

    results = run_batch(
        MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND,
        language="nl-NL", download_root=tmp_path / "models",
    )

    assert results[0].hypothesis_text == "hallo wereld"
    assert _FakeProcessor.last_call_kwargs["language"] == "nl-NL"
    assert _FakeProcessor.last_call_kwargs["sampling_rate"] == 16000


def test_run_batch_applies_threads_despite_being_a_gpu_only_engine(monkeypatch, tmp_path):
    """ADR-0009 §2: a profile may only declare `threads` overridable if the
    adapter genuinely applies it. It does — mel feature extraction is
    `torch.stft` on the CPU device and runs inside every timed call — so
    unlike `parakeet._load`, which only sets it for `backend="cpu"`, this one
    sets it unconditionally. Without that the declaration would be a silent
    knob."""
    _patch_transformers(monkeypatch)

    captured = {}
    monkeypatch.setattr(torch, "set_num_threads", lambda n: captured.setdefault("threads", n))

    run_batch(MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND, threads=7,
              download_root=tmp_path / "models")

    assert captured["threads"] == 7


def test_run_batch_runs_a_warm_up_call_before_the_timed_loop(monkeypatch, tmp_path):
    """Measured 2.22s first call vs 0.09s for the identical second one on
    Apple Silicon MPS — that one-off JIT/kernel-compile cost must not land
    inside utterance #1's own processing_time_s."""
    _patch_transformers(monkeypatch)

    calls = []
    original = _FakeModel.generate
    monkeypatch.setattr(_FakeModel, "generate", lambda self, **kw: (calls.append(kw), original(self, **kw))[1])

    run_batch(MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND, download_root=tmp_path / "models")

    assert len(calls) == 2  # one warm-up on silence + one real utterance


# --- run_streaming wiring ---


def test_run_streaming_never_calls_the_shared_bounded_window_helper(monkeypatch, tmp_path):
    """ADR-0013's central claim in test form: this engine has a real
    incremental path, so routing it through
    `streaming.run_windowed_local_agreement_streaming` would throw away the
    entire reason for the adapter. Guards against a future "refactor to
    share code with parakeet"."""
    _patch_transformers(monkeypatch)

    def _explode(*_args, **_kwargs):
        raise AssertionError("nemotron must not use the bounded-window re-decode path")

    monkeypatch.setattr("oesb_runner.streaming.run_windowed_local_agreement_streaming", _explode)

    traces = run_streaming(MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND, streaming_latency_ms=320)

    assert len(traces) == 1
    assert traces[0].final_text == "hallo wereld"


def test_run_streaming_commits_every_emitted_word(monkeypatch, tmp_path):
    """A streaming RNNT emits left-to-right and never revises, so
    `committed_word_count` is every whole word published so far — no
    local-agreement approximation, unlike the re-decode engines."""
    _patch_transformers(monkeypatch)

    traces = run_streaming(MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND, streaming_latency_ms=320)

    for update in traces[0].updates:
        assert update.committed_word_count == len(update.text.split())


def test_run_streaming_partials_are_monotonic_and_perfectly_stable(monkeypatch, tmp_path):
    """The two metric consequences ADR-0013 §5 commits to, asserted rather
    than asserted-in-prose: every partial extends the previous one word for
    word (nothing is ever rewritten), so `partial_stability` is exactly 1.0."""
    _patch_transformers(monkeypatch)

    traces = run_streaming(MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND, streaming_latency_ms=320)

    previous: list[str] = []
    for update in traces[0].updates:
        words = update.text.split()
        assert words[:len(previous)] == previous
        previous = words
    assert partial_stability.compute(traces) == 1.0


def test_run_streaming_rejects_an_unsupported_latency_before_loading_weights(monkeypatch, tmp_path):
    _patch_transformers(monkeypatch)

    with pytest.raises(ValueError, match="not supported by this checkpoint"):
        run_streaming(MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND, streaming_latency_ms=160)


def test_run_streaming_ignores_chunk_ms(monkeypatch, tmp_path):
    """`cli.py` passes one fixed kwarg set to every streaming adapter, so
    this one receives `chunk_ms` and must demonstrably do nothing with it
    — the chunk size is fixed by the checkpoint's geometry."""
    _patch_transformers(monkeypatch)

    at_250 = run_streaming(MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND,
                           streaming_latency_ms=320, chunk_ms=250)
    at_2000 = run_streaming(MODEL, [_fake_utterance(tmp_path)], backend=FAKE_BACKEND,
                            streaming_latency_ms=320, chunk_ms=2000)

    assert len(at_250[0].updates) == len(at_2000[0].updates)
    assert at_250[0].final_text == at_2000[0].final_text


# --- run_concurrency (ADR-0012 shape) ---


class _FakeConcurrentModel(_FakeModel):
    instances_generated_from: ClassVar[set] = set()

    def generate(self, **kwargs):
        _FakeConcurrentModel.instances_generated_from.add(id(self))
        return super().generate(**kwargs)


def _patch_load_with_fake_instances(monkeypatch) -> list:
    """Patches `_load` itself rather than the transformers auto-classes,
    for the same reason `test_adapter_parakeet.py` documents: transformers'
    lazy-module machinery makes attribute patching unreliable here, and
    `_load` is the exact seam these tests need."""
    created: list = []

    def _fake_load(model_name, backend, threads, download_root, language=None, streaming_latency_ms=None):
        instance = _FakeConcurrentModel()
        created.append(instance)
        return _FakeProcessor(), instance, "nl-NL"

    monkeypatch.setattr(nemotron, "_load", _fake_load)
    monkeypatch.setattr(nemotron, "decode_pcm", lambda *a, **k: [0.0])
    return created


def test_run_concurrency_builds_one_model_instance_per_worker(monkeypatch, tmp_path):
    """Nemotron is NOT safe to share one instance across threads — both
    generate() overrides keep per-call state on the instance, and the 3.5
    one even rebinds and deletes `self.get_audio_features` around each
    call. N workers therefore means N full instances (an N-fold VRAM cost),
    not one shared model with a worker-count knob."""
    from oesb_runner.adapters.nemotron import run_concurrency

    created = _patch_load_with_fake_instances(monkeypatch)

    run_concurrency(MODEL, _fake_utterances(tmp_path), concurrency=4, duration_s=0.02, backend=FAKE_BACKEND)

    assert len(created) == 4
    assert len({id(m) for m in created}) == 4


def test_run_concurrency_each_worker_only_calls_its_own_instance(monkeypatch, tmp_path):
    from oesb_runner.adapters.nemotron import run_concurrency

    _FakeConcurrentModel.instances_generated_from = set()
    created = _patch_load_with_fake_instances(monkeypatch)

    run_concurrency(MODEL, _fake_utterances(tmp_path), concurrency=3, duration_s=0.05, backend=FAKE_BACKEND)

    assert _FakeConcurrentModel.instances_generated_from
    assert _FakeConcurrentModel.instances_generated_from <= {id(m) for m in created}


def test_run_concurrency_returns_calls_with_the_utterances_own_duration(monkeypatch, tmp_path):
    from oesb_runner.adapters.nemotron import run_concurrency

    _patch_load_with_fake_instances(monkeypatch)

    calls = run_concurrency(
        MODEL, _fake_utterances(tmp_path, n=1), concurrency=1, duration_s=0.02, backend=FAKE_BACKEND
    )

    assert calls
    assert all(c.audio_duration_s == 1.0 for c in calls)


# --- real audio (slow) ---


@requires_pack
@pytest.mark.slow
@requires_gpu
def test_run_batch_transcribes_real_dutch_audio_within_wer_tolerance(tmp_path):
    """End-to-end proof against real Dutch FLEURS audio — this adapter's
    actual reason for existing (Babbl's realtime Dutch STT hardware
    question). Measured on Apple Silicon MPS: WER 0.131, RTF 0.072x. The
    bound below is loose on purpose: it proves real Dutch comes out, it
    does not pin an accuracy number."""
    pack = load_pack(PACK_DIR)
    transcriptions = run_batch(
        MODEL, pack.utterances, backend=_real_gpu_backend(), language="nl-NL",
        download_root=tmp_path / "models",
    )
    by_id = {t.utterance_id: t for t in transcriptions}
    pairs = [
        (normalize("goesb-nl-v1", u.reference_text), normalize("goesb-nl-v1", by_id[u.utterance_id].hypothesis_text))
        for u in pack.utterances
    ]

    assert wer.compute(pairs) < 0.5
    assert rtf.compute(sum(t.processing_time_s for t in transcriptions), pack.total_duration_s) > 0


@requires_pack
@pytest.mark.slow
@requires_gpu
def test_run_streaming_on_real_audio_collapses_first_partial_and_first_final_latency(tmp_path):
    """ADR-0013 §5's second documented consequence, on real audio rather
    than fakes: because every published word is already final, the first
    non-empty partial IS the first finalized word — the two latency metrics
    are identical, not merely close. A leaderboard that ranks them as
    separate columns across all five engines is comparing different things
    (docs/specs/metrics.md)."""
    pack = load_pack(PACK_DIR)
    traces = run_streaming(
        MODEL, pack.utterances[:3], backend=_real_gpu_backend(), language="nl-NL",
        streaming_latency_ms=320, download_root=tmp_path / "models",
    )

    assert first_partial_latency.compute(traces) == first_final_latency.compute(traces)
    assert partial_stability.compute(traces) == 1.0
    pairs = [
        (normalize("goesb-nl-v1", u.reference_text), normalize("goesb-nl-v1", t.final_text))
        for u, t in zip(pack.utterances[:3], traces, strict=True)
    ]
    assert wer.compute(pairs) < 0.6
