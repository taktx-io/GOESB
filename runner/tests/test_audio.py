import pytest

# numpy/soundfile are only pulled in by the vosk/whisper-cpp extras (CI's
# base `pip install -e ".[dev]"` doesn't install either) — gate collection
# of this whole module on them the same way test_adapter_vosk.py etc. do,
# rather than importing numpy unconditionally at module top level (that
# broke CI on every OS: bare ModuleNotFoundError before this importorskip
# line ever ran).
numpy = pytest.importorskip("numpy", reason="requires `pip install goesb-runner[vosk]` (for numpy)")
soundfile = pytest.importorskip("soundfile", reason="requires `pip install goesb-runner[vosk]` (for soundfile)")
np = numpy

from oesb_runner.audio import decode_pcm


def test_decode_pcm_int16_is_nonzero_for_a_float32_source_wav(tmp_path):
    """Regression test: soundfile/libsndfile (confirmed with 0.14.0/1.2.2)
    silently returns an all-zero array when asked to convert a 32-bit-float
    WAV (e.g. any FLEURS clip) directly to int16 via `sf.read(dtype="int16")`
    — this fed silent, empty audio into vosk for every FLEURS-sourced pack
    without raising or warning. decode_pcm must never reproduce that: it
    always reads via float64 and does the int16 scaling itself."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    wav_path = tmp_path / "float32_source.wav"
    soundfile.write(wav_path, sine, sample_rate, subtype="FLOAT")
    assert soundfile.info(str(wav_path)).subtype == "FLOAT"  # confirm the fixture is genuinely float32-native

    int16_samples = decode_pcm(wav_path, sample_rate, dtype="int16")
    assert int16_samples.dtype == np.int16
    assert np.count_nonzero(int16_samples) > len(int16_samples) * 0.9  # a sine wave, not silence
    # 0.3 amplitude sine -> expect a peak around 0.3 * 32768 =~ 9830, loosely bounded
    assert 5000 < int(np.abs(int16_samples).max()) < 12000

    float32_samples = decode_pcm(wav_path, sample_rate, dtype="float32")
    assert float32_samples.dtype == np.float32
    assert np.count_nonzero(float32_samples) > len(float32_samples) * 0.9


def test_decode_pcm_rejects_unsupported_dtype(tmp_path):
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)
    wav_path = tmp_path / "silence.wav"
    soundfile.write(wav_path, silence, sample_rate, subtype="FLOAT")

    with pytest.raises(ValueError, match="unsupported dtype"):
        decode_pcm(wav_path, sample_rate, dtype="float64")


def test_decode_pcm_resamples_when_native_rate_differs_from_target(tmp_path):
    """Real report, confirmed by direct measurement: this used to be a
    silent no-op — a 48kHz Common Voice clip fed straight into whisper.cpp
    (16kHz-only, no internal resampling) played back 3x too fast, producing
    fluent-sounding but 100%+ WER hallucinated garbage. Verifies both the
    output length (duration preserved at the new rate) and that the tone's
    actual frequency content survives the resample, not just its length."""
    native_rate = 48000
    target_rate = 16000
    duration_s = 1.0
    freq_hz = 440
    t = np.linspace(0, duration_s, int(native_rate * duration_s), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

    wav_path = tmp_path / "48khz_source.wav"
    soundfile.write(wav_path, sine, native_rate, subtype="FLOAT")
    assert soundfile.info(str(wav_path)).samplerate == native_rate

    samples = decode_pcm(wav_path, target_rate, dtype="float32")

    expected_len = int(native_rate * duration_s) * target_rate // native_rate
    assert abs(len(samples) - expected_len) <= 1

    # A naive "just truncate/stride the raw samples" bug would still change
    # length correctly but corrupt pitch — check the dominant frequency in
    # the resampled signal is still ~440Hz, not e.g. ~1320Hz (what you'd
    # get misinterpreting 48kHz-native samples as a 16kHz stream, the
    # actual failure mode this fixes).
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / target_rate)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - freq_hz) < 5


def test_decode_pcm_no_resample_needed_when_rate_already_matches(tmp_path):
    """Guard against a regression where resampling always runs (and always
    subtly degrades audio) even when the native rate already matches."""
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "already_16k.wav"
    soundfile.write(wav_path, sine, sample_rate, subtype="FLOAT")

    samples = decode_pcm(wav_path, sample_rate, dtype="float32")

    assert len(samples) == sample_rate
