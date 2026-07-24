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

    int16_samples = decode_pcm(wav_path, dtype="int16")
    assert int16_samples.dtype == np.int16
    assert np.count_nonzero(int16_samples) > len(int16_samples) * 0.9  # a sine wave, not silence
    # 0.3 amplitude sine -> expect a peak around 0.3 * 32768 =~ 9830, loosely bounded
    assert 5000 < int(np.abs(int16_samples).max()) < 12000

    float32_samples = decode_pcm(wav_path, dtype="float32")
    assert float32_samples.dtype == np.float32
    assert np.count_nonzero(float32_samples) > len(float32_samples) * 0.9


def test_decode_pcm_rejects_unsupported_dtype(tmp_path):
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)
    wav_path = tmp_path / "silence.wav"
    soundfile.write(wav_path, silence, sample_rate, subtype="FLOAT")

    with pytest.raises(ValueError, match="unsupported dtype"):
        decode_pcm(wav_path, dtype="float64")
