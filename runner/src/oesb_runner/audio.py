"""Minimal audio introspection and decoding.

`flac_duration_s` is dependency-free by design: FLAC duration, parsed straight
from the STREAMINFO metadata block (no ffmpeg/libsndfile required just to know
how long a pack's clips are). `decode_pcm` is the one exception — adapters
that need actual PCM samples (vosk, whisper.cpp) lazy-import `soundfile` for
it, same optional-dependency pattern as the runtime adapters themselves, so
importing this module never requires it.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def flac_duration_s(path: str | Path) -> float:
    """Duration in seconds, read from a FLAC file's STREAMINFO block."""
    with open(path, "rb") as f:
        if f.read(4) != b"fLaC":
            raise ValueError(f"not a FLAC file: {path}")
        while True:
            header = f.read(4)
            if len(header) < 4:
                raise ValueError(f"FLAC file has no STREAMINFO block: {path}")
            is_last = header[0] & 0x80
            block_type = header[0] & 0x7F
            length = int.from_bytes(header[1:4], "big")
            body = f.read(length)
            if block_type == 0:  # STREAMINFO
                # bytes 10..18 of STREAMINFO pack sample_rate(20)/channels(3)/
                # bits_per_sample(5)/total_samples(36) into 64 bits.
                packed = struct.unpack(">Q", body[10:18])[0]
                sample_rate = packed >> 44
                total_samples = packed & 0xF_FFFF_FFFF
                if sample_rate == 0:
                    raise ValueError(f"FLAC STREAMINFO has zero sample rate: {path}")
                return total_samples / sample_rate
            if is_last:
                raise ValueError(f"FLAC file has no STREAMINFO block: {path}")


def decode_pcm(path: str | Path, dtype: str = "int16") -> np.ndarray:
    """Decode an audio file to a 1-D mono PCM array at its native sample rate.

    `dtype` is `"int16"` (what vosk's `AcceptWaveform` expects as raw bytes)
    or `"float32"` (what pywhispercpp's `transcribe()` accepts directly).
    Every GOESB pack shipped so far is already mono at the profile's target
    rate (see each pack's `pack.yaml` `audio.sample_rate_hz`), so this does
    not resample — a documented assumption, not a silent one.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - exercised only without an extra
        raise RuntimeError(
            "soundfile is not installed; run `pip install goesb-runner[vosk]` "
            "or `pip install goesb-runner[whisper-cpp]`"
        ) from exc

    samples, _sample_rate = sf.read(str(path), dtype=dtype, always_2d=False)
    if samples.ndim > 1:  # collapse stereo to mono by averaging channels
        samples = samples.mean(axis=1).astype(samples.dtype)
    return samples
