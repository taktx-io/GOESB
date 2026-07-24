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


def wav_duration_s(path: str | Path) -> float:
    """Duration in seconds, read straight from a WAV file's RIFF chunks —
    no extra dependency, same spirit as `flac_duration_s` below. Parsed by
    hand rather than via the stdlib `wave` module: `wave` only understands
    integer-PCM format tag 1 and raises on anything else, but datasets
    published as 32-bit float WAV (format tag 3 — e.g. FLEURS) are common
    and don't need decoding just to know their length."""
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            raise ValueError(f"not a RIFF/WAV file: {path}")
        f.read(4)  # overall chunk size, unused
        if f.read(4) != b"WAVE":
            raise ValueError(f"not a WAVE file: {path}")

        sample_rate: int | None = None
        block_align: int | None = None
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            chunk_id, chunk_size = header[:4], int.from_bytes(header[4:8], "little")
            if chunk_id == b"fmt ":
                fmt = f.read(chunk_size)
                # channels(2)@2, sample_rate(4)@4, block_align(2)@12
                sample_rate = struct.unpack("<I", fmt[4:8])[0]
                block_align = struct.unpack("<H", fmt[12:14])[0]
            elif chunk_id == b"data":
                if sample_rate is None or block_align is None:
                    raise ValueError(f"WAV data chunk came before fmt chunk: {path}")
                frames = chunk_size / block_align
                return frames / sample_rate
            else:
                f.seek(chunk_size, 1)
            if chunk_size % 2:
                f.seek(1, 1)  # RIFF chunks are word-aligned; skip the pad byte
        raise ValueError(f"WAV file has no data chunk: {path}")


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
