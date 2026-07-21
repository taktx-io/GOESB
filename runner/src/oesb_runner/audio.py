"""Minimal, dependency-free audio introspection.

Only what the runner needs before it ever hands audio to a runtime adapter:
FLAC duration, parsed straight from the STREAMINFO metadata block (no ffmpeg/
libsndfile required just to know how long a pack's clips are).
"""
from __future__ import annotations

import struct
from pathlib import Path


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
