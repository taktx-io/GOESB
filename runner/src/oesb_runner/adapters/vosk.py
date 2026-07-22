"""`vosk` batch runtime adapter (docs/02-architecture.md §4).

Optional dependency (`pip install oesb-runner[vosk]`) — `vosk`/`soundfile`
are only imported inside `run_batch`, matching the lazy-import pattern used
by every other adapter.
"""
from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from pathlib import Path

from ..audio import decode_pcm
from ..pack import Utterance
from . import Transcription, register

_MODEL_URLS = {
    "vosk-model-small-en-us-0.15":
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
}
_SAMPLE_RATE = 16000


def _resolve_model_dir(model_name: str, download_root: Path) -> Path:
    """Download + unzip the pinned vosk model into `download_root` if not
    already there. `vosk.Model(model_path=...)` needs an already-extracted
    local directory — unlike faster-whisper's HF snapshot download, vosk has
    no built-in fetch-by-pinned-version — so this does it explicitly: a
    declarative fetch of a named, version-pinned asset, not arbitrary code
    (ADR-0004)."""
    model_dir = download_root / model_name
    if model_dir.exists():
        return model_dir
    try:
        url = _MODEL_URLS[model_name]
    except KeyError:
        raise ValueError(
            f"unknown vosk model {model_name!r}; known models: {sorted(_MODEL_URLS)}"
        ) from None

    download_root.mkdir(parents=True, exist_ok=True)
    zip_path = download_root / f"{model_name}.zip"
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310 - fixed, pinned https URL
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(download_root)
    zip_path.unlink()
    return model_dir


@register("vosk", benchmark_type="batch")
def run_batch(
    model_name: str,
    utterances: list[Utterance],
    *,
    quantization: str = "int8",
    beam_size: int = 5,
    temperature: float = 0.0,
    vad: bool = True,
    threads: int = 4,
    download_root: str | Path | None = None,
) -> list[Transcription]:
    """Transcribe every utterance once, batch-style, and time each call.

    `quantization`/`beam_size`/`temperature`/`vad`/`threads` are accepted for
    call-shape parity with the other batch adapters (docs/03-roadmap.md M2
    exit criterion: adapters swap without core changes) but unused — vosk's
    Kaldi decoder has no equivalent tunables exposed through its Python API.
    """
    try:
        import vosk
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "vosk is not installed; run `pip install oesb-runner[vosk]`"
        ) from exc
    vosk.SetLogLevel(-1)  # silence vosk's own stderr logging

    root = Path(download_root) if download_root is not None else Path.home() / ".oesb" / "models" / model_name
    model_dir = _resolve_model_dir(model_name, root)
    model = vosk.Model(model_path=str(model_dir))

    results: list[Transcription] = []
    for utterance in utterances:
        samples = decode_pcm(utterance.audio_path, dtype="int16")
        recognizer = vosk.KaldiRecognizer(model, _SAMPLE_RATE)

        start = time.perf_counter()
        recognizer.AcceptWaveform(samples.tobytes())
        hypothesis_text = json.loads(recognizer.FinalResult()).get("text", "")
        elapsed = time.perf_counter() - start

        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results
