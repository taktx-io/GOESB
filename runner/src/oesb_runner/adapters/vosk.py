"""`vosk` batch runtime adapter (docs/02-architecture.md §4).

Optional dependency (`pip install goesb-runner[vosk]`) — `vosk`/`soundfile`
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
from . import Transcription, log_progress, register

_MODEL_URLS = {
    "vosk-model-small-en-us-0.15":
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "vosk-model-small-es-0.42":
        "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "vosk-model-small-fr-0.22":
        "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "vosk-model-small-de-0.15":
        "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "vosk-model-small-pt-0.3":
        "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "vosk-model-small-nl-0.22":
        "https://alphacephei.com/vosk/models/vosk-model-small-nl-0.22.zip",
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
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(download_root)
    zip_path.unlink()
    return model_dir


@register("vosk", benchmark_type="batch", applied_parameters=frozenset())
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
    language: str | None = None,
    backend: str = "cpu",
) -> list[Transcription]:
    """Transcribe every utterance once, batch-style, and time each call.

    `quantization`/`beam_size`/`temperature`/`vad`/`threads`/`language` are
    accepted for call-shape parity with the other batch adapters
    (docs/03-roadmap.md M2 exit criterion: adapters swap without core
    changes) but unused — vosk's Kaldi decoder has no equivalent tunables
    exposed through its Python API, and each vosk model is already
    per-language (the profile's `model.name` picks the language, not a
    runtime parameter). `backend` (ADR-0008) is accepted for the same
    call-shape-parity reason but unused — this adapter is genuinely
    CPU-only (no `backends` declared at registration means it defaults to
    cpu-only, so the CLI never calls in with anything else)."""
    try:
        import vosk
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "vosk is not installed; run `pip install goesb-runner[vosk]`"
        ) from exc
    vosk.SetLogLevel(-1)  # silence vosk's own stderr logging

    root = Path(download_root) if download_root is not None else Path.home() / ".goesb" / "models" / model_name
    model_dir = _resolve_model_dir(model_name, root)
    model = vosk.Model(model_path=str(model_dir))

    results: list[Transcription] = []
    for i, utterance in enumerate(utterances, start=1):
        samples = decode_pcm(utterance.audio_path, dtype="int16")
        recognizer = vosk.KaldiRecognizer(model, _SAMPLE_RATE)

        start = time.perf_counter()
        recognizer.AcceptWaveform(samples.tobytes())
        hypothesis_text = json.loads(recognizer.FinalResult()).get("text", "")
        elapsed = time.perf_counter() - start

        log_progress(i, len(utterances), utterance.utterance_id, elapsed)
        results.append(Transcription(
            utterance_id=utterance.utterance_id,
            hypothesis_text=hypothesis_text,
            processing_time_s=elapsed,
        ))
    return results
