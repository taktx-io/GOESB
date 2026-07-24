"""Auto-download audio for packs with a known, ungated public source.

GOESB never *hosts* audio (privacy-first) — but for open packs built from a
plain-HTTPS, no-account-needed corpus (FLEURS, LibriSpeech dev-clean), the
runner can stream the same public archive the pack was built from and pull
out just the clips its own manifest.jsonl already lists. This is exactly
what scripts/fetch_fleurs_subset.py and scripts/fetch_librispeech_subset.py
do when building a pack from scratch; this module is the shared "just fetch
these already-known filenames" half of that, reused by both the runner
(`goesb run`, auto-fetch) and those scripts (initial pack authoring).

Packs whose audio.source.type isn't one of AUTO_FETCH_SOURCE_TYPES (custom
corpora, Common Voice's consent-gated download, or no declared source at
all) aren't auto-fetchable — audio.source.fetch_instructions is the
always-present fallback for those.
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

FLEURS_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
LIBRISPEECH_BASE_URL = "https://www.openslr.org/resources/12"

AUTO_FETCH_SOURCE_TYPES = frozenset({"fleurs", "librispeech"})


def _stream_extract(
    url: str, wanted_names: set[str], audio_dir: Path, name_filter: Callable[[str], bool]
) -> set[str]:
    """Stream a remote .tar.gz and extract only members whose basename is in
    `wanted_names`, stopping as soon as every one has been found."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    collected: set[str] = set()
    with urllib.request.urlopen(url) as resp:  # nosec B310 - fixed public dataset URL
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not name_filter(member.name) or not member.isfile():
                    continue
                name = Path(member.name).name
                if name not in wanted_names:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                (audio_dir / name).write_bytes(extracted.read())
                collected.add(name)
                if collected == wanted_names:
                    break
    return collected


def fetch_fleurs_audio(params: dict[str, Any], wanted_names: set[str], audio_dir: Path) -> set[str]:
    language = params["language"]
    split = params.get("split", "dev")
    url = f"{FLEURS_BASE_URL}/{language}/audio/{split}.tar.gz"
    return _stream_extract(url, wanted_names, audio_dir, name_filter=lambda _name: True)


def fetch_librispeech_audio(params: dict[str, Any], wanted_names: set[str], audio_dir: Path) -> set[str]:
    speaker, chapter = params["speaker"], params["chapter"]
    split = params.get("split", "dev-clean")
    url = f"{LIBRISPEECH_BASE_URL}/{split}.tar.gz"
    prefix = f"LibriSpeech/{split}/{speaker}/{chapter}/"
    return _stream_extract(url, wanted_names, audio_dir, name_filter=lambda name: name.startswith(prefix))


_PROVIDERS: dict[str, Callable[[dict[str, Any], set[str], Path], set[str]]] = {
    "fleurs": fetch_fleurs_audio,
    "librispeech": fetch_librispeech_audio,
}


def auto_fetch_audio(source: dict[str, Any], wanted_names: set[str], audio_dir: Path) -> set[str] | None:
    """Returns the set of filenames actually fetched, or None if `source`
    isn't one the runner knows how to auto-fetch — the caller should fall
    back to printing source.get("fetch_instructions") in that case."""
    provider = _PROVIDERS.get(source.get("type"))
    if provider is None:
        return None
    return provider(source.get("params", {}), wanted_names, audio_dir)
