"""Auto-download audio for packs with a known, ungated public source.

GOESB never *hosts* audio (privacy-first) — but for open packs built from a
plain-HTTPS, no-account-needed corpus (FLEURS, LibriSpeech dev-clean), the
runner can stream the same public archive the pack was built from and pull
out just the clips its own manifest.jsonl already lists. This is exactly
what scripts/fetch_fleurs_subset.py and scripts/fetch_librispeech_subset.py
do when building a pack from scratch; this module is the shared "just fetch
these already-known filenames" half of that, reused by both the runner
(`goesb run`, auto-fetch) and those scripts (initial pack authoring).

`mozilla_data_collective` (ADR-0010) is the one auto-fetchable exception to
"ungated": Common Voice via the Mozilla Data Collective platform, gated
behind a personal API key rather than a plain URL. The wizard's
`_preflight_pack_credentials` step resolves that key before this module is
ever called; `fetch_common_voice_audio` below assumes it's already in
`os.environ` by the time it runs.

Packs whose audio.source.type isn't one of AUTO_FETCH_SOURCE_TYPES (other
custom/consent-gated corpora, or no declared source at all) aren't
auto-fetchable — audio.source.fetch_instructions is the always-present
fallback for those.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .remote import CACHE_ROOT

FLEURS_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
LIBRISPEECH_BASE_URL = "https://www.openslr.org/resources/12"

AUTO_FETCH_SOURCE_TYPES = frozenset({"fleurs", "librispeech", "mozilla_data_collective"})


class GatedFetchAuthError(RuntimeError):
    """Raised when a gated source (ADR-0010) rejects the credential used to
    fetch it — a bad, expired, or revoked API key, or a dataset the account
    hasn't been granted access to — as opposed to a network or other
    transient failure. Callers (`cli._resolve_pack_audio`) use this to
    report a clearer message than a generic auto-fetch failure, without
    ever letting the underlying traceback (which could echo request/response
    detail) reach the user."""


def _stream_extract(
    url: str, wanted_names: set[str], audio_dir: Path, name_filter: Callable[[str], bool]
) -> set[str]:
    """Stream a remote .tar.gz and extract only members whose basename is in
    `wanted_names`, stopping as soon as every one has been found."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    collected: set[str] = set()
    with (
        urllib.request.urlopen(url) as resp,  # nosec B310 - fixed public dataset URL
        tarfile.open(fileobj=resp, mode="r|gz") as tar,
    ):
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


def _extract_matching(archive_path: Path, wanted_names: set[str], audio_dir: Path) -> set[str]:
    """Pull just `wanted_names` (matched by basename) out of a downloaded
    .tar.gz/.tgz/.zip archive — the datacollective package hands back the
    whole archive, not an already-filtered extraction, so this is the
    counterpart to `_stream_extract` for a locally-downloaded file rather
    than a streamed remote one."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    collected: set[str] = set()
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if info.is_dir() or name not in wanted_names:
                    continue
                with zf.open(info) as src, (audio_dir / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                collected.add(name)
                if collected == wanted_names:
                    break
        return collected

    with tarfile.open(archive_path, mode="r:*") as tar:
        for member in tar:
            if not member.isfile():
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


def fetch_common_voice_audio(params: dict[str, Any], wanted_names: set[str], audio_dir: Path) -> set[str]:
    """Fetch a Common Voice subset via the Mozilla Data Collective platform
    (ADR-0010) — gated behind a personal API key (`MDC_API_KEY`), unlike
    FLEURS/LibriSpeech above. `datacollective` is a guarded optional
    dependency, imported lazily here, same style as the STT adapters'
    faster_whisper/vosk/pywhispercpp imports — it is not a hard dependency
    of the base install."""
    try:
        from datacollective import download_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datacollective is not installed; run `pip install datacollective`"
        ) from exc

    dataset_id = params["dataset_id"]
    try:
        archive_path = download_dataset(dataset_id, show_progress=False)
    except (PermissionError, ValueError) as exc:
        # PermissionError: MDC API returned 403 (key rejected, or the
        # dataset's terms haven't been accepted on the account behind the
        # key). ValueError: no API key present at all (datacollective reads
        # MDC_API_KEY itself). Both are "the credential is the problem", not
        # a network/dataset-id problem — see GatedFetchAuthError.
        raise GatedFetchAuthError(
            f"Mozilla Data Collective rejected the {dataset_id!r} request: {exc}"
        ) from exc

    return _extract_matching(archive_path, wanted_names, audio_dir)


_PROVIDERS: dict[str, Callable[[dict[str, Any], set[str], Path], set[str]]] = {
    "fleurs": fetch_fleurs_audio,
    "librispeech": fetch_librispeech_audio,
    "mozilla_data_collective": fetch_common_voice_audio,
}


def auto_fetch_audio(source: dict[str, Any], wanted_names: set[str], audio_dir: Path) -> set[str] | None:
    """Returns the set of filenames actually fetched, or None if `source`
    isn't one the runner knows how to auto-fetch — the caller should fall
    back to printing source.get("fetch_instructions") in that case."""
    provider = _PROVIDERS.get(source.get("type"))
    if provider is None:
        return None
    return provider(source.get("params", {}), wanted_names, audio_dir)


def shared_audio_dir(source: dict[str, Any]) -> Path:
    """Where auto-fetched audio for this exact `(source.type,
    source.params)` lives: `~/.goesb/cache/audio/<hash>`, keyed on content
    identity rather than any one pack's directory. `load_pack()` looks up
    audio strictly by the filename each manifest.jsonl entry names — it
    never scans the directory — so every sibling pack whose audio.source
    matches (e.g. every engine/size combo generated for one language, all
    pointing at the same FLEURS split) can point straight at this same
    folder and read the exact same files: nothing to fetch twice, nothing
    to copy or link."""
    canonical = json.dumps(
        {"type": source.get("type"), "params": source.get("params", {})}, sort_keys=True
    )
    key = hashlib.sha256(canonical.encode()).hexdigest()
    return CACHE_ROOT / "audio" / key
