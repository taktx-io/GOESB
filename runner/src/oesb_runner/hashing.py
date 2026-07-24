"""Content hashing helpers for immutable, verifiable benchmark assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_asset_sha256(data: dict[str, Any], exclude: tuple[str, ...] = ("sha256",)) -> str:
    """Hash of a profile/pack dict's canonical JSON, with e.g. its own `sha256`
    field excluded so the hash can be recomputed and compared against it."""
    canonical = {k: v for k, v in data.items() if k not in exclude}
    return sha256_bytes(json.dumps(canonical, sort_keys=True).encode("utf-8"))


def sha256_dir(path: str | Path) -> str:
    """Fingerprint a directory tree (e.g. a downloaded model snapshot): hash of
    each file's relative path + its own content hash, combined in sorted order."""
    h = hashlib.sha256()
    root = Path(path)
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(file_path.relative_to(root)).encode("utf-8"))
        h.update(sha256_file(file_path).encode("utf-8"))
    return h.hexdigest()


def sha256_module_source(module: Any) -> str:
    """Hash of a Python module's own source file — used to identify exactly
    which reviewed, in-tree adapter/plugin code produced a result (ADR-0004).

    Frozen builds (PyInstaller etc.) never extract plain .py source to disk
    — modules import straight out of the bundled archive, so there's no live
    file to hash. Falls back to a manifest precomputed at build time and
    bundled as package data (see scripts/generate_frozen_adapter_hashes.py,
    run before freezing) — normal and editable installs never need it, since
    a real source file exists on disk either way."""
    import inspect
    import sys

    if getattr(sys, "frozen", False):
        return _frozen_module_hash(module)

    source_file = inspect.getsourcefile(module)
    if source_file is None:
        raise ValueError(f"module has no source file to hash: {module!r}")
    return sha256_file(source_file)


def _frozen_module_hash(module: Any) -> str:
    import json
    from importlib import resources

    manifest = json.loads(
        resources.files("oesb_runner").joinpath("_frozen_adapter_hashes.json").read_text()
    )
    try:
        return manifest[module.__name__]
    except KeyError:
        raise ValueError(
            f"no precomputed hash for {module.__name__!r} in the frozen build's "
            "manifest — regenerate it with scripts/generate_frozen_adapter_hashes.py"
        ) from None
