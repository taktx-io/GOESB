"""GOESB runner command-line interface.

See docs/02-architecture.md and the roadmap for context. `run` implements the
M1 slice: local batch run -> normalized WER/CER/RTF/CPU/RAM -> signed, hashed
result document on disk (docs/03-roadmap.md M1).
"""
from __future__ import annotations

import base64
import difflib
import functools
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import psutil
import questionary
import typer
import yaml
from packaging.version import Version
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.console import Console
from rich.table import Table

from . import __version__, credentials, cuda_runtime
from . import energy as energy_probe
from .adapters import (
    get_adapter,
    get_applied_parameters,
    get_supported_backends,
    registered_adapters,
)
from .audio_sources import (
    AUTO_FETCH_SOURCE_TYPES,
    GatedFetchAuthError,
    MissingDependencyError,
    auto_fetch_audio,
    shared_audio_dir,
)
from .environment import _capture_cpu, _capture_gpu, _run, capture_environment
from .hashing import canonical_asset_sha256, sha256_dir, sha256_module_source
from .identity import (
    Identity,
    clear_identity,
    compute_discriminator,
    load_identity,
    save_identity,
)
from .metrics import (
    cer,
    cpu_ram,
    end_of_speech_latency,
    first_final_latency,
    first_partial_latency,
    gpu_pct,
    partial_stability,
    rtf,
    streaming_responsiveness,
    temperature,
    throughput,
    update_frequency,
    wer,
)
from .metrics import energy as energy_metric
from .normalization import normalize
from .pack import load_pack
from .remote import DEFAULT_API_URL, fetch_pack, fetch_profile
from .schema_validation import (
    unmet_min_runner_version,
    unrecognized_pack_source_type,
    validate_against,
)
from .signing import (
    generate_ephemeral_keypair,
    public_key_bytes_for,
    sign_payload_sha256,
    sign_with_key,
    verify_result_document,
)
from .stats import relative_std, summarize

app = typer.Typer(help="Open Edge Speech Benchmark runner")


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    """Open Edge Speech Benchmark runner. Run with no command for an
    interactive wizard."""
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty():
        # Piped/scripted/CI invocation with no subcommand - show help
        # instead of hanging on a prompt nothing will ever answer.
        typer.echo(ctx.get_help())
        raise typer.Exit()
    _run_wizard()


def _reexec(args: list[str]) -> None:
    """Re-run this same `goesb` invocation as a fresh process with real argv
    — reuses Click's own argument parsing/defaults for whichever subcommand
    the wizard picked instead of duplicating its logic, and streams output
    live instead of capturing it (unlike calling the command function
    in-process)."""
    result = subprocess.run([sys.argv[0], *args], check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def _matching_packs(packs: list[dict], language: str) -> list[dict]:
    """Packs whose language matches a profile's language (ADR-0011), or
    every pack if none do — same fallback the wizard has always used so an
    unmatched profile still lets you pick something rather than
    dead-ending."""
    return [p for p in packs if p["language"] == language] or packs


# Bulk-generated profile ids are a clean <engine>-<size>-<lang>-<benchmark_type>
# grid (e.g. "whisper-medium-en-batch", "vosk-small-es-batch",
# "vosk-small-en-streaming") — confirmed across every profile in the
# official set. Anything not matching this (e.g. a hand-authored profile
# outside the bulk set) just doesn't get a matrix cell. One compiled
# pattern per matrix-shaped benchmark type — concurrency has no language
# axis (see _wizard_run_concurrency's own docstring) so it was never a
# candidate here.
_MATRIX_BENCHMARK_TYPES = ("batch", "streaming")
_MATRIX_ID_RE = {
    benchmark_type: re.compile(
        rf"^(whisper|whispercpp|vosk)-(tiny|base|small|medium|large-v3)-([a-z]{{2}})-{benchmark_type}$"
    )
    for benchmark_type in _MATRIX_BENCHMARK_TYPES
}

_MATRIX_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
_MATRIX_COLUMNS = (
    [("whisper", size) for size in _MATRIX_SIZES]
    + [("whispercpp", size) for size in _MATRIX_SIZES]
    + [("vosk", "small"), ("vosk", "medium")]
)

# faster-whisper's streaming adapter re-decodes the *entire growing*
# buffer every chunk (no bounded window) — unlike vosk's genuinely
# incremental decode, this both runs slower than realtime on modest
# hardware (measured RTF 3.19x on an Apple M1 Pro) and, worse, makes its
# own reported latency numbers dishonest once RTF > 1: `emit_time_s`
# (faster_whisper.py's run_streaming) assumes each chunk starts decoding
# the instant its audio "arrives," ignoring backlog from earlier chunks
# still catching up — so first_final_latency understates what a real
# deployment would experience. Excluded from the wizard's matrix (not
# deleted — still directly runnable via `goesb run` for testing the
# bounded-sliding-window rewrite meant to replace this) until that
# rewrite lands and the numbers are trustworthy again.
_MATRIX_STREAMING_EXCLUDED_PROFILE_IDS = frozenset({"whisper-medium-en-streaming"})


@dataclass
class _Matrix:
    languages: list[str]  # sorted BCP-47 tags, the grid's rows
    columns: list[tuple[str, str]]  # (engine, size) pairs that exist, the grid's columns
    cells: dict[tuple[str, str], str]  # (language, "<engine>-<size>") -> profile_id


def _build_matrix(profiles: list[dict], benchmark_type: str = "batch") -> _Matrix:
    """Groups profiles of one matrix-shaped benchmark type (batch or
    streaming — see `_MATRIX_BENCHMARK_TYPES`) into the wizard's language x
    engine/size grid. A language missing most columns (e.g. the single
    Dutch example profile) just has fewer entries in `cells` — the grid
    renders whatever exists, no "unavailable" placeholders."""
    id_re = _MATRIX_ID_RE[benchmark_type]
    excluded = _MATRIX_STREAMING_EXCLUDED_PROFILE_IDS if benchmark_type == "streaming" else frozenset()
    by_lang: dict[str, dict[str, str]] = {}
    for p in profiles:
        if p["id"] in excluded:
            continue
        match = id_re.match(p["id"])
        if match is None:
            continue
        engine, size, _lang_code = match.groups()
        by_lang.setdefault(p["language"], {})[f"{engine}-{size}"] = p["id"]

    columns = [
        (engine, size) for engine, size in _MATRIX_COLUMNS
        if any(f"{engine}-{size}" in cols for cols in by_lang.values())
    ]
    cells = {
        (language, column_key): profile_id
        for language, cols in by_lang.items()
        for column_key, profile_id in cols.items()
    }
    return _Matrix(languages=sorted(by_lang), columns=columns, cells=cells)


def _matrix_cell_exists(matrix: _Matrix, row: int, col: int) -> bool:
    """row/col are 1-indexed grid positions (0 is reserved for headers)."""
    language = matrix.languages[row - 1]
    engine, size = matrix.columns[col - 1]
    return (language, f"{engine}-{size}") in matrix.cells


def _toggle_selection(selected: set[tuple[int, int]], matrix: _Matrix, row: int, col: int) -> set[tuple[int, int]]:
    """Pure selection-state transition for one space-press at grid position
    (row, col), row 0 / col 0 being the header row/column. A header toggles
    every existing cell in its whole row/column at once — clearing them if
    all are already selected, else selecting all of them. A body cell
    toggles itself alone; toggling a cell with no profile (a gap in a
    sparse row) is a no-op. Returns a new set — callers hold the current
    selection and reassign it, this never mutates its input."""
    if row == 0 and col == 0:
        return selected
    if row == 0:
        column_cells = {(r, col) for r in range(1, len(matrix.languages) + 1) if _matrix_cell_exists(matrix, r, col)}
        return selected - column_cells if column_cells <= selected else selected | column_cells
    if col == 0:
        row_cells = {(row, c) for c in range(1, len(matrix.columns) + 1) if _matrix_cell_exists(matrix, row, c)}
        return selected - row_cells if row_cells <= selected else selected | row_cells
    if not _matrix_cell_exists(matrix, row, col):
        return selected
    return selected - {(row, col)} if (row, col) in selected else selected | {(row, col)}


def _selection_to_profile_ids(selected: set[tuple[int, int]], matrix: _Matrix) -> list[str]:
    """Maps selected (row, col) body cells (headers, row/col 0, are never
    themselves "selected" — toggling one just selects/clears its cells)
    back to profile ids, deduped and sorted."""
    profile_ids: set[str] = set()
    for row, col in selected:
        if row == 0 or col == 0:
            continue
        language = matrix.languages[row - 1]
        engine, size = matrix.columns[col - 1]
        profile_id = matrix.cells.get((language, f"{engine}-{size}"))
        if profile_id is not None:
            profile_ids.add(profile_id)
    return sorted(profile_ids)


_MATRIX_ENGINE_SHORT = {"whisper": "fw", "whispercpp": "wc", "vosk": "vk"}
_MATRIX_SIZE_SHORT = {"tiny": "T", "base": "B", "small": "S", "medium": "M", "large-v3": "L"}
_MATRIX_COLUMN_WIDTH = 7
_MATRIX_ROW_HEADER_WIDTH = 8
_MATRIX_LEGEND = (
    "fw=faster-whisper  wc=whisper-cpp  vk=vosk    T=tiny B=base S=small M=medium L=large-v3\n"
    "Arrows to move, space to toggle a cell/row/column, enter to confirm, escape to go back."
)


def _ask_matrix(matrix: _Matrix) -> list[str] | None:
    """The real 2D grid picker: arrow keys move a cursor over language
    (rows) x engine/size (columns), space toggles whatever's under the
    cursor — a single cell, or (on a header) that whole row/column — enter
    confirms, escape backs out (same as cancelling, since this is the
    batch wizard's first step). Built directly on prompt_toolkit (which
    questionary itself is a thin wrapper over — Question.ask() is just
    `self.application.run()`) since questionary's own prompts have no 2D
    navigation or escape handling to build on."""
    n_rows, n_cols = len(matrix.languages), len(matrix.columns)
    selected: set[tuple[int, int]] = set()
    cursor = [0, 0]  # [row, col]; 0 is the header row/column

    def render() -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []

        def cell(text: str, row: int, col: int) -> None:
            # The real terminal cursor (via the SetCursorPosition sentinel)
            # is enough to show position — no style/color on top of it, so
            # nothing competes with the "[x]" selection marker for attention.
            if cursor == [row, col]:
                tokens.append(("[SetCursorPosition]", ""))
            tokens.append(("", text))

        cell(" " * _MATRIX_ROW_HEADER_WIDTH, 0, 0)
        for c, (engine, size) in enumerate(matrix.columns, start=1):
            label = f"{_MATRIX_ENGINE_SHORT[engine]}-{_MATRIX_SIZE_SHORT[size]}"
            cell(label.center(_MATRIX_COLUMN_WIDTH), 0, c)
        tokens.append(("", "\n"))

        for r, language in enumerate(matrix.languages, start=1):
            cell(language.ljust(_MATRIX_ROW_HEADER_WIDTH), r, 0)
            for c in range(1, n_cols + 1):
                if not _matrix_cell_exists(matrix, r, c):
                    glyph = ""
                elif (r, c) in selected:
                    glyph = "[x]"
                else:
                    glyph = "[ ]"
                cell(glyph.center(_MATRIX_COLUMN_WIDTH), r, c)
            tokens.append(("", "\n"))
        return tokens

    bindings = KeyBindings()

    @bindings.add(Keys.Up, eager=True)
    def _move_up(event) -> None:
        cursor[0] = max(0, cursor[0] - 1)
        event.app.invalidate()

    @bindings.add(Keys.Down, eager=True)
    def _move_down(event) -> None:
        cursor[0] = min(n_rows, cursor[0] + 1)
        event.app.invalidate()

    @bindings.add(Keys.Left, eager=True)
    def _move_left(event) -> None:
        cursor[1] = max(0, cursor[1] - 1)
        event.app.invalidate()

    @bindings.add(Keys.Right, eager=True)
    def _move_right(event) -> None:
        cursor[1] = min(n_cols, cursor[1] + 1)
        event.app.invalidate()

    @bindings.add(" ", eager=True)
    def _toggle(event) -> None:
        nonlocal selected
        selected = _toggle_selection(selected, matrix, cursor[0], cursor[1])
        event.app.invalidate()

    @bindings.add(Keys.ControlM, eager=True)
    def _confirm(event) -> None:
        event.app.exit(result=_selection_to_profile_ids(selected, matrix))

    @bindings.add(Keys.Escape, eager=True)
    def _back(event) -> None:
        event.app.exit(result=None)

    @bindings.add(Keys.ControlC, eager=True)
    @bindings.add(Keys.ControlQ, eager=True)
    def _cancel(event) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    app = Application(
        layout=Layout(Window(FormattedTextControl(render))),
        key_bindings=bindings,
        style=questionary.styles.DEFAULT_STYLE,
        full_screen=False,
    )
    try:
        return app.run()
    except KeyboardInterrupt:
        return None


@functools.cache
def _load_profile_for_wizard(profile_id: str, profiles_dir: str, api_url: str) -> dict | None:
    """Best-effort full profile load for wizard-side preflight steps —
    local dir first, else `fetch_profile`. Memoized per (profile_id,
    profiles_dir, api_url) for the life of this process: `fetch_profile`
    itself now hits the network unconditionally on every call (a later,
    deliberate fix for stale-cache staleness — see remote.py's own
    docstring — that inadvertently broke this function's *own* original
    "second call is a cheap disk read" contract, since a single wizard run
    calls this multiple times for the same profile, e.g. by both
    `_preflight_engines` and `_wizard_engine_parameters`). This restores
    that contract at the right layer: still one fresh network fetch per
    profile per `goesb` process (each `_reexec`'d run gets its own fresh
    fetch, so staleness-across-runs is unaffected), just not one fetch per
    *call* within that same process's own preflight sequence. Returns
    `None` on any network failure so callers can degrade gracefully rather
    than crash the wizard — the per-combo `run()` call surfaces the real
    error properly when it actually runs."""
    profile_path = Path(profiles_dir) / profile_id / "profile.yaml"
    try:
        if profile_path.exists():
            return _load_yaml(profile_path)
        return fetch_profile(profile_id, api_url)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None


@functools.cache
def _load_pack_for_wizard(pack_id: str, packs_dir: str, api_url: str) -> dict | None:
    """Best-effort full pack.yaml load for wizard-side preflight steps — the
    credential check needs audio.source.credential, which _pack_rows'
    lightweight rows (id/visibility/version/profile_id) don't carry. Same
    local-dir-first-else-fetch shape, same per-process memoization, and the
    same "degrade, never crash the wizard" contract as
    `_load_profile_for_wizard` above."""
    pack_path = Path(packs_dir) / pack_id / "pack.yaml"
    try:
        if pack_path.exists():
            return _load_yaml(pack_path)
        return _load_yaml(fetch_pack(pack_id, api_url) / "pack.yaml")
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None


def _preflight_pack_credentials(
    combos: list[tuple[str, str]], packs_dir: str, api_url: str
) -> list[tuple[str, str]] | None:
    """Before the batch loop starts (ADR-0010) — same rationale as
    _preflight_engines, just for gated-pack API credentials instead of
    engine installs: ask once per distinct env_var across the whole batch,
    not once per combo buried hours into an unattended run. A credential
    already resolvable (env or ~/.goesb/credentials.json) is never
    prompted for again. Declining (empty answer) drops only the combos
    needing that env_var — same continue-past-failure spirit
    _preflight_engines already uses. Returns None (the wizard's own abort
    convention) if the prompt itself is cancelled."""
    pack_cache: dict[str, dict | None] = {}

    def _pack(pack_id: str) -> dict | None:
        if pack_id not in pack_cache:
            pack_cache[pack_id] = _load_pack_for_wizard(pack_id, packs_dir, api_url)
        return pack_cache[pack_id]

    credential_by_env_var: dict[str, dict] = {}
    for _profile_id, pack_id in combos:
        pack = _pack(pack_id)
        if pack is None:
            continue
        credential = pack.get("audio", {}).get("source", {}).get("credential")
        if credential:
            credential_by_env_var.setdefault(credential["env_var"], credential)

    unresolved_env_vars: set[str] = set()
    for env_var, credential in credential_by_env_var.items():
        # Truthy, not `is not None`: a blank string in the on-disk store
        # (e.g. a stale entry from before this exact check existed, or any
        # future write path that isn't as careful as this function's own
        # decline-drops-the-value branch below) must never look
        # "resolved" — it would silently skip the prompt forever while the
        # actual auto-fetch keeps failing downstream with a credential
        # error, with no obvious link back to this step. Matches
        # `credentials.load_credential`'s own environ check, which already
        # treats a blank env var the same way.
        existing = credentials.load_credential(env_var)
        if existing:
            # A hit from the on-disk store (as opposed to an env var the
            # user already exported themselves) is data, not an
            # environment mutation — load_credential never touches
            # os.environ. Every downstream consumer trusts os.environ
            # directly (`_reexec`'s subprocess inherits it; audio_sources.
            # fetch_common_voice_audio's own docstring literally assumes
            # it's "already in os.environ by the time it runs"), so
            # skipping the prompt without also exporting here meant a
            # credential saved on run N was silently never usable on run
            # N+1 — every subsequent run failed downstream with a
            # "missing API key" error that gave no hint the credential had,
            # in fact, already been found and considered resolved.
            os.environ[env_var] = existing
            continue  # already set in the environment or saved from a prior run

        typer.echo(credential["instructions"], err=True)
        typer.echo(f"Sign up: {credential['signup_url']}", err=True)
        value = questionary.password(
            f"Paste your {env_var} (leave blank to skip these packs):"
        ).ask()
        if value is None:
            return None  # Ctrl-C/abort — same convention as the rest of the wizard
        value = value.strip()
        if not value:
            unresolved_env_vars.add(env_var)
            continue

        os.environ[env_var] = value  # _reexec's subprocess inherits this
        credentials.save_credential(env_var, value)

    if not unresolved_env_vars:
        return combos

    kept = []
    for profile_id, pack_id in combos:
        pack = _pack(pack_id) or {}
        credential = pack.get("audio", {}).get("source", {}).get("credential")
        if credential and credential["env_var"] in unresolved_env_vars:
            typer.echo(
                f"  {profile_id}  x  {pack_id} — skipping (no {credential['env_var']} credential provided)",
                err=True,
            )
            continue
        kept.append((profile_id, pack_id))
    return kept


def _preflight_engines(
    combos: list[tuple[str, str]], profiles_dir: str, api_url: str
) -> list[tuple[str, str]]:
    """Before the batch loop starts — which re-execs each combo as its own
    fresh subprocess — make sure every distinct engine the batch needs is
    installed, prompting once per engine right now instead of letting each
    subprocess discover it needs a Y/n answer on its own. A batch spanning
    several engines could otherwise stall for however long it takes
    someone to notice a prompt sitting unanswered hours into an
    unattended run. Combos whose engine install is declined or fails are
    dropped and reported, same continue-past-failure spirit as the rest
    of the batch — just resolved up front rather than discovered mid-run."""
    runtime_by_profile: dict[str, str | None] = {}
    for profile_id, _pack_id in combos:
        if profile_id in runtime_by_profile:
            continue
        profile = _load_profile_for_wizard(profile_id, profiles_dir, api_url)
        runtime_by_profile[profile_id] = profile["runtime"]["name"] if profile else None

    unavailable_engines: set[str] = set()
    for runtime_name in sorted({r for r in runtime_by_profile.values() if r is not None}):
        try:
            _ensure_engine_installed(runtime_name)
        except typer.Exit:
            unavailable_engines.add(runtime_name)

    if not unavailable_engines:
        return combos

    kept = []
    for profile_id, pack_id in combos:
        runtime_name = runtime_by_profile.get(profile_id)
        if runtime_name in unavailable_engines:
            typer.echo(f"  {profile_id}  x  {pack_id} — skipping ({runtime_name!r} not installed)", err=True)
            continue
        kept.append((profile_id, pack_id))
    return kept


def _parse_param_sweep(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def _format_combo_label(profile_id: str, pack_id: str, overrides: dict[str, str]) -> str:
    suffix = "   " + "  ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else ""
    return f"{profile_id}  x  {pack_id}{suffix}"


def _profile_param_default(profile: dict, param_name: str) -> Any:
    model_cfg = profile.get("model", {})
    if param_name in model_cfg:
        return model_cfg[param_name]
    return profile.get("configuration", {}).get(param_name)


_SUGGESTED_CONCURRENCY_SWEEP = [1, 4, 8, 16]

# How much extra throughput the next doubling has to buy to be worth
# running — below this, `_run_concurrency_auto_sweep` calls the previous
# level the knee and stops. 15%: comfortably above run-to-run measurement
# noise (the existing FR-5.3 tolerance check flags >15% relative std as
# noteworthy on its own) but well below a real scaling step, which roughly
# doubles throughput until contention sets in.
_CONCURRENCY_PLATEAU_GAIN = 0.15


def _suggested_concurrency_sweep(profile: dict) -> str:
    """An example sweep shown in the wizard's `concurrency` prompt for users
    who want to type their own instead of the auto-detected default.
    Clamped to whatever ceiling this profile's own `overridable.concurrency`
    declares — e.g. whisper-cpp's tighter per-instance memory cost (ADR-0012
    addendum) caps it lower than faster-whisper's shared-model harness."""
    domain = profile.get("overridable", {}).get("concurrency", {})
    ceiling = domain.get("range", {}).get("max")
    levels = [v for v in _SUGGESTED_CONCURRENCY_SWEEP if ceiling is None or v <= ceiling]
    return ",".join(str(v) for v in levels) or "1"


def _concurrency_ceiling(profile_id: str, profiles_dir: str, api_url: str) -> int | None:
    profile = _load_profile_for_wizard(profile_id, profiles_dir, api_url)
    domain = (profile or {}).get("overridable", {}).get("concurrency", {})
    return domain.get("range", {}).get("max")


def _latest_result_throughput(results_dir: str, profile_id: str, pack_id: str) -> float | None:
    """Read back the throughput metric from the result `run` just wrote to
    disk for this (profile, pack) — `_reexec` streams its child's output
    live rather than capturing it, so this is how the auto-sweep's plateau
    decision gets at a level's outcome without re-deriving it. `None` if no
    matching file exists (e.g. the run failed before writing one) or the
    result has no throughput metric — either way the caller can't judge a
    plateau and should stop climbing rather than guess."""
    matches = sorted(
        Path(results_dir).glob(f"{profile_id}__{pack_id}__*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        return None
    result = json.loads(matches[-1].read_text())
    return result.get("metrics", {}).get("throughput", {}).get("value")


def _next_concurrency_level(level: int) -> int:
    """1, 2, 4, 8, then +4 a level: doubling covers the cheap, uninteresting
    low end in a handful of runs, but by the time levels are large enough
    to be near a real knee, doubling's own step size has gotten coarse
    right where resolution matters most (8 -> 16 is already a +100% jump,
    hiding any real ceiling at e.g. 10, 11, or 12 between them). Switching
    to a fixed +4 step once level >= 8 trades that away for real
    resolution near the plateau, at the cost of more levels (and more wall
    clock) to reach a very high ceiling -- an acceptable trade for this
    tool's actual target hardware (edge/local devices, not racks of GPUs;
    every real sweep run so far this session plateaued in the single-to-
    low-double digits)."""
    return level + 4 if level >= 8 else level * 2


def _run_concurrency_auto_sweep(
    profile_id: str,
    pack_id: str,
    repeats: str,
    backend: str,
    hardware_id: str,
    other_overrides: dict[str, str],
    profiles_dir: str = "profiles",
    api_url: str = DEFAULT_API_URL,
    results_dir: str = "runs/results",
) -> list[tuple[str, str, dict[str, str], bool]]:
    """Climbs concurrency (1, 2, 4, 8, 12, 16, ... — see
    `_next_concurrency_level`), reading back each level's throughput to
    decide whether the next level is still worth running — stops once TWO
    CONSECUTIVE levels each buy less than `_CONCURRENCY_PLATEAU_GAIN` more
    throughput than the one before (the "does this hardware stay fast
    under load" knee this benchmark_type exists to find), or the profile's
    own `overridable.concurrency.range.max` ceiling is reached. Two
    consecutive low-gain levels, not just one: a single `duration_s`
    window's throughput reading carries real run-to-run measurement
    noise, and stopping on the first low reading risks calling a false
    plateau from noise alone rather than the hardware's actual ceiling —
    a second low reading in a row confirms it's real. A level that
    recovers back above the gain floor resets the counter; it wasn't a
    plateau, just a noisy dip. Replaces guessing a static sweep with the
    actual number for this hardware — every level explored is still a
    normal, independently-submittable `run` result; only the SET of
    levels to run is decided at runtime instead of upfront. Returns one
    outcome tuple per level explored, same shape
    `_wizard_confirm_and_run`'s manual-sweep path already produces."""
    ceiling = _concurrency_ceiling(profile_id, profiles_dir, api_url)
    outcomes: list[tuple[str, str, dict[str, str], bool]] = []
    prev_throughput: float | None = None
    consecutive_low_gain = 0
    level = 1
    while True:
        overrides = {**other_overrides, "concurrency": str(level)}
        args = ["run", profile_id, pack_id, "--repeats", repeats, "--hardware", hardware_id]
        if backend != "cpu":
            args += ["--backend", backend]
        for key, value in overrides.items():
            args += ["--param", f"{key}={value}"]

        try:
            _reexec(args)
        except typer.Exit:
            outcomes.append((profile_id, pack_id, overrides, False))
            break  # a failed level can't inform whether to keep climbing

        outcomes.append((profile_id, pack_id, overrides, True))
        throughput = _latest_result_throughput(results_dir, profile_id, pack_id)

        if prev_throughput is not None and throughput is not None and prev_throughput > 0:
            gain = (throughput - prev_throughput) / prev_throughput
            typer.echo(
                f"  concurrency={level}: {throughput:.2f} audio-s/s ({gain:+.0%} vs the previous level)",
                err=True,
            )
            if gain < _CONCURRENCY_PLATEAU_GAIN:
                consecutive_low_gain += 1
                if consecutive_low_gain >= 2:
                    typer.echo(
                        f"  plateaued at concurrency={level} — two low-gain levels in a row, "
                        "stopping the auto-sweep",
                        err=True,
                    )
                    break
            else:
                consecutive_low_gain = 0

        if ceiling is not None and level >= ceiling:
            break

        prev_throughput = throughput
        next_level = _next_concurrency_level(level)
        level = min(next_level, ceiling) if ceiling is not None else next_level

    return outcomes


def _wizard_engine_parameters(
    combos: list[tuple[str, str]], profiles_dir: str, api_url: str
) -> list[tuple[str, str, dict[str, str]]] | None:
    """The parameter step (ADR-0009 §3), between engine preflight and the
    repeats prompt: for each engine present in the selection, ask about
    parameters overridable in *all* of that engine's selected profiles —
    grouping is per engine, not per batch, so an engine-specific parameter
    can never leak onto an engine that lacks it (a mixed whisper+vosk
    selection asks the whisper questions and runs vosk cells as-is).
    Enter (empty input) means "use each profile's own default" — no
    --param is appended for that key at all, so a full Enter-through
    reproduces today's behavior byte-for-byte. A single value overrides
    that engine's cells; a comma-separated list (`1,4,8`) sweeps them —
    cells x values, and values x values if more than one parameter is
    swept for the same engine. Returns expanded
    (profile_id, pack_id, param_overrides) triples, or None if the user
    aborts a prompt."""
    profile_ids = {profile_id for profile_id, _pack_id in combos}
    profiles_by_id = {
        profile_id: _load_profile_for_wizard(profile_id, profiles_dir, api_url)
        for profile_id in profile_ids
    }

    combos_by_engine: dict[str | None, list[tuple[str, str]]] = {}
    for profile_id, pack_id in combos:
        profile = profiles_by_id.get(profile_id)
        engine = profile["runtime"]["name"] if profile else None
        combos_by_engine.setdefault(engine, []).append((profile_id, pack_id))

    expanded: list[tuple[str, str, dict[str, str]]] = []
    for engine in sorted(combos_by_engine, key=lambda e: e or ""):
        engine_combos = combos_by_engine[engine]
        if engine is None:
            # Profile couldn't be resolved — pass through untouched, let
            # run() itself surface the real error for this combo.
            expanded.extend((pid, pack, {}) for pid, pack in engine_combos)
            continue

        overridable_sets = [
            set(profiles_by_id[pid].get("overridable", {})) for pid, _pack in engine_combos
        ]
        common_params = set.intersection(*overridable_sets) if overridable_sets else set()
        if not common_params:
            expanded.extend((pid, pack, {}) for pid, pack in engine_combos)
            continue

        sweeps: dict[str, list[str]] = {}
        for param_name in sorted(common_params):
            profile_for_default = profiles_by_id[engine_combos[0][0]]
            if param_name == "concurrency":
                example = _suggested_concurrency_sweep(profile_for_default)
                prompt = (
                    f"[{engine}] concurrency (Enter to auto-detect the useful max, "
                    f"or your own comma list e.g. {example}):"
                )
            else:
                default = _profile_param_default(profile_for_default, param_name)
                prompt = f"[{engine}] {param_name} (default {default}):"
            raw = questionary.text(prompt).ask()
            if raw is None:
                return None
            raw = raw.strip()
            if param_name == "concurrency" and not raw:
                raw = "auto"
            if not raw:
                continue  # Enter: no override for this parameter at all
            if param_name == "concurrency" and raw.lower() == "auto":
                # A sentinel, not a real value — `_run_concurrency_auto_sweep`
                # expands it into concrete levels at run time, once
                # throughput is actually measured, so it never reaches
                # _resolve_one_param's int-range domain check below.
                sweeps[param_name] = ["auto"]
                continue
            sweeps[param_name] = _parse_param_sweep(raw)

        if not sweeps:
            expanded.extend((pid, pack, {}) for pid, pack in engine_combos)
            continue

        # Validate every swept value against every affected profile's own
        # domain now — before run 1, not run 12 of 15 (ADR-0008 error
        # philosophy: explicit, early, never silent).
        for pid, _pack in engine_combos:
            profile = profiles_by_id[pid]
            for param_name, values in sweeps.items():
                for raw_value in values:
                    if param_name == "concurrency" and raw_value == "auto":
                        continue
                    try:
                        _resolve_one_param(profile, param_name, raw_value)
                    except ValueError as exc:
                        typer.echo(f"--param error for {pid}: {exc}", err=True)
                        raise typer.Exit(code=1) from exc

        param_names = list(sweeps)
        for pid, pack in engine_combos:
            for combo_values in product(*(sweeps[p] for p in param_names)):
                expanded.append((pid, pack, dict(zip(param_names, combo_values, strict=True))))

    return expanded


def _choose_packs_for_language(
    language: str, matching_packs: list[dict], packs_dir: str, api_url: str
) -> list[str] | None:
    """A single-match language (rare after ADR-0011 — most languages now
    have several eligible packs) returns immediately, no prompt. When more
    than one pack matches the language (e.g. an ungated FLEURS pack and a
    gated Common-Voice pack, both nl-NL), ask which pack(s) to run instead
    of silently taking the first match — a checkbox, not a single pick,
    since running more than one pack in one batch is a reasonable thing to
    want. Scoped to `language`, not a single profile: `matching_packs` is
    already purely a function of language (`_matching_packs`), so every
    profile sharing that language gets the exact same choice set -- the
    caller asks this once per distinct language in the whole matrix
    selection and reuses the answer for every profile/engine matching it,
    instead of repeating an identical prompt once per profile (a batch
    spanning several engines for one language used to ask the same
    question that many times). The first *ungated* pack comes pre-checked
    (falling back to matching_packs[0] only if every match needs a
    credential) — every profile that has ever had just one match had an
    ungated one, so this is the actual old default, not just
    matching_packs[0]: that's `_pack_rows`' local-dir listing order
    (alphabetical), which is incidental and would otherwise silently
    default to whichever pack's id happens to sort first — confirmed as a
    real bug during manual testing, where an alphabetically-earlier gated
    pack ended up pre-checked ahead of the ungated one it was supposed to
    match. Returns None if the prompt itself is aborted (Ctrl-C) — same
    convention as the rest of the wizard, propagated by the caller to bail
    the whole run. Returns [] if every choice is unchecked — that's a
    deliberate decline, not an abort: the caller drops every combo for
    this language silently, same continue-past-decline spirit as a
    declined credential or engine prompt elsewhere in this preflight."""
    if len(matching_packs) == 1:
        return [matching_packs[0]["id"]]

    pack_info = []
    for pack_row in matching_packs:
        pack_id = pack_row["id"]
        full_pack = _load_pack_for_wizard(pack_id, packs_dir, api_url)
        gated = bool((full_pack or {}).get("audio", {}).get("source", {}).get("credential"))
        pack_info.append((pack_id, gated))

    default_pack_id = next((pid for pid, gated in pack_info if not gated), pack_info[0][0])
    choices = [
        questionary.Choice(f"{pid} ({'gated' if gated else 'open'})", value=pid, checked=pid == default_pack_id)
        for pid, gated in pack_info
    ]

    return questionary.checkbox(
        f"Multiple packs match language {language!r} — pick one or more (space to toggle):",
        choices=choices,
    ).ask()


def _wizard_run_matrix(benchmark_type: str) -> None:
    """Shared run flow for every matrix-shaped benchmark type (batch,
    streaming — `_MATRIX_BENCHMARK_TYPES`): a language x engine/size matrix
    picker (single cells, whole rows, or whole columns — one selection runs
    one benchmark, more runs a batch), packs resolved automatically
    (one-pack-per-profile), one shared repeats value, then a single
    confirmed queue. A bad combo (e.g. a missing model download) must not
    abort the rest of the queue, so each `_reexec` is run in isolation and
    reported rather than propagated. Concurrency doesn't fit this shape
    (no language axis — see `_wizard_run_concurrency`'s own docstring), so
    it stays a separate flow."""
    # Once here, not once per _reexec'd `run` below -- see
    # _SKIP_OUTDATED_CHECK_ENV_VAR's docstring on _warn_if_runner_outdated.
    _warn_if_runner_outdated(DEFAULT_API_URL, offline=False)
    os.environ[_SKIP_OUTDATED_CHECK_ENV_VAR] = "1"

    profiles = _profile_rows(DEFAULT_API_URL, "profiles", offline=False)
    if not profiles:
        typer.echo("no profiles found (checked the API and ./profiles)", err=True)
        return
    matrix = _build_matrix(profiles, benchmark_type)
    if not matrix.columns:
        typer.echo(
            f"no {benchmark_type}-matrix profiles found (none matched the "
            f"<engine>-<size>-<lang>-{benchmark_type} id pattern)", err=True,
        )
        return

    typer.echo(_MATRIX_LEGEND, err=True)
    profile_ids = _ask_matrix(matrix)
    if not profile_ids:
        return

    packs = _pack_rows(DEFAULT_API_URL, "packs", offline=False)
    profiles_by_id = {p["id"]: p for p in profiles}
    combos: list[tuple[str, str]] = []
    # Cached per language, not asked fresh per profile: a matrix selection
    # spanning several engines for the same language would otherwise ask
    # an identical "which pack(s)" question once per engine.
    pack_ids_by_language: dict[str, list[str]] = {}
    for profile_id in profile_ids:
        language = profiles_by_id[profile_id]["language"]
        if language not in pack_ids_by_language:
            matching_packs = _matching_packs(packs, language)
            if not matching_packs:
                typer.echo(f"no packs found for language {language!r} — skipping", err=True)
                pack_ids_by_language[language] = []
                continue
            chosen_pack_ids = _choose_packs_for_language(language, matching_packs, "packs", DEFAULT_API_URL)
            if chosen_pack_ids is None:
                return  # aborted the pack picker
            pack_ids_by_language[language] = chosen_pack_ids
        combos.extend((profile_id, pack_id) for pack_id in pack_ids_by_language[language])
    if not combos:
        return

    _wizard_confirm_and_run(combos)


def _wizard_run() -> None:
    """Batch run flow — see `_wizard_run_matrix`."""
    _wizard_run_matrix("batch")


def _wizard_run_streaming() -> None:
    """Streaming run flow — see `_wizard_run_matrix`."""
    _wizard_run_matrix("streaming")


def _wizard_run_concurrency() -> None:
    """Concurrency/load benchmark flow (ADR-0012) — no language/pack matrix
    at all: these profiles measure performance under simultaneous load, not
    transcription accuracy, so which pack backs the audio has no effect on
    the result. Picks one canonical open pack automatically instead of
    asking the user to choose something that doesn't matter; concurrency
    level(s) are swept the same way beam_size etc. already are, via
    `_wizard_engine_parameters` reading the profile's own `overridable`
    block (shared with `_wizard_run` in `_wizard_confirm_and_run`)."""
    _warn_if_runner_outdated(DEFAULT_API_URL, offline=False)
    os.environ[_SKIP_OUTDATED_CHECK_ENV_VAR] = "1"

    profiles = _profile_rows(DEFAULT_API_URL, "profiles", offline=False)
    concurrency_profiles = [p for p in profiles if p["benchmark_type"] == "concurrency"]
    if not concurrency_profiles:
        typer.echo("no concurrency profiles found (checked the API and ./profiles)", err=True)
        return

    profile_ids = questionary.checkbox(
        "Which engine/model to load-test? (space to toggle)",
        choices=[questionary.Choice(p["id"], value=p["id"]) for p in concurrency_profiles],
    ).ask()
    if not profile_ids:
        return

    packs = _pack_rows(DEFAULT_API_URL, "packs", offline=False)
    open_packs = [p for p in packs if p.get("visibility") == "open"]
    if not open_packs:
        typer.echo("no open packs found to use as filler audio", err=True)
        return
    # Prefer librispeech-en when present -- small, always-available, and
    # already what several adapter tests use as their own filler content --
    # falling back to whatever open pack sorts first otherwise. Any open
    # pack works equally well here; this just picks one deterministically
    # instead of asking the user to choose something with no effect on the
    # result (ADR-0012).
    pack_id = next(
        (p["id"] for p in open_packs if p["id"] == "librispeech-en"),
        min(open_packs, key=lambda p: p["id"])["id"],
    )
    typer.echo(
        f"Using {pack_id!r} as filler audio — content is irrelevant to this "
        "benchmark type (ADR-0012).",
        err=True,
    )

    combos = [(profile_id, pack_id) for profile_id in profile_ids]
    _wizard_confirm_and_run(combos)


def _wizard_confirm_and_run(combos: list[tuple[str, str]]) -> None:
    """Shared tail of every wizard run flow, from pack-credential/engine
    preflight through the final confirmed `_reexec` queue — used by both
    `_wizard_run_matrix` (batch and streaming's shared language matrix) and
    `_wizard_run_concurrency`, which differ only in how `combos` itself
    gets built. A bad combo (e.g. a missing model download) must not abort
    the rest of the queue, so each `_reexec` is run in isolation and
    reported rather than propagated."""
    combos = _preflight_pack_credentials(combos, "packs", DEFAULT_API_URL)
    if not combos:
        return

    combos = _preflight_engines(combos, "profiles", DEFAULT_API_URL)
    if not combos:
        return

    runtime_by_profile = {
        profile_id: (
            (_load_profile_for_wizard(profile_id, "profiles", DEFAULT_API_URL) or {})
            .get("runtime", {})
            .get("name")
        )
        for profile_id in {profile_id for profile_id, _pack_id in combos}
    }
    gpu = _capture_gpu({})
    backend_by_runtime = _wizard_pick_backends(runtime_by_profile, gpu)
    if backend_by_runtime is None:
        return

    def combo_backend(profile_id: str) -> str:
        runtime_name = runtime_by_profile.get(profile_id)
        return backend_by_runtime.get(runtime_name, "cpu") if runtime_name else "cpu"

    # Asked after the backend, not before: a GPU backend choice gives the
    # hardware guess a GPU catalog entry to prefer instead of always
    # preselecting the CPU one (see _guess_hardware_label_for_backends).
    # One prompt per DISTINCT backend actually used, not one global prompt
    # -- a batch mixing cuda for one engine and cpu for another ran on
    # physically different hardware for each (see
    # _pick_hardware_ids_by_backend).
    distinct_backends = sorted({combo_backend(profile_id) for profile_id, _pack_id in combos})
    hardware_by_backend = _pick_hardware_ids_by_backend(distinct_backends, gpu)
    if hardware_by_backend is None:
        return

    expanded = _wizard_engine_parameters(combos, "profiles", DEFAULT_API_URL)
    if not expanded:
        return

    repeats = questionary.text("Repeats (applied to every run in the batch):", default="2").ask()
    if repeats is None:
        return

    has_auto_sweep = any(overrides.get("concurrency") == "auto" for _p, _pk, overrides in expanded)
    total_runs = len(expanded) * int(repeats)
    auto_note = " — auto-sweeps will run more than this" if has_auto_sweep else ""
    typer.echo(f"About to run {len(expanded)} benchmark(s) ({total_runs} runs incl. {repeats} repeats){auto_note}:")
    for profile_id, pack_id, overrides in expanded:
        typer.echo(f"  {_format_combo_label(profile_id, pack_id, overrides)}")
    if len(expanded) > 20 and not questionary.confirm(
        f"That's {len(expanded)} combos — really proceed?", default=False
    ).ask():
        return
    if not questionary.confirm("Proceed?", default=True).ask():
        return

    outcomes: list[tuple[str, str, dict[str, str], bool]] = []
    for profile_id, pack_id, overrides in expanded:
        backend = combo_backend(profile_id)
        if overrides.get("concurrency") == "auto":
            other_overrides = {k: v for k, v in overrides.items() if k != "concurrency"}
            typer.echo(f"Auto-detecting max useful concurrency for {profile_id} x {pack_id} ...")
            outcomes.extend(
                _run_concurrency_auto_sweep(
                    profile_id, pack_id, repeats, backend, hardware_by_backend[backend], other_overrides
                )
            )
            continue
        args = [
            "run", profile_id, pack_id, "--repeats", repeats,
            "--hardware", hardware_by_backend[backend],
        ]
        if backend != "cpu":
            args += ["--backend", backend]
        for key, value in overrides.items():
            args += ["--param", f"{key}={value}"]
        try:
            _reexec(args)
            outcomes.append((profile_id, pack_id, overrides, True))
        except typer.Exit:
            outcomes.append((profile_id, pack_id, overrides, False))

    typer.echo("Batch summary:")
    for profile_id, pack_id, overrides, ok in outcomes:
        typer.echo(f"  {'✓' if ok else '✗'} {_format_combo_label(profile_id, pack_id, overrides)}")


def _wizard_validate() -> None:
    path = questionary.path("Path to a profile.yaml or pack.yaml:").ask()
    if path:
        _reexec(["validate", path])


def _wizard_submit() -> None:
    """Submit one or more result files — questionary's checkbox already
    supports select/deselect-all (press 'a') and invert ('i') natively, no
    custom widget needed, unlike the matrix picker. All chosen files go in
    one `_submit_paths()` batch under a single call-home token (rather than
    one token per file — see that function's docstring for why); one
    rejected result must not block the rest. Deletion is offered only for
    files that actually made it, and defaults to "no" — deleting a local
    result file isn't undoable."""
    results_dir = Path("runs/results")
    result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        typer.echo(f"no result files found under {results_dir}", err=True)
        return
    chosen = questionary.checkbox(
        "Pick result(s) to submit:",
        choices=[questionary.Choice(str(p), value=str(p)) for p in result_files],
    ).ask()
    if not chosen:
        return

    submitted: list[str] = []
    for result_path, accepted, message in _submit_paths(chosen, DEFAULT_API_URL):
        typer.echo(message, err=True)
        if accepted:
            submitted.append(result_path)

    if not submitted:
        return

    typer.echo(f"Submitted {len(submitted)}/{len(chosen)} result file(s).", err=True)
    if questionary.confirm(
        f"Delete the {len(submitted)} submitted result file(s) now?", default=False
    ).ask():
        for result_path in submitted:
            Path(result_path).unlink(missing_ok=True)
        typer.echo(f"Deleted {len(submitted)} file(s).", err=True)


def _run_wizard() -> None:
    actions = {
        "Run benchmark(s)": _wizard_run,
        "Run streaming benchmark(s)": _wizard_run_streaming,
        "Run concurrency/load benchmark(s)": _wizard_run_concurrency,
        "List available profiles": lambda: _reexec(["list-profiles"]),
        "List available packs": lambda: _reexec(["list-packs"]),
        "Validate a profile/pack file": _wizard_validate,
        "Submit a result": _wizard_submit,
        "Show environment fingerprint": lambda: _reexec(["env"]),
        "Print version": lambda: typer.echo(f"goesb-runner {__version__}"),
        "Exit": None,
    }
    while True:
        choice = questionary.select("What would you like to do?", choices=list(actions)).ask()
        if choice is None or choice == "Exit":
            break
        actions[choice]()


# FR-5.3: deviations must be surfaced, not hidden. This is the documented
# default tolerance on the primary metric's relative std across repeats
# (docs/specs/environment-capture.md "Reproducibility tolerance").
DEFAULT_TOLERANCE_REL_STD = 0.05

# Latency metrics are pooled per-utterance samples (p50/p95 across the pack),
# not one aggregate scalar per repeat like WER/RTF — kept separate from
# per_repeat_metrics below because they aggregate differently.
LATENCY_METRIC_IDS = {
    first_partial_latency.METRIC_ID,
    first_final_latency.METRIC_ID,
    end_of_speech_latency.METRIC_ID,
}

_METRIC_UNITS = {
    wer.METRIC_ID: wer.UNIT,
    cer.METRIC_ID: cer.UNIT,
    rtf.METRIC_ID: rtf.UNIT,
    throughput.METRIC_ID: throughput.UNIT,
    cpu_ram.CPU_METRIC_ID: cpu_ram.CPU_UNIT,
    cpu_ram.RAM_METRIC_ID: cpu_ram.RAM_UNIT,
    energy_metric.METRIC_ID: energy_metric.UNIT,
    temperature.METRIC_ID: temperature.UNIT,
    gpu_pct.METRIC_ID: gpu_pct.UNIT,
    first_partial_latency.METRIC_ID: first_partial_latency.UNIT,
    first_final_latency.METRIC_ID: first_final_latency.UNIT,
    end_of_speech_latency.METRIC_ID: end_of_speech_latency.UNIT,
    update_frequency.METRIC_ID: update_frequency.UNIT,
    partial_stability.METRIC_ID: partial_stability.UNIT,
    streaming_responsiveness.METRIC_ID: streaming_responsiveness.UNIT,
}


@app.command()
def version() -> None:
    """Print the runner version."""
    typer.echo(f"goesb-runner {__version__}")


@app.command()
def env() -> None:
    """Capture and print the reproducibility environment fingerprint."""
    typer.echo(json.dumps(capture_environment(), indent=2))


@app.command()
def validate(path: str) -> None:
    """Validate a profile or pack YAML file against its JSON Schema."""
    data = yaml.safe_load(Path(path).read_text())
    schema_filename = (
        "benchmark-pack.schema.json" if "profile_id" in data
        else "benchmark-profile.schema.json"
    )
    errors = validate_against(data, schema_filename)
    if errors:
        typer.echo(f"INVALID ({schema_filename}):", err=True)
        for e in errors:
            typer.echo(f"  - {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"valid ({schema_filename})")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _resolve_pack_audio(
    pack_dir: Path, pack_yaml: dict, audio_dir: str | None, offline: bool
) -> Path:
    """Figure out where this pack's audio actually is — auto-fetching it if
    necessary — and return that directory. This is the single source of
    truth the caller must pass straight to `load_pack()`: resolution and
    the fetch destination must never be computed twice/separately, since
    that's exactly how they drifted apart in 0.2.4 (fetched into the
    shared cache dir, then loaded from the pack's own empty directory)."""
    source = pack_yaml.get("audio", {}).get("source", {})
    auto_fetchable = False
    if audio_dir:
        resolved_audio_dir = Path(audio_dir)
    elif (pack_dir / "audio").exists():
        return pack_dir / "audio"  # already populated (manual or a prior direct fetch) — use it as-is
    elif source.get("type") in AUTO_FETCH_SOURCE_TYPES:
        # Nothing here yet and this source is auto-fetchable: point
        # straight at the shared, content-addressed cache instead of this
        # pack's own directory. load_pack() looks up audio strictly by the
        # filename each manifest.jsonl entry names — it never scans the
        # directory — so every sibling pack whose audio.source matches
        # (e.g. every engine/size combo generated for one language, all
        # pointing at the same FLEURS split) can share this exact folder:
        # the fetch happens at most once total across all of them, and
        # there's nothing to copy or link afterwards.
        resolved_audio_dir = shared_audio_dir(source)
        auto_fetchable = True
    else:
        resolved_audio_dir = pack_dir / "audio"

    manifest_path = pack_dir / "manifest.jsonl"
    wanted_names: set[str] | None = None
    if manifest_path.exists():
        wanted_names = {
            json.loads(line)["relative_path"]
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        }

    if resolved_audio_dir.exists():
        # A directory existing isn't proof it's complete: an auto-fetch
        # interrupted mid-stream (network blip, Ctrl-C) can leave it
        # partially — or, if interrupted on the very first clip, entirely —
        # empty. shared_audio_dir() reuses this exact path across every
        # sibling pack, so trusting bare existence here would permanently
        # poison every one of them the moment any single fetch is cut
        # short; check completeness against the manifest before trusting it.
        existing_names = {p.name for p in resolved_audio_dir.iterdir()}
        if wanted_names is None or wanted_names <= existing_names:
            return resolved_audio_dir

    fetch_instructions = source.get("fetch_instructions")
    if offline:
        typer.echo(f"No audio at {resolved_audio_dir} and --offline was given", err=True)
        if fetch_instructions:
            typer.echo(f"To fetch it:\n{fetch_instructions}", err=True)
        raise typer.Exit(code=1)

    if not auto_fetchable or wanted_names is None:
        typer.echo(
            "Don't know how to auto-fetch audio for this pack" +
            (f" — to fetch it manually:\n{fetch_instructions}" if fetch_instructions
             else " and no fetch_instructions were provided either."),
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"No audio at {resolved_audio_dir} yet — attempting auto-fetch "
        f"(source type: {source['type']}) ...",
        err=True,
    )
    # One retry, only after a successful on-the-spot install: the wizard
    # re-execs `run` as a fresh subprocess per combo (`_reexec`), so without
    # this a missing dependency like `datacollective` would prompt and fail
    # identically for every combo that needs it instead of getting fixed
    # once. attempted_install guards against looping forever if the install
    # "succeeds" but the package still somehow isn't importable.
    attempted_install = False
    while True:
        try:
            fetched = auto_fetch_audio(source, wanted_names, resolved_audio_dir)
            break
        except GatedFetchAuthError as exc:
            typer.echo(f"Auto-fetch failed — credential rejected: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except MissingDependencyError as exc:
            if not attempted_install and _offer_install(exc.package):
                attempted_install = True
                continue
            typer.echo(
                f"{exc.package} is not installed; run `{_suggest_install_command(exc.package)}` and retry.",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        except Exception as exc:  # a bad/expired key or network
            # failure here must report cleanly and fail just this combo, never
            # surface as a raw traceback (ADR-0010) — mirrors the missing-clips
            # branch below, which already fails this one combo without
            # aborting the rest of the batch.
            typer.echo(f"Auto-fetch failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    missing = wanted_names - fetched
    if missing:
        typer.echo(
            f"auto-fetch only found {len(fetched)}/{len(wanted_names)} clips — "
            f"missing: {sorted(missing)}",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"Fetched {len(fetched)} audio files into {resolved_audio_dir}", err=True)
    return resolved_audio_dir


# runtime.name in a profile IS the pip extra name (see pyproject.toml
# [project.optional-dependencies]) — this only maps it to the actual
# importable module name, since that's the one thing pip's own naming
# doesn't tell us.
_ENGINE_MODULE_NAMES = {
    "faster-whisper": "faster_whisper",
    "vosk": "vosk",
    "whisper-cpp": "pywhispercpp",
}


def _install_package(spec: str) -> subprocess.CompletedProcess:
    """Install `spec` into this interpreter's environment. Tries plain pip
    first (the common case), then bootstraps pip via the stdlib's own
    `ensurepip` and retries, then falls back to `uv pip install` against
    this exact interpreter — needed because pipx's `uv` backend creates
    venvs with no `pip` module inside them at all (confirmed: plain `pip
    install` fails there with "No module named pip", on every platform,
    not just one machine), and `uv` itself needs no pip in the target
    environment to install into it. `uv` is guaranteed present wherever a
    uv-backed pipx venv exists — same cross-platform installer provides
    both."""
    result = subprocess.run([sys.executable, "-m", "pip", "install", spec], check=False)
    if result.returncode == 0:
        return result

    if subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=False).returncode == 0:
        result = subprocess.run([sys.executable, "-m", "pip", "install", spec], check=False)
        if result.returncode == 0:
            return result

    uv_path = shutil.which("uv")
    if uv_path is not None:
        return subprocess.run([uv_path, "pip", "install", "--python", sys.executable, spec], check=False)

    return result


def _is_pipx_install() -> bool:
    """True iff this exact interpreter is a pipx-managed venv for
    goesb-runner (`.../pipx/venvs/goesb-runner/...`) — pipx sets no env var
    inside the venv it creates, so a path-substring check on sys.executable
    is the only runtime signal available. Only used to pick the right
    *suggested* command when an automatic install isn't possible or was
    declined; `_install_package` itself targets sys.executable directly and
    already works the same regardless of pipx vs a plain pip/venv install."""
    parts = Path(sys.executable).resolve().parts
    return "pipx" in parts and "goesb-runner" in parts


def _suggest_install_command(package: str) -> str:
    """Best manual fix for `package` missing from this interpreter's
    environment — pipx apps live in an isolated venv, so the plain `pip
    install` a generic error would suggest is invisible to them and leaves
    the same failure recurring forever (see the Ubuntu report that prompted
    this: `pipx install <package>` looked like it worked but installed into
    its own throwaway venv, not goesb-runner's)."""
    if getattr(sys, "frozen", False):
        return f'the standalone binary can\'t install packages — switch to a `pip install goesb-runner` install to use {package}'
    if _is_pipx_install():
        return f"pipx inject goesb-runner {package}"
    return f"pip install {package}"


def _offer_install(package: str) -> bool:
    """Interactively install `package` into this exact interpreter's venv,
    mirroring `_ensure_engine_installed`'s UX — returns True iff it's now
    importable, so the caller can retry whatever needed it. Never offers in
    a frozen (PyInstaller) binary or a non-interactive run; the caller falls
    back to `_suggest_install_command` in both cases."""
    if getattr(sys, "frozen", False) or not sys.stdin.isatty():
        return False
    if not questionary.confirm(f"{package!r} isn't installed — install it now?", default=True).ask():
        return False
    typer.echo(f"Installing {package} ...", err=True)
    result = _install_package(package)
    if result.returncode != 0:
        typer.echo("Install failed.", err=True)
        return False
    importlib.invalidate_caches()
    typer.echo("Installed.", err=True)
    return True


def _ensure_engine_installed(runtime_name: str) -> None:
    """If `runtime_name`'s adapter dependency isn't importable yet, offer to
    pip-install its extra on the spot — pinned to this exact goesb-runner
    version, so it can never silently upgrade the runner itself — instead
    of just telling the user to do it by hand and re-run. Standalone
    PyInstaller binaries already bundle exactly one engine each; this
    doesn't apply to them."""
    module_name = _ENGINE_MODULE_NAMES.get(runtime_name)
    if module_name is None or importlib.util.find_spec(module_name) is not None:
        return  # unknown engine (let the adapter itself raise) or already installed

    if getattr(sys, "frozen", False):
        typer.echo(
            f"Engine {runtime_name!r} isn't available in this standalone binary — "
            f"download goesb-{runtime_name}-<platform> from the latest release instead.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Engine {runtime_name!r} isn't installed yet.", err=True)
    if not sys.stdin.isatty():
        typer.echo(f'Install it: pip install "goesb-runner[{runtime_name}]"', err=True)
        raise typer.Exit(code=1)

    if not questionary.confirm(f"Install goesb-runner[{runtime_name}] now?", default=True).ask():
        typer.echo(f'Install it yourself: pip install "goesb-runner[{runtime_name}]"', err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Installing goesb-runner[{runtime_name}]=={__version__} ...", err=True)
    result = _install_package(f"goesb-runner[{runtime_name}]=={__version__}")
    if result.returncode != 0:
        typer.echo("Install failed — install it yourself and retry.", err=True)
        raise typer.Exit(code=result.returncode)
    importlib.invalidate_caches()
    typer.echo("Installed.", err=True)


def _ensure_cuda_runtime_ready(runtime_name: str, backend: str) -> None:
    """Real report: a fresh Ubuntu box with an NVIDIA driver but no (or a
    mismatched) system CUDA Toolkit crashed the first time faster-whisper
    actually used `--backend cuda`, needing a separate manual `pip install
    nvidia-cublas-cu12` to fix — ctranslate2 dlopen's cuBLAS lazily and
    declares no pip dependency on it (confirmed against the actual PyPI
    wheel). Offers that exact install right here, mirroring
    `_ensure_engine_installed`'s UX, instead of letting the user hit the
    crash first. Scoped to exactly `cuda_runtime.py`'s own coverage
    (faster-whisper's ctranslate2 backend, Linux only) — a no-op for cpu,
    any other engine, any other platform, or when cuBLAS is already
    loadable some other way (a full system CUDA Toolkit, conda).

    Real report: checks via `cublas_loadable()` alone used to re-prompt
    "install nvidia-cublas-cu12?" on every single run even after it had
    already been installed via the pip wheel -- `cublas_loadable()` only
    does a bare `dlopen("libcublas.so.12")` against the OS loader's
    *default* search path, which a pip-installed wheel's
    `nvidia/cublas/lib/` is never on (see `_pip_cublas_lib_path`'s own
    docstring). `preload_installed_cublas()` is the function that actually
    checks the pip-wheel install location too (loading it explicitly by
    absolute path first) -- using the narrower check here for "is there
    anything to do" was always wrong, it just happened to look right on a
    machine with a full system CUDA Toolkit instead of the pip wheel."""
    if backend != "cuda" or runtime_name != "faster-whisper":
        return
    if not cuda_runtime.cuda_libs_supported() or cuda_runtime.preload_installed_cublas():
        return
    _offer_install(cuda_runtime.CUBLAS_PACKAGE)


def _profile_rows(api_url: str, profiles_dir: str, offline: bool) -> list[dict]:
    """Each row: id, language, benchmark_type, version. API first (unless
    --offline), local --profiles-dir as fallback — shared by list-profiles
    and the interactive wizard."""
    rows: list[dict] = []
    if not offline:
        try:
            data = _get_json(f"{api_url.rstrip('/')}/profiles", timeout=10)
            rows = [
                {
                    "id": p["id"], "language": p.get("language") or "-",
                    "benchmark_type": p["benchmark_type"], "version": p["version"],
                }
                for p in data["profiles"]
            ]
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            typer.echo(f"could not reach {api_url} ({exc}) — falling back to local {profiles_dir!r}", err=True)

    if not rows:
        local_dir = Path(profiles_dir)
        if local_dir.exists():
            for entry in sorted(local_dir.iterdir()):
                yaml_path = entry / "profile.yaml"
                if yaml_path.exists():
                    prof = _load_yaml(yaml_path)
                    rows.append({
                        "id": prof["id"], "language": prof.get("language") or "-",
                        "benchmark_type": prof["benchmark_type"], "version": prof["version"],
                    })
    return rows


def _pack_rows(api_url: str, packs_dir: str, offline: bool) -> list[dict]:
    """Each row: id, visibility, version, language — shared by list-packs
    and the interactive wizard (which filters by language, ADR-0011)."""
    rows: list[dict] = []
    if not offline:
        try:
            data = _get_json(f"{api_url.rstrip('/')}/packs", timeout=10)
            rows = [
                {
                    "id": p["id"], "visibility": p["visibility"],
                    "version": p["version"], "language": p.get("language") or "-",
                }
                for p in data["packs"]
            ]
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            typer.echo(f"could not reach {api_url} ({exc}) — falling back to local {packs_dir!r}", err=True)

    if not rows:
        local_dir = Path(packs_dir)
        if local_dir.exists():
            for entry in sorted(local_dir.iterdir()):
                yaml_path = entry / "pack.yaml"
                if yaml_path.exists():
                    pack = _load_yaml(yaml_path)
                    rows.append({
                        "id": pack["id"], "visibility": pack["visibility"],
                        "version": pack["version"],
                        "language": pack.get("metadata", {}).get("language") or "-",
                    })
    return rows


def _hardware_rows(api_url: str, hardware_dir: str, offline: bool) -> list[dict]:
    """Each row: id, display_name, vendor, category — shared by list-hardware
    and the wizard's hardware picker. Live by default (not cached forever
    like fetch_profile/fetch_pack): a pinned profile/pack is immutable once
    referenced by id+version+sha, but the hardware catalog is a growing
    collection where picker-time staleness matters."""
    rows: list[dict] = []
    if not offline:
        try:
            data = _get_json(f"{api_url.rstrip('/')}/hardware/catalog", timeout=10)
            rows = [
                {
                    "id": h["id"], "display_name": h["display_name"],
                    "vendor": h["vendor"], "category": h["category"],
                }
                for h in data["hardware"]
            ]
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            typer.echo(f"could not reach {api_url} ({exc}) — falling back to local {hardware_dir!r}", err=True)

    if not rows:
        local_dir = Path(hardware_dir)
        if local_dir.exists():
            for entry in sorted(local_dir.iterdir()):
                yaml_path = entry / "hardware.yaml"
                if yaml_path.exists():
                    hw = _load_yaml(yaml_path)
                    rows.append({
                        "id": hw["id"], "display_name": hw["display_name"],
                        "vendor": hw["vendor"], "category": hw["category"],
                    })
    return rows


_CPU_MODEL_NOISE_RE = re.compile(r"\(R\)|\(TM\)|®|™|\bCPU\b|@\s*[\d.]+\s*GHz", re.IGNORECASE)


def _normalize_cpu_model(raw: str) -> str:
    """Strip the register-mark/clock-speed cruft a raw CPU probe string
    carries (e.g. "Intel(R) Xeon(R) CPU E3-1240 v6 @ 3.70GHz") down to
    something close to the catalog's plain display_name ("Intel Xeon
    E3-1240 v6") — good enough for difflib to line the two up even before
    accounting for whatever wording differences remain."""
    return re.sub(r"\s+", " ", _CPU_MODEL_NOISE_RE.sub("", raw)).strip()


def _guess_hardware_id(rows: list[dict]) -> str | None:
    """Best-effort local-CPU match against the catalog — the shared match
    logic behind both _guess_hardware_label and doctor's public-result-
    coverage check. Deliberately CPU-only: hardware_id is meant to reflect
    whichever compute path a run actually took (hardware/README.md), and
    doctor's own call site has no profile/backend context to work with at
    all. The wizard, which DOES know the chosen backend by the time it
    asks for hardware, uses the separate GPU-aware
    _guess_hardware_label_for_backends below instead of this function
    directly when a GPU backend was picked. Returns None (no match) rather
    than guess wrong — under virtualization in particular the probed
    string is unrecoverable (e.g. "QEMU Virtual CPU version 2.5+"), and
    difflib's cutoff is exactly what keeps a string that different from
    ever matching."""
    model = _capture_cpu({}).get("model")
    if not model:
        return None
    query = _normalize_cpu_model(model)
    cpu_rows = [r for r in rows if r.get("category") == "cpu"]
    display_names = [r["display_name"] for r in cpu_rows]
    match = difflib.get_close_matches(query, display_names, n=1, cutoff=0.6)
    if not match:
        return None
    return cpu_rows[display_names.index(match[0])]["id"]


def _guess_hardware_label(rows: list[dict]) -> str | None:
    """_guess_hardware_id, formatted as the wizard picker's own label
    string ("display_name (vendor)") so it can be used directly as
    questionary.autocomplete's default= — a suggestion the user still has
    to press Enter on, never a silent auto-assign."""
    guessed_id = _guess_hardware_id(rows)
    if guessed_id is None:
        return None
    row = next(r for r in rows if r["id"] == guessed_id)
    return f"{row['display_name']} ({row['vendor']})"


def _guess_gpu_hardware_id(rows: list[dict], gpu: dict[str, Any]) -> str | None:
    """GPU sibling of _guess_hardware_id -- same difflib match, against
    category == "gpu" catalog rows and the nvidia-smi-probed GPU model
    name (_capture_gpu's "model" key) instead of the CPU model string."""
    model = gpu.get("model")
    if not model:
        return None
    gpu_rows = [r for r in rows if r.get("category") == "gpu"]
    display_names = [r["display_name"] for r in gpu_rows]
    match = difflib.get_close_matches(model, display_names, n=1, cutoff=0.6)
    if not match:
        return None
    return gpu_rows[display_names.index(match[0])]["id"]


def _guess_hardware_label_for_backends(
    rows: list[dict], gpu: dict[str, Any] | None, backend_by_runtime: dict[str, str] | None
) -> str | None:
    """Which guess to preselect in the wizard's hardware picker, now that
    it's asked after the compute backend instead of before it.
    backend_by_runtime is _wizard_pick_backends' return value, which never
    includes cpu entries (see its own docstring) -- so a non-empty dict
    means at least one engine in this batch will actually run GPU-backed,
    and the CPU catalog entry would be the wrong preselection.

    Deliberately does NOT fall back to the CPU guess when a GPU backend
    was chosen but nothing in the catalog matches the probed GPU model --
    suggesting CPU there is exactly the wrong-preselection bug this
    exists to fix; no guess (forcing a manual search) beats a wrong one,
    same philosophy _guess_hardware_id's own docstring states."""
    if backend_by_runtime and gpu is not None:
        guessed_id = _guess_gpu_hardware_id(rows, gpu)
        if guessed_id is None:
            return None
        row = next(r for r in rows if r["id"] == guessed_id)
        return f"{row['display_name']} ({row['vendor']})"
    return _guess_hardware_label(rows)


# prompt_toolkit's default completion-menu style leaves the entry text
# color unset, so it falls through to whatever the terminal/theme decides —
# unreadable against the menu's own grey background in some themes. Fixed
# explicit colors instead of relying on that fallback.
_COMPLETION_MENU_STYLE = questionary.Style([
    ("completion-menu", "bg:#333333 fg:#eeeeee"),
    ("completion-menu.completion", "bg:#333333 fg:#eeeeee"),
    ("completion-menu.completion.current", "bg:#5f5faf fg:#ffffff bold"),
])


def _pick_hardware_id(
    api_url: str,
    hardware_dir: str,
    offline: bool,
    *,
    gpu: dict[str, Any] | None = None,
    backend_by_runtime: dict[str, str] | None = None,
    question: str | None = None,
) -> str | None:
    """Searchable hardware picker for _wizard_run. questionary.autocomplete
    only takes plain-string choices (no separate display/value like
    select()'s Choice), so this keeps its own label->id mapping and
    resolves an unmatched/blank answer to the catalog's 'custom' escape
    hatch. gpu/backend_by_runtime (both optional -- omit for a plain
    CPU-only guess) let the caller thread through what compute backend was
    actually chosen, so the preselection can prefer a GPU catalog match
    over the CPU one -- see _guess_hardware_label_for_backends. question
    overrides the default prompt text -- used when a batch mixes backends
    (see _pick_hardware_ids_by_backend) and asking the same generic
    question twice in a row would leave it unclear which answer applies
    to which backend."""
    rows = _hardware_rows(api_url, hardware_dir, offline)
    if not rows:
        return "custom"

    labels_by_id = {r["id"]: f"{r['display_name']} ({r['vendor']})" for r in rows}
    ids_by_label = {v: k for k, v in labels_by_id.items()}
    other_label = "Other / not yet in the catalog"
    choices = sorted(ids_by_label) + [other_label]

    guessed_label = _guess_hardware_label_for_backends(rows, gpu, backend_by_runtime)
    if guessed_label is not None:
        typer.echo(
            f"Detected: {guessed_label} — press Enter to accept, or type to search for a different one.",
            err=True,
        )

    answer = questionary.autocomplete(
        question or "What hardware did you run this on? (type to search)",
        choices=choices,
        default=guessed_label or "",
        match_middle=True,
        style=_COMPLETION_MENU_STYLE,
    ).ask()
    if answer is None:
        return None

    resolved = ids_by_label.get(answer)
    if resolved is not None:
        return resolved
    if answer != other_label:
        # Typed free text that doesn't exactly match any catalog label
        # (e.g. "intel-n150" instead of the shown "Intel N150 (Intel)") —
        # falls back to custom either way (no dead end), but silently
        # doing so meant results meant for a real catalog entry ended up
        # filed under "custom" with no indication anything was off until
        # someone noticed on the leaderboard later.
        typer.echo(
            f"'{answer}' doesn't match a catalog entry — recording hardware as 'custom' instead.",
            err=True,
        )
    return "custom"


def _pick_hardware_ids_by_backend(
    backends: list[str], gpu: dict[str, Any] | None
) -> dict[str, str] | None:
    """One hardware prompt per distinct backend actually used in this
    wizard batch, not one global prompt for the whole thing -- a batch
    mixing `cuda` for one engine and `cpu` for another ran on physically
    different hardware for each, and a single shared --hardware would
    misattribute one of them. `backends` should already be deduplicated
    (see _wizard_run). Returns backend -> hardware_id, or None if any one
    prompt is aborted (Ctrl-C), same bail-the-whole-run convention as
    every other wizard preflight step."""
    result: dict[str, str] = {}
    for backend in sorted(backends):
        # A single-entry fake "chosen backends" map is exactly what
        # _guess_hardware_label_for_backends needs to prefer the GPU guess
        # for a non-cpu backend, or the plain CPU guess for "cpu" -- same
        # contract _wizard_run's real backend_by_runtime already satisfies.
        fake_backend_by_runtime = {} if backend == "cpu" else {"_": backend}
        noun = "CPU" if backend == "cpu" else "GPU"
        hardware_id = _pick_hardware_id(
            DEFAULT_API_URL,
            "hardware",
            offline=False,
            gpu=gpu,
            backend_by_runtime=fake_backend_by_runtime,
            question=f"What {noun} did you run the {backend!r}-backend benchmarks on? (type to search)",
        )
        if hardware_id is None:
            return None
        result[backend] = hardware_id
    return result


@app.command("list-profiles")
def list_profiles_cmd(
    api_url: str = typer.Option(DEFAULT_API_URL, help="Where to list official profiles from."),
    profiles_dir: str = typer.Option(
        "profiles", help="Also used as a fallback (or with --offline) to list local profiles."
    ),
    offline: bool = typer.Option(False, "--offline", help="List local profiles only, skip the API call."),
) -> None:
    """List profile ids you can pass to `goesb run` (id, language, type, version)."""
    rows = _profile_rows(api_url, profiles_dir, offline)
    if not rows:
        typer.echo("no profiles found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"{'ID':<32} {'LANGUAGE':<10} {'TYPE':<10} VERSION")
    for r in rows:
        typer.echo(f"{r['id']:<32} {r['language']:<10} {r['benchmark_type']:<10} {r['version']}")


@app.command("list-packs")
def list_packs_cmd(
    api_url: str = typer.Option(DEFAULT_API_URL, help="Where to list official packs from."),
    packs_dir: str = typer.Option(
        "packs", help="Also used as a fallback (or with --offline) to list local packs."
    ),
    offline: bool = typer.Option(False, "--offline", help="List local packs only, skip the API call."),
) -> None:
    """List pack ids you can pass to `goesb run` (id, language, visibility, version)."""
    rows = _pack_rows(api_url, packs_dir, offline)
    if not rows:
        typer.echo("no packs found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"{'ID':<36} {'LANGUAGE':<10} {'VISIBILITY':<12} VERSION")
    for r in rows:
        typer.echo(f"{r['id']:<36} {r['language']:<10} {r['visibility']:<12} {r['version']}")


@app.command("list-hardware")
def list_hardware_cmd(
    api_url: str = typer.Option(DEFAULT_API_URL, help="Where to list the official hardware catalog from."),
    hardware_dir: str = typer.Option(
        "hardware", help="Also used as a fallback (or with --offline) to list local hardware entries."
    ),
    offline: bool = typer.Option(False, "--offline", help="List local hardware entries only, skip the API call."),
) -> None:
    """List hardware ids you can pass to `goesb run --hardware` (id, display name, vendor, category)."""
    rows = _hardware_rows(api_url, hardware_dir, offline)
    if not rows:
        typer.echo("no hardware entries found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"{'ID':<32} {'DISPLAY NAME':<40} {'VENDOR':<14} CATEGORY")
    for r in rows:
        typer.echo(f"{r['id']:<32} {r['display_name']:<40} {r['vendor']:<14} {r['category']}")


def _cuda_device_count() -> int | None:
    """`None` means "couldn't check" (ctranslate2 not installed, or the
    probe itself raised) — distinct from `0`, which means ctranslate2 loaded
    fine but genuinely sees no usable CUDA device (the actual "cuBLAS/cuDNN
    missing" signal `goesb doctor` cares about). Read-only: this never
    initializes a model, just asks ctranslate2 what it can see."""
    try:
        import ctranslate2
    except ImportError:
        return None
    try:
        return ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001 - a probe must never itself crash `doctor`
        return None


def _ready_backends(runtime_name: str, benchmark_type: str, gpu: dict[str, Any] | None) -> frozenset[str]:
    """Backends actually usable on this machine right now for this engine
    — `get_supported_backends` alone is declared-only (what the adapter
    knows how to drive), not "verified working", the same gap
    `_doctor_engine_line` reports as text but nothing downstream can act
    on. Reuses `doctor`'s own probes (whisper-cpp's per-backend build-info
    check, faster-whisper's ctranslate2 CUDA device count) so the wizard
    never offers a backend certain to fail. Always includes "cpu"."""
    backends = get_supported_backends(runtime_name, benchmark_type)
    module_name = _ENGINE_MODULE_NAMES.get(runtime_name)
    if module_name is None or importlib.util.find_spec(module_name) is None:
        return frozenset()  # not installed — _preflight_engines handles that separately
    if backends == {"cpu"}:
        return backends

    if runtime_name == "whisper-cpp":
        try:
            from pywhispercpp.model import Model

            from .adapters import whisper_cpp as whisper_cpp_adapter
        except ImportError:
            return frozenset({"cpu"})
        ready = {"cpu"}
        for gpu_backend in backends - {"cpu"}:
            check = whisper_cpp_adapter._BACKEND_AVAILABILITY_CHECK.get(gpu_backend)
            if check is None or not check(Model):
                continue
            if gpu_backend == "cuda" and gpu is None:
                continue  # compiled in, but no NVIDIA GPU detected — matches doctor's own caveat
            ready.add(gpu_backend)
        return frozenset(ready)

    if runtime_name == "faster-whisper":
        if gpu is not None and (_cuda_device_count() or 0) > 0:
            return frozenset({"cpu", "cuda"})
        return frozenset({"cpu"})

    return backends  # unknown engine shape — trust declared support


def _wizard_pick_backends(
    runtime_by_profile: dict[str, str | None], gpu: dict[str, Any] | None
) -> dict[str, str] | None:
    """One backend prompt per distinct engine in the batch, offered only
    among backends `_ready_backends` confirms are actually usable here —
    a user picking a backend certain to fail on this machine is exactly
    the gap `goesb doctor` already reports but the wizard never asked
    about. An engine with only cpu available is never prompted — full
    Enter-through-everything still reproduces today's cpu-only behavior
    byte-for-byte. Returns runtime_name -> chosen backend (cpu entries
    omitted, since that's the `run` default already), or None if the
    user aborts a prompt."""
    chosen: dict[str, str] = {}
    for runtime_name in sorted({r for r in runtime_by_profile.values() if r is not None}):
        ready = _ready_backends(runtime_name, "batch", gpu)
        if ready <= {"cpu"}:
            continue
        backend = questionary.select(
            f"[{runtime_name}] compute backend:",
            choices=sorted(ready),
            default="cpu",
        ).ask()
        if backend is None:
            return None
        if backend != "cpu":
            chosen[runtime_name] = backend
    return chosen


def _doctor_engine_line(runtime_name: str, benchmark_type: str, gpu: dict[str, Any] | None) -> str:
    label = f"  {runtime_name} ({benchmark_type})"
    backends = get_supported_backends(runtime_name, benchmark_type)
    module_name = _ENGINE_MODULE_NAMES.get(runtime_name)
    installed = module_name is not None and importlib.util.find_spec(module_name) is not None

    if not installed:
        return (
            f"{label}: not installed — supports {sorted(backends)} once installed "
            f'(`pip install "goesb-runner[{runtime_name}]"`).'
        )
    if backends == {"cpu"}:
        return f"{label}: cpu ready (installed, cpu-only engine)."

    if runtime_name == "whisper-cpp":
        # Unlike faster-whisper below, this doesn't need gpu (nvidia-smi)
        # at all for most backends — whisper.cpp's own build-info string
        # (system_info(), a static method, no model file needed) says
        # directly which GPU backend(s) this exact compiled binary has,
        # independent of whether an NVIDIA device is even present to
        # report. Real per-backend signal instead of the "can't be
        # checked without running a real transcription" this used to say.
        # Guarded by its own try/except, distinct from `installed` above:
        # that's an importlib.util.find_spec check, which can disagree
        # with a real import in a broken/partial install — must report
        # that clearly rather than let it surface as doctor's generic "an
        # internal probe failed" catch-all.
        try:
            from pywhispercpp.model import Model

            from .adapters import whisper_cpp as whisper_cpp_adapter
        except ImportError:
            return (
                f"{label}: cpu ready; gpu readiness unknown (pywhispercpp reports as "
                "installed but isn't actually importable — a broken or partial install)."
            )
        parts = []
        for gpu_backend in sorted(backends - {"cpu"}):
            check = whisper_cpp_adapter._BACKEND_AVAILABILITY_CHECK.get(gpu_backend)
            if check is None or not check(Model):
                parts.append(f"{gpu_backend} unavailable (not compiled into this build)")
            elif gpu_backend == "cuda" and gpu is None:
                parts.append(f"{gpu_backend} compiled in, but no NVIDIA GPU detected — unlikely to work")
            elif gpu_backend == "cuda":
                parts.append(f"{gpu_backend} ready ({gpu['model']} detected)")
            else:
                parts.append(f"{gpu_backend} ready")
        return f"{label}: cpu ready; " + "; ".join(parts)

    if gpu is None:
        return f"{label}: cpu ready; cuda unavailable (no NVIDIA GPU detected)."

    if runtime_name == "faster-whisper":
        cuda_count = _cuda_device_count()
        if cuda_count is None:
            return f"{label}: cpu ready; cuda readiness unknown (ctranslate2 not installed or probe failed)."
        if cuda_count > 0:
            return f"{label}: cpu ready; cuda ready ({cuda_count} device(s) visible to ctranslate2)."
        missing = (
            f"{label}: cpu ready; {gpu['model']} detected (driver {gpu['driver']}) but "
            "ctranslate2 sees 0 usable CUDA devices — cuBLAS is likely missing or "
            "mismatched with that driver. "
        )
        if cuda_runtime.cuda_libs_supported():
            return missing + (
                'Run `goesb run --backend cuda` again — it now offers to `pip install '
                '"goesb-runner[cuda]"` automatically (a standalone cuBLAS wheel, no '
                "system CUDA Toolkit install needed), or continue on --backend cpu."
            )
        return missing + (
            "Install the CUDA Toolkit + cuDNN matching your driver version "
            "(https://developer.nvidia.com/cudnn) to unlock --backend cuda, or continue "
            "on --backend cpu."
        )
    return f"{label}: installed, supports {sorted(backends)}."


def _detect_non_nvidia_gpu() -> str | None:
    """Advisory-only presence check for a non-NVIDIA GPU, so `doctor`
    doesn't say "GPU: none detected" on every Mac and every AMD/Intel
    Linux box just because `_capture_gpu`'s nvidia-smi probe is the only
    one that exists — misleading specifically on the hardware where
    whisper-cpp's Metal/Vulkan support would actually matter. Deliberately
    separate from environment.py's capture_environment(), which stays
    NVIDIA-only and unchanged — that one is part of the signed result
    document; this is purely doctor's own display (ADR-0008: "detection
    informs the human, it never changes the experiment"). Presence only,
    not a readiness claim the way the NVIDIA branch below is."""
    system = platform.system()
    if system == "Darwin":
        # Every Mac since the Metal era (2012+) has a Metal-capable GPU,
        # integrated or discrete — unlike NVIDIA, no separate probe is
        # needed to know one exists; getting the exact model needs
        # `system_profiler SPDisplaysDataType`, which takes ~1s and isn't
        # worth it just to confirm what's already certain.
        return "Apple GPU (Metal-capable; this doesn't identify the exact model)"
    if system == "Linux":
        info = _run(["lspci", "-mm"])
        if not info:
            return None
        for line in info.splitlines():
            if "VGA compatible controller" not in line and "3D controller" not in line:
                continue
            # lspci -mm quotes each field; vendor and device name are the
            # last two quoted fields on the line.
            fields = re.findall(r'"([^"]*)"', line)
            if len(fields) >= 2:
                return f"{fields[-2]} {fields[-1]} (via lspci)"
        return None
    if system == "Windows":
        info = _run(["wmic", "path", "win32_VideoController", "get", "name"])
        if not info:
            return None
        names = [line.strip() for line in info.splitlines()[1:] if line.strip()]
        return f"{names[0]} (via wmic)" if names else None
    return None


def _hardware_result_gaps(hardware_id: str, api_url: str) -> list[str] | None:
    """Official profiles — for engines actually installed here — with zero
    submitted public results yet for `hardware_id`. The ADR-0008-promised
    half of `doctor` ("which (profile x backend) combinations have no
    verified public result yet"), scoped to profile x hardware only:
    LeaderboardEntry (api/src/oesb_api/schemas.py) doesn't expose which
    backend a result used, only runtime_name, so backend-level granularity
    isn't knowable from the leaderboard API as it exists today. Returns
    None (not an empty list) on any network failure, so the caller can
    tell "checked, nothing missing" apart from "couldn't check"."""
    installed_runtimes = {
        runtime_name
        for runtime_name, _benchmark_type in registered_adapters()
        if (module_name := _ENGINE_MODULE_NAMES.get(runtime_name)) is not None
        and importlib.util.find_spec(module_name) is not None
    }
    if not installed_runtimes:
        return []
    try:
        profiles_data = _get_json(f"{api_url.rstrip('/')}/profiles", timeout=10)
        candidate_ids = {
            p["id"] for p in profiles_data["profiles"] if p.get("runtime") in installed_runtimes
        }
        if not candidate_ids:
            return []
        leaderboard_data = _get_json(
            f"{api_url.rstrip('/')}/leaderboards?hardware={hardware_id}&limit=500", timeout=10
        )
        covered_ids = {r["profile_id"] for r in leaderboard_data["results"]}
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError):
        return None
    return sorted(candidate_ids - covered_ids)


@app.command()
def doctor() -> None:
    """Report detected accelerators and which `--backend` values this
    machine can actually use, per installed engine — without running
    anything (ADR-0008: detection informs the human, it never changes the
    experiment). Reuses the same GPU probe environment capture uses."""
    try:
        typer.echo("GOESB doctor — detecting hardware and backend readiness (nothing is run).\n", err=True)

        unavailable: dict[str, str] = {}
        gpu = _capture_gpu(unavailable)
        if gpu is None:
            typer.echo(f"GPU: none detected via nvidia-smi ({unavailable.get('gpu', 'no probe available')}).", err=True)
            other_gpu = _detect_non_nvidia_gpu()
            if other_gpu is not None:
                typer.echo(f"Non-NVIDIA GPU detected: {other_gpu}.", err=True)
        else:
            typer.echo(f"GPU: {gpu['model']} — driver {gpu['driver']}, {gpu['vram']} VRAM (via nvidia-smi).", err=True)

        typer.echo("\nPer-engine backend readiness:", err=True)
        for runtime_name, benchmark_type in registered_adapters():
            typer.echo(_doctor_engine_line(runtime_name, benchmark_type, gpu), err=True)

        hardware_rows = _hardware_rows(DEFAULT_API_URL, "hardware", offline=False)
        guessed_hardware_id = _guess_hardware_id(hardware_rows)
        if guessed_hardware_id is None:
            typer.echo(
                "\nPublic result coverage: couldn't confidently match this CPU to a "
                "catalog entry — skipping.",
                err=True,
            )
        else:
            gaps = _hardware_result_gaps(guessed_hardware_id, DEFAULT_API_URL)
            if gaps is None:
                typer.echo(
                    f"\nPublic result coverage for {guessed_hardware_id!r}: couldn't reach "
                    f"{DEFAULT_API_URL} — skipping.",
                    err=True,
                )
            elif not gaps:
                typer.echo(
                    f"\nPublic result coverage for {guessed_hardware_id!r}: every official "
                    "profile for your installed engine(s) already has a public result.",
                    err=True,
                )
            else:
                shown, remaining = gaps[:10], len(gaps) - 10
                typer.echo(
                    f"\n{len(gaps)} official profile(s) for your installed engine(s) have no "
                    f"public result yet on {guessed_hardware_id!r} (doesn't distinguish cpu "
                    "vs cuda — the leaderboard doesn't expose that yet):",
                    err=True,
                )
                for profile_id in shown:
                    typer.echo(f"  {profile_id}", err=True)
                if remaining > 0:
                    typer.echo(f"  ... and {remaining} more", err=True)
    except Exception as exc:  # noqa: BLE001 - a report command must never itself crash or leave a partial state
        typer.echo(f"\ndoctor: an internal probe failed unexpectedly ({exc}) — this is a bug, please report it.", err=True)


def _sample_during(fn, interval_s: float = 0.2, gpu_interval_s: float = 1.0):
    """Run `fn()` while sampling CPU/RAM/temperature/GPU in the background,
    and RAPL energy once before and once after (a monotonic counter, so a
    single before/after delta is what's needed, not periodic sampling — see
    energy.py). Returns (result, cpu_ram_samples, temp_samples_c,
    rapl_uj_delta, gpu_samples). `temp_samples_c`/`gpu_samples` are empty and
    `rapl_uj_delta` is `None` on platforms without hwmon/RAPL/an NVIDIA GPU —
    callers treat that exactly like any other "not yet implemented" metric
    gap, never a fabricated zero.

    GPU utilisation is sampled on its own, coarser `gpu_interval_s` cadence
    (default 1s, vs 200ms for everything else) rather than every tick.
    `gpu_pct.sample_gpu_pct()` reads NVML directly now (in-process, no
    subprocess spawn) so this is no longer working around per-call
    overhead the way it originally was — kept coarser anyway since GPU
    utilisation doesn't meaningfully change tick-to-tick the way CPU load
    can, so nothing is lost by sampling it less often.
    """
    samples: list[cpu_ram.Sample] = []
    temp_samples_c: list[float] = []
    gpu_samples: list[float] = []
    stop = threading.Event()
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime baseline

    def sampler() -> None:
        ticks_per_gpu_sample = max(1, round(gpu_interval_s / interval_s))
        tick = 0
        while not stop.is_set():
            samples.append(cpu_ram.sample_process_tree(proc))
            temp_c = energy_probe.sample_hwmon_temp_c()
            if temp_c is not None:
                temp_samples_c.append(temp_c)
            if tick % ticks_per_gpu_sample == 0:
                gpu_c = gpu_pct.sample_gpu_pct()
                if gpu_c is not None:
                    gpu_samples.append(gpu_c)
            tick += 1
            stop.wait(interval_s)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    rapl_start_uj = energy_probe.read_rapl_uj()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    rapl_end_uj = energy_probe.read_rapl_uj()
    samples.append(cpu_ram.sample_process_tree(proc))
    temp_c = energy_probe.sample_hwmon_temp_c()
    if temp_c is not None:
        temp_samples_c.append(temp_c)

    rapl_uj_delta = (
        rapl_end_uj - rapl_start_uj
        if rapl_start_uj is not None and rapl_end_uj is not None
        else None
    )
    return result, samples, temp_samples_c, rapl_uj_delta, gpu_samples


def _parse_param_overrides(param: list[str]) -> dict[str, str]:
    """Parse repeatable `--param KEY=VALUE` options into a dict of raw
    string values — not yet type-coerced or domain-checked, since that
    needs the profile's own `overridable` declaration (see
    `_resolve_parameters`)."""
    overrides: dict[str, str] = {}
    for raw in param:
        key, sep, value = raw.partition("=")
        if not sep:
            raise ValueError(f"--param must be KEY=VALUE, got {raw!r}")
        overrides[key] = value
    return overrides


def _coerce_param_value(raw: str, default: Any) -> Any:
    """Coerce a `--param` CLI string into the type of its profile default —
    bool needs explicit true/false parsing (Python's `bool("false")` is
    True, a classic footgun)."""
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"expected true/false, got {raw!r}")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"expected an integer, got {raw!r}") from None
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"expected a number, got {raw!r}") from None
    return raw


def _check_param_domain(param_name: str, value: Any, domain: dict) -> None:
    if "allowed" in domain and value not in domain["allowed"]:
        raise ValueError(f"{param_name}={value!r} not in allowed values {domain['allowed']}")
    if "range" in domain:
        lo, hi = domain["range"]["min"], domain["range"]["max"]
        if not (lo <= value <= hi):
            raise ValueError(f"{param_name}={value!r} outside range [{lo}, {hi}]")


def _resolve_one_param(profile: dict, param_name: str, raw_value: str | None) -> dict:
    """Resolve a single profile-declared overridable parameter to
    `{"value": ..., "default": ...}`, validating `raw_value` (a `--param`
    CLI string, or None to just take the default) against the profile's
    declared domain. Raises ValueError with a human-readable message on
    any failure — callers (the CLI, the wizard's preflight) decide how to
    present it."""
    overridable = profile.get("overridable", {})
    if param_name not in overridable:
        raise ValueError(
            f"{param_name!r} is not declared overridable by profile "
            f"{profile['id']!r} (overridable: {sorted(overridable) or 'none'})"
        )
    model_cfg = profile.get("model", {})
    configuration = profile.get("configuration", {})
    if param_name in model_cfg:
        default = model_cfg[param_name]
    elif param_name in configuration:
        default = configuration[param_name]
    else:
        raise ValueError(
            f"profile {profile['id']!r} declares {param_name!r} overridable "
            "but it isn't set in model/configuration"
        )
    if raw_value is None:
        return {"value": default, "default": default}
    value = _coerce_param_value(raw_value, default)
    _check_param_domain(param_name, value, overridable[param_name])
    return {"value": value, "default": default}


def _validate_overridable_against_adapter(profile: dict) -> None:
    """No silent knobs (ADR-0009 §2): a profile may declare a parameter
    overridable only if its adapter genuinely applies it. Several adapters
    accept extra kwargs purely for call-shape parity and ignore them
    (whisper-cpp: beam_size/vad/quantization; vosk: everything) — a
    profile declaring one of those would sign results asserting a value
    that had no effect on the run, which is worse than not having the
    feature. This is authoring-mistake protection (a correctly-generated
    profile never trips it) — not expressible in JSON Schema alone, since
    the schema has no notion of which adapter a runtime.name maps to."""
    overridable = profile.get("overridable", {})
    if not overridable:
        return
    runtime_name = profile["runtime"]["name"]
    benchmark_type = profile["benchmark_type"]
    applied = get_applied_parameters(runtime_name, benchmark_type)
    silent = set(overridable) - applied
    if silent:
        raise ValueError(
            f"profile declares {sorted(silent)} overridable, but adapter {runtime_name!r} "
            f"(benchmark_type={benchmark_type!r}) doesn't apply "
            f"{'it' if len(silent) == 1 else 'them'} — accepted for call-shape parity "
            "only, so declaring it would sign a result asserting a value with no effect"
        )


def _resolve_parameters(profile: dict, param_overrides: dict[str, str]) -> dict[str, dict]:
    """Resolve every profile-declared overridable parameter for this run —
    `--param` overrides, else the profile default — validating each
    against its declared domain (ADR-0009 §2). Returns
    `{key: {"value", "default"}}` for every eligible parameter, overridden
    or not, so results stay self-describing without a profile-catalog
    lookup. Hard errors before anything runs — no silent fallback, no
    clamping (ADR-0008 error philosophy)."""
    overridable = profile.get("overridable", {})
    unknown = set(param_overrides) - set(overridable)
    if unknown:
        raise ValueError(
            "--param given for parameter(s) not declared overridable by "
            f"this profile: {sorted(unknown)} (overridable: {sorted(overridable) or 'none'})"
        )
    return {
        key: _resolve_one_param(profile, key, param_overrides.get(key))
        for key in overridable
    }


@app.command()
def run(
    profile_id: str,
    pack_id: str,
    repeats: int = typer.Option(2, min=1, help="Number of repeats (FR-5.3 tolerance check needs >=2)."),
    profiles_dir: str = typer.Option("profiles"),
    packs_dir: str = typer.Option("packs"),
    audio_dir: str = typer.Option(None, help="Defaults to <packs_dir>/<pack_id>/audio"),
    results_dir: str = typer.Option("runs/results"),
    model_override: str = typer.Option(
        None, help="Override the profile's model name (e.g. 'tiny' for a local smoke test)."
    ),
    models_root: str = typer.Option(
        None, help="Where the runtime adapter downloads/caches model weights."
    ),
    external_energy_wh: float = typer.Option(
        None,
        help="Manually-read external power-meter energy (Wh) for this run, "
        "overriding RAPL where RAPL is unavailable (e.g. non-Linux) or "
        "simply preferred — a declarative user-supplied value, not code "
        "(ADR-0004).",
    ),
    api_url: str = typer.Option(
        DEFAULT_API_URL,
        help="Where to fetch an official profile/pack from when it isn't found "
        "under --profiles-dir/--packs-dir. Fetched profiles/packs are cached "
        "under ~/.goesb/cache — offline after the first fetch, same as model "
        "weights already work.",
    ),
    offline: bool = typer.Option(
        False, "--offline",
        help="Never fetch a profile/pack over the network; fail if not found locally.",
    ),
    hardware_id: str = typer.Option(
        None, "--hardware",
        help="Catalog hardware id you actually ran this on (see `goesb list-hardware`); "
        "leave unset to skip. Not validated locally — the auto-detected "
        "environment.cpu/gpu fields are captured either way; this is a "
        "user-asserted override for when auto-detection is wrong (e.g. under "
        "virtualization, where the guest OS cannot see the real hardware).",
    ),
    param: list[str] = typer.Option(  # noqa: B008 — typer's documented default_factory pattern
        default_factory=list,
        help="KEY=VALUE, repeatable — override a profile-declared overridable "
        "parameter (ADR-0009) for this run only, e.g. --param beam_size=8. "
        "The key must be in the profile's `overridable` block and the value "
        "within its declared domain; anything else is a hard error before "
        "anything runs, never a silent fallback.",
    ),
    backend: str = typer.Option(
        "cpu",
        help="Compute backend to run on (ADR-0008) — never auto-selected, "
        "always this exact value. Run `goesb doctor` to see which backends "
        "this machine can actually use. Requesting one this profile's "
        "runtime doesn't support is a hard error before anything runs.",
    ),
) -> None:
    """Run a benchmark for a profile + pack and emit a signed result document."""
    _warn_if_runner_outdated(api_url, offline=offline)

    profile_path = Path(profiles_dir) / profile_id / "profile.yaml"
    pack_dir = Path(packs_dir) / pack_id

    if profile_path.exists():
        profile = _load_yaml(profile_path)
    elif offline:
        typer.echo(f"profile {profile_id!r} not found under {profiles_dir!r} and --offline was given", err=True)
        raise typer.Exit(code=1)
    else:
        typer.echo(f"profile {profile_id!r} not found locally, fetching from {api_url} ...", err=True)
        try:
            profile = fetch_profile(profile_id, api_url)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            typer.echo(f"could not fetch profile {profile_id!r} from {api_url}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    profile_errors = validate_against(profile, "benchmark-profile.schema.json")
    if profile_errors:
        typer.echo(f"profile {profile_id} failed validation: {profile_errors}", err=True)
        raise typer.Exit(code=1)

    # Resolved before anything heavier happens (engine install, pack/audio
    # fetch, model download) — an unknown key or out-of-domain value must
    # fail immediately, not after the user's sat through a multi-GB model
    # download (ADR-0008 error philosophy: explicit, early, never silent).
    try:
        _validate_overridable_against_adapter(profile)
        parameters = _resolve_parameters(profile, _parse_param_overrides(param))
    except ValueError as exc:
        typer.echo(f"--param error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # Same "explicit, early, never silent" placement as the --param check
    # above (ADR-0008): a backend this profile's runtime doesn't support
    # must fail before the engine install prompt, not after, and never
    # silently fall back to whatever the underlying library would have
    # auto-selected.
    supported_backends = get_supported_backends(profile["runtime"]["name"], profile["benchmark_type"])
    if backend not in supported_backends:
        typer.echo(
            f"--backend {backend!r} is not supported by {profile['runtime']['name']!r} "
            f"({profile['benchmark_type']!r}) — this runtime supports: "
            f"{', '.join(sorted(supported_backends))}.",
            err=True,
        )
        raise typer.Exit(code=1)

    _ensure_engine_installed(profile["runtime"]["name"])
    _ensure_cuda_runtime_ready(profile["runtime"]["name"], backend)

    if not (pack_dir / "pack.yaml").exists():
        if offline:
            typer.echo(f"pack {pack_id!r} not found under {packs_dir!r} and --offline was given", err=True)
            raise typer.Exit(code=1)
        typer.echo(
            f"pack {pack_id!r} not found locally, fetching metadata from {api_url} "
            "(audio still needs its own fetch step — see the pack's fetch_instructions) ...",
            err=True,
        )
        try:
            pack_dir = fetch_pack(pack_id, api_url)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            typer.echo(f"could not fetch pack {pack_id!r}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    pack_yaml = _load_yaml(pack_dir / "pack.yaml")
    required_version = unmet_min_runner_version(pack_yaml, __version__)
    if required_version is not None:
        typer.echo(
            f"pack {pack_id} requires goesb-runner >= {required_version}, this install "
            f"is {__version__} — run `pip install --upgrade goesb-runner` and try again.",
            err=True,
        )
        raise typer.Exit(code=1)
    pack_errors = validate_against(pack_yaml, "benchmark-pack.schema.json")
    if pack_errors:
        unknown_type = unrecognized_pack_source_type(pack_yaml)
        if unknown_type is not None:
            typer.echo(
                f"pack {pack_id} uses audio source type {unknown_type!r}, which this "
                "version of goesb-runner doesn't know about — run `pip install "
                "--upgrade goesb-runner` and try again.",
                err=True,
            )
        else:
            typer.echo(f"pack {pack_id} failed validation: {pack_errors}", err=True)
        raise typer.Exit(code=1)
    # ADR-0011: eligibility is decided by language, not by a pack pinning
    # one exact profile_id — that field is informational only now.
    #
    # ADR-0012 exception: a `concurrency` profile measures load behavior,
    # not transcription accuracy — audio content (and its language) is
    # incidental, only its duration matters, and such a profile never
    # declares a `language` at all. This check exists to catch an
    # accidentally-mismatched language for a *scored* profile; it has
    # nothing to verify for one that isn't scored.
    if profile["benchmark_type"] == "concurrency":
        pass
    else:
        profile_language = profile.get("language")
        pack_language = pack_yaml["metadata"]["language"]
        if profile_language is None:
            typer.echo(
                f"profile {profile_id} has no `language` declared — cannot verify "
                f"pack {pack_id} ({pack_language!r}) is eligible for it (ADR-0011).",
                err=True,
            )
            raise typer.Exit(code=1)
        if pack_language != profile_language:
            typer.echo(
                f"pack {pack_id} is {pack_language!r}, profile {profile_id} is "
                f"{profile_language!r} — languages must match exactly (ADR-0011).",
                err=True,
            )
            raise typer.Exit(code=1)

    resolved_audio_dir = _resolve_pack_audio(pack_dir, pack_yaml, audio_dir, offline)
    pack = load_pack(pack_dir, audio_dir=resolved_audio_dir)

    typer.echo(
        f"Loaded {len(pack.utterances)} utterances "
        f"({pack.total_duration_s:.1f}s) from {pack_id}",
        err=True,
    )

    environment = capture_environment()

    benchmark_type = profile["benchmark_type"]
    runtime_name = profile["runtime"]["name"]
    adapter = get_adapter(runtime_name, benchmark_type=benchmark_type)
    runtime_hash = sha256_module_source(sys.modules[adapter.__module__])

    model_cfg = dict(profile["model"])
    configuration = dict(profile.get("configuration", {}))
    # Apply this run's resolved overrides (ADR-0009) before anything below
    # reads model_cfg/configuration, so every downstream use — the adapter
    # call, config_sha256, the result's own model/configuration echo — sees
    # the actual value used, with zero separate plumbing.
    for key, entry in parameters.items():
        if key in model_cfg:
            model_cfg[key] = entry["value"]
        else:
            configuration[key] = entry["value"]
    model_name = model_override or model_cfg["name"]
    # 2-letter code from the profile's BCP-47 language (e.g. "es-419" -> "es")
    # so the adapter can condition the decoder on the actual spoken
    # language, rather than the engine's own default — whisper.cpp in
    # particular defaults to English when no language is given, which for
    # non-English audio produces English-translation-flavored hallucinated
    # output instead of a real transcription in the target language.
    profile_language = profile.get("language")
    language_code = profile_language.split("-")[0].lower() if profile_language else None
    # A `concurrency` profile never scores text (no WER/CER, see the
    # pooled_metric_ids/ADR-0011 notes above) and so declares no
    # `normalization` block at all -- nothing below ever calls normalize()
    # for it, these are simply unused in that branch.
    if profile["benchmark_type"] == "concurrency":
        ruleset_id = None
        norm_options = {}
    else:
        ruleset_id = profile["normalization"]["ruleset_id"]
        norm_options = {
            k: v for k, v in profile["normalization"].items()
            if k in ("lowercase", "remove_punctuation", "expand_numbers")
        }

    models_root_path = Path(models_root) if models_root else Path.home() / ".goesb" / "models" / model_name
    models_root_path.mkdir(parents=True, exist_ok=True)

    # `real_time_factor` is a single corpus-aggregate scalar for
    # batch/streaming (today's unchanged behavior), but for `concurrency`
    # it's the per-call distribution across every worker in the run — the
    # actual answer to "does an individual request stay fast under load,"
    # not just a mean. Same metric id, pooled like LATENCY_METRIC_IDS
    # (p50/p95 always attached, not gated on repeats > 1) only for this
    # one benchmark_type.
    pooled_metric_ids = LATENCY_METRIC_IDS | (
        {rtf.METRIC_ID} if profile["benchmark_type"] == "concurrency" else set()
    )
    scalar_metrics = [m for m in profile["metrics"] if m not in pooled_metric_ids]
    per_repeat_metrics: dict[str, list[float]] = {m: [] for m in scalar_metrics}
    latency_samples_ms: dict[str, list[float]] = {
        m: [] for m in profile["metrics"] if m in pooled_metric_ids
    }
    # One entry per utterance per repeat — reference vs what the engine
    # actually produced, captured before normalize() strips casing and
    # punctuation for WER scoring. The aggregate WER alone can't tell you
    # whether it's low because the engine is genuinely good or because
    # normalization happens to be hiding garbage output; this is the only
    # place that raw comparison survives past this function. Written as its
    # own JSONL file below, next to but separate from the result document —
    # never merged into it or covered by payload_sha256/signature, so it's
    # inherently excluded from `submit` regardless of pack visibility.
    utterance_log: list[dict[str, Any]] = []

    for repeat in range(1, repeats + 1):
        typer.echo(f"Repeat {repeat}/{repeats} ...", err=True)

        if benchmark_type == "batch":

            def _do_transcribe():
                return adapter(
                    model_name,
                    pack.utterances,
                    quantization=model_cfg.get("quantization", "int8"),
                    beam_size=model_cfg.get("beam_size", 5),
                    temperature=model_cfg.get("temperature", 0.0),
                    vad=model_cfg.get("vad", True),
                    threads=configuration.get("threads", 4),
                    download_root=models_root_path,
                    language=language_code,
                    backend=backend,
                )

            transcriptions, samples, temp_samples_c, rapl_uj_delta, gpu_samples = _sample_during(_do_transcribe)
            by_id = {t.utterance_id: t for t in transcriptions}

            pairs = []
            for utterance in pack.utterances:
                hyp = by_id[utterance.utterance_id].hypothesis_text
                utterance_log.append({
                    "repeat": repeat,
                    "utterance_id": utterance.utterance_id,
                    "reference_text": utterance.reference_text,
                    "hypothesis_text": hyp,
                })
                pairs.append((
                    normalize(ruleset_id, utterance.reference_text, **norm_options),
                    normalize(ruleset_id, hyp, **norm_options),
                ))

            total_processing_s = sum(t.processing_time_s for t in transcriptions)
            computed = {
                "wer": wer.compute(pairs),
                "cer": cer.compute(pairs),
                "real_time_factor": rtf.compute(total_processing_s, pack.total_duration_s),
                "throughput": throughput.compute(pack.total_duration_s, total_processing_s),
                "cpu_pct": cpu_ram.reduce_cpu_pct(samples),
                "ram_mb": cpu_ram.reduce_peak_ram_mb(samples),
            }
            if external_energy_wh is not None:
                computed["energy_wh"] = external_energy_wh
            elif rapl_uj_delta is not None:
                computed["energy_wh"] = energy_metric.compute(rapl_uj_delta)
            if temp_samples_c:
                computed["temperature_c"] = temperature.reduce_peak_temp_c(temp_samples_c)
            if gpu_samples:
                computed["gpu_pct"] = gpu_pct.reduce_mean_gpu_pct(gpu_samples)
            for metric_id, values in per_repeat_metrics.items():
                if metric_id in computed:
                    values.append(computed[metric_id])

        elif benchmark_type == "streaming":

            def _do_transcribe():
                return adapter(
                    model_name,
                    pack.utterances,
                    chunk_ms=configuration.get("chunk_ms", 1000),
                    quantization=model_cfg.get("quantization", "int8"),
                    beam_size=model_cfg.get("beam_size", 5),
                    temperature=model_cfg.get("temperature", 0.0),
                    vad=model_cfg.get("vad", True),
                    threads=configuration.get("threads", 4),
                    download_root=models_root_path,
                    language=language_code,
                    backend=backend,
                )

            traces, samples, temp_samples_c, rapl_uj_delta, gpu_samples = _sample_during(_do_transcribe)
            by_id = {t.utterance_id: t for t in traces}

            pairs = []
            for utterance in pack.utterances:
                hyp = by_id[utterance.utterance_id].final_text
                utterance_log.append({
                    "repeat": repeat,
                    "utterance_id": utterance.utterance_id,
                    "reference_text": utterance.reference_text,
                    "hypothesis_text": hyp,
                })
                pairs.append((
                    normalize(ruleset_id, utterance.reference_text, **norm_options),
                    normalize(ruleset_id, hyp, **norm_options),
                ))

            total_processing_s = sum(t.processing_time_s for t in traces)
            this_repeat_latency = {
                first_partial_latency.METRIC_ID: first_partial_latency.compute(traces),
                first_final_latency.METRIC_ID: first_final_latency.compute(traces),
                end_of_speech_latency.METRIC_ID: end_of_speech_latency.compute(traces),
            }
            update_freq = update_frequency.compute(traces)
            stability = partial_stability.compute(traces)
            computed = {
                "wer": wer.compute(pairs),
                "cer": cer.compute(pairs),
                "real_time_factor": rtf.compute(total_processing_s, pack.total_duration_s),
                "throughput": throughput.compute(pack.total_duration_s, total_processing_s),
                "cpu_pct": cpu_ram.reduce_cpu_pct(samples),
                "ram_mb": cpu_ram.reduce_peak_ram_mb(samples),
                "update_frequency": update_freq,
                "partial_stability": stability,
                "streaming_responsiveness": streaming_responsiveness.compute(
                    update_frequency_hz=update_freq,
                    partial_stability=stability,
                    first_partial_latency_p50_ms=summarize(
                        this_repeat_latency[first_partial_latency.METRIC_ID]
                    )["p50"],
                ),
            }
            if external_energy_wh is not None:
                computed["energy_wh"] = external_energy_wh
            elif rapl_uj_delta is not None:
                computed["energy_wh"] = energy_metric.compute(rapl_uj_delta)
            if temp_samples_c:
                computed["temperature_c"] = temperature.reduce_peak_temp_c(temp_samples_c)
            if gpu_samples:
                computed["gpu_pct"] = gpu_pct.reduce_mean_gpu_pct(gpu_samples)
            for metric_id, values in per_repeat_metrics.items():
                if metric_id in computed:
                    values.append(computed[metric_id])
            for metric_id, values in latency_samples_ms.items():
                values.extend(this_repeat_latency[metric_id])

        elif benchmark_type == "concurrency":

            def _do_run_concurrency():
                return adapter(
                    model_name,
                    pack.utterances,
                    concurrency=configuration.get("concurrency", 1),
                    duration_s=configuration.get("duration_s", 30),
                    quantization=model_cfg.get("quantization", "int8"),
                    beam_size=model_cfg.get("beam_size", 5),
                    temperature=model_cfg.get("temperature", 0.0),
                    vad=model_cfg.get("vad", True),
                    threads=configuration.get("threads", 4),
                    download_root=models_root_path,
                    language=language_code,
                    backend=backend,
                )

            calls, samples, temp_samples_c, rapl_uj_delta, gpu_samples = _sample_during(_do_run_concurrency)

            # No WER/CER, no reference/hypothesis pairing, nothing appended
            # to utterance_log — this benchmark type never scores accuracy.
            total_audio_s = sum(c.audio_duration_s for c in calls)
            wall_s = configuration.get("duration_s", 30)
            computed = {
                "throughput": throughput.compute(total_audio_s, wall_s),
                "cpu_pct": cpu_ram.reduce_cpu_pct(samples),
                "ram_mb": cpu_ram.reduce_peak_ram_mb(samples),
            }
            if external_energy_wh is not None:
                computed["energy_wh"] = external_energy_wh
            elif rapl_uj_delta is not None:
                computed["energy_wh"] = energy_metric.compute(rapl_uj_delta)
            if temp_samples_c:
                computed["temperature_c"] = temperature.reduce_peak_temp_c(temp_samples_c)
            if gpu_samples:
                computed["gpu_pct"] = gpu_pct.reduce_mean_gpu_pct(gpu_samples)
            for metric_id, values in per_repeat_metrics.items():
                if metric_id in computed:
                    values.append(computed[metric_id])
            # real_time_factor is pooled per-call here (see pooled_metric_ids
            # above), not a single corpus-aggregate scalar like batch/streaming.
            if rtf.METRIC_ID in latency_samples_ms:
                latency_samples_ms[rtf.METRIC_ID].extend(
                    rtf.compute(c.processing_time_s, c.audio_duration_s) for c in calls
                )

        else:
            typer.echo(f"unsupported benchmark_type: {benchmark_type!r}", err=True)
            raise typer.Exit(code=1)

    metrics_block = {}
    for metric_id, values in per_repeat_metrics.items():
        if not values:
            continue  # e.g. energy_wh/temperature_c on a platform with no
            # RAPL/hwmon (macOS, Windows) and no --external-energy-wh given
        summary = summarize(values)
        metrics_block[metric_id] = {"value": summary["value"], "unit": _METRIC_UNITS[metric_id]}
        if len(values) > 1:
            metrics_block[metric_id]["spread"] = {
                k: v for k, v in summary.items() if k != "value"
            }
            metrics_block[metric_id]["per_repeat"] = values

    for metric_id, samples_ms in latency_samples_ms.items():
        summary = summarize(samples_ms)
        metrics_block[metric_id] = {
            "value": summary["value"],
            "unit": _METRIC_UNITS[metric_id],
            # Always attached (not gated on repeats > 1 like scalar metrics
            # above): the schema requires p50/p95 for every "ms" metric,
            # since these are pooled per-utterance samples, not per-repeat.
            "spread": {k: v for k, v in summary.items() if k != "value"},
        }

    primary_metric = profile["scoring"]["primary_metric"]
    primary_values = latency_samples_ms.get(primary_metric) or per_repeat_metrics.get(primary_metric)
    if primary_values and len(primary_values) > 1:
        primary_summary = summarize(primary_values)
        rel_std = relative_std(primary_summary)
        if rel_std > DEFAULT_TOLERANCE_REL_STD:
            typer.echo(
                f"WARNING: {primary_metric} relative std {rel_std:.1%} exceeds "
                f"tolerance {DEFAULT_TOLERANCE_REL_STD:.0%} across {len(primary_values)} samples "
                "(FR-5.3: surfaced, not hidden)",
                err=True,
            )

    resolved_config = {"model": model_cfg | {"name": model_name}, "configuration": configuration}
    config_sha256 = canonical_asset_sha256(resolved_config, exclude=())
    model_sha256 = sha256_dir(models_root_path)

    result = {
        # 0.3: runtime.backend (ADR-0008) — 0.2 was already claimed by
        # ADR-0009's `parameters` field (see CHANGELOG [0.3.0]); backend
        # didn't ride that bump, so this is its own.
        # 0.4: adds optional top-level `comment`/`submitted_by` — set at
        # `goesb submit` time, not here, but the const still has to match
        # what `submit` will attach or local self-verification would fail.
        "schema_version": "0.4",
        "profile": {
            "id": profile["id"],
            "version": profile["version"],
            "sha256": canonical_asset_sha256(profile, exclude=()),
        },
        "pack": {
            "id": pack_yaml["id"],
            "version": pack_yaml["version"],
            "sha256": pack_yaml["sha256"],
            "visibility": pack_yaml["visibility"],
        },
        "runtime": {
            "name": runtime_name,
            "version": profile["runtime"].get("min_version", "unknown"),
            "sha256": runtime_hash,
            "backend": backend,
        },
        "model": {
            "name": model_name,
            "quantization": model_cfg.get("quantization", "unknown"),
            "sha256": model_sha256,
        },
        "config_sha256": config_sha256,
        "environment": environment,
        "metrics": metrics_block,
        "repeats": repeats,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runner": {"version": __version__},
    }
    if hardware_id:
        result["hardware_id"] = hardware_id
    if parameters:
        result["parameters"] = parameters

    # Nothing named payload_sha256/signature exists on `result` yet, so this
    # hashes exactly the content those two fields will end up covering.
    payload_sha256 = canonical_asset_sha256(result, exclude=())
    result["payload_sha256"] = payload_sha256
    result["signature"] = sign_payload_sha256(payload_sha256)

    result_errors = validate_against(result, "benchmark-result.schema.json")
    if result_errors:
        typer.echo(f"assembled result failed its own schema: {result_errors}", err=True)
        raise typer.Exit(code=1)
    if not verify_result_document(result):
        # Would indicate a bug in the hash/sign wiring above, not user error.
        typer.echo("BUG: freshly-signed result failed self-verification", err=True)
        raise typer.Exit(code=1)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = result["timestamp"].replace(":", "").replace("-", "")
    out_path = out_dir / f"{profile_id}__{pack_id}__{ts_slug}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    utterances_path = out_dir / f"{profile_id}__{pack_id}__{ts_slug}.utterances.jsonl"
    utterances_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in utterance_log) + "\n"
    )

    typer.echo(f"Wrote {out_path}", err=True)
    typer.echo(f"Wrote {utterances_path} ({len(utterance_log)} utterance(s))", err=True)

    table = Table(title="Results", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Spread (±std)", justify="right")
    table.add_column("Unit")
    for metric_id, block in metrics_block.items():
        spread = f"± {block['spread']['std']:.4f}" if "spread" in block else "—"
        table.add_row(metric_id, f"{block['value']:.4f}", spread, block["unit"])
    Console().print(table)


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_json(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310 - caller-controlled --api-url
        return json.loads(resp.read())


_SKIP_OUTDATED_CHECK_ENV_VAR = "_GOESB_SKIP_OUTDATED_CHECK"


def _warn_if_runner_outdated(api_url: str, *, offline: bool) -> None:
    """Best-effort `goesb run` preflight: if the platform currently
    requires a newer runner than this install, fail before any profile/
    pack fetch, engine install, or the benchmark itself -- rather than
    burning through a long run only to have its result rejected by
    `goesb submit` later (see MIN_RUNNER_VERSION on the API side).

    `run` has never required network access and this must not change
    that: a short timeout and a silent return on ANY failure (offline
    machine, unreachable API, timeout) means this only ever HELPS when a
    connection happens to be available, never blocks or slows down a
    genuinely offline run. --offline skips it outright, same as it
    already skips profile/pack fetches.

    A wizard batch re-execs `goesb run` as a fresh subprocess per combo
    (_reexec) -- without this env-var skip, a batch of N combos would
    pay this round-trip N times (fresh DNS/TLS per process, no shared
    cache) for a check that only needs to happen once. _wizard_run does
    its own check up front and sets this in its own environment, which
    every _reexec'd child inherits."""
    if offline or os.environ.get(_SKIP_OUTDATED_CHECK_ENV_VAR) == "1":
        return
    try:
        health = _get_json(f"{api_url.rstrip('/')}/health", timeout=3)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return
    min_runner_version = health.get("min_runner_version")
    if min_runner_version and Version(__version__) < Version(min_runner_version):
        typer.echo(
            f"This goesb-runner ({__version__}) is older than what {api_url} currently "
            f"accepts (minimum {min_runner_version}) — its result would be rejected at "
            "submit time anyway, so stopping now instead of running the full benchmark: "
            "pip install --upgrade goesb-runner",
            err=True,
        )
        raise typer.Exit(code=1)


def _submit_paths(
    result_paths: list[str],
    api_url: str,
    *,
    comment: str | None = None,
    identity: Identity | None = None,
) -> list[tuple[str, bool, str]]:
    """Submit every path under ONE shared call-home token (ADR-0005) —
    returns (path, accepted, message) per input path, in the same order.

    The API's per-IP rate limit (tokens.py) counts token *issuance*, not
    results ingested: fetching one token per file (the original approach)
    means a batch of N results costs N units of quota, the same as N
    separate spam attempts would. Fetching a single token and submitting
    every file as one array to `POST /benchmark/batch` costs exactly 1,
    regardless of N — this is the actual fix, not a bigger rate limit.
    A result that fails locally (edited since `goesb run` wrote it) never
    reaches the network at all; one rejected-by-the-API result never blocks
    its siblings.

    comment/identity apply to every result in this batch identically (one
    submission event, same as the shared token/keypair below) and are
    attached to the in-memory copy only — the local file `run` wrote is
    never touched. Since they're added after the file's own payload_sha256
    was computed, payload_sha256 is recomputed over the now-larger document
    before it gets (re-)signed a few lines down, same as every other
    field-then-hash-then-sign sequence in this module."""
    try:
        health = _get_json(f"{api_url.rstrip('/')}/health", timeout=10)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        msg = f"could not reach {api_url} to check compatibility: {exc}"
        return [(p, False, msg) for p in result_paths]

    min_runner_version = health.get("min_runner_version")
    if min_runner_version and Version(__version__) < Version(min_runner_version):
        msg = (
            f"This goesb-runner ({__version__}) is older than what {api_url} currently "
            f"accepts (minimum {min_runner_version}) — upgrade before submitting: "
            "pip install --upgrade goesb-runner"
        )
        return [(p, False, msg) for p in result_paths]

    outcomes: list[tuple[str, bool, str]] = []
    ready: list[tuple[str, dict]] = []
    for path in result_paths:
        result = json.loads(Path(path).read_text())
        recomputed = canonical_asset_sha256(result, exclude=("payload_sha256", "signature"))
        if recomputed != result.get("payload_sha256"):
            message = (
                f"{path} content does not match its own payload_sha256 "
                "(edited since `goesb run` wrote it?) — refusing to submit"
            )
            outcomes.append((path, False, message))
            continue

        if comment is not None:
            result["comment"] = comment
        if identity is not None:
            result["submitted_by"] = {"callsign": identity.callsign, "discriminator": identity.discriminator}
        if comment is not None or identity is not None:
            result["payload_sha256"] = canonical_asset_sha256(result, exclude=("payload_sha256", "signature"))
            mutated_errors = validate_against(result, "benchmark-result.schema.json")
            if mutated_errors:
                outcomes.append((path, False, f"{path}: comment/identity produced an invalid document: {mutated_errors}"))
                continue

        ready.append((path, result))

    if not ready:
        return outcomes

    private_key = generate_ephemeral_keypair()
    public_key_b64 = base64.b64encode(public_key_bytes_for(private_key)).decode("ascii")

    try:
        token = _post_json(f"{api_url.rstrip('/')}/runner-tokens", {"public_key": public_key_b64}, timeout=10)
    except urllib.error.HTTPError as exc:
        msg = f"failed to obtain a submission token: {exc.code} {exc.read().decode()}"
        return outcomes + [(p, False, msg) for p, _ in ready]
    except urllib.error.URLError as exc:
        msg = f"could not reach {api_url}: {exc.reason}"
        return outcomes + [(p, False, msg) for p, _ in ready]

    for _path, result in ready:
        result["signature"] = sign_with_key(result["payload_sha256"], private_key, token["token_id"])

    try:
        response = _post_json(
            f"{api_url.rstrip('/')}/benchmark/batch",
            {"token_id": token["token_id"], "results": [r for _, r in ready]},
            timeout=60,
        )
    except urllib.error.HTTPError as exc:
        msg = f"batch submission rejected: {exc.code} {exc.read().decode()}"
        return outcomes + [(p, False, msg) for p, _ in ready]
    except urllib.error.URLError as exc:
        msg = f"could not reach {api_url}: {exc.reason}"
        return outcomes + [(p, False, msg) for p, _ in ready]

    for (path, _result), item in zip(ready, response["results"], strict=True):
        if item.get("accepted"):
            outcomes.append((path, True, f"Submitted: {item}"))
        else:
            outcomes.append((path, False, f"submission rejected: {item.get('detail')}"))

    return outcomes


def _prompt_new_identity(callsign: str) -> Identity:
    """Prompts for the secret, derives+persists the discriminator, and
    returns the Identity — the one path both an explicit `--callsign
    <new-name>` and a freshly-typed callsign at the interactive prompt
    funnel through. `.ask()` returning None is the Ctrl-C/abort convention
    used throughout this module (see e.g. line ~417)."""
    secret = questionary.password(
        f"Secret passphrase for '{callsign}' (not stored, used only to distinguish "
        "identical callsigns from different people):"
    ).ask()
    if secret is None:
        raise typer.Exit(code=1)
    new_identity = Identity(callsign, compute_discriminator(callsign, secret))
    save_identity(new_identity)
    return new_identity


def resolve_identity(callsign: str | None, anonymous: bool) -> Identity | None:
    """Five-case resolution order — see identity.py for why the secret
    itself never touches disk or the network:

    1. --anonymous: skip identity for this submission only, saved identity
       (if any) untouched.
    2. --callsign matching what's already saved: reuse it, no prompt at all
       (idempotent, safe to script).
    3. --callsign that's new/different: needs a fresh secret to derive a
       new discriminator, so it requires a TTY even if the rest of the
       invocation is non-interactive.
    4. No --callsign, non-interactive (no TTY): silently reuse whatever's
       saved, or None if nothing's ever been set — never hangs a script
       waiting on input.
    5. No --callsign, interactive: prompt every time (not silently
       automatic), pre-filled with the saved callsign as the default.
       Enter confirms it as-is (no secret needed, the discriminator's
       already on disk); clearing the field submits anonymously for this
       run only, without touching the saved file; typing something new
       falls through to the same secret prompt as case 3.
    """
    if anonymous:
        return None

    saved = load_identity()

    if callsign:
        if saved and saved.callsign == callsign:
            return saved
        if not sys.stdin.isatty():
            typer.echo(
                "setting a new callsign needs an interactive session — run `goesb submit` "
                "once at a terminal, or omit --callsign to reuse the last saved one",
                err=True,
            )
            raise typer.Exit(code=1)
        return _prompt_new_identity(callsign)

    if not sys.stdin.isatty():
        return saved

    entered = questionary.text(
        "Credit this submission with a callsign? (Enter to keep the current one, "
        "clear the field to submit anonymously this time):",
        default=saved.callsign if saved else "",
    ).ask()
    if entered is None:
        raise typer.Exit(code=1)
    entered = entered.strip()
    if not entered:
        return None
    if saved and entered == saved.callsign:
        return saved
    return _prompt_new_identity(entered)


@app.command()
def submit(
    result_paths: list[str] = typer.Argument(  # noqa: B008
        ..., metavar="RESULT_PATH...", help="One or more result files to submit."
    ),
    api_url: str = typer.Option(
        DEFAULT_API_URL, help="Base URL of the GOESB API to submit the result(s) to."
    ),
    callsign: str | None = typer.Option(
        None,
        "--callsign",
        help="Credit this submission to a callsign. A new/different callsign prompts for a "
        "secret passphrase (used once, never stored) and is persisted as the default for "
        "future submits; omit to reuse whatever's already saved.",
    ),
    comment: str | None = typer.Option(
        None, "--comment", help="Optional note (max 500 chars) attached to every result in this submission."
    ),
    anonymous: bool = typer.Option(
        False, "--anonymous", help="Submit without credit this time, even if a callsign is saved locally."
    ),
) -> None:
    """Sign one or more locally-produced results for public submission and
    POST them to the API (ADR-0005).

    Producing a result (`goesb run`) never requires network access; this is
    the separate, explicit submission step. Every path given here shares a
    single ephemeral keypair and a single call-home token — the private key
    never touches disk or leaves this machine, and submitting many results
    in one sitting costs the same rate-limit quota as submitting one (see
    `_submit_paths`). A result that fails locally or is rejected by the API
    never blocks its siblings; the command exits non-zero if any path
    failed, even though the others may have succeeded.
    """
    identity = resolve_identity(callsign, anonymous)
    outcomes = _submit_paths(result_paths, api_url, comment=comment, identity=identity)
    for _path, _accepted, message in outcomes:
        typer.echo(message, err=True)
    if any(not accepted for _, accepted, _ in outcomes):
        raise typer.Exit(code=1)


@app.command("set-identity")
def set_identity_command(
    callsign: str = typer.Argument(..., help="The callsign to credit future submissions to."),
) -> None:
    """Set (or change) the callsign `goesb submit` will offer as its default,
    prompting once for a secret passphrase to derive the public discriminator
    (see identity.py — the secret itself is never stored)."""
    identity = _prompt_new_identity(callsign)
    typer.echo(f"Saved identity: {identity.callsign}#{identity.discriminator}")


@app.command("clear-identity")
def clear_identity_command() -> None:
    """Remove the locally-saved callsign — the next `goesb submit` will ask
    fresh, with no pre-filled default."""
    clear_identity()
    typer.echo("Cleared local identity.")


if __name__ == "__main__":
    app()
