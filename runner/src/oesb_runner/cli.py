"""GOESB runner command-line interface.

See docs/02-architecture.md and the roadmap for context. `run` implements the
M1 slice: local batch run -> normalized WER/CER/RTF/CPU/RAM -> signed, hashed
result document on disk (docs/03-roadmap.md M1).
"""
from __future__ import annotations

import base64
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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

from . import __version__
from . import energy as energy_probe
from .adapters import get_adapter
from .audio_sources import AUTO_FETCH_SOURCE_TYPES, auto_fetch_audio
from .environment import capture_environment
from .hashing import canonical_asset_sha256, sha256_dir, sha256_module_source
from .metrics import (
    cer,
    cpu_ram,
    end_of_speech_latency,
    first_final_latency,
    first_partial_latency,
    partial_stability,
    rtf,
    streaming_responsiveness,
    temperature,
    update_frequency,
    wer,
)
from .metrics import energy as energy_metric
from .normalization import normalize
from .pack import load_pack
from .remote import DEFAULT_API_URL, fetch_pack, fetch_profile
from .schema_validation import validate_against
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


def _matching_packs(packs: list[dict], profile_id: str) -> list[dict]:
    """Packs targeting `profile_id`, or every pack if none do — same
    fallback the wizard has always used so an unmatched profile still lets
    you pick something rather than dead-ending."""
    return [p for p in packs if p["profile_id"] == profile_id] or packs


# Bulk-generated batch profile ids are a clean <engine>-<size>-<lang>-batch
# grid (e.g. "whisper-medium-en-batch", "vosk-small-es-batch") — confirmed
# across every profile in the official set. Anything not matching this (e.g.
# the one streaming profile) just doesn't get a matrix cell, same as any
# hand-authored profile outside the bulk set.
_MATRIX_ID_RE = re.compile(r"^(whisper|whispercpp|vosk)-(tiny|base|small|medium|large-v3)-([a-z]{2})-batch$")

_MATRIX_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
_MATRIX_COLUMNS = (
    [("whisper", size) for size in _MATRIX_SIZES]
    + [("whispercpp", size) for size in _MATRIX_SIZES]
    + [("vosk", "small")]
)


@dataclass
class _Matrix:
    languages: list[str]  # sorted BCP-47 tags, the grid's rows
    columns: list[tuple[str, str]]  # (engine, size) pairs that exist, the grid's columns
    cells: dict[tuple[str, str], str]  # (language, "<engine>-<size>") -> profile_id


def _build_matrix(profiles: list[dict]) -> _Matrix:
    """Groups batch profiles into the wizard's language x engine/size grid.
    A language missing most columns (e.g. the single Dutch example profile)
    just has fewer entries in `cells` — the grid renders whatever exists,
    no "unavailable" placeholders."""
    by_lang: dict[str, dict[str, str]] = {}
    for p in profiles:
        match = _MATRIX_ID_RE.match(p["id"])
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


def _wizard_run() -> None:
    """The wizard's sole run flow: a language x engine/size matrix picker
    (single cells, whole rows, or whole columns — one selection runs one
    benchmark, more runs a batch), packs resolved automatically
    (one-pack-per-profile), one shared repeats value, then a single
    confirmed queue. A bad combo (e.g. a missing model download) must not
    abort the rest of the queue, so each `_reexec` is run in isolation and
    reported rather than propagated."""
    profiles = _profile_rows(DEFAULT_API_URL, "profiles", offline=False)
    if not profiles:
        typer.echo("no profiles found (checked the API and ./profiles)", err=True)
        return
    matrix = _build_matrix(profiles)
    if not matrix.columns:
        typer.echo("no batch-matrix profiles found (none matched the <engine>-<size>-<lang>-batch id pattern)", err=True)
        return

    profile_ids: list[str] | None = None
    hardware_id: str | None = None
    while profile_ids is None:
        typer.echo(_MATRIX_LEGEND, err=True)
        profile_ids = _ask_matrix(matrix)
        if not profile_ids:
            return
        hardware_id = _pick_hardware_id(DEFAULT_API_URL, "hardware", offline=False, allow_back=True)
        if hardware_id is None:
            return
        if hardware_id is _WIZARD_BACK:
            profile_ids = None  # loop re-shows the grid, selection cleared

    packs = _pack_rows(DEFAULT_API_URL, "packs", offline=False)
    combos: list[tuple[str, str]] = []
    for profile_id in profile_ids:
        matching_packs = _matching_packs(packs, profile_id)
        if not matching_packs:
            typer.echo(f"no packs found for {profile_id!r} — skipping", err=True)
            continue
        combos.append((profile_id, matching_packs[0]["id"]))
    if not combos:
        return

    repeats = questionary.text("Repeats (applied to every run in the batch):", default="2").ask()
    if repeats is None:
        return

    typer.echo(f"About to run {len(combos)} benchmark(s):")
    for profile_id, pack_id in combos:
        typer.echo(f"  {profile_id}  x  {pack_id}")
    if not questionary.confirm("Proceed?", default=True).ask():
        return

    outcomes: list[tuple[str, str, bool]] = []
    for profile_id, pack_id in combos:
        try:
            _reexec(["run", profile_id, pack_id, "--repeats", repeats, "--hardware", hardware_id])
            outcomes.append((profile_id, pack_id, True))
        except typer.Exit:
            outcomes.append((profile_id, pack_id, False))

    typer.echo("Batch summary:")
    for profile_id, pack_id, ok in outcomes:
        typer.echo(f"  {'✓' if ok else '✗'} {profile_id}  x  {pack_id}")


def _wizard_validate() -> None:
    path = questionary.path("Path to a profile.yaml or pack.yaml:").ask()
    if path:
        _reexec(["validate", path])


def _wizard_submit() -> None:
    results_dir = Path("runs/results")
    result_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        typer.echo(f"no result files found under {results_dir}", err=True)
        return
    result_path = questionary.select(
        "Pick a result to submit:",
        choices=[questionary.Choice(str(p), value=str(p)) for p in result_files],
    ).ask()
    if result_path:
        _reexec(["submit", result_path])


def _run_wizard() -> None:
    actions = {
        "Run benchmark(s)": _wizard_run,
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
    cpu_ram.CPU_METRIC_ID: cpu_ram.CPU_UNIT,
    cpu_ram.RAM_METRIC_ID: cpu_ram.RAM_UNIT,
    energy_metric.METRIC_ID: energy_metric.UNIT,
    temperature.METRIC_ID: temperature.UNIT,
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
    """Each row: id, visibility, version, profile_id — shared by list-packs
    and the interactive wizard (which filters by profile_id)."""
    rows: list[dict] = []
    if not offline:
        try:
            data = _get_json(f"{api_url.rstrip('/')}/packs", timeout=10)
            rows = [
                {
                    "id": p["id"], "visibility": p["visibility"],
                    "version": p["version"], "profile_id": p["profile_id"],
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
                        "version": pack["version"], "profile_id": pack["profile_id"],
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


# Sentinel _pick_hardware_id returns (instead of an id or None) when the
# caller passed allow_back=True and the user picked the back option — lets
# _wizard_run distinguish "go back a step" from "cancel entirely" (plain
# None) without a real prior-step stack.
_WIZARD_BACK = object()

_HARDWARE_BACK_LABEL = "« Back to language/engine selection »"

# prompt_toolkit's default completion-menu style leaves the entry text
# color unset, so it falls through to whatever the terminal/theme decides —
# unreadable against the menu's own grey background in some themes. Fixed
# explicit colors instead of relying on that fallback.
_COMPLETION_MENU_STYLE = questionary.Style([
    ("completion-menu", "bg:#333333 fg:#eeeeee"),
    ("completion-menu.completion", "bg:#333333 fg:#eeeeee"),
    ("completion-menu.completion.current", "bg:#5f5faf fg:#ffffff bold"),
])


def _pick_hardware_id(api_url: str, hardware_dir: str, offline: bool, allow_back: bool = False) -> str | None:
    """Searchable hardware picker for _wizard_run. questionary.autocomplete
    only takes plain-string choices (no separate display/value like
    select()'s Choice), so this keeps its own label->id mapping and
    resolves an unmatched/blank answer to the catalog's 'custom' escape
    hatch. allow_back=True adds a literal back-option choice, since
    questionary has no Escape handling to hook into here the way the
    matrix picker does."""
    rows = _hardware_rows(api_url, hardware_dir, offline)
    if not rows:
        return "custom"

    labels_by_id = {r["id"]: f"{r['display_name']} ({r['vendor']})" for r in rows}
    ids_by_label = {v: k for k, v in labels_by_id.items()}
    other_label = "Other / not yet in the catalog"
    choices = sorted(ids_by_label) + [other_label]
    if allow_back:
        choices = [_HARDWARE_BACK_LABEL, *choices]

    answer = questionary.autocomplete(
        "What hardware did you run this on? (type to search)",
        choices=choices,
        match_middle=True,
        style=_COMPLETION_MENU_STYLE,
    ).ask()
    if answer is None:
        return None
    if allow_back and answer == _HARDWARE_BACK_LABEL:
        return _WIZARD_BACK
    return ids_by_label.get(answer, "custom")


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
    """List pack ids you can pass to `goesb run` (id, visibility, version)."""
    rows = _pack_rows(api_url, packs_dir, offline)
    if not rows:
        typer.echo("no packs found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"{'ID':<36} {'VISIBILITY':<12} VERSION")
    for r in rows:
        typer.echo(f"{r['id']:<36} {r['visibility']:<12} {r['version']}")


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


def _sample_during(fn, interval_s: float = 0.2):
    """Run `fn()` while sampling CPU/RAM/temperature in the background, and
    RAPL energy once before and once after (a monotonic counter, so a single
    before/after delta is what's needed, not periodic sampling — see
    energy.py). Returns (result, cpu_ram_samples, temp_samples_c,
    rapl_uj_delta). `temp_samples_c` is empty and `rapl_uj_delta` is `None`
    on platforms without hwmon/RAPL (macOS, Windows, RAPL-less Linux) —
    callers treat that exactly like any other "not yet implemented" metric
    gap, never a fabricated zero.
    """
    samples: list[cpu_ram.Sample] = []
    temp_samples_c: list[float] = []
    stop = threading.Event()
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime baseline

    def sampler() -> None:
        while not stop.is_set():
            samples.append(cpu_ram.sample_process_tree(proc))
            temp_c = energy_probe.sample_hwmon_temp_c()
            if temp_c is not None:
                temp_samples_c.append(temp_c)
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
    return result, samples, temp_samples_c, rapl_uj_delta


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
) -> None:
    """Run a benchmark for a profile + pack and emit a signed result document."""
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

    _ensure_engine_installed(profile["runtime"]["name"])

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
    pack_errors = validate_against(pack_yaml, "benchmark-pack.schema.json")
    if pack_errors:
        typer.echo(f"pack {pack_id} failed validation: {pack_errors}", err=True)
        raise typer.Exit(code=1)
    if pack_yaml["profile_id"] != profile_id:
        typer.echo(
            f"pack {pack_id} targets profile {pack_yaml['profile_id']!r}, not {profile_id!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    resolved_audio_dir = Path(audio_dir) if audio_dir else (pack_dir / "audio")
    if not resolved_audio_dir.exists():
        source = pack_yaml.get("audio", {}).get("source", {})
        fetch_instructions = source.get("fetch_instructions")
        if offline:
            typer.echo(f"No audio at {resolved_audio_dir} and --offline was given", err=True)
            if fetch_instructions:
                typer.echo(f"To fetch it:\n{fetch_instructions}", err=True)
            raise typer.Exit(code=1)

        manifest_path = pack_dir / "manifest.jsonl"
        if source.get("type") not in AUTO_FETCH_SOURCE_TYPES or not manifest_path.exists():
            typer.echo(
                "Don't know how to auto-fetch audio for this pack" +
                (f" — to fetch it manually:\n{fetch_instructions}" if fetch_instructions
                 else " and no fetch_instructions were provided either."),
                err=True,
            )
            raise typer.Exit(code=1)

        wanted_names = {
            json.loads(line)["relative_path"]
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        }
        typer.echo(
            f"No audio at {resolved_audio_dir} yet — attempting auto-fetch "
            f"(source type: {source['type']}) ...",
            err=True,
        )
        fetched = auto_fetch_audio(source, wanted_names, resolved_audio_dir)
        missing = wanted_names - fetched
        if missing:
            typer.echo(
                f"auto-fetch only found {len(fetched)}/{len(wanted_names)} clips — "
                f"missing: {sorted(missing)}",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Fetched {len(fetched)} audio files into {resolved_audio_dir}", err=True)

    pack = load_pack(pack_dir, audio_dir=Path(audio_dir) if audio_dir else None)

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
    model_name = model_override or model_cfg["name"]
    # 2-letter code from the profile's BCP-47 language (e.g. "es-419" -> "es")
    # so the adapter can condition the decoder on the actual spoken
    # language, rather than the engine's own default — whisper.cpp in
    # particular defaults to English when no language is given, which for
    # non-English audio produces English-translation-flavored hallucinated
    # output instead of a real transcription in the target language.
    profile_language = profile.get("language")
    language_code = profile_language.split("-")[0].lower() if profile_language else None
    ruleset_id = profile["normalization"]["ruleset_id"]
    norm_options = {
        k: v for k, v in profile["normalization"].items()
        if k in ("lowercase", "remove_punctuation", "expand_numbers")
    }
    configuration = profile.get("configuration", {})

    models_root_path = Path(models_root) if models_root else Path.home() / ".goesb" / "models" / model_name
    models_root_path.mkdir(parents=True, exist_ok=True)

    scalar_metrics = [m for m in profile["metrics"] if m not in LATENCY_METRIC_IDS]
    per_repeat_metrics: dict[str, list[float]] = {m: [] for m in scalar_metrics}
    latency_samples_ms: dict[str, list[float]] = {
        m: [] for m in profile["metrics"] if m in LATENCY_METRIC_IDS
    }

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
                )

            transcriptions, samples, temp_samples_c, rapl_uj_delta = _sample_during(_do_transcribe)
            by_id = {t.utterance_id: t for t in transcriptions}

            pairs = []
            for utterance in pack.utterances:
                hyp = by_id[utterance.utterance_id].hypothesis_text
                pairs.append((
                    normalize(ruleset_id, utterance.reference_text, **norm_options),
                    normalize(ruleset_id, hyp, **norm_options),
                ))

            total_processing_s = sum(t.processing_time_s for t in transcriptions)
            computed = {
                "wer": wer.compute(pairs),
                "cer": cer.compute(pairs),
                "real_time_factor": rtf.compute(total_processing_s, pack.total_duration_s),
                "cpu_pct": cpu_ram.reduce_cpu_pct(samples),
                "ram_mb": cpu_ram.reduce_peak_ram_mb(samples),
            }
            if external_energy_wh is not None:
                computed["energy_wh"] = external_energy_wh
            elif rapl_uj_delta is not None:
                computed["energy_wh"] = energy_metric.compute(rapl_uj_delta)
            if temp_samples_c:
                computed["temperature_c"] = temperature.reduce_peak_temp_c(temp_samples_c)
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
                )

            traces, samples, temp_samples_c, rapl_uj_delta = _sample_during(_do_transcribe)
            by_id = {t.utterance_id: t for t in traces}

            pairs = []
            for utterance in pack.utterances:
                hyp = by_id[utterance.utterance_id].final_text
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
            for metric_id, values in per_repeat_metrics.items():
                if metric_id in computed:
                    values.append(computed[metric_id])
            for metric_id, values in latency_samples_ms.items():
                values.extend(this_repeat_latency[metric_id])

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
        "schema_version": "0.1",
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

    typer.echo(f"Wrote {out_path}", err=True)
    for metric_id, block in metrics_block.items():
        spread = f" ± {block['spread']['std']:.4f}" if "spread" in block else ""
        typer.echo(f"  {metric_id}: {block['value']:.4f}{spread} {block['unit']}")


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


@app.command()
def submit(
    result_path: str,
    api_url: str = typer.Option(
        DEFAULT_API_URL, help="Base URL of the GOESB API to submit the result to."
    ),
) -> None:
    """Sign a locally-produced result for public submission and POST it to
    the API (ADR-0005).

    Producing a result (`goesb run`) never requires network access; this is
    the separate, explicit submission step. A fresh keypair is generated
    in-memory for this submission only — the private key never touches disk
    or leaves this machine — and the API is asked to vouch for its public
    key with a short-lived, single-use token, which is what actually signs
    the result. Re-uses the file's own `payload_sha256` unchanged (content,
    and therefore the hash, doesn't depend on who signs it) after confirming
    the file hasn't been altered since `goesb run` wrote it.
    """
    result = json.loads(Path(result_path).read_text())

    try:
        health = _get_json(f"{api_url.rstrip('/')}/health", timeout=10)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        typer.echo(f"could not reach {api_url} to check compatibility: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    min_runner_version = health.get("min_runner_version")
    if min_runner_version and Version(__version__) < Version(min_runner_version):
        typer.echo(
            f"This goesb-runner ({__version__}) is older than what {api_url} currently "
            f"accepts (minimum {min_runner_version}) — upgrade before submitting: "
            "pip install --upgrade goesb-runner",
            err=True,
        )
        raise typer.Exit(code=1)

    recomputed = canonical_asset_sha256(result, exclude=("payload_sha256", "signature"))
    if recomputed != result.get("payload_sha256"):
        typer.echo(
            f"{result_path} content does not match its own payload_sha256 "
            "(edited since `goesb run` wrote it?) — refusing to submit",
            err=True,
        )
        raise typer.Exit(code=1)

    private_key = generate_ephemeral_keypair()
    public_key_b64 = base64.b64encode(public_key_bytes_for(private_key)).decode("ascii")

    try:
        token = _post_json(f"{api_url.rstrip('/')}/runner-tokens", {"public_key": public_key_b64}, timeout=10)
    except urllib.error.HTTPError as exc:
        typer.echo(f"failed to obtain a submission token: {exc.code} {exc.read().decode()}", err=True)
        raise typer.Exit(code=1) from exc
    except urllib.error.URLError as exc:
        typer.echo(f"could not reach {api_url}: {exc.reason}", err=True)
        raise typer.Exit(code=1) from exc

    result["signature"] = sign_with_key(recomputed, private_key, token["token_id"])

    try:
        response = _post_json(f"{api_url.rstrip('/')}/benchmark", result, timeout=30)
    except urllib.error.HTTPError as exc:
        typer.echo(f"submission rejected: {exc.code} {exc.read().decode()}", err=True)
        raise typer.Exit(code=1) from exc
    except urllib.error.URLError as exc:
        typer.echo(f"could not reach {api_url}: {exc.reason}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Submitted: {response}", err=True)


if __name__ == "__main__":
    app()
