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

from . import __version__
from . import energy as energy_probe
from .adapters import get_adapter, get_applied_parameters
from .audio_sources import AUTO_FETCH_SOURCE_TYPES, auto_fetch_audio, shared_audio_dir
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


def _load_profile_for_wizard(profile_id: str, profiles_dir: str, api_url: str) -> dict | None:
    """Best-effort full profile load for wizard-side preflight steps —
    local dir first, else fetch_profile (cached under ~/.goesb/cache after
    the first fetch, so a second call within the same wizard run, e.g. by
    both _preflight_engines and _wizard_engine_parameters, is a cheap disk
    read, not a repeat network round-trip). Returns None on any network
    failure so callers can degrade gracefully rather than crash the
    wizard — the per-combo `run()` call surfaces the real error properly
    when it actually runs."""
    profile_path = Path(profiles_dir) / profile_id / "profile.yaml"
    try:
        if profile_path.exists():
            return _load_yaml(profile_path)
        return fetch_profile(profile_id, api_url)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None


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
            default = _profile_param_default(profiles_by_id[engine_combos[0][0]], param_name)
            raw = questionary.text(f"[{engine}] {param_name} (default {default}):").ask()
            if raw is None:
                return None
            raw = raw.strip()
            if not raw:
                continue  # Enter: no override for this parameter at all
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

    combos = _preflight_engines(combos, "profiles", DEFAULT_API_URL)
    if not combos:
        return

    expanded = _wizard_engine_parameters(combos, "profiles", DEFAULT_API_URL)
    if not expanded:
        return

    repeats = questionary.text("Repeats (applied to every run in the batch):", default="2").ask()
    if repeats is None:
        return

    total_runs = len(expanded) * int(repeats)
    typer.echo(f"About to run {len(expanded)} benchmark(s) ({total_runs} runs incl. {repeats} repeats):")
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
        args = ["run", profile_id, pack_id, "--repeats", repeats, "--hardware", hardware_id]
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
    ruleset_id = profile["normalization"]["ruleset_id"]
    norm_options = {
        k: v for k, v in profile["normalization"].items()
        if k in ("lowercase", "remove_punctuation", "expand_numbers")
    }

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
        "schema_version": "0.2",
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


def _submit_paths(result_paths: list[str], api_url: str) -> list[tuple[str, bool, str]]:
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
    its siblings."""
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


@app.command()
def submit(
    result_paths: list[str] = typer.Argument(  # noqa: B008
        ..., metavar="RESULT_PATH...", help="One or more result files to submit."
    ),
    api_url: str = typer.Option(
        DEFAULT_API_URL, help="Base URL of the GOESB API to submit the result(s) to."
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
    outcomes = _submit_paths(result_paths, api_url)
    for _path, _accepted, message in outcomes:
        typer.echo(message, err=True)
    if any(not accepted for _, accepted, _ in outcomes):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
