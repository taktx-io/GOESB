"""OESB runner command-line interface.

See docs/02-architecture.md and the roadmap for context. `run` implements the
M1 slice: local batch run -> normalized WER/CER/RTF/CPU/RAM -> signed, hashed
result document on disk (docs/03-roadmap.md M1).
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import psutil
import typer
import yaml

from . import __version__
from .adapters import get_adapter
from .environment import capture_environment
from .hashing import canonical_asset_sha256, sha256_dir, sha256_module_source
from .metrics import cer, cpu_ram, rtf, wer
from .normalization import normalize
from .pack import load_pack
from .schema_validation import validate_against
from .signing import sign_payload_sha256, verify_result_document
from .stats import relative_std, summarize

app = typer.Typer(help="Open Edge Speech Benchmark runner")

# FR-5.3: deviations must be surfaced, not hidden. This is the documented
# default tolerance on the primary metric's relative std across repeats
# (docs/specs/environment-capture.md "Reproducibility tolerance").
DEFAULT_TOLERANCE_REL_STD = 0.05


@app.command()
def version() -> None:
    """Print the runner version."""
    typer.echo(f"oesb-runner {__version__}")


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


def _sample_during(fn, interval_s: float = 0.2):
    """Run `fn()` while sampling CPU/RAM in the background; return (result, samples)."""
    samples: list[cpu_ram.Sample] = []
    stop = threading.Event()
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime baseline

    def sampler() -> None:
        while not stop.is_set():
            samples.append(cpu_ram.sample_process_tree(proc))
            stop.wait(interval_s)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    samples.append(cpu_ram.sample_process_tree(proc))
    return result, samples


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
) -> None:
    """Run a benchmark for a profile + pack and emit a signed result document."""
    profile_path = Path(profiles_dir) / profile_id / "profile.yaml"
    pack_dir = Path(packs_dir) / pack_id

    profile = _load_yaml(profile_path)
    profile_errors = validate_against(profile, "benchmark-profile.schema.json")
    if profile_errors:
        typer.echo(f"profile {profile_id} failed validation: {profile_errors}", err=True)
        raise typer.Exit(code=1)

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

    pack = load_pack(pack_dir, audio_dir=Path(audio_dir) if audio_dir else None)

    typer.echo(
        f"Loaded {len(pack.utterances)} utterances "
        f"({pack.total_duration_s:.1f}s) from {pack_id}",
        err=True,
    )

    environment = capture_environment()

    runtime_name = profile["runtime"]["name"]
    adapter = get_adapter(runtime_name)
    runtime_hash = sha256_module_source(sys.modules[adapter.__module__])

    model_cfg = dict(profile["model"])
    model_name = model_override or model_cfg["name"]
    ruleset_id = profile["normalization"]["ruleset_id"]
    norm_options = {
        k: v for k, v in profile["normalization"].items()
        if k in ("lowercase", "remove_punctuation", "expand_numbers")
    }
    configuration = profile.get("configuration", {})

    models_root_path = Path(models_root) if models_root else Path.home() / ".oesb" / "models" / model_name
    models_root_path.mkdir(parents=True, exist_ok=True)

    per_repeat_metrics: dict[str, list[float]] = {m: [] for m in profile["metrics"]}

    for repeat in range(1, repeats + 1):
        typer.echo(f"Repeat {repeat}/{repeats} ...", err=True)

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
            )

        transcriptions, samples = _sample_during(_do_transcribe)
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
        for metric_id in per_repeat_metrics:
            if metric_id in computed:
                per_repeat_metrics[metric_id].append(computed[metric_id])

    metrics_block = {}
    for metric_id, values in per_repeat_metrics.items():
        if not values:
            continue  # not yet implemented (e.g. energy_wh) — known M1/M2 gap
        summary = summarize(values)
        metrics_block[metric_id] = {
            "value": summary["value"],
            "unit": "ratio" if metric_id in ("wer", "cer", "real_time_factor") else
                    ("%" if metric_id == "cpu_pct" else "MB"),
        }
        if len(values) > 1:
            metrics_block[metric_id]["spread"] = {
                k: v for k, v in summary.items() if k != "value"
            }
            metrics_block[metric_id]["per_repeat"] = values

    primary_metric = profile["scoring"]["primary_metric"]
    if primary_metric in metrics_block and len(per_repeat_metrics[primary_metric]) > 1:
        primary_summary = summarize(per_repeat_metrics[primary_metric])
        rel_std = relative_std(primary_summary)
        if rel_std > DEFAULT_TOLERANCE_REL_STD:
            typer.echo(
                f"WARNING: {primary_metric} relative std {rel_std:.1%} exceeds "
                f"tolerance {DEFAULT_TOLERANCE_REL_STD:.0%} across {repeats} repeats "
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
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runner": {"version": __version__},
    }

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


if __name__ == "__main__":
    app()
