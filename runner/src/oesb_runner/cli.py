"""GOESB runner command-line interface.

See docs/02-architecture.md and the roadmap for context. `run` implements the
M1 slice: local batch run -> normalized WER/CER/RTF/CPU/RAM -> signed, hashed
result document on disk (docs/03-roadmap.md M1).
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import psutil
import typer
import yaml
from packaging.version import Version

from . import __version__
from . import energy as energy_probe
from .adapters import get_adapter
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
        fetch_instructions = pack_yaml.get("audio", {}).get("source", {}).get("fetch_instructions")
        if fetch_instructions:
            typer.echo(f"No audio at {resolved_audio_dir} yet. To fetch it:\n{fetch_instructions}", err=True)

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
        "http://127.0.0.1:8000", help="Base URL of the GOESB API to submit the result to."
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
