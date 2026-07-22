# Environment capture specification

Reproducibility is GOESB's defining quality (NFR-1). The runner captures a
structured **environment fingerprint** with every result. This spec lists what is
captured, why it matters, and how it is used.

## Purpose

Two results are only comparable if the conditions that shape them are known. The
fingerprint lets a reader answer "would I get these numbers on my setup?" and
lets the platform detect when a "reproduction" actually changed a variable
(driver, thermal headroom, quantization).

## Captured fields

### Hardware
- CPU (model, cores/threads, base/boost clocks, microcode)
- GPU (model, VRAM, driver)
- NPU (model, driver) where present
- RAM (total, type, speed)
- Storage (type, model) for model-load characterisation
- Firmware / BIOS version
- Cooling (active/passive; fan present) — explains throttling
- Power source (AC/battery) and, where available, power-delivery limits

### Software
- Operating system + version
- Kernel version
- Runtime (name + version) — e.g. faster-whisper 1.x
- Driver versions (GPU/NPU/audio)
- Compiler / toolchain where relevant to native runtimes
- Optional Docker image reference (digest) for containerised runs

### Model
- Model name + version
- Quantization
- Beam size
- Language
- VAD on/off + settings
- Temperature
- Chunk size
- Thread settings
- Streaming settings (for streaming/conversation)

## Format & integrity

- The fingerprint is a JSON object with a `schema_version`.
- It is embedded in the result document and covered by the result hash and
  signature (see [ADR-0004](../adr/0004-runner-security-model.md)).
- Best-effort capture: fields that cannot be read on a platform are recorded as
  `null` with a reason, never silently omitted — a missing field is itself
  information (e.g. no power telemetry available).

## Reproducibility tolerance

Identical profile + pack + fingerprint should yield metrics within a documented
tolerance. The runner records repeat count and spread; the platform flags results
whose variance exceeds tolerance rather than hiding it (FR-5.3). Thermal state is
a common cause of drift, which is why cooling and temperature are first-class.

## Roadmap

The scaffolded `goesb env` command returns a partial fingerprint (OS + Python).
Full hardware/thermal/power capture lands in **M1**; cross-platform energy/
thermal probes landed in **M2**: RAPL (`intel-rapl` sysfs) for `energy_wh`
and hwmon for `temperature_c`/the `cooling` fingerprint field
(`runner/src/oesb_runner/energy.py`), both Linux-only by construction
(macOS/Windows have no equivalent userspace interface without elevated
privileges) and unit-tested against synthetic sysfs fixtures. **Not yet
proven against real Linux hardware** — M2 was implemented and CI-verified
from a macOS/arm64 dev machine; a real-hardware Linux run is a tracked
follow-up, not silently assumed working. NVML for NVIDIA GPU energy/thermal
remains unimplemented.
