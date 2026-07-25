#!/usr/bin/env python3
"""Generate the seed hardware/ catalog (hardware/<id>/hardware.yaml, one file
per entry, per hardware/README.md). Re-run to add entries to ENTRIES below;
existing files are skipped unless --force, matching generate_bulk_assets.py's
idempotency convention."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
HARDWARE_DIR = ROOT / "hardware"

ENTRIES = [
    # CPUs
    {"id": "raspberry-pi-4-model-b", "display_name": "Raspberry Pi 4 Model B (BCM2711)",
     "vendor": "Raspberry Pi", "category": "cpu", "cores": 4, "threads": 4,
     "max_clock_ghz": 1.8, "release_year": 2019},
    {"id": "raspberry-pi-5", "display_name": "Raspberry Pi 5 (BCM2712)",
     "vendor": "Raspberry Pi", "category": "cpu", "cores": 4, "threads": 4,
     "max_clock_ghz": 2.4, "release_year": 2023},
    {"id": "intel-celeron-n4100", "display_name": "Intel Celeron N4100",
     "vendor": "Intel", "category": "cpu", "cores": 4, "threads": 4,
     "base_clock_ghz": 1.1, "max_clock_ghz": 2.4, "release_year": 2017},
    {"id": "intel-core-i5-8250u", "display_name": "Intel Core i5-8250U",
     "vendor": "Intel", "category": "cpu", "cores": 4, "threads": 8,
     "base_clock_ghz": 1.6, "max_clock_ghz": 3.4, "release_year": 2017},
    {"id": "intel-core-i7-9700k", "display_name": "Intel Core i7-9700K",
     "vendor": "Intel", "category": "cpu", "cores": 8, "threads": 8,
     "base_clock_ghz": 3.6, "max_clock_ghz": 4.9, "release_year": 2018},
    {"id": "intel-core-i5-1135g7", "display_name": "Intel Core i5-1135G7",
     "vendor": "Intel", "category": "cpu", "cores": 4, "threads": 8,
     "base_clock_ghz": 2.4, "max_clock_ghz": 4.2, "release_year": 2020},
    {"id": "intel-core-i7-1265u", "display_name": "Intel Core i7-1265U",
     "vendor": "Intel", "category": "cpu", "cores": 10, "threads": 12,
     "base_clock_ghz": 1.8, "max_clock_ghz": 4.8, "release_year": 2022},
    {"id": "intel-core-i9-13900k", "display_name": "Intel Core i9-13900K",
     "vendor": "Intel", "category": "cpu", "cores": 24, "threads": 32,
     "base_clock_ghz": 3.0, "max_clock_ghz": 5.8, "release_year": 2022},
    {"id": "intel-xeon-e3-1240-v6", "display_name": "Intel Xeon E3-1240 v6",
     "vendor": "Intel", "category": "cpu", "cores": 4, "threads": 8,
     "base_clock_ghz": 3.7, "max_clock_ghz": 4.1, "release_year": 2017},
    {"id": "intel-xeon-e5-2680-v4", "display_name": "Intel Xeon E5-2680 v4",
     "vendor": "Intel", "category": "cpu", "cores": 14, "threads": 28,
     "base_clock_ghz": 2.4, "max_clock_ghz": 3.3, "release_year": 2016},
    {"id": "intel-xeon-silver-4210", "display_name": "Intel Xeon Silver 4210",
     "vendor": "Intel", "category": "cpu", "cores": 10, "threads": 20,
     "base_clock_ghz": 2.2, "max_clock_ghz": 3.2, "release_year": 2019},
    {"id": "intel-xeon-gold-6248", "display_name": "Intel Xeon Gold 6248",
     "vendor": "Intel", "category": "cpu", "cores": 20, "threads": 40,
     "base_clock_ghz": 2.5, "max_clock_ghz": 3.9, "release_year": 2019},
    {"id": "amd-ryzen-5-3600", "display_name": "AMD Ryzen 5 3600",
     "vendor": "AMD", "category": "cpu", "cores": 6, "threads": 12,
     "base_clock_ghz": 3.6, "max_clock_ghz": 4.2, "release_year": 2019},
    {"id": "amd-ryzen-7-5800x", "display_name": "AMD Ryzen 7 5800X",
     "vendor": "AMD", "category": "cpu", "cores": 8, "threads": 16,
     "base_clock_ghz": 3.8, "max_clock_ghz": 4.7, "release_year": 2020},
    {"id": "amd-ryzen-5-5600u", "display_name": "AMD Ryzen 5 5600U",
     "vendor": "AMD", "category": "cpu", "cores": 6, "threads": 12,
     "base_clock_ghz": 2.3, "max_clock_ghz": 4.2, "release_year": 2021},
    {"id": "amd-ryzen-9-7950x", "display_name": "AMD Ryzen 9 7950X",
     "vendor": "AMD", "category": "cpu", "cores": 16, "threads": 32,
     "base_clock_ghz": 4.5, "max_clock_ghz": 5.7, "release_year": 2022},
    {"id": "amd-epyc-7402p", "display_name": "AMD EPYC 7402P",
     "vendor": "AMD", "category": "cpu", "cores": 24, "threads": 48,
     "base_clock_ghz": 2.8, "max_clock_ghz": 3.35, "release_year": 2019},
    {"id": "amd-epyc-9354", "display_name": "AMD EPYC 9354",
     "vendor": "AMD", "category": "cpu", "cores": 32, "threads": 64,
     "base_clock_ghz": 3.25, "max_clock_ghz": 3.8, "release_year": 2023},
    {"id": "apple-m1", "display_name": "Apple M1",
     "vendor": "Apple", "category": "cpu", "cores": 8, "threads": 8, "release_year": 2020},
    {"id": "apple-m1-pro", "display_name": "Apple M1 Pro",
     "vendor": "Apple", "category": "cpu", "cores": 10, "threads": 10, "release_year": 2021},
    {"id": "apple-m2", "display_name": "Apple M2",
     "vendor": "Apple", "category": "cpu", "cores": 8, "threads": 8, "release_year": 2022},
    {"id": "apple-m2-pro", "display_name": "Apple M2 Pro",
     "vendor": "Apple", "category": "cpu", "cores": 12, "threads": 12, "release_year": 2023},
    {"id": "apple-m3", "display_name": "Apple M3",
     "vendor": "Apple", "category": "cpu", "cores": 8, "threads": 8, "release_year": 2023},
    {"id": "apple-m3-max", "display_name": "Apple M3 Max",
     "vendor": "Apple", "category": "cpu", "cores": 16, "threads": 16, "release_year": 2023},
    {"id": "apple-m4", "display_name": "Apple M4",
     "vendor": "Apple", "category": "cpu", "cores": 10, "threads": 10, "release_year": 2024},
    {"id": "apple-m4-pro", "display_name": "Apple M4 Pro",
     "vendor": "Apple", "category": "cpu", "cores": 14, "threads": 14, "release_year": 2024},
    # GPUs
    {"id": "nvidia-jetson-nano", "display_name": "NVIDIA Jetson Nano",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 4, "release_year": 2019},
    {"id": "nvidia-jetson-orin-nano-8gb", "display_name": "NVIDIA Jetson Orin Nano 8GB",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 8, "release_year": 2023},
    {"id": "nvidia-jetson-agx-orin-64gb", "display_name": "NVIDIA Jetson AGX Orin 64GB",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 64, "release_year": 2022},
    {"id": "nvidia-rtx-3060", "display_name": "NVIDIA GeForce RTX 3060",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 12, "release_year": 2021},
    {"id": "nvidia-rtx-4070", "display_name": "NVIDIA GeForce RTX 4070",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 12, "release_year": 2023},
    {"id": "nvidia-rtx-4090", "display_name": "NVIDIA GeForce RTX 4090",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 24, "release_year": 2022},
    {"id": "nvidia-t4", "display_name": "NVIDIA T4",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 16, "release_year": 2018},
    {"id": "nvidia-a10", "display_name": "NVIDIA A10",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 24, "release_year": 2021},
    {"id": "nvidia-a100-40gb", "display_name": "NVIDIA A100 40GB",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 40, "release_year": 2020},
    {"id": "nvidia-quadro-p2000", "display_name": "NVIDIA Quadro P2000",
     "vendor": "NVIDIA", "category": "gpu", "vram_gb": 5, "release_year": 2017},
    # Rockchip ARM SoCs — common in edge SBCs (Orange Pi 5, Radxa Rock 5,
    # Firefly ITX-3588Q, NanoPi M4, and similar boards built around them).
    {"id": "rockchip-rk3588", "display_name": "Rockchip RK3588",
     "vendor": "Rockchip", "category": "cpu", "cores": 8, "threads": 8,
     "max_clock_ghz": 2.4, "release_year": 2022,
     "notes": "4x Cortex-A76 + 4x Cortex-A55, Mali-G610 GPU, ~6 TOPS NPU."},
    {"id": "rockchip-rk3588s", "display_name": "Rockchip RK3588S",
     "vendor": "Rockchip", "category": "cpu", "cores": 8, "threads": 8,
     "max_clock_ghz": 2.4, "release_year": 2022,
     "notes": "RK3588 variant with fewer PCIe/video lanes, same core layout and NPU."},
    {"id": "rockchip-rk3568", "display_name": "Rockchip RK3568",
     "vendor": "Rockchip", "category": "cpu", "cores": 4, "threads": 4,
     "max_clock_ghz": 2.0, "release_year": 2021,
     "notes": "4x Cortex-A55, ~1 TOPS NPU on some SKUs."},
    {"id": "rockchip-rk3566", "display_name": "Rockchip RK3566",
     "vendor": "Rockchip", "category": "cpu", "cores": 4, "threads": 4,
     "max_clock_ghz": 1.8, "release_year": 2021},
    {"id": "rockchip-rk3399", "display_name": "Rockchip RK3399",
     "vendor": "Rockchip", "category": "cpu", "cores": 6, "threads": 6,
     "max_clock_ghz": 2.0, "release_year": 2016,
     "notes": "2x Cortex-A72 + 4x Cortex-A55, no NPU — original Rock Pi 4 / NanoPi M4 generation."},
    # Qualcomm edge-AI SoCs — Hexagon-NPU parts aimed at on-device voice/AI
    # (QCS610 specifically targets far-field voice assistants), plus one
    # laptop-class reference (Snapdragon X Elite) for on-device speech apps.
    {"id": "qualcomm-qcs6490", "display_name": "Qualcomm QCS6490",
     "vendor": "Qualcomm", "category": "cpu", "cores": 8, "threads": 8,
     "max_clock_ghz": 2.7, "release_year": 2021,
     "notes": "Kryo 670 octa-core, Hexagon 770 NPU (~12 TOPS) — common in edge-AI/voice dev kits (e.g. Qualcomm RB3 Gen2)."},
    {"id": "qualcomm-qcm6490", "display_name": "Qualcomm QCM6490",
     "vendor": "Qualcomm", "category": "cpu", "cores": 8, "threads": 8,
     "max_clock_ghz": 2.7, "release_year": 2021,
     "notes": "Same silicon as QCS6490, module packaging (no display/camera ISP exposed)."},
    {"id": "qualcomm-qcs8550", "display_name": "Qualcomm QCS8550",
     "vendor": "Qualcomm", "category": "cpu", "cores": 8, "threads": 8,
     "max_clock_ghz": 3.36, "release_year": 2023,
     "notes": "Snapdragon 8 Gen 2-based edge-AI SoC, higher-tier Hexagon NPU."},
    {"id": "qualcomm-qcs8250", "display_name": "Qualcomm QCS8250",
     "vendor": "Qualcomm", "category": "cpu", "cores": 8, "threads": 8,
     "max_clock_ghz": 2.84, "release_year": 2020,
     "notes": "Snapdragon 865-based edge-AI SoC — Qualcomm RB5 dev kit."},
    {"id": "qualcomm-qcs610", "display_name": "Qualcomm QCS610",
     "vendor": "Qualcomm", "category": "cpu", "cores": 8, "threads": 8,
     "max_clock_ghz": 2.0, "release_year": 2019,
     "notes": "Lower-power edge-AI SoC aimed specifically at far-field voice/smart-speaker designs."},
    {"id": "qualcomm-snapdragon-x-elite", "display_name": "Qualcomm Snapdragon X Elite",
     "vendor": "Qualcomm", "category": "cpu", "cores": 12, "threads": 12,
     "max_clock_ghz": 4.3, "release_year": 2024,
     "notes": "Laptop-class Oryon cores, ~45 TOPS Hexagon NPU — relevant for on-device speech apps on ARM Windows laptops."},
    # Escape hatch
    {"id": "custom", "display_name": "Other / not yet in the catalog",
     "vendor": "Other", "category": "other",
     "notes": "Pick this when your real hardware isn't listed yet — consider "
              "opening a PR to add it. The auto-detected environment.cpu/gpu "
              "fields still capture whatever the OS reports."},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing entries.")
    args = parser.parse_args()

    written = 0
    for entry in ENTRIES:
        entry_dir = HARDWARE_DIR / entry["id"]
        entry_path = entry_dir / "hardware.yaml"
        if entry_path.exists() and not args.force:
            continue
        entry_dir.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))
        written += 1

    print(f"Wrote {written}/{len(ENTRIES)} hardware entries under {HARDWARE_DIR}")


if __name__ == "__main__":
    main()
