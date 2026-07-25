# Hardware Catalog

A curated, git-reviewed list of real CPU/GPU models a runner user can assert
they ran a benchmark on, instead of relying solely on the OS-probed hardware
string in `environment.cpu`/`environment.gpu` — which under virtualization is
not just imprecise but unrecoverable (a guest OS cannot see the host's real
CPU, e.g. a real Xeon E3-1240 v6 host reports to a QEMU/KVM guest as "QEMU
Virtual CPU version 2.5+"). Each entry has a unique `id`, a `display_name`,
`vendor`, and `category` (`cpu`, `gpu`, `accelerator`, or `other`).

The `goesb run --hardware <id>` flag (and the interactive wizard's searchable
picker) let a user assert which catalog entry they actually ran on; that id
becomes `hardware_id` on the result document. Auto-detection still runs and
is kept as a diagnostic field (`environment.cpu.model`) — it's just no longer
what leaderboards/filtering key off.

Not every real machine is in the catalog yet — pick `custom` (`hardware/custom/hardware.yaml`)
if yours isn't listed, and consider opening a PR to add it.

## Silicon, not products — with one deliberate exception

Entries track the compute chip/module, not the branded board or mini-PC
built around it: dozens of vendors sell boards around the same reference
RK3588 or Snapdragon QCS6490 silicon with connector/case differences that
don't change ASR compute — one `rockchip-rk3588` entry covers all of them,
rather than a row per board (Radxa Rock 5B, Orange Pi 5, ...). Raspberry Pi
is the deliberate exception: nobody buys a bare BCM2711, the Pi Foundation
controls the whole config, and "Raspberry Pi 4" — not its SoC part number —
is the compute identity the community actually means. The rule is "track
the unit that's the real, referred-to compute identity," not "always the
silicon" — for most SBC-class chips that's the chip; for Pi it's the board.

## `category` is the compute path, not a silicon taxonomy

`category` records which compute resource a benchmark on this hardware
actually exercises, not whether the part is "narrowly a CPU." A full SoC
with both CPU cores and a GPU (e.g. an NVIDIA Jetson module) is `gpu` if
results on it run via CUDA, `cpu` if they run on its ARM cores — pick
whichever path is the one actually being benchmarked, don't split one
physical chip into two catalog rows for its two paths. A chip's on-die NPU
(AMD XDNA, Intel's NPU in Meteor/Lunar Lake) never gets its own row either
— the chip is already covered by its `cpu` entry, and "which execution
path a given result used" is a property of the *run* (which runtime/
adapter it used), not of the hardware identity. `accelerator` is for
add-in cards that aren't a CPU or GPU at all (Hailo, Coral Edge TPU) — real
compute hardware sold and discussed independently of a host chip.

Devices that never run inference themselves — voice satellites that just
stream audio to a server (ESP32-S3 boards, Home Assistant's Voice Preview
Edition) — don't belong in this catalog at all, under any category: they
can never be the `hardware_id` behind a real result.

Naming convention: `hardware/<id>/hardware.yaml`, one file per entry, e.g.
`hardware/intel-xeon-e3-1240-v6/hardware.yaml`.

See `runner/src/oesb_runner/schemas/benchmark-hardware.schema.json` for the
contract.
