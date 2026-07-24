# Hardware Catalog

A curated, git-reviewed list of real CPU/GPU models a runner user can assert
they ran a benchmark on, instead of relying solely on the OS-probed hardware
string in `environment.cpu`/`environment.gpu` — which under virtualization is
not just imprecise but unrecoverable (a guest OS cannot see the host's real
CPU, e.g. a real Xeon E3-1240 v6 host reports to a QEMU/KVM guest as "QEMU
Virtual CPU version 2.5+"). Each entry has a unique `id`, a `display_name`,
`vendor`, and `category` (`cpu`, `gpu`, or `other`).

The `goesb run --hardware <id>` flag (and the interactive wizard's searchable
picker) let a user assert which catalog entry they actually ran on; that id
becomes `hardware_id` on the result document. Auto-detection still runs and
is kept as a diagnostic field (`environment.cpu.model`) — it's just no longer
what leaderboards/filtering key off.

Not every real machine is in the catalog yet — pick `custom` (`hardware/custom/hardware.yaml`)
if yours isn't listed, and consider opening a PR to add it.

Naming convention: `hardware/<id>/hardware.yaml`, one file per entry, e.g.
`hardware/intel-xeon-e3-1240-v6/hardware.yaml`.

See `runner/src/oesb_runner/schemas/benchmark-hardware.schema.json` for the
contract.
