# spikeengine

Host runtime and packaged bitstreams for the **STDP-capable parallel-lane
SpikeEngine** — a board-variant fork of the SuperNeuroMAT SpikeEngine FPGA
accelerator that adds **on-chip STDP learning**.

The engine is destination-partitioned across *K* lanes, stores spike history in
a source-indexed BRAM (eliminating the wide-mux congestion of a flat history
vector), and runs the STDP weight-update as an initiation-interval-1 pipeline.
On-chip STDP **training is bit-exact** against the `superneuromat` software
simulator — verified in RTL simulation and on **real Basys3 silicon** (the full
64-pixel / 10-class digits classifier trains bit-exact, all 640 weights matching
software). The packaged Basys3 build uses **wide fixed-point** (16-bit weights,
24-bit membrane) so the fine multi-tap STDP increments survive per-update
quantization; on the digits task, over the full 899-image test set, it reaches
**58.4% top-1 accuracy** — an exact match to the `superneuromat` software
reference computed the same way. (The tutorial's own headline "0.909" is
one-vs-rest `(TP+TN)/total` with a multi-answer decode: a 10-class problem
scored as ten independent binary ones, so the true negatives inflate it. 58.4%
is the accuracy.) The rate-readout
*inference* is, like training, **bit-exact** against software (an earlier
apparent ~1-count drift was traced to a host-side weight-loading bug, not a
hardware limitation — see the digits example's module docstring).

## Install

```bash
pip install -e .                # core runtime (pyserial)
pip install -e ".[spikeengine-examples]"    # + superneuromat, numpy, scikit-learn, jupyter for the notebook
```

## Quick start

```python
from superneuromat import spikeengine as se

dev = se.connect(port="auto", board="basys3")   # autodetect the board's UART
dev.soft_reset()                                # clear runtime state (no reprogram)
info = se.load_network(dev, snn, frac_bits=11, weight_w=16, data_w=24)   # map a superneuromat.SNN onto the board
spikes = se.run_schedule(dev, schedule, total_ticks,
                         frac_bits=11, n_neurons=info["n_neurons"])

# read hardware-learned weights straight off the chip after training
weights = se.read_weights(dev, info["entry_index"], frac_bits=11)   # {(post,pre): weight}
dev.close()
```

## Programming a board

Default flow is Vivado JTAG (volatile load):

```python
from superneuromat.spikeengine import program
program.program(board="basys3")   # loads the packaged bitstream
```

or manually:

```
vivado -mode batch -source src/superneuromat/spikeengine/scripts/program_infer_basys3_stdp_cap256x8.tcl \
    -tclargs <path-to>/snm_infer_basys3_stdp_cap256x8.bit
```

## Building a bitstream from RTL (instead of using the packaged one)

`program.program(board)` above loads the packaged, pre-built bitstream in seconds. The
full RTL sources, Vivado constraints, and build scripts are ALSO bundled in the package
(`rtl/`, `constraints/`, `scripts/build_infer_*.tcl`) if you want to run a real
synth -> place -> route -> write_bitstream flow yourself instead -- e.g. after modifying
the RTL, or just to reproduce the packaged build from source:

```python
from superneuromat.spikeengine import build

result = build.build_bitstream("basys3")   # real Vivado run, 10-40+ minutes
print(result.wns_ns, result.lut, result.bram, result.drc_errors)
print(result.bitstream_path)   # the FRESH bitstream (packaged one is untouched)
print(result.outdir)           # every artifact this build generated: checkpoints
                                # (post_synth.dcp/post_route.dcp) + the full
                                # post_route_*.rpt report set (timing/util/DRC/power)
print(build.rtl_source_dir())  # the RTL this (or any) build synthesizes from

program.program("basys3", bitstream=result.bitstream_path)   # program the FRESH build
```

This is an opt-in, heavy path -- never invoked silently -- and requires a local Vivado
install (raises `build.VivadoNotFoundError` if one isn't found, pointing back at the
packaged-bitstream fallback). Supports all three boards: `basys3`, `sp701`, `zcu104`.
Hardware-validated 2026-07-28: from-RTL rebuilds of all three reproduced their known-good
timing exactly (basys3 WNS +0.174ns, sp701 +0.514ns, zcu104 +1.176ns, all DRC-clean), and
a fresh basys3 build was programmed onto real hardware and passed a functional test (all
4 OR-gate truth-table rows correct on-chip).

See `spikeengine/examples/build_vs_packaged.ipynb` for a runnable side-by-side comparison
of both paths across all three boards.

## Examples

`spikeengine/examples/digits_stdp_e2e.py` (and the accompanying notebook)
train an 8×8-digits STDP classifier on the board and verify every tick's output
spikes match a fixed-point-faithful `superneuromat` reference:

```bash
python -m superneuromat.spikeengine.examples.digits_stdp_e2e --board basys3 --port auto
python -m superneuromat.spikeengine.examples.digits_stdp_e2e --no-hardware   # software reference only
```

Two smaller, notebook-only examples (no on-chip STDP -- fixed weights, pure inference)
port this project's own introductory SuperNeuroMAT tutorials
(`board_variants/npu_stdp_dev/notebooks/tutorials/`) onto real hardware, cross-checking
every test case against the software reference:

- `spikeengine/examples/logic_gates.ipynb` -- OR and AND gates, all 4 truth-table rows.
- `spikeengine/examples/bars_and_stripes.ipynb` -- the 3x3 bar/stripe pattern detector
  (9 input + 6 hidden + 2 output + 1 cancel neuron), 9 test patterns including the
  cancel-neuron veto case.

Both default to `RUN_HARDWARE = True` / `BOARD = 'basys3'` / `PORT = 'auto'` in their
first config cell -- set `RUN_HARDWARE = False` to run the software reference only.

- `spikeengine/examples/build_vs_packaged.ipynb` -- build-from-RTL vs. use-packaged-bitstream,
  for all three boards; see "Building a bitstream from RTL" above.

## New in 0.3.0: citation-graph datasets + bigger models

This version adds a **citation-graph SNN dataset family** (microseer / miniseer /
cora / citeseer) that runs on hardware with on-chip STDP, plus two architecture
changes that make the larger graphs fit: **decoupled per-lane synapse capacity**
(`SYN_CAP_PER_LANE`) and **Option A — neuron state in Block RAM**
(`SNM_NEURON_STATE_BRAM`). See **[USER_MANUAL.md](USER_MANUAL.md)** for the full
guide (what's implemented, how to use it end to end). Dataset catalogue:
`spikeengine.list_datasets()` / `get_dataset(name)`.

| dataset | neurons | board | on hardware | notebook / example |
|---|---|---|---|---|
| microseer | 90 | basys3 | ✅ **top-1 41.7%** (20/48), 48/48 SW==HW | `examples/citation_gnn_fpga` |
| miniseer | 2,116 | zcu104 | ✅ **top-1 55.0%** (66/120), 120/120 SW==HW | `--dataset miniseer` |
| cora | 2,715 | zcu104 | ✅ **top-1 62.9%** (88/140), 140/140 SW==HW | `--dataset cora` |
| citeseer | 3,318 | zcu104 | ✅ **top-1 41.7%** (50/120), 120/120 SW==HW | `--dataset citeseer` |

| pubmed | 19,720 | — | software only (too large) | — |

Accuracy is **top-1** — the single highest-weight predicted topic against the
true label. These are multiclass tasks, so top-1 is the accuracy figure; the
SGNN benchmark's own one-vs-rest number is considerably higher because it
counts a true negative for every topic correctly not predicted. See
`docs/USER_MANUAL.md` §2b before quoting either.

"SW==HW" is per-paper agreement between the FPGA and the fixed-point-faithful
software reference; agreement was exact on every paper of every dataset, so
both report identical accuracy. Basys3/microseer and SP701/microseer have been
re-validated under this import path; the ZCU104 figures were measured through
the pre-merge path (see `SPIKEENGINE_VERSIONS.md`).

## Packaged boards

| board  | config          | WNS       | status |
|--------|-----------------|-----------|--------|
| basys3 | N=256 / K=8, 16-bit weights | +0.174 ns | hardware-validated: on-chip training + inference bit-exact (640/640 weights); digits one-vs-rest 0.909 (top-1 58.4%); LEDs show output classes; soft-reset; weight readback |
| sp701  | N=352 / K=8 | +0.514 ns | **legacy 8-bit build** (the wide-fixed-point `-generic` was silently dropped before the 2026-07-29 board-top parameter-forwarding fix); timing-closed/DRC-clean in Vivado, not board-validated. Rebuild wide with `scripts/build_infer_sp701_stdp_custom.tcl`. Citation datasets do not fit SP701 — use ZCU104. |
| zcu104 | N=1024 / K=16 | +1.176 ns | **legacy 8-bit build** (same param-forwarding caveat). For the citation datasets use the per-dataset wide-fixed-point + Option A bitstreams in `bitstreams/zcu104/` (catalogued in `datasets.py`), not this legacy build. |

## What's in the package

- `runtime.py` — `InferEngineStdpDevice`, the UART command-protocol host runtime
  (config load, `run_tick`, `read_spikes`, `soft_reset`, STDP-table load,
  `read_synapse` weight readback).
- `network.py` — `load_network` / `run_schedule` / `read_weights` helpers
  mapping a `superneuromat.SNN` + tick schedule onto the board, and reading
  hardware-trained weights back off it directly (not just inferred via
  spike-train determinism).
- `boards.py` — board geometry + packaged-bitstream catalogue.
- `program.py` — Vivado programming helper.
- `rtl/`, `scripts/`, `constraints/` — the Verilog sources and Vivado
  build/program Tcl (rebuild a bitstream, retarget another board).
- `bitstreams/` — the timing-closed bitstreams.
