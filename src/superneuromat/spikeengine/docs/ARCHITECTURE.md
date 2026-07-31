# SpikeEngine — Architecture Reference (Part 2: Implementation Details)

Companion to **[USER_MANUAL.md](USER_MANUAL.md)** (Part 1: how to use the
package). This document explains how SpikeEngine is implemented: the RTL
architecture, the wire protocol, fixed-point encoding, resource consumption,
and validated size limits. Every number here is either read directly from the
RTL/source in this package or measured in a Vivado build or a real hardware
run — none are estimated unless explicitly marked "estimated."

**Contents**
1. [Source-code map and how the pieces link](#1-source-code-map-and-how-the-pieces-link)
2. [RTL file hierarchy](#2-rtl-file-hierarchy) (2.1 top-level I/O and the debug port)
3. [GUI (separate component)](#3-gui-separate-component--out-of-scope-for-this-package)
4. [Datasets tested](#4-datasets-tested--summary-full-detail-in-user_manualmd-2)
5. [The controller](#5-the-controller-snm_infer_cmd_ctrl_stdpv)
6. [Lanes (the "NPU") and how they cooperate](#6-lanes-the-npu--neuron-processing-unit-and-how-they-cooperate)
7. [Neuron and synapse update logic](#7-neuron-and-synapse-update-logic) (7.1 per-neuron state, 7.2 membrane pipeline, 7.3 synapse gather)
8. [Communication and I/O: host ↔ FPGA](#8-communication-and-io-host--fpga) (8.1 physical link, 8.2 opcodes, 8.3 bulk vs regular, 8.4 inputs, 8.5 outputs)
9. [STDP implementation](#9-stdp-implementation)
10. [Fixed-point representation and tradeoffs](#10-fixed-point-representation-and-tradeoffs)
11. [Bit-widths, max sizes, and RTL parameter reference](#11-bit-widths-max-sizes-and-rtl-parameter-reference)
12. [Resource consumption (LUT/FF/BRAM/URAM)](#12-resource-consumption-lut--ff--bram--uram--what-scales-with-what)
13. [Design rationale — why this, not that](#13-design-rationale--why-this-not-that)
14. [Current limitations and future work](#14-current-limitations-and-future-work)
15. [How the Tcl / Vivado build flow works](#15-how-the-tcl--vivado-build-flow-works)
16. [Dependencies](#16-dependencies--precisely-what-is-needed-for-what)
17. [How to inspect the RTL and generated bitstreams](#17-how-to-inspect-the-rtl-and-generated-bitstreams-yourself)

---

## 1. Source-code map and how the pieces link

Complete file listing of the installed package, every file annotated. The
package is `superneuromat.spikeengine`, at `src/superneuromat/spikeengine/`
in the repo. Repo root holds `pyproject.toml` (packaging, dependencies,
the `spikeengine-gui` entry point) and `LICENSE` (BSD-3-Clause).

```
src/superneuromat/spikeengine/
│
├── __init__.py                  public API. connect() resolves board/dataset geometry
│                                  (N_MAX, NUM_LANES, weight_w, data_w) and returns an
│                                  InferEngineStdpDevice; re-exports the catalogues,
│                                  load_network/run_schedule/read_weights, bitstream_path().
├── _transport.py                serial layer, vendored so the package is self-contained.
│                                  SerialTransport (8-byte frame xfer), autodetect_port()
│                                  (USB VID:PID match, then per-port READ_STATUS probe when
│                                  a board exposes several UART nodes), SNMError, opcode
│                                  constants shared with the serial-core protocol.
├── runtime.py                   the wire protocol. InferEngineDevice (config load, input
│                                  injection, tick execution, spike/weight readback) and
│                                  InferEngineStdpDevice (adds STDP tap table, global
│                                  enable, soft_reset, learned-weight readback). Also the
│                                  bulk-streaming path: begin_bulk/end_bulk/_stream_commands.
├── network.py                   SNN -> device mapping. load_network() walks a
│                                  superneuromat.SNN, computes the lane/local-index layout,
│                                  quantizes to fixed point, and writes it; run_schedule()
│                                  replays a {tick: {neuron: current}} schedule;
│                                  read_weights(); quantize_raw().
├── boards.py                    board catalogue (StdpBoard): part number, N_MAX/NUM_LANES,
│                                  baud, packaged bitstream path, datapath widths,
│                                  frac_bits default, hardware-validation record per board.
├── datasets.py                  citation-dataset catalogue (Dataset): size, lane count,
│                                  required SYN_CAP_PER_LANE, per-dataset bitstream, build
│                                  WNS, and BOTH accuracy metrics (sw_accuracy = one-vs-rest,
│                                  sw_top1 = strict top-1) with their definitions.
├── capacity.py                  pre-Vivado feasibility: estimate() models BRAM/URAM/LUT/FF
│                                  from a network's real size, required_syn_cap_per_lane()
│                                  computes max-lane in-degree, recommend_board() picks a
│                                  target, validate_network_fits() is the host-side guard
│                                  that refuses an over-capacity load BEFORE any wire write.
├── program.py                   Vivado JTAG programmer. program() resolves the board's
│                                  program_infer_*.tcl, copies the .bit to a short local
│                                  path (OneDrive file-lock workaround), runs Vivado in
│                                  batch mode, and verifies PROGRAM_DONE in the log.
├── build.py                     Vivado RTL->bitstream builder. build_bitstream() reproduces
│                                  a packaged board config from source; rtl_source_dir()
│                                  exposes the plain-text Verilog for inspection.
├── README.md                    short overview + quickstart.
│
├── docs/
│   ├── USER_MANUAL.md           Part 1: how to use the package -- install, workflow,
│   │                             dataset results, fixed-point recipe, troubleshooting, scope.
│   └── ARCHITECTURE.md          Part 2 (this file): how it works internally.
│
├── rtl/                         Verilog RTL, plain text, hand-written and parametrized.
│   │                             Per-file detail in §2.1; hierarchy in §2.
│   ├── snm_infer_top_stdp.v     top: engine + SPI slave, the device-independent core top.
│   ├── snm_infer_engine_stdp.v  engine wrapper: command controller + multilane array.
│   ├── snm_infer_cmd_ctrl_stdp.v command decoder FSM: opcode/selector dispatch, tick
│   │                             sequencing, input-vector expansion, LED spike monitor.
│   ├── snm_infer_multilane_stdp.v the K-lane array + the 3-phase tick barrier
│   │                             (B_INFER -> B_STDP -> B_SHIFT) and the spike broadcast.
│   ├── snm_infer_lane_stdp.v    one lane = gather stage + neuron-update stage.
│   ├── snm_gather_lane_stdp.v   per-lane synapse table (syn_row, BRAM/URAM) walk,
│   │                             weighted-input accumulation, and the II=1 STDP update
│   │                             pipeline with its source-indexed spike-history BRAM.
│   ├── snm_neuron_update_lane.v per-neuron membrane pipeline: leak toward reset, saturating
│   │                             input/synapse fold, threshold compare, refractory. Holds
│   │                             the optional Option A (SNM_NEURON_STATE_BRAM) state move.
│   ├── snm_spi_slave.v          64-bit SPI command frame slave, reused unmodified.
│   ├── snm_bram_fifo.v          BRAM-backed FIFO (the UART command buffer).
│   ├── snm_config.vh            generated parameter header (N_MAX, NUM_LANES, widths).
│   └── fpga/                    board-specific tops + the UART bridge (FPGA-only glue).
│       ├── snm_infer_fpga_top_stdp.v   generic FPGA top; also used directly by Basys3.
│       ├── snm_infer_sp701_top_stdp.v  SP701 top (pin/clock wrapper).
│       ├── snm_infer_zcu104_top_stdp.v ZCU104 top (pin/clock wrapper).
│       ├── uart_to_spi_master.v UART<->SPI bridge: 8 UART bytes <-> one 64-bit SPI
│       │                         frame, plus the 8192-byte command FIFO that makes
│       │                         host-side streaming possible (§2.2).
│       ├── uart_rx.v            UART receiver.
│       └── uart_tx.v            UART transmitter.
│
├── constraints/                 per-board Vivado pin/timing constraints.
│   ├── basys3.xdc               Basys3 (Artix-7 xc7a35t).
│   ├── sp701.xdc                SP701 (Spartan-7 xc7s100).
│   └── zcu104.xdc               ZCU104 (Zynq UltraScale+ xczu7ev).
│
├── scripts/                     Vivado Tcl. Build scripts honour SPIKEENGINE_BUILD_DIR
│   │                             (defaults under TEMP) and copy results back (§15).
│   ├── build_infer_basys3_stdp_cap256x8.tcl      fixed packaged Basys3 build (N=256, K=8).
│   ├── build_infer_sp701_stdp_maxcap352x8.tcl    fixed packaged SP701 build (N=352, K=8).
│   ├── build_infer_zcu104_stdp_maxcap1024x16.tcl fixed packaged ZCU104 build (N=1024, K=16).
│   ├── build_infer_basys3_stdp_custom.tcl        parametric Basys3 build: N/K/SYN_CAP via
│   │                                              -tclargs (no Python wrapper; §1.2).
│   ├── build_infer_sp701_stdp_custom.tcl         parametric SP701 build.
│   ├── build_infer_zcu104_stdp_custom.tcl        parametric ZCU104 build.
│   ├── build_dataset_zcu104.tcl                  ZCU104 dataset builds with Option A
│   │                                              (neuron state in BRAM) enabled.
│   ├── program_infer_basys3_stdp_cap256x8.tcl    JTAG program, Basys3.
│   ├── program_infer_sp701_stdp_maxcap352x8.tcl  JTAG program, SP701.
│   ├── program_infer_zcu104_stdp_maxcap1024x16.tcl JTAG program, ZCU104.
│   └── fpga_waivers.tcl                          documented, benign build-message waivers;
│                                                  masks no timing or functional problem.
│
├── bitstreams/                  pre-built, timing-closed .bit files (the no-Vivado path).
│   ├── basys3/snm_infer_basys3_stdp_cap256x8.bit          packaged default, N=256/K=8, W16/D24.
│   ├── sp701/snm_infer_sp701_stdp_maxcap352x8.bit         packaged default, N=352/K=8, W16/D24.
│   ├── sp701/snm_infer_sp701_microseer_N90_K8_cap200_w16.bit microseer, wide FP + Option A;
│   │                                                       hardware-validated (48/48).
│   ├── zcu104/snm_infer_zcu104_stdp_maxcap1024x16.bit     packaged default; LEGACY 8-bit build.
│   ├── zcu104/snm_infer_zcu104_miniseer_N2116_K8_bramA.bit miniseer, WNS +0.382 ns.
│   ├── zcu104/snm_infer_zcu104_cora_N2715_K8_bramA.bit     cora, WNS +0.498 ns.
│   └── zcu104/snm_infer_zcu104_citeseer_N3318_K8_bramA.bit citeseer, SYN_CAP=7500, WNS +0.889 ns.
│
├── examples/                    runnable end-to-end examples. The two main ones ship as
│   │                             both a script and a notebook of the same content.
│   ├── __init__.py
│   ├── digits_stdp_e2e.py       sklearn digits: on-chip STDP training (bit-exact vs
│   │                             SuperNeuroMAT) then on-chip rate-readout inference.
│   ├── digits_stdp_e2e.ipynb    notebook form of the above.
│   ├── citation_gnn_fpga.py     citation-graph SNN classifier on hardware, with a
│   │                             fixed-point-faithful software cross-check per paper.
│   │                             Also measure_timestep() for per-tick wall-clock time.
│   ├── citation_gnn_fpga.ipynb  notebook form of the above.
│   ├── logic_gates.ipynb        minimal AND/OR/XOR networks; smallest working example.
│   ├── bars_and_stripes.ipynb   small pattern-classification demo.
│   └── build_vs_packaged.ipynb  compares a from-RTL build against the packaged bitstream.
│
├── gui/                         PyQt5 desktop application (§3). NPU-STDP boards only.
│   ├── __init__.py
│   ├── snm_gui.py               the application: board selection, connect/program, network
│   │                             tables, run controls, live spike view. main() is the
│   │                             `spikeengine-gui` entry point.
│   ├── snm_npu_stdp.py          NPU-STDP board registry + LaneEngineStdpDevice, which
│   │                             subclasses this package's InferEngineStdpDevice.
│   ├── snm_npu.py               STUB. Non-STDP NPU is not supported in this copy; every
│   │                             call raises with a clear message (is_npu() stays real).
│   ├── snm_driver.py            STUB. The classic single-core engine is not supported here;
│   │                             kept so module-level imports and `except SNMError` hold.
│   ├── snm_network.py           GUI-side network builder/editor model.
│   ├── snm_boards.py            board-registry access, reads boards.yaml.
│   ├── snm_config.py            generated host parameter values.
│   ├── snm_presets.py           prebuilt example networks the GUI can load directly.
│   ├── snm_snn_io.py            JSON import/export ("superneuromat-fpga-snn" v1).
│   ├── snm_digits_example.py    digits STDP classifier wired for the GUI; reuses the
│   │                             validated examples/digits_stdp_e2e.py recipe.
│   ├── snm_capacity_example.py  full-chip ZCU104 capacity exercise.
│   ├── network_view.py          2D matplotlib network rendering.
│   ├── pyvista_network_view.py  optional 3D view; falls back to 2D if pyvista is absent.
│   ├── graph_layout.py          force-relaxation node layout (2D and 3D).
│   └── boards.yaml              GUI board definitions.
│
└── tests/                       pytest. Software-only unless noted.
    ├── test_package.py          import/API surface, board catalogue, packaged bitstreams
    │                             exist, software reference runs. Contains the one
    │                             hardware test (skipped unless SE_STDP_PORT is set).
    ├── test_capacity.py         capacity model against real Vivado build numbers, and
    │                             proof the capacity guard fires before any wire write.
    └── test_datasets_program_connect.py dataset catalogue, both accuracy metrics,
                                  program.py's per-board script lookup, and connect()'s
                                  dataset= geometry/width override precedence.
```

### 1.1 How Python and RTL relate

They do **not** link at compile time — there is no code generation, no
Python-to-Verilog translation step anywhere in this package. The RTL is
fixed Verilog with `parameter`s (`N_MAX`, `NUM_LANES`, `SYN_CAP_PER_LANE`,
`WEIGHT_W`, `DATA_W`, ...); Python's only lever over the hardware's shape is
choosing values for those parameters and handing them to Vivado. Once a
bitstream is built, the RTL and the Python side are two independent things
connected only by (a) the UART wire protocol (§8) and (b) a **convention**
that Python's device object is told the same geometry the bitstream was
built with — the bitstream has no self-describing header the host can read
back to confirm this automatically.

### 1.2 Building RTL into a bitstream from Python — API and CLI

**Python API** (`build.py`) — reproduces each board's **fixed, packaged**
configuration only (verified against `build.py`'s own
`_BUILD_SCRIPT_BY_BOARD` mapping: no parameters are passed, it always builds
the same script Vivado already produced the packaged `.bit` from):
```python
from superneuromat.spikeengine import build

result = build.build_bitstream("basys3")     # -> build_infer_basys3_stdp_cap256x8.tcl
print(result.wns_ns, result.lut, result.bram, result.drc_errors)
print(result.bitstream_path)                 # the freshly built .bit
print(result.outdir)                         # every artifact: checkpoints + reports
```
It locates a local Vivado install (`find_vivado()`, raises
`VivadoNotFoundError` with a clear message if none is found), runs
`vivado -mode batch -nojournal -source <fixed script>` as a subprocess with
**no `-tclargs`**, and parses the printed summary line and report files.
`result.outdir` contains everything Vivado wrote — nothing is hidden.

**Custom geometry (e.g. a new dataset size) — Tcl only, no Python wrapper.**
The parametric scripts used to build this version's citation-dataset
bitstreams (`scripts/build_infer_basys3_stdp_custom.tcl`,
`scripts/build_infer_sp701_stdp_custom.tcl`,
`scripts/build_infer_zcu104_stdp_custom.tcl`,
`scripts/build_dataset_zcu104.tcl`) accept `N_MAX`/`NUM_LANES`/
`SYN_CAP_PER_LANE` via `-tclargs`, but **`build.build_bitstream()` does not
call them** — they must be invoked directly:
```
vivado -mode batch -nojournal -source scripts/build_dataset_zcu104.tcl \
    -tclargs <N_MAX> <NUM_LANES> <SYN_CAP_PER_LANE> <label>
```
All build scripts (fixed or parametric) follow the same shape: read the RTL
file list, apply `-generic` overrides, run synth → place → route →
`write_bitstream`, print a `TIMING board=... wns=... luts=...` /
`BUILD_DONE ... bitstream=...` summary line (§15 has the full step-by-step).
Wiring the parametric scripts into `build.py` as a proper API
(`build_bitstream(board, n_max=..., num_lanes=..., syn_cap_per_lane=...)`)
is a reasonable follow-up, not yet done.

**Programming a built (or packaged) bitstream:**
```python
from superneuromat.spikeengine import program
program.program("basys3")                                  # packaged bitstream
program.program("zcu104", bitstream=result.bitstream_path)  # a fresh custom build
```
which runs `vivado -mode batch -source scripts/program_infer_<board>_*.tcl
-tclargs <bit path>` and checks for a printed `PROGRAM_DONE` (not just process
exit code) before returning.

### 1.3 Where to start reading

- `runtime.py` — the wire protocol (host↔FPGA communication, §8).
- `rtl/snm_infer_engine_stdp.v` — the top-level RTL module (instantiates
  `cmd_ctrl` + `multilane`, §2).
- `network.py` — how a `superneuromat.SNN` becomes wire commands (§8.4).
- `datasets.py` / `boards.py` — what's been built and validated (§4), and the
  geometry/datapath each board or dataset bitstream expects.

---

## 2. RTL file hierarchy

```
fpga/snm_infer_fpga_top_stdp.v     GENERIC board top: clock/reset handling, instantiates:
  └─ snm_infer_top_stdp.v          mid wrapper: SPI slave + engine, instantiates:
       ├─ snm_spi_slave.v          8-byte SPI command/response framing (§8)
       └─ snm_infer_engine_stdp.v  top-level engine, instantiates:
            ├─ snm_infer_cmd_ctrl_stdp.v      the controller (§5)
            └─ snm_infer_multilane_stdp.v     K parallel lanes + barrier sync (§6), each:
                 └─ snm_infer_lane_stdp.v     one lane = gather + neuron-update
                      ├─ snm_gather_lane_stdp.v      synapse table walk + accumulate + STDP
                      └─ snm_neuron_update_lane.v    membrane/threshold/refractory pipeline
snm_bram_fifo.v                    generic BRAM-inferring FIFO, used internally
fpga/uart_rx.v, uart_tx.v          UART byte framing
fpga/uart_to_spi_master.v          UART <-> internal SPI-like command bus bridge (§8)
```

**Per-board top-level wiring is not uniform** (checked directly, this is not
a symmetric `{basys3,sp701,zcu104}` naming pattern):
- **Basys3** has no dedicated wrapper file — the Vivado build's `-top` is
  `snm_infer_fpga_top_stdp` itself (confirmed in
  `scripts/build_infer_basys3_stdp_cap256x8.tcl`). Its clock is already a
  plain single-ended 100 MHz pin, so no extra clocking IP is needed.
- **SP701 and ZCU104** each get their own wrapper
  (`fpga/snm_infer_sp701_top_stdp.v`, `fpga/snm_infer_zcu104_top_stdp.v`)
  that adds differential-clock input buffering + an MMCM to derive the
  100 MHz core clock, then instantiates `snm_infer_fpga_top_stdp.v`
  underneath.

Every `.v` file is plain ASCII text — confirmed via `file rtl/*.v`. No IP
encryption, no generated/opaque netlists. Open any file directly to read it.

### 2.1 Every RTL file, what it does, and its key functions

Every module below is a single `module` declaration (one module per file) —
verified via `grep "^module" rtl/**/*.v`. Descriptions are drawn from each
file's own header comment, not paraphrased from memory.

**Core engine (`rtl/`):**

| file | module | what it does |
|---|---|---|
| `snm_infer_top_stdp.v` | `snm_infer_top_stdp` | Mid-level wrapper: instantiates `snm_spi_slave` + `snm_infer_engine_stdp`, exposes only the SPI pins + status/reset at this level. |
| `snm_infer_engine_stdp.v` | `snm_infer_engine_stdp` | Top-level engine: instantiates `snm_infer_cmd_ctrl_stdp` (§5) + `snm_infer_multilane_stdp` (§6), wires the command bus between them. |
| `snm_infer_cmd_ctrl_stdp.v` | `snm_infer_cmd_ctrl_stdp` | The controller FSM (§5): decodes commands, dispatches config writes/reads, drives `OP_RUN_START`/`OP_ENGINE_RESET`, owns the LED spike-monitor mux. |
| `snm_infer_multilane_stdp.v` | `snm_infer_multilane_stdp` | Instantiates `NUM_LANES` copies of `snm_infer_lane_stdp`, runs the 3-phase tick barrier (§6): `B_INFER → B_STDP → B_SHIFT`. |
| `snm_infer_lane_stdp.v` | `snm_infer_lane_stdp` | One lane: instantiates `snm_gather_lane_stdp` + `snm_neuron_update_lane`, wires the per-tick accumulator between them. |
| `snm_gather_lane_stdp.v` | `snm_gather_lane_stdp` | Synapse table walk, spike accumulation, and the STDP weight-update pipeline for this lane (§7.3, §9). |
| `snm_neuron_update_lane.v` | `snm_neuron_update_lane` | The 3/4/5-stage membrane pipeline (§7.2) — 4 internal functions: `leak_toward_reset` (membrane decay toward reset value), `saturating_fold` (adds synaptic current, clamps on overflow), `saturating_add` (same-width saturating add, used for external input), `leak_and_input` (combines the two for the 3-stage pipeline variant). |
| `snm_spi_slave.v` | `snm_spi_slave` | SPI-mode-0 (CPOL=0, CPHA=0) slave: decodes the 64-bit `[opcode\|selector\|address\|data]` command frame (§8.1) off the internal SPI bus. |
| `snm_bram_fifo.v` | `snm_bram_fifo` | Generic BRAM-backed FIFO (first-word-fall-through), used internally for the UART command-byte FIFO and any depth-N_MAX event queues — deep queues cheaply, without spending flip-flops. |
| `snm_config.vh` | (header, not a module) | Compile-time defines (`` `ifdef`` pipeline-stage selectors, URAM-packing flag) shared by files that need them — included directly in each file that uses one of its defines, since Vivado compiles each `.v` file as its own compilation unit (a `` `define`` in one file's own `` `include`` is invisible to another file otherwise). |

**Board/transport layer (`rtl/fpga/`):**

| file | module | what it does |
|---|---|---|
| `snm_infer_fpga_top_stdp.v` | `snm_infer_fpga_top_stdp` | Generic board top (§2, used directly by Basys3): clock buffering, reset sync, instantiates `snm_infer_top_stdp` + the UART↔SPI bridge. |
| `snm_infer_sp701_top_stdp.v` | `snm_infer_sp701_top_stdp` | SP701-specific wrapper: differential 200 MHz input buffering + MMCM to derive the 100 MHz core clock, then instantiates `snm_infer_fpga_top_stdp`. |
| `snm_infer_zcu104_top_stdp.v` | `snm_infer_zcu104_top_stdp` | ZCU104-specific wrapper: same role as the SP701 one, for ZCU104's 125 MHz differential input clock. |
| `uart_to_spi_master.v` | `uart_to_spi_master` | The UART↔SPI bridge (§2.2 below): converts 8 UART bytes each direction into one SPI command/response frame, driving the unmodified `snm_spi_slave`. |
| `uart_rx.v` | `uart_rx` | Plain 8N1 UART receiver: 1 start bit, 8 data bits LSB-first, 1 stop bit, mid-bit oversampling; `rx_valid` pulses one clock when a byte is ready. |
| `uart_tx.v` | `uart_tx` | Plain 8N1 UART transmitter: assert `tx_start` with `tx_data` valid; `tx_busy` stays high until the stop bit completes. |

### 2.2 Why UART↔SPI, and the path to an ASIC

**Why this two-layer transport** (UART on the host-facing USB link, SPI
internally on-chip, bridged by `uart_to_spi_master.v`) rather than one
protocol end to end: `snm_spi_slave.v` — the actual command decoder — is
SPI-mode-0, a fixed-frame, edge-clocked protocol with no start/stop framing
of its own. USB-to-board connectivity on these dev boards is UART (a
simple, ubiquitous, driver-free serial link every FTDI/CP210x bridge chip
exposes), not SPI. `uart_to_spi_master.v` exists purely to convert one to
the other: it accumulates 8 incoming UART bytes into the exact 64-bit frame
`snm_spi_slave` expects, drives `spi_cs_n`/`spi_sclk`/`spi_mosi` as if it
were a real SPI master, and shifts the 8 response bytes captured on
`spi_miso` back out over UART. `snm_spi_slave` itself is **used completely
unmodified** — the bridge is a "thin shifter," not a reimplementation, so
the command decoder's behavior is identical regardless of which physical
link reaches it.

**Why keep SPI as the *internal* protocol rather than collapsing straight
to a UART-native command decoder:** this is a deliberate choice for a
future ASIC target, not an accident of incremental development. SPI is a
synchronous, low-pin-count protocol that maps directly onto typical ASIC
I/O pads (a handful of pins: SCLK/MOSI/MISO/CS, no UART baud-rate clock
recovery circuitry needed on-die) and is a standard way to talk to a
custom chip from an external host or microcontroller. Keeping
`snm_spi_slave` as the one true on-chip command interface means: an ASIC
implementation could expose SPI pins directly (dropping the UART bridge
entirely, since UART only exists to reach these FPGA dev boards over USB)
without touching the command decoder or anything upstream of it — the
entire engine (`snm_infer_engine_stdp` down through the lanes) is written
against the SPI-slave's command bus, not against UART framing, so it is
already transport-agnostic in the direction that matters for an ASIC.

### 2.3 Top-level I/O — pin counts, direction, and the debug port

No module in this design has any `inout` (bidirectional) port — every port
at every level is a plain `input` or `output`. Verified by reading each top
module's port list directly:

| board top | inputs | outputs | notes |
|---|---|---|---|
| Basys3 (`snm_infer_fpga_top_stdp`, used directly) | `clk`, `rst_btn`, `uart_rx` (3×1-bit) | `uart_tx` (1-bit), `led[15:0]` (16-bit) | 3 in / 17 out bits |
| SP701 (`snm_infer_sp701_top_stdp`) | `SYSCLK_P`, `SYSCLK_N`, `CPU_RESET`, `uart_rx` (4×1-bit) | `uart_tx` (1-bit), `led[7:0]` (8-bit) | 4 in / 9 out bits |
| ZCU104 (`snm_infer_zcu104_top_stdp`) | `CLK_125_P`, `CLK_125_N`, `GPIO_PB_SW0`, `uart_rx` (4×1-bit) | `uart_tx` (1-bit), `led[3:0]` (4-bit) | 4 in / 5 out bits |

So the entire physical interface, on every board, is: a clock (differential
on SP701/ZCU104), a reset button, and the two UART wires — everything else
(load a network, run a tick, read weights, ...) rides over those two UART
pins as the 8-byte command frames in §8.1. The `led[N:0]` output is the one
extra physical signal, and it doubles as **the debug port**: at the engine
level it comes from `spike_mon[15:0]` (`snm_infer_engine_stdp.v`'s own
comment: `LED[0] = this neuron id`), a live, host-independent window into
whichever `SPIKE_MON_BASE`-selected range of the spike vector you build with
— you can watch a chosen neuron range fire in real time with no UART
traffic at all. There is no other dedicated debug port (no ILA, no JTAG
debug core) in this RTL; JTAG on these boards is used only for bitstream
programming, not runtime introspection.

---

## 3. GUI

The desktop GUI is **part of this package**: `src/superneuromat/spikeengine/gui/`. Install it
with `pip install superneuromat`, launch it with the `spikeengine-gui` console
script (entry point in `pyproject.toml`, resolves to `spikeengine.gui.snm_gui:main`).

The canonical source the GUI is copied from is
`board_variants/npu_stdp_dev/gui/fpga_gui_app/` — that directory is kept around as
the original development copy, but `src/superneuromat/spikeengine/gui/` is the one that ships.
Every `*.py` and `*.yaml` file from the canonical copy is present in the package
copy (verified file-for-file 2026-07-30; the only files in the canonical copy not
present are `__pycache__/*.pyc`, which are compiled bytecode caches, not source —
they regenerate automatically and are never something a copy needs to carry).

**What changed on the way in** — the canonical copy supports three board
"kinds" (classic single-core SuperNeuroMAT3, non-STDP NPU, NPU-STDP), two of
which (classic, non-STDP NPU) hard-depend at import time on the separate
`superneuromat.spikeengine` package (a different, non-lane wire protocol this
package does not implement). Rather than pull that dependency in or leave those
two paths as import-time crashes, they were narrowed out of the package copy:
- `snm_gui.py` — the board-selector dropdown no longer populates classic-board
  entries (the population loop is commented out); only NPU-STDP boards are
  offered.
- `snm_npu_stdp.py` — rewritten so `LaneEngineStdpDevice` subclasses this
  package's own `spikeengine.runtime.InferEngineStdpDevice` instead of the
  other package's `LaneEngineDevice`. Redundant method overrides were dropped
  since the base class already provides byte-identical wire-packing logic.
  `load_network` (GUI-specific network-format loading) and the module-level
  board-lookup helpers are kept.
  `snm_npu.py` and `snm_driver.py` — replaced with stubs. Every function raises
  a clear `RuntimeError` naming the missing dependency if actually called;
  nothing in the package copy's UI can reach a code path that would call them,
  since classic/non-STDP-NPU are never offered in the dropdown. This keeps
  `from . import snm_driver as snm` / `from . import snm_npu` importing cleanly
  without silently faking data.

NPU-STDP is a strict superset of what non-STDP NPU offered, so nothing is lost
functionally by dropping it here — a user who specifically needs the classic
engine or non-STDP NPU still has the canonical
`board_variants/npu_stdp_dev/gui/fpga_gui_app/` copy, which keeps its dependency
on `superneuromat.spikeengine`.

**Verification status**: `import spikeengine.gui.snm_gui` succeeds standalone
(including all PyQt5 imports and the rewired NPU-STDP path) — confirmed by
direct import test, not just `py_compile`. Actual widget construction
(`SNNConsole()`) was **not** verified interactively: this development
environment's offscreen Qt/VTK path segfaults on OpenGL context creation
(`vtkWin32OpenGLRenderWindow: failed to get valid pixel format`), a pre-existing
environment limitation unrelated to the rewire. This is a real gap — the module
loads correctly, but the window has not been visually exercised — and should be
closed by running the GUI on a machine with a real display before signoff.

---

## 4. Datasets tested — summary (full detail in USER_MANUAL.md §2)

| dataset | neurons | synapses | board | status |
|---|---|---|---|---|
| digits (8×8), logic_gates, bars_and_stripes | small | — | Basys3 | hardware-validated per `boards.py` (dated 2026-07-27) — not re-run this session |
| microseer | 90 | 996 | Basys3, **SP701** | hardware-validated on both: 48/48 papers SW==HW on each (one-vs-rest 0.6424, identical on both boards) |
| miniseer | 2,116 | 31,456 | ZCU104 | hardware-validated: 120/120 papers SW==HW, one-vs-rest 0.7667 |
| cora | 2,715 | 46,788 | ZCU104 | hardware-validated: 140/140 papers SW==HW, one-vs-rest 0.8173 |
| citeseer | 3,318 | 47,616 | ZCU104 | hardware-validated: 120/120 papers SW==HW, one-vs-rest 0.5806 (`SYN_CAP_PER_LANE=7500`, WNS +0.889 ns) |
| pubmed | 19,720 | 206,710 | none | not feasible on current hardware (§14); software reference only |

---

## 5. The controller (`snm_infer_cmd_ctrl_stdp.v`)

A single FSM per engine instance, states (RTL `localparam`, authoritative):

```
S_IDLE  S_DECODE  S_VEC_EXPAND  S_TICK_PRE  S_TICK_WAIT  S_TICK_CAP  S_RESP  S_SYN_RD_WAIT
```

It decodes the incoming 64-bit command word (opcode/selector/address/data, §8)
in `S_DECODE`, dispatches config writes/reads directly (single cycle where
possible), and for `OP_RUN_START` transitions through `S_TICK_PRE` →
`S_TICK_WAIT` (waiting on the multilane engine's `tick_done`) → `S_TICK_CAP`
before responding. `S_SYN_RD_WAIT` exists because a synapse readback has a
2-cycle BRAM read latency the controller must wait out before returning data.
The controller also owns `OP_ENGINE_RESET` (soft-reset: pulses a datapath
reset without a bitstream reload) and the LED spike-monitor mux
(`SPIKE_MON_BASE`).

---

## 6. Lanes (the "NPU" / Neuron Processing Unit) and how they cooperate

**Terminology note:** the RTL/code calls this unit a "lane"
(`snm_infer_lane_stdp`); when discussed as a processing unit it is referred to
as an NPU lane — same thing, two names.

**Partitioning:** destination-partitioned. `lane = dst_neuron_id % NUM_LANES`.
Each lane owns `LOCAL_D = ceil(N_MAX / NUM_LANES)` neurons and the full set of
their incoming synapses (`syn_row`, sized by the build parameter
`SYN_CAP_PER_LANE` — this is what the "sparse capacity" flexibility feature
controls, decoupled from `N_MAX` since this version).

**A lane = two sub-modules:**
- `snm_gather_lane_stdp.v` — walks the lane's synapse table
  (`dst_ptr[local_dst] .. dst_ptr[local_dst+1]`), accumulates weighted
  incoming spikes per destination neuron, and runs the STDP weight-update
  pipeline.
- `snm_neuron_update_lane.v` — the per-neuron membrane/threshold/refractory
  pipeline (§7), consuming the gather stage's accumulator.

**Barrier synchronization (multilane, `snm_infer_multilane_stdp.v`):** a tick
is a 3-phase barrier-synchronous FSM (`B_INFER → B_STDP → B_SHIFT`), each
phase gated by its own per-lane "done" bit-vector — no phase starts until
**every** lane has finished the previous one:

```verilog
// snm_infer_multilane_stdp.v — the 3-phase tick barrier
localparam [1:0] B_INFER = 2'd1, B_STDP = 2'd2, B_SHIFT = 2'd3;
reg [NUM_LANES-1:0] infer_seen, stdp_seen, hist_refresh_seen;
...
B_INFER: begin                            // 1. all lanes run inference in parallel
    infer_seen <= infer_seen | lane_tick_done;
    if (&infer_seen_next) begin           // wait for EVERY lane's bit to be set
        stdp_start_r <= 1'b1;             // pulse: all lanes' spike_out now valid
        bstate <= B_STDP;
    end
end
B_STDP: begin                             // 2. STDP weight update, same wait-for-all pattern
    stdp_seen <= stdp_seen | lane_stdp_done;
    if (&stdp_seen_next) begin
        hist_refresh_start_r <= 1'b1;     // pulse: begin every lane's history refresh
        bstate <= B_SHIFT;
    end
end
B_SHIFT: begin                            // 3. STDP-history window refresh, same pattern
    hist_refresh_seen <= hist_refresh_seen | lane_hist_refresh_done;
    if (&hist_refresh_seen_next) begin
        tick_done <= 1'b1;                // only NOW is the tick complete
        bstate <= B_IDLE;
    end
end
```

The reason for three separate barriers, not one: phase 1 must fully settle
(every lane's `spike_out` valid for this tick) before phase 2 can start,
because STDP needs the complete, final spike vector — no lane can start
learning from a neighbor's partially-computed result. Phase 3 (history
shift) runs after STDP for the same reason in reverse: the history window
used by STDP must reflect ticks *before* this one, so the shift-in of this
tick's result is deferred until STDP has already used the old window.

**How neurons/synapses stay in sync across lanes:** every lane receives a
broadcast copy of the full spike vector each tick (`spike_frozen`, replicated
per lane — the O(N·NUM_LANES) flip-flop cost discussed in §12) so any lane can
look up any source neuron's spike bit for its own gather, without cross-lane
communication during the tick. Config writes (`cfg_*`) are similarly broadcast
to all lanes but gated by a `cfg_lane` selector so only the addressed lane
actually writes.

---

## 7. Neuron and synapse update logic

### 7.1 Per-neuron state — what's stored, and where

Every neuron in a lane has: `threshold`, `leak`, `reset_state`,
`refrac_period`, `input_enable` (config, host-loaded once), `input_value`
(per-tick external stimulus, host-loaded), and `vmem`/`refrac_count`
(runtime state, engine-owned, persists across ticks). All eight arrays are
`[0:LOCAL_D-1]`, declared in `snm_neuron_update_lane.v`.

`vmem`/`refrac_count` move to Block RAM under the opt-in Option A build
(`` `ifdef SNM_NEURON_STATE_BRAM``, see §12/§13) — every other field stays in
distributed fabric. `input_value` is deliberately excluded from that move
even though it would also cut flip-flop count, because it is the one
host-written runtime array: a multi-cycle BRAM clear-on-reset would race a
host rewrite (§13 has the fuller rationale). The relevant code, showing the
config-write path is a single always-block driver (config survives
soft-reset; `input_value`'s bulk reset on `!reset_n` is separate and
unaffected by the Option A define):

```verilog
// snm_neuron_update_lane.v
always @(posedge clk) begin
`ifdef SNM_NEURON_STATE_BRAM
    if (cfg_param_we) begin
        if (cfg_param_field == PF_THR  || cfg_param_field == PF_ALL)
            threshold[cfg_param_idx]     <= cfg_param_threshold;
        if (cfg_param_field == PF_LEAK || cfg_param_field == PF_ALL)
            leak[cfg_param_idx]          <= cfg_param_leak;
        if (cfg_param_field == PF_RST  || cfg_param_field == PF_ALL)
            reset_state[cfg_param_idx]   <= cfg_param_reset_state;
        if (cfg_param_field == PF_RP   || cfg_param_field == PF_ALL)
            refrac_period[cfg_param_idx] <= cfg_param_refrac_period;
    end
    if (!reset_n) begin
        for (cfg_k = 0; cfg_k < LOCAL_D; cfg_k = cfg_k + 1)
            input_value[cfg_k] <= {DATA_W{1'b0}};
    end else if (in_we) begin
        input_value[in_idx] <= in_value;
    end
`else
    ... // same logic, input_value/threshold/leak/etc. all plain fabric arrays
```

### 7.2 Membrane update pipeline

`snm_neuron_update_lane.v` walks one neuron per cycle through a 3/4/5-stage
pipeline (stage count is a per-board timing-closure tunable, `` `ifdef
SNM_INFER_NEURON_PIPE_{3,4,5}STAGE``; all boards currently build 5-stage):

1. Read this neuron's state/config (the arrays in §7.1).
2. **Leak toward reset** — `leak_toward_reset(vmem, reset_state, leak)`
   moves the membrane potential toward its reset value by `leak` per tick,
   clamped so it never overshoots past `reset_state`:
   ```verilog
   // snm_neuron_update_lane.v — leak_toward_reset
   if (vmem_ext > reset_ext) begin
       candidate = vmem_ext - leak_ext;
       leak_toward_reset = (candidate < reset_ext) ? reset_in : candidate[DATA_W-1:0];
   end else if (vmem_ext < reset_ext) begin
       candidate = vmem_ext + leak_ext;
       leak_toward_reset = (candidate > reset_ext) ? reset_in : candidate[DATA_W-1:0];
   end else begin
       leak_toward_reset = reset_in;
   end
   ```
3. Optionally saturating-add external `input_value`, if `input_enable &&
   input_valid` this tick (same saturating-add primitive as step 4).
4. **Fold in synaptic current** — `saturating_fold(leaked_vmem,
   synapse_accumulator)` adds the gather stage's accumulated current,
   clamping to the `DATA_W`-bit signed range instead of wrapping on
   overflow:
   ```verilog
   // snm_neuron_update_lane.v — saturating_fold
   sum = {{(ACCUM_W+1-DATA_W){base[DATA_W-1]}}, base} + {accum[ACCUM_W-1], accum};
   if (sum > max_value) saturating_fold = {1'b0, {(DATA_W-1){1'b1}}};       // clamp high
   else if (sum < min_value) saturating_fold = {1'b1, {(DATA_W-1){1'b0}}};  // clamp low
   else saturating_fold = sum[DATA_W-1:0];
   ```
5. **Decide**: if `refrac_count != 0`, suppress the spike but still
   integrate (refractory). Else if `final > threshold`, fire
   (`spike_out=1`, reset `vmem`, load `refrac_count = refrac_period`). Else
   no spike, keep `final` as the new `vmem`.

### 7.3 Synapse gather

`snm_gather_lane_stdp.v`: for each destination neuron, walks its contiguous
range in the lane's synapse table (bounds given by `dst_ptr[local_dst]` ..
`dst_ptr[local_dst+1]`), and for each `(src, weight)` entry checks
`spike_frozen[src]` (this tick's frozen input-spike vector, §6); if set,
accumulates `weight` into that destination neuron's running sum for this
tick. The STDP weight-update pipeline (§9) runs in the same module, after
the barrier described in §6.

---

## 8. Communication and I/O: host ↔ FPGA

Everything about how the host talks to the board — physical link, on-chip
framing, opcodes, and how inputs go in / outputs come back — grouped here
since they're one continuous path: `runtime.py` method call → 8-byte frame →
`cmd_ctrl` decode → RTL action → 8-byte response → return value.

### 8.1 Physical link and on-chip bridge

**Physical path:** USB-UART (not raw SPI over pins — SPI is internal to the
FPGA). The board exposes a USB-UART bridge; the host talks a byte-serial
link to it. `_transport.py` matches the runtime UART by USB VID:PID.
Verified directly on hardware this session: **both SP701 and ZCU104 present
as FTDI devices (VID:PID `0403:6011`)** — an earlier version of this
document claimed SP701 used a Silicon Labs CP210x bridge; that was wrong,
corrected after physically probing the board's enumerated COM ports. Boards
with a multi-channel FTDI chip (SP701's FT4232H exposes 3 UART-class ports,
ZCU104's FT4232H exposes JTAG on one interface plus 3 UART-class ports) need
the actual runtime UART identified by probing each candidate port with a
real command round-trip — `_transport.py` disambiguates by declared
interface index where possible, but the code path was validated by direct
trial against physical hardware, not by trusting the interface-index
convention alone.

**On-chip bridge** (`fpga/uart_to_spi_master.v`): converts UART bytes to an
internal SPI-Mode-0 command bus that drives the **unmodified**
`snm_spi_slave.v` (reuse of existing infrastructure, §13). One host
transaction = exactly 8 bytes each direction, forming a 64-bit command word:

```
byte0 = opcode        byte1 = selector       byte2:3 = address (hi:lo)
byte4:7 = data (32-bit, MSB first)
```

Response: the 8 bytes captured on the same frame — status in byte0, response
data in byte4..7. The host sends a command frame, then a NOP frame to read the
previous response (the SPI slave protocol is request-then-poll).

### 8.2 Opcodes and selectors

**Opcodes** (`snm_infer_cmd_ctrl_stdp.v`, authoritative):

| opcode | value | purpose |
|---|---|---|
| `OP_WRITE_CONFIG` | `0x01` | write a config field (selector picks which, see `IL_*` below) |
| `OP_READ_CONFIG` | `0x02` | read back a config field (used for synapse-weight readback) |
| `OP_RUN_START` | `0x05` | advance one tick (blocks until `tick_done`) |
| `OP_READ_STATUS` | `0x09` | read the status word |
| `OP_READ_OUTPUT` | `0x0a` | read one 32-neuron word of this tick's spike vector |
| `OP_INPUT_VECTOR_WRITE` | `0x0c` | write a 32-neuron input mask word |
| `OP_CLEAR_ERROR` | `0x10` | clear the error/status latch |
| `OP_ENGINE_RESET` | `0x11` | soft-reset (clears runtime state, not config, no reprogram) |

`IL_*` selectors (used with `OP_WRITE_CONFIG`/`OP_READ_CONFIG`) address which
config field: `IL_THRESHOLD 0x06`, `IL_LEAK 0x07`, `IL_RESET_STATE 0x08`,
`IL_REFRAC 0x09`, `IL_INPUT_ENABLE 0x0A`, `IL_DPTR 0x0D`, `IL_SYN 0x0E`,
`IL_INPUT_VALUE 0x0F`, `IL_SYN_ADDR_HI 0x19`, `IL_INPUT_VEC_VAL 0x1A`,
`IL_SET_LANE 0x1B`, `IL_STDP_TABLE 0x1C` (window-slot apos/aneg), `IL_STDP_ENABLE 0x1D`.

### 8.3 Regular vs. bulk commands

Every `runtime.py` method (e.g. `write_synapse`, `run_tick`) issues one
command and waits for its response — simple, robust, one round-trip latency
each. `begin_bulk()`/`end_bulk()` instead **stream** a batch of command words
over the UART without waiting for each individual response, then collect all
responses at the end — used by `load_network()` and
`run_schedule(streaming=True)` because loading tens of thousands of synapses
one-at-a-time would be prohibitively slow (each Windows serial round-trip has
real overhead independent of the FPGA's actual command latency). Bulk mode is
where the transient `InferEngineError: stream stalled` (USER_MANUAL §9) can
occur on very long bursts; it does not corrupt state (see `retry` pattern in
`examples/citation_gnn_fpga.py`).

### 8.3.1 The full host→FPGA path, step by step

What physically happens to one 8-byte command word, from a Python call to the
lane memory it lands in:

1. **Host builds the word.** `runtime.py`'s `_cmd_word(op, sel, addr, data)`
   packs `opcode<<56 | sel<<48 | addr<<32 | data` into a 64-bit integer, then
   `.to_bytes(8, "big")`.
2. **Host writes bytes to the serial port.** In lock-step mode this is one
   `write` + blocking `read` per command. In bulk mode many words are
   concatenated and pushed with a single large `write_raw()` call, so one USB
   transfer carries hundreds of commands instead of one.
3. **UART receives, byte by byte.** `uart_rx.v` deserializes at 4 Mbaud.
4. **Bytes queue in the on-chip FIFO.** `uart_to_spi_master.v` buffers them in
   a BRAM-backed FIFO, `CMD_FIFO_DEPTH = 8192` bytes = **1024 command frames**.
   This buffer is the entire reason streaming works: the host can keep sending
   while the engine is still busy with earlier commands, instead of the link
   idling through every round trip.
5. **Bridge shifts one frame out as SPI.** Once 8 bytes are available, the
   bridge clocks them into `snm_spi_slave.v` as a 64-bit SPI Mode-0 frame.
   SPI speed is irrelevant here — the UART is the bottleneck by a wide margin.
6. **Command controller decodes and acts.** `snm_infer_cmd_ctrl_stdp.v`
   dispatches on opcode/selector: a config write goes straight to the
   addressed lane's memory (synapse table BRAM, or the neuron parameter arrays
   in fabric); `OP_RUN_START` starts a tick; a read schedules a response.
7. **Response returns the same way**, 8 bytes back over SPI→UART→USB. In bulk
   mode responses are drained asynchronously rather than one at a time.

Nothing is buffered on-chip beyond that single 1024-frame FIFO. Once a frame
leaves it, it is decoded and applied in the same pass — there is no staging
area and no separate commit step.

### 8.3.2 How many commands fit in 1 ms

All boards run the UART at **4 Mbaud** (`boards.py`). One UART byte costs 10
bit-times (8 data + start + stop), and one command is 8 bytes:

```
byte rate    = 4,000,000 / 10          = 400,000 bytes/s
frame rate   = 400,000 / 8             =  50,000 commands/s
per 1 ms     = 50,000 / 1000           =      50 commands
```

**So roughly 50 command words per millisecond, per direction** — that is the
hard link ceiling, and it is the same whether or not streaming is used.
Streaming does not raise this number; it lets you *reach* it. Without
streaming, throughput is set by round-trip latency (each command waits for its
own response, and Windows USB-serial latency dominates), which measured
**2,245 commands/s** in the case recorded in `uart_to_spi_master.v`'s header —
about 4.5% of the wire limit, or ~2 commands per millisecond.

Two consequences worth planning around:

- **Loading a network is link-bound, not compute-bound.** A 47,616-synapse
  dataset needs >47,616 command words, so ≥ ~1 second on the wire at best.
  This is why per-paper reloads dominate the citation-example runtime.
- **Per-tick input is cheap by comparison.** Setting stimulus for N neurons
  costs one `IL_INPUT_VALUE` write per neuron whose current changes, plus
  `ceil(N_MAX/32)` mask words, plus one `OP_RUN_START`. For a 1,156-input
  layer with every input active that is ~1,193 frames ≈ 24 ms streamed — but
  a typical sparse event bin touches far fewer neurons, and unchanged
  `input_value` registers do not need rewriting at all.

The host keeps at most `1024 - max(16, 1024/8) = 896` frames in flight
(`runtime.py:_stream_window`), sized to stay below the FIFO depth so it can
never overflow while still covering the link's bandwidth-delay product.

### 8.4 Inputs — how spikes/current are given to neurons

Two distinct input mechanisms exist:
1. **Config load** (`load_network` in `network.py`): writes the network's
   synapses (`OP_WRITE_CONFIG`/`IL_SYN` + `IL_DPTR` per lane), per-neuron
   parameters (`IL_THRESHOLD`/`IL_LEAK`/`IL_RESET_STATE`/`IL_REFRAC`/
   `IL_INPUT_ENABLE`), and the STDP tap table (`IL_STDP_TABLE`,
   `IL_STDP_ENABLE`). This happens once per run (or once per paper, for the
   citation examples, which reload fresh initial weights each time).
2. **Per-tick stimulus** (`run_schedule` in `network.py`, or
   `runtime.input_vector()`/`write_synapse`-style direct calls): injects an
   external current into specific neurons for a specific tick via
   `IL_INPUT_VALUE` (the per-neuron current magnitude, a signed `DATA_W`-bit
   fixed-point value) and `OP_INPUT_VECTOR_WRITE` (a 32-neuron bitmask marking
   which neurons' `input_value` applies this tick, gated by that neuron's
   persistent `input_enable`). `input_value` is a **persistent** register — a
   neuron injected on tick T and NOT re-addressed on tick T+1 must be
   explicitly re-zeroed, or its stale value reapplies whenever `input_valid`
   is asserted for any other neuron (`run_schedule` handles this correctly;
   direct low-level use must replicate it — see its docstring).

For the citation datasets specifically, one input spike (constant magnitude
2.0, quantized) is injected into the test paper's neuron at tick 0; everything
after that is graph propagation + STDP, not further external input.

### 8.5 Outputs — how results are read back

- **Spikes**: `read_spikes()` issues `OP_READ_OUTPUT` once per 32-neuron word
  (`spike_words = ceil(N_MAX/32)`) and assembles the full boolean vector in
  global neuron order.
- **Learned/current synapse weights**: `OP_READ_CONFIG`/`IL_SYN` with the
  synapse's lane + entry index (`read_synapse` in `runtime.py`) returns
  `(source_neuron_id, raw_weight)`; the host divides by `2**frac_bits` to get
  the real value. This is how the citation examples classify a paper — they
  read back the paper→topic synapse weights after the run and pick the
  strongest.
- **Status/error**: `read_status()` (`OP_READ_STATUS`) returns a status word;
  `clear_error()` (`OP_CLEAR_ERROR`) clears it.
- **LEDs** (hardware-only, visual): `spike_mon` is a 16-bit window into the
  spike vector starting at the build parameter `SPIKE_MON_BASE`, wired
  directly to board LEDs — a live, no-host-needed view of a chosen neuron
  range (typically the output/class neurons).

---

## 9. STDP implementation

STDP is **global** on hardware — there is no per-synapse enable bit on the
wire (`load_network` stores only `(src, weight)` per synapse entry). One
`stdp_global_enable` bit (config `IL_STDP_ENABLE`) turns learning on/off for
the whole engine. The tap table (`IL_STDP_TABLE`) holds `apos`/`aneg` values
per window-slot (`STDP_WINDOW` slots, default 5), applied based on each
synapse's `hist_mem` lookup (per-lane, source-indexed spike-history BRAM,
tracking whether each source fired within the last `STDP_WINDOW` ticks). The
update happens as an initiation-interval-1 pipeline in the gather stage,
after the barrier described in §6 (so it only runs once every lane's spike
vector for this tick is final).

This global behavior was checked against the software model's *selective*
per-synapse STDP (only some synapses have `enable_stdp=True` in the SGNN
model): forcing global STDP and comparing against the model's own
selective-STDP classification gives identical results (verified on a
20-paper microseer subset, float precision, 20/20 agreement). This is why
the hardware-faithful software reference in `examples/citation_gnn_fpga.py`
forces `snn.enable_stdp = [True] * len(...)` before inference. The full
120/140-paper hardware runs (miniseer/cora, §4) additionally confirm the
FPGA's global-STDP output matches this same software reference on every
paper.

---

## 10. Fixed-point representation and tradeoffs

### 10.1 Encoding: decimals and negative values

All values are **signed integers on a `1/2**frac_bits` grid** — plain
two's-complement fixed-point, no floating point anywhere in the datapath.

- **Decimals**: `quantize_raw(value, frac_bits) = round(value * 2**frac_bits)`
  (`network.py`) maps a real number onto the nearest integer on that grid;
  the device stores only the integer. Reading back divides by the same
  scale (`read_weights`: `raw / 2**frac_bits`). `frac_bits` is a **host-side
  convention only** — nothing in the RTL "knows" where the binary point is;
  it just does signed integer arithmetic. Get `frac_bits` wrong between load
  and readback and every value is silently scaled wrong, with no error.
- **Negative values**: standard two's complement, sign-extended in the RTL
  by replicating the MSB (see §7.2's `leak_toward_reset`/`saturating_fold`
  excerpts: `{vmem_in[DATA_W-1], vmem_in}` is exactly this). An N-bit signed
  field represents `[-2^(N-1), 2^(N-1)-1]` raw integers.

### 10.2 Limits per field — and a real overflow-behavior split

Not every fixed-point field behaves the same on overflow. Verified directly
against `network.py`/`runtime.py`:

| field | width | write path | **on overflow** |
|---|---|---|---|
| synapse weight | `weight_w` | `_clamp_weight(quantize_raw(...), weight_w)` | **saturates** (clamps to max/min magnitude) |
| STDP taps (apos/aneg) | `weight_w` | same `_clamp_weight` path | **saturates** |
| threshold / leak / reset_state | `data_w` | `quantize_raw(...) & ((1<<data_w)-1)` | **wraps** (bitmask, not clamped) |
| per-tick input current | `data_w` | `runtime.write_input_value`: `value & self._data_mask` | **wraps** (bitmask, not clamped) |

The wrap-on-overflow paths are a real, previously-hit hazard, not a
theoretical one: an in-repo fix note (`network.py`, 2026-07-27) documents a
concrete incident where a teacher current of 100 at `frac_bits=10` produced
raw value 102400, and an earlier hardcoded `& 0xFFFF` mask wrapped it to
36864 — silently injecting current equivalent to 36, not 100, and a
threshold-99 neuron that should have fired never did. The fix widened the
mask to the actual `data_w`, but **the underlying wrap-instead-of-saturate
behavior is unchanged** for `data_w`-wide fields — only the synapse-weight
path saturates. See §14 for this as a standing limitation.

**Concrete limits at this version's shipped citation-dataset config**
(`weight_w=16`, `data_w=24`, `frac_bits=13`):

| field | raw range | real-value range | resolution |
|---|---|---|---|
| synapse weight (`weight_w=16`) | [-32768, 32767] | **[-4.0, 4.0)** (clamped) | `2⁻¹³ ≈ 1.22×10⁻⁴` |
| threshold / leak / reset_state / input current (`data_w=24`) | [-8388608, 8388607] | **[-1024.0, 1024.0)** (wraps if exceeded) | `2⁻¹³ ≈ 1.22×10⁻⁴` |

Both share the same resolution (same `frac_bits`), but the membrane-domain
fields have a **256× larger range** than weights, because `data_w=24` is
wider than `weight_w=16` — this headroom is exactly why the citation-dataset
recipe (§9/§13) rescales only the *driving weight* (which travels the
`weight_w`-limited synapse path) down to 2.0, while thresholds (which travel
the wider `data_w` path) can stay at their natural values without hitting
the wrap hazard.

---

## 11. Bit-widths, max sizes, and RTL parameter reference

| parameter | meaning | default | wide/dataset builds |
|---|---|---|---|
| `N_MAX` | max neuron count the build supports | 1024 | dataset-specific (90–3,318 built this version) |
| `NUM_LANES` (K) | parallel lane count | 16 | 8 (all current dataset builds) or 16 |
| `WEIGHT_W` | synapse weight bit-width (signed) | 8 | **16** (all wide/dataset builds) |
| `DATA_W` | membrane/accumulator bit-width (signed) | 16 | **24** (all wide/dataset builds) |
| `REF_W` | refractory-period/count bit-width | 8 | 8 |
| `ACCUM_W` | per-tick synapse accumulator width | 32 | 32 |
| `STDP_WINDOW` | STDP history depth (ticks) | 5 | 5 |
| `SYN_CAP_PER_LANE` | synapse-table depth per lane | `LOCAL_D * N_MAX` (dense) | build-specific, sized to the real per-lane in-degree + margin (§14) |
| `SPIKE_MON_BASE` | first neuron shown on LED[0] | 0 | 64 (all board tops) |

**Max all-to-all (dense) configuration validated on real hardware:**
Basys3, `N_MAX=256, NUM_LANES=8` (`SYN_CAP_PER_LANE=8192` = dense worst case),
`WEIGHT_W=16, DATA_W=24`. WNS +0.174 ns, LUT 20,437/20,800 (98.25%), FF
34,363/41,600 (82.6%), BRAM 50/50 (100%) — re-verified against
`post_route_util.rpt`/`post_route_timing.rpt` this session. Hardware
validation (on-chip STDP training bit-exact, 640/640 weights, digits
classifier) is per `boards.py`'s existing record (dated 2026-07-27); not
re-run this session.

**Max sparse configuration validated on real hardware (as of this document):**
ZCU104 `cora`, `N_MAX=2715, NUM_LANES=8, SYN_CAP_PER_LANE=6436`. WNS +0.498
ns, LUT 112,873/230,400 (48.99%), FF 178,557/460,800 (38.75%), BRAM 30/312
(9.62%), URAM 8/96 (8.33%). Hardware-validated: 140/140 test papers exact
SW==HW agreement. (citeseer, `N_MAX=3318`, is a larger sparse build — see
USER_MANUAL.md for its final validated status once the rebuild described
there completes.)

---

## 12. Resource consumption (LUT / FF / BRAM / URAM) — what scales with what

Measured this version via real Vivado builds (not estimated):

All numbers below are Slice LUTs / Slice (CLB) Registers from
`post_route_util.rpt`, read directly from each build's report file.

| build | LUT | FF | BRAM (36k-eq. tiles) | URAM | WNS |
|---|---|---|---|---|---|
| Basys3, N=256/K=8 dense (original) | 20,437 (98.3%) | 34,363 (82.6%) | 50 (100%) | — | +0.174 ns |
| Basys3, N=384/K=8, Option A | 20,508 (98.6%) | 32,228 (77.5%) | 58 | — | +0.317 ns |
| Basys3, N=512/K=8, Option A OFF | 31,354 (150.7%) | 56,675 (136.2%) | — | — | (over budget, synth-only) |
| Basys3, N=512/K=8, Option A ON | 25,682 (123.5%) | 39,938 (96.0%) | 30 (60%) | — | (over LUT budget, synth-only) |
| ZCU104, N=2,116/K=8 (miniseer) | 88,833 (38.6%) | 141,624 (30.7%) | 30 (9.6%) | 8 (8.3%) | +0.382 ns |
| ZCU104, N=2,715/K=8 (cora) | 112,873 (49.0%) | 178,557 (38.8%) | 30 (9.6%) | 8 (8.3%) | +0.498 ns |
| ZCU104, N=3,318/K=8, `SYN_CAP_PER_LANE=7500` (citeseer) | 133,744 (58.1%) | 214,142 (46.5%) | 30 (9.6%) | 8 (8.3%) | +0.889 ns |

The N=512/K=8 pair is a controlled comparison (same RTL, same build script,
only the `SNM_NEURON_STATE_BRAM` define toggled) showing Option A's effect:
FF drops from 136.2% to 96.0% of budget at the same neuron count. Basys3
cannot fit N=512 either way (LUT stays over budget); this pair is a synthesis
comparison, not a claim that N=512 builds on Basys3. The ZCU104 rows are
full place-and-route results and are hardware-validated (§4).

**What consumes LUT/FF, per lane** (measured via hierarchical utilization,
N=512/K=8, one lane, `LOCAL_D=64`): `u_gather` ≈ 1,974 LUT / 2,653 FF (synapse
walk, accumulate, STDP pipeline); `u_neuron` ≈ 1,593 LUT / 4,225 FF (membrane
pipeline + per-neuron config storage). **LUT/FF scale ~O(N)** — each neuron
costs roughly the same fixed per-neuron logic regardless of `NUM_LANES,`
because `NUM_LANES × LOCAL_D = N_MAX` (splitting neurons across more lanes
does not reduce total logic, only reduces `LOCAL_D` per lane and increases
parallelism). Reducing `NUM_LANES` does **not** meaningfully reduce total
LUT/FF for a fixed `N_MAX`.

**What consumes BRAM/URAM:** the synapse table (`syn_row`, sized by
`SYN_CAP_PER_LANE × NUM_LANES`, packed into URAM on ZCU104 builds via
`` `ifdef SNM_SYN_MEM_ULTRA``), the STDP source-indexed history memories, and
(Option A builds) `vmem`/`refrac_count`. This is the axis the sparse-capacity
decoupling feature (`SYN_CAP_PER_LANE` as an independent build parameter)
targets: a sparse graph needs far less than the dense `LOCAL_D × N_MAX`
worst-case sizing.

**Option A's effect:** moving `vmem`/`refrac_count` to Block RAM cut FF by
roughly one third at the same `N_MAX` (Basys3 N=512: FF 137%→96% of budget)
by removing the per-neuron flip-flops those two fields previously occupied —
this is what raised the closable neuron ceiling.

---

## 13. Design rationale — why this, not that

Documented, sourced reasoning behind specific choices. Only included where a
source comment or a directly-measured tradeoff backs it — not inferred after
the fact.

- **Per-lane replication of `spike_frozen` and STDP history (the O(N·K) cost
  behind §12's resource numbers) is a deliberate area-for-throughput
  tradeoff, not an oversight.** `snm_gather_lane_stdp.v`'s own comment: at
  `NUM_LANES=16` with 4 reads/lane, that is 64 concurrent read ports —
  "no realistic shared SRAM/BRAM macro provides" that. Serving it from one
  shared copy would mean arbitrating lane access, "directly undermining the
  II=1 pipelining" and dropping throughput "by roughly NUM_LANES× in the
  worst case." Per-lane replication trades area for throughput, "the same
  tradeoff register files/caches make," and the comment explicitly says it
  "should stay this way unless a future redesign is willing to give back
  STDP throughput for it." One piece IS free to remove without that
  tradeoff: on boards without URAM packing, only one of the two ping-pong
  history memories is ever read — the RTL already does not read the other.
- **Option A moves only `vmem`/`refrac_count` to Block RAM, not all six
  per-neuron fields.** Verified this session (§12): forcing all six into
  block RAM cost ~6 BRAM tiles/lane and blew Basys3's budget, spilling into
  pathological distributed logic (measured: LUT 20,437→45,022 at N=256).
  `input_value` specifically stays in fabric because it is the only
  host-written runtime array — a multi-cycle BRAM clear-on-reset would race
  a host rewrite (found by the soft-reset test failing after the first
  attempt at this change; see the RTL's own comment on `input_value`).
- **The driving weights for the citation datasets are rescaled to 2.0, not
  left at the model's original 100.0.** Verified against the original
  `configs/*/default_*.yaml` and `gnn_citation_networks.py` (§10 has the
  full fixed-point tradeoff): the model's neurons are threshold-1, so any positive
  driving current produces the same spiking decision — the exact magnitude
  is free to choose. 100.0 does not fit a 16-bit weight alongside the
  model's own 0.0001-magnitude STDP taps at any single `frac_bits`; 2.0
  does, without changing which neurons fire.
- **STDP is one global enable bit, not a per-synapse flag on the wire.**
  Each synapse table entry is `(source_id, weight)` only (`IL_SYN`, §8) —
  there is no per-entry enable bit in that format. This was checked against
  the classification task specifically (not assumed to be harmless in
  general): forcing global STDP in software and comparing against the SGNN
  model's own selective STDP gives identical classification (§9, 20/20
  agreement verified this session), which is why this format choice does
  not compromise the citation-dataset results.
- **What is NOT documented with a stated rationale:** why destination
  partitioning (`lane = dst % NUM_LANES`) was chosen over source
  partitioning, or why UART-bridged-to-internal-SPI was chosen as the host
  link over a different transport. No comment or design note stating a
  considered alternative was found in this codebase for either. Do not
  treat their absence from this list as evidence they were the only options
  considered — it means no written rationale was found, not that none
  exists.

---

## 14. Current limitations and future work

- **pubmed (19,720 neurons) does not fit any available board.** LUT scales
  ~O(N) at roughly 40 LUT/neuron (K=8, measured); pubmed would need
  ≈750,000+ LUT, well beyond ZCU104's 230,400. Fixing this needs either (a)
  event-driven/address-event spike representation instead of a dense N-wide
  spike vector (removes the fixed per-neuron logic cost for inactive
  neurons), or (b) DRAM-backed state on ZCU104's PS side. Neither is
  implemented in this version.
- **SP701's *packaged* bitstream is still a legacy 8-bit build**
  (`WEIGHT_W=8, DATA_W=16` — the wide-fixed-point `-generic` overrides were
  silently dropped by a board-top parameter-forwarding gap, fixed in RTL
  this version). A correct wide rebuild (`WEIGHT_W=16, DATA_W=24`, Option A)
  IS hardware-validated on SP701 (§4: microseer, 48/48 SW==HW, one-vs-rest
  0.6424, WNS +0.175 ns) — but that bitstream is not yet the packaged
  default. No citation dataset larger than microseer fits SP701's
  ~1,350-neuron LUT ceiling — those need ZCU104.
- **No event-driven spike path.** All current builds carry a dense, N-wide
  spike vector per tick regardless of how many neurons actually fired — the
  dominant scaling cost (§12).
- **No CI / automated hardware regression.** Verification in this version is
  by direct, logged hardware runs during development, not a continuously
  running test pipeline.
- **Transient UART packet loss** on very long bulk-load bursts (observed rate
  ≈0.03–0.7% of commands over multi-hour sessions) — mitigated by an
  application-level retry (`examples/citation_gnn_fpga.hardware_infer`), not
  fixed at the protocol/transport level.
- **Non-weight fixed-point fields wrap on overflow instead of saturating.**
  `threshold`/`leak`/`reset_state`/per-tick input current are all masked
  (`& ((1<<data_w)-1)`), not clamped like synapse weights are (§10.2). A
  value whose magnitude exceeds the `data_w`/`frac_bits` range at load time
  silently wraps to an unrelated value with no error raised — a real,
  previously-hit incident (documented in `network.py`, 2026-07-27: a
  teacher current of 100 wrapped to the equivalent of 36 under an earlier,
  narrower mask). The current code has the correct mask width for the
  device's actual `data_w`, but the wrap-not-saturate behavior itself is
  unchanged; a caller that computes values outside the valid range for
  these fields gets silent corruption, not an exception. Fixing this
  properly means adding a host-side range check (mirroring
  `capacity.validate_network_fits`'s "refuse before any wire write"
  pattern) for these fields specifically, which does not exist yet.
- **Small dataset-capacity variance run-to-run.** The exact per-lane synapse
  capacity a citation-graph model needs can shift by roughly 1% between
  Python process runs even with `PYTHONHASHSEED=0` set (observed: a citeseer
  build sized at 6,842 needed 6,881 in a later run). The host-side capacity
  guard (`capacity.validate_network_fits`) catches this before any wire write
  — it never corrupts data — but it means a bitstream built at the exact
  minimum capacity can occasionally reject a network that "should" fit.
  Recommendation: build with headroom above the measured minimum (this
  version rebuilt citeseer at 7,500 against a measured 6,881 requirement).

---

## 15. How the Tcl / Vivado build flow works

Every `scripts/build_infer_<board>_stdp_*.tcl` and `scripts/build_dataset_zcu104.tcl`
follows the same shape:
1. Read the RTL file list (`rtl/*.v` + `rtl/fpga/<board>_top>.v`) with
   `read_verilog`.
2. Read the board's `.xdc` constraints with `read_xdc`.
3. `synth_design -top <board_top> -part <part> -generic <NAME>=<VALUE> ...`
   for every RTL parameter (`N_MAX`, `NUM_LANES`, `WEIGHT_W`, `DATA_W`,
   `SYN_CAP_PER_LANE`, `SPIKE_MON_BASE`, `STDP_WINDOW`) plus
   `-verilog_define` flags selecting pipeline-stage/URAM `` `ifdef `` options.
4. `place_design`, `phys_opt_design` (with forced replication on high-fanout
   config-bus nets — a recurring congestion point at higher `NUM_LANES`),
   `route_design`.
5. `report_utilization`/`report_timing_summary`/`report_drc`, then
   `write_bitstream`.
6. Print a machine-parseable summary line (`TIMING board=... wns=... luts=...`)
   that `build.py`'s report parser reads.

`build.build_bitstream(board)` invokes this via `vivado -mode batch -nojournal
-source <script>` as a subprocess, parses the summary line and report files,
and returns a result object (WNS, LUT, BRAM, DRC error count, bitstream path,
full output directory with every checkpoint/report). `program.program(board,
bitstream=...)` similarly invokes a `program_infer_<board>_*.tcl` script over
`vivado -mode batch`, which opens the JTAG hardware target, finds the matching
device by part name, and calls `program_hw_devices`; it prints `PROGRAM_DONE`
on success, which the Python wrapper checks for explicitly (not just process
exit code) before returning.

---

## 16. Dependencies — precisely what is needed for what

| capability | requires |
|---|---|
| Install + use the core runtime (`connect`, `load_network`, `run_schedule`, ...) | `pyserial` only (declared core dependency) |
| Run the bundled examples/notebooks | `pip install "superneuromat[spikeengine-examples]"` → adds `superneuromat`, `numpy`, `scikit-learn`, `matplotlib`, `jupyter` |
| Run the citation-GNN examples | the above, **plus** a local checkout of `sgnn-superneuro` (external, not on PyPI, not bundled — set `SGNN_REPO` env var to its path) |
| **Program a board with a packaged bitstream** | a local **Vivado** install (JTAG programming path; `program.program()` shells out to `vivado -mode batch`) |
| **Build a bitstream from RTL** | a local **Vivado** install (`build.build_bitstream()`) — not needed if you only use packaged bitstreams |
| **Modify the RTL and re-verify in simulation** | **Icarus Verilog (`iverilog`/`vvp`) is NOT a package dependency** — it was used only as a development-time tool during this session's RTL regression testing (sim parity checks before promoting RTL changes). It is not invoked by any shipped code path, not declared in `pyproject.toml`, and not required to install or use `spikeengine`. Anyone modifying the RTL is free to use any Verilog simulator; Icarus is simply what was used here. |
| The legacy PyQt5 GUI (`fpga_gui_app/`, separate from this package) | `PyQt5`, `matplotlib` — not part of `spikeengine`'s dependency graph |

No dependency is silently required — every heavy/external requirement
(Vivado, `SGNN_REPO`) fails with an explicit, actionable error
(`VivadoNotFoundError`, `SystemExit` with the missing-path message) rather
than an unclear import error.

---

## 17. How to inspect the RTL and generated bitstreams yourself

```python
from superneuromat.spikeengine import build
print(build.rtl_source_dir())     # -> .../site-packages/spikeengine/rtl
                                   # open any .v file directly -- plain text, commented

result = build.build_bitstream("basys3")
print(result.outdir)              # every artifact: post_synth.dcp, post_route.dcp,
                                   # post_route_util.rpt, post_route_timing.rpt,
                                   # post_route_drc.rpt, the .bit itself
```

The packaged (pre-built) bitstreams live at
`spikeengine.boards.get_board(name).bitstream_path()` /
`spikeengine.datasets.get_dataset(name).bitstream_path()`; both are ordinary
`.bit` files, inspectable with standard Vivado bitstream tools if desired.
