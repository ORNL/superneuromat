# SpikeEngine — User Manual (Part 1: How to Use It)

A practical guide to running spiking neural networks on FPGA hardware with the
`spikeengine` package: what it is, which datasets are implemented, and how to use
it end to end. If you are new, read sections 1–4; if you just want to run a
dataset, jump to section 5.

**This is Part 1 (usage).** For implementation details — the wire protocol,
opcodes, the lane/NPU architecture, STDP RTL, fixed-point encoding, resource
consumption, validated size limits, and the source-code map — see
**[ARCHITECTURE.md](ARCHITECTURE.md)** (Part 2).

---

## 1. What this is (and why you'd use it)

`spikeengine` is a host runtime + FPGA accelerator for **spiking neural networks
(SNNs)** with **on-chip STDP learning**. You describe a network as a
`superneuromat.SNN` in Python; the package maps it onto a real FPGA, runs the
simulation on-chip (tick by tick, with STDP if enabled), and reads results back.

The engine is **destination-partitioned across K lanes** (each lane owns a slice
of the neurons and their incoming synapses) and runs the STDP update as a
one-result-per-cycle pipeline. On-chip behavior is **bit-exact against the
`superneuromat` software simulator** — proven in RTL simulation and on real
silicon.

**The core idea a new user needs:** the hardware is *configurable*. Neuron count
(`N_MAX`), lane count (`NUM_LANES`), and — new in this version — **synapse
capacity per lane (`SYN_CAP_PER_LANE`)** are build parameters. That flexibility
is what lets networks of very different sizes fit different boards. A small dense
network fits a Basys3; a few-thousand-neuron sparse citation graph fits a ZCU104.
The package includes a **capacity estimator and board recommender** so you don't
have to guess.

---

## 2. Datasets implemented

Two families are implemented and shipped as examples.

### 2a. Introductory / image tasks (Basys3)
| example | task | on hardware | notes |
|---|---|---|---|
| `digits_stdp_e2e` | 8×8 digit classifier, **on-chip STDP training** | ✅ validated | 640/640 weights bit-exact vs software; one-vs-rest 0.909, top-1 58.4% |
| `logic_gates` | OR / AND gates | ✅ validated | all 4 truth-table rows |
| `bars_and_stripes` | 3×3 bar/stripe detector | ✅ validated | 9 patterns incl. the cancel-neuron veto |

### 2b. Citation-graph SNNs (SGNN) — the new dataset family
These are graph-structured classifiers (papers → topics) from the
`sgnn-superneuro` benchmark, run on hardware with STDP. This is the flagship new
capability of this version.

**Accuracy = top-1.** These are multiclass tasks (each paper gets one topic
out of k), so top-1 — the single highest-weight predicted topic against the
true label — is the accuracy figure. The one-vs-rest column is **not** an
accuracy in the usual sense and is much higher; it is kept only for
comparability with the SGNN benchmark, which reports it. See the definitions
below the table before quoting either number.

| dataset | neurons | synapses | board | test papers | **SW top-1** | **HW top-1** | per-paper SW==HW | one-vs-rest (SGNN metric) |
|---|---|---|---|---|---|---|---|---|
| **microseer** | 90 | 996 | Basys3, SP701 | 48 | **41.7%** (20/48) | **41.7%** (20/48) | 48/48 (100%) | 0.6424 |
| **miniseer** | 2,116 | 31,456 | ZCU104 | 120 | **55.0%** (66/120) | **55.0%** (66/120) | 120/120 (100%) | 0.7667 |
| **cora** | 2,715 | 46,788 | ZCU104 | 140 | **62.9%** (88/140) | **62.9%** (88/140) | 140/140 (100%) | 0.8173 |
| **citeseer** | 3,318 | 47,616 | ZCU104 | 120 | **41.7%** (50/120) | **41.7%** (50/120) | 120/120 (100%) | 0.5806 |
| pubmed | 19,720 | 206,710 | — | — | — | ❌ not feasible | — | — |

**Why the SW and HW columns are identical:** they are not copied — the FPGA
was run independently on every test paper and its prediction compared against
the software reference paper-by-paper. Agreement was exact on every paper of
every dataset (final column), so both metrics necessarily evaluate to the same
value. The hardware is bit-exact with the fixed-point software model, not
merely close to it.

**Metric definitions** (both are computed by the SGNN benchmark's own
`calculate_accuracy()`, `gnn_citation_networks.py`):

- **Top-1 accuracy** — USE THIS (`Results.legacy`, line 383). The single
  highest-weight predicted topic versus the true label, one decision per
  paper. This is what "accuracy" means for a multiclass task and the figure
  directly comparable to a conventional published number.
- **One-vs-rest accuracy** — NOT a multiclass accuracy (`Results.accuracy`,
  lines 346-348). `(TP+TN)/(TP+TN+FP+FN)` accumulated over every *(paper,
  topic)* pair, treating each topic as an independent binary problem. The
  denominator is `n_papers × n_topics`, and every topic the model correctly
  did NOT predict counts as a true negative — so with k topics the true
  negatives dominate and the score is inflated by construction. microseer is
  41.7% top-1 but 0.6424 one-vs-rest; the gap is the metric, not the model.
  Reported only because it is the figure the SGNN repo persists to
  `results.json`, so results stay comparable with that benchmark. Do not
  quote it as accuracy.

Note the SGNN repo does not use one consistent definition of "accuracy": a
third, unrelated formulation appears in `spawn_bo.py:104`. Compare against
the specific metric a given source reports, not the word alone.

Recipe for all runs: driving weights rescaled to 2.0, `frac_bits=13`,
`weight_w=16`, `data_w=24`, global STDP. Run over each dataset's FULL
test-paper set. These supersede a preliminary 40-paper subset estimate
(miniseer/cora ≈0.788).

### What these numbers are, and are not (measured 2026-07-31)

**They reproduce the `sgnn-superneuro` reference implementation, not the
published paper's tables.** Both claims were checked; only the first holds.

Against the reference implementation, running its own native evaluation path
on this platform, microseer gives **20/48 = 41.7% top-1 — identical to ours**.
The chain is exact end to end: FPGA == our software reference == the repo.

Against the paper (Zhu et al., *npj Unconventional Computing* 2026, Table 1,
SuperNeuroMAT test column):

| dataset | paper | ours | |
|---|---|---|---|
| CiteSeer | 42.40% | 42.5% | matches |
| MiniSeer | 58.34% | 55.0% | −3.3 |
| MicroSeer | 58.33% | 41.7% | −16.6 |

The two large datasets agree; MicroSeer does not. Cause, measured rather than
assumed: **the test split differs.** MicroSeer has 84 papers, of which the
repo selects 48 — and that selection contains **zero AI papers and one ML
paper**, so 5 of 6 topics are scored on an HCI/Agents-skewed sample. The paper
describes a 10/10/80 split (~67 test papers). On 48 papers each one is worth
2.1%, so a different split moves the figure several points. MiniSeer and
CiteSeer are 25x and 39x larger, where a split cannot distort the balance
nearly as much.

Three further things verified against the paper's own text:

* **Features are correctly OFF.** The paper states the algorithm "does not
  make use of features from the citation graphs". Enabling them measures
  *worse* anyway (microseer 41.7% -> 35.4%, miniseer 55.0% -> 30.8%): the
  feature->topic synapses are wired all-to-all at one uniform weight with STDP
  disabled, so they cannot discriminate and merely add ties.
* **Fresh state per paper is correct.** The paper states the weights and
  initial network state "are the same before" each evaluation. Reusing one
  model across papers (the repo's non-spawn path) lets each paper's STDP
  contaminate the next and scores **10.4% — below the 16.7% chance line**.
* **The paper's "lowest weight" rule is a documentation error.** Line 36 says
  the lowest-weight synapse is the prediction; line 48 of the same page says
  the strongest connection is. The code uses highest. Decoding by lowest
  measures 0.0% (microseer) and 5.0% (miniseer), far below chance -- so
  highest is right and the sentence is wrong.

**Why accuracy is limited in absolute terms:** a large share of papers produce
no differentiating signal at all. Their paper neuron fires once, no topic
neuron fires back inside the STDP window, so every paper->topic synapse
receives only the uniform depression term and lands on the same floor value
(-0.0025). Six identical weights is a 6-way tie, resolved by topic load order.
That is 18/48 papers on microseer, 25/120 on miniseer, 51/120 on citeseer.
It is a property of the SGNN model at `simtime=10`, not of this engine.

**citeseer note:** the first citeseer bitstream was built with
`SYN_CAP_PER_LANE=6842`, sized from an earlier deterministic run; the exact
requirement measured at hardware-test time was 6,881 (0.57% higher) — the
host-side capacity guard correctly refused the load with zero data written
(no corruption), and the bitstream was rebuilt with `SYN_CAP_PER_LANE=7500`
for headroom, then hardware-validated at that capacity. See
ARCHITECTURE.md §13 for why this run-to-run variance occurs.

**Why pubmed is excluded:** at 19,720 neurons it needs ~4× more LUTs than the
largest available board (ZCU104) provides. It is out of reach of the current
*dense-spike* architecture and would require event-driven spikes or DRAM-backed
state (see section 8, Roadmap). It ships as a **software-only** reference.

---

## 3. Install

```bash
pip install -e .                # core runtime (pyserial)
pip install -e ".[spikeengine-examples]"    # + superneuromat, numpy, scikit-learn, jupyter
```

For the citation-GNN datasets you also need the `sgnn-superneuro` sources
(dataset builders, external, not on PyPI — set `SGNN_REPO` to its path). That
checkout has its own Python dependencies, **not installed by the core
package**. They are now declared as an extra, so one command covers them:

```bash
pip install -e ".[datasets]"
```

That resolves to `networkx`, `pandas`, `PyYAML`, `wrapt`, `tqdm`,
`scikit-learn`, `tabulate`, `matplotlib` — the full closure, re-measured
2026-07-31 by installing into a clean venv and re-running until nothing was
missing. The earlier list in this section named only five of them and pinned
`networkx<=2.8.7`; that pin is not required here (verified against networkx
3.6.1 with NumPy 2.x). If anything is still missing you get a message naming
the package and the command to install it, not a bare traceback.

### Windows: enable long paths before cloning

The repository contains paths over 260 characters under
`board_variants/.../fpga_src/sourcecode/rtl/`. With git's default
`core.longpaths=false`, `git clone` on Windows fails partway through checkout
with `Filename too long` and leaves an incomplete working tree — including,
depending on where it stops, **no bitstreams at all**, since they sort after
`board_variants/`. The clone itself reports success, so this is easy to miss.

```bash
git config --global core.longpaths true
```

Or clone to a short root such as `C:\dev\`. To confirm a good checkout:

```bash
find . -name "*.bit" | wc -l      # expect 7
```

---

## 4. The core workflow (any network)

```python
from superneuromat import spikeengine as se

# 1. connect to a board over its UART (autodetect the port)
# IMPORTANT: if the programmed bitstream is a DATASET build (its N_MAX/NUM_LANES
# differ from the board's packaged default -- true for every citation dataset),
# pass dataset=... so the device is sized correctly. Omitting it uses the
# board's default geometry AND datapath width, which will silently corrupt
# synapse addressing and truncate 16-bit weights against a dataset bitstream.
dev = se.connect(port="auto", board="zcu104", dataset="miniseer")

# 2. clear runtime state without reprogramming (safe between runs)
dev.soft_reset()

# 3. map a superneuromat.SNN onto the board
info = se.load_network(dev, snn, frac_bits=13, weight_w=16, data_w=24)

# 4. run a tick schedule (inject inputs at given ticks, run N ticks with STDP)
spikes = se.run_schedule(dev, schedule, total_ticks,
                         frac_bits=13, n_neurons=info["n_neurons"])

# 5. read hardware-learned weights straight off the chip
weights = se.read_weights(dev, info["entry_index"], frac_bits=13)  # {(post,pre): w}
dev.close()
```

**Prefer the context-manager form** — it closes the port even if something
raises partway through. A leaked port stays locked on Windows, so the next
run fails to open it:

```python
with se.connect(port="auto", board="zcu104", dataset="miniseer") as dev:
    dev.soft_reset()
    info = se.load_network(dev, snn, frac_bits=13, weight_w=16, data_w=24)
    spikes = se.run_schedule(dev, schedule, total_ticks,
                             frac_bits=13, n_neurons=info["n_neurons"])
    weights = se.read_weights(dev, info["entry_index"], frac_bits=13)
# port is closed here, on success or on exception
```

`load_network` **validates capacity before writing anything** — if the network
exceeds the loaded bitstream's `SYN_CAP_PER_LANE`, it raises `CapacityError`
rather than silently corrupting a neighboring neuron's synapses. Always let it
guard the load.

---

## 5. Running a dataset

### 5a. Which board fits my network?
Ask the package before building or connecting:

```python
from superneuromat import spikeengine as se
board = se.recommend_board(snn, num_lanes=8)   # -> 'basys3' | 'sp701' | 'zcu104'
cap   = se.required_syn_cap_per_lane(post_neuron_ids, num_lanes=8)
```

Rule of thumb from real builds (16-bit weights, STDP, K=8):

| board | LUTs | neuron ceiling* | good for |
|---|---|---|---|
| Basys3 (xc7a35t) | 20.8k | ~256 (≈384 with Option A) | digits, logic, **microseer** |
| SP701 (xc7s100) | 64k | ~1.5k | mid-size models |
| ZCU104 (xczu7ev) | 230k | **~5.5k** | **miniseer / cora / citeseer** |

\* neuron count is **LUT-bound**, not synapse-bound. Reducing synapses frees
BRAM but does *not* raise the neuron ceiling — that takes a bigger board.

#### Board-default bitstreams do not all use the same numeric precision

Each board ships one *default* bitstream, used when you connect without naming
a dataset. They were not all built with the same datapath:

| board | default bitstream | weights | membrane |
|---|---|---|---|
| basys3 | `snm_infer_basys3_stdp_cap256x8` | **16-bit** | **24-bit** |
| sp701 | `snm_infer_sp701_stdp_maxcap352x8` | **16-bit** | **24-bit** |
| zcu104 | `snm_infer_zcu104_stdp_maxcap1024x16` | 8-bit | 16-bit |

8-bit weights give 256 representable values against 65,536 for 16-bit, and the
narrower membrane saturates sooner. Nothing silently misbehaves:
`load_network()` reads the width **from the device** and quantizes to match, so
a narrow board stays self-consistent — just coarser. All published accuracy
figures in section 2b were measured on 16/24, because every citation dataset
carries its own wide bitstream and `connect(dataset=...)` selects it.

SP701's default was rebuilt wide and hardware-validated on 2026-07-31. If you
care about weight resolution on ZCU104, use a wide dataset build or build your
own. A wide default build script exists
(`scripts/build_infer_zcu104_stdp_maxcap1024x16.tcl`, which emits a
`..._wide16.bit`); it is not the packaged default. ZCU104 was left narrow
deliberately (2026-07-31) rather than swapped in unvalidated: the wide
variant in particular needs its URAM packing re-checked, since widening an
entry from 18 to 26 bits drops packing from 4 to 2 entries per 72-bit row and
roughly doubles the tiles required against a 96-tile budget.

### 5b. Program the board and run
```python
from superneuromat.spikeengine import program
program.program(board="zcu104", bitstream="<dataset bitstream>.bit")
```
Then use the section-4 workflow, or run the packaged citation example
(`examples/citation_gnn_fpga`) which builds the dataset graph, loads it, runs
per-paper inference with STDP, reads back the paper→topic weights, and classifies
— cross-checking every paper against the software reference.

```bash
python -m superneuromat.spikeengine.examples.citation_gnn_fpga --dataset miniseer --board zcu104 --port auto
python -m superneuromat.spikeengine.examples.citation_gnn_fpga --dataset microseer --no-hardware   # SW reference
```

---

## 6. Fixed-point recipe (important for accuracy)

The hardware datapath is 16-bit weight / 24-bit membrane integer. Networks map
correctly when you (a) keep the driving/teacher weights small (they only need to
push a neuron over threshold), and (b) choose `frac_bits` so the smallest STDP
tap survives quantization. For the citation datasets the validated recipe is:

- rescale graph/input driving weights to **2.0**
- `frac_bits = 13`, `weight_w = 16`, `data_w = 24`
- STDP is **global** on hardware (every enabled synapse), which is proven to give
  the same classification as the software model's selective STDP for these tasks.

The digits example uses `frac_bits = 11` (its dynamic range is different). When in
doubt, run `--no-hardware` first: it executes the fixed-point-faithful software
path and must match your expected accuracy before you trust the board.

---

## 7. Building a bitstream from RTL (optional)

Packaged bitstreams load in seconds. To build your own (e.g. a custom
`N_MAX`/`SYN_CAP_PER_LANE` for a new network size), the full RTL + Vivado scripts
are bundled:

```python
from superneuromat.spikeengine import build
result = build.build_bitstream("basys3")          # reproduce the packaged build
print(result.wns_ns, result.lut, result.bram, result.bitstream_path)
```

For the large sparse dataset builds (ZCU104, with the Option A neuron-state-in-BRAM
optimization) use the parametric scripts:

```
vivado -mode batch -source scripts/build_dataset_zcu104.tcl -tclargs <N> <K> <SYN_CAP> <label>
```

This is an opt-in, heavy path (10–40+ min, longer for N≥2000) and needs a local
Vivado install.

---

## 8. How the new version scales (architecture notes)

Two changes in this version are what make the larger datasets possible:

1. **Decoupled synapse capacity** (`SYN_CAP_PER_LANE`). Previously the per-lane
   synapse table was sized for the dense N² worst case. It is now a build
   parameter, so a *sparse* graph (few synapses per neuron) can hold many more
   neurons in the same BRAM/URAM budget. A host-side guard
   (`capacity.validate_network_fits`) enforces the bound before any wire write.

2. **Option A — neuron state in Block RAM.** Per-neuron membrane/refractory state
   used to live in fabric flip-flops, making logic scale O(N) and capping neuron
   count. Moving that state into Block RAM (opt-in `SNM_NEURON_STATE_BRAM`, proven
   bit-exact) cut flip-flop usage ~35% and raised the neuron ceiling. On Basys3 it
   lifted the closable size from N≈256 to N=384 (timing-closed, WNS +0.317 ns);
   on ZCU104 it is what lets the 2–3k-neuron citation graphs fit with headroom.

**Roadmap (not in this version):** the remaining wall at very large N (pubmed) is
the *dense spike representation* — each lane carries an N-wide spike vector.
Event-driven (address-event) spikes and/or DRAM-backed state are the paths to
10k+ neurons.

---

## 9. Troubleshooting

- **`CapacityError` on load** — your network's max per-lane in-degree exceeds the
  bitstream's `SYN_CAP_PER_LANE`. Use `recommend_board` / build a bitstream with a
  larger cap. (For citation graphs the "topic hub" neurons — every paper wires to
  every topic — set this floor.)
- **Board not detected** — pass the COM port explicitly (`port="COM17"`); the
  UART/FTDI device can change enumeration after a reprogram/replug.
- **Accuracy differs from software** — run `--no-hardware` to confirm the
  fixed-point reference first; check `frac_bits` and the driving-weight scale
  (section 6).
- **Wrong neuron on the LEDs** — the `SPIKE_MON_BASE` build generic selects which
  neuron drives LED[0]; set it to your first output neuron.
- **`KeyError` from `load_network`, or garbage results, against a dataset
  bitstream** — the device was constructed with the wrong geometry (board
  default instead of the bitstream's own `N_MAX`/`NUM_LANES`/`weight_w`). Pass
  `dataset=...` to `connect()` (section 4). This was the root cause of an
  observed all-topics-tied classification result during bring-up: weights
  loaded correctly in address but were truncated to 8 bits because the device
  inherited the board's legacy `weight_w`.
- **`InferEngineError: stream stalled: sent N/N, recv M/N` during
  `load_network`** — a transient UART packet loss during a large bulk config
  load (observed at a rate of roughly 0.03–0.7% of commands over multi-hour
  sessions reloading tens of thousands of synapses per inference). Not a
  protocol bug. `examples.citation_gnn_fpga.hardware_infer` retries up to twice
  automatically (`dev.soft_reset()` fully clears device state before each
  retry, so this is safe); if you call `load_network` directly, wrap it in the
  same retry pattern for long unattended runs.
- **Vivado programming does not report `PROGRAM_DONE`** — confirm the correct
  JTAG target is found (`open_hw_target` / `get_hw_devices` output in the
  Vivado log should show the board's part, e.g. `xczu7` for ZCU104); a stale
  `hw_server` session or another tool holding the JTAG cable is the usual
  cause. Also confirm `program.program(board, bitstream=...)` resolved the
  board-specific programmer script (it globs `scripts/program_infer_<board>_*.tcl`)
  rather than silently falling back to the Basys3 script.

---

## 10. Scope — what this package does and does not do

**Covered:**
- On-chip STDP learning (a global tap table, applied every tick to enabled
  synapses) and on-chip inference, on real FPGA hardware.
- A destination-partitioned, K-lane parallel architecture (`NUM_LANES` is a
  build parameter); lanes process their neurons/synapses independently.
- Compiling a bitstream from RTL and programming a board, both via Vivado,
  driven from Python (`build.build_bitstream`, `program.program`).
- Inspecting the RTL before compiling: `build.rtl_source_dir()` returns the
  path to the plain-text Verilog sources every build reads from (not
  encrypted IP, not a generated blob — ordinary, commented `.v` files you can
  open directly).
- A capacity estimator and board recommender (`capacity.py`) calibrated
  against real Vivado builds.
- Measuring on-chip timestep duration against the "1 ms" SuperNeuroMAT tick
  convention: `examples.citation_gnn_fpga.measure_timestep(dev)` reads the
  hardware cycle counters and reports microseconds/tick.

**Explicitly NOT covered:**
- **The RTL is hand-written, parametrized Verilog — this package does not
  generate RTL from Python.** Flexibility comes from module parameters
  (`N_MAX`, `NUM_LANES`, `SYN_CAP_PER_LANE`, ...) passed to Vivado as
  `-generic` overrides at synthesis time, not from Python-side code
  generation.
- **SP701 only runs the smallest citation dataset (microseer).** Its LUT
  budget (~1,350-neuron ceiling) is smaller than every other citation
  dataset. A wide-fixed-point + Option A rebuild for microseer IS
  hardware-validated on SP701 (48/48 test papers, one-vs-rest 0.6424,
  identical to Basys3's result). The rebuilt packaged default is independently
  validated at N352/K8, W16/D24 (40/40 tick outputs and 640/640 learned weights
  bit-exact). Use `se.connect(board="sp701", dataset="microseer")` only when
  reproducing the dataset-specific Microseer result.
- **pubmed (19,720 neurons)** does not fit any available board with the
  current architecture (§8). Ships as a software-only reference.
- **The desktop GUI** (`spikeengine.gui`, `pip install "superneuromat[gui]"`) ships
  as part of this package (see ARCHITECTURE.md §3 for the full rewire
  history). It supports NPU-STDP boards only — the classic single-core engine
  and non-STDP NPU are stubbed out, since neither has an equivalent in this
  package's wire protocol. Module import (including all PyQt5 imports and the
  rewired device path) is verified; interactive widget construction was not
  visually verified in this development environment (a VTK/OpenGL offscreen
  rendering limitation, unrelated to the GUI code itself) — run it on a
  machine with a real display before relying on it for a demo.
- No CI / automated regression against real hardware — verification in this
  version is by direct, logged hardware runs, not a continuously-running test
  pipeline.

**Hardware-verified vs. unit-tested — do not conflate the two:**
- **Hardware-verified** (real board, this session's logged runs): every
  dataset/board combination marked `hardware_validated=True` in `boards.py`/
  `datasets.py` (§2), plus the digits/logic_gates/bars_and_stripes examples
  per `boards.py`'s existing record. This is the only category that confirms
  the *bitstream* is correct.
- **Unit-tested** (`tests/`, software-only, no board needed): `datasets.py`'s
  catalogue lookups and bitstream-path resolution, `program.py`'s
  board-specific Tcl script selection, and `connect()`'s `dataset=` geometry/
  weight-width override logic (`tests/test_datasets_program_connect.py`) —
  plus the pre-existing capacity-guard and import/API tests
  (`test_capacity.py`, `test_package.py`). These confirm the *host-side
  Python logic* is correct; they say nothing about the bitstream, since no
  wire traffic occurs. `test_hardware_bit_exact` is the one test that
  exercises real hardware, and is skipped unless `SE_STDP_PORT` is set.

## 11. Reference

- Public API: `connect`, `soft_reset`, `load_network`, `run_schedule`,
  `read_weights`, `recommend_board`, `required_syn_cap_per_lane`,
  `estimate_capacity`, `program.program`, `build.build_bitstream`.
- Examples: `examples/` (digits, logic_gates, bars_and_stripes, and the citation
  notebooks).
- Boards: `basys3`, `sp701`, `zcu104`.
