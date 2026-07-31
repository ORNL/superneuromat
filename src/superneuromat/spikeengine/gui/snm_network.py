"""High-level SNN builder for the SuperNeuroMAT3 FPGA.

Lets you describe a network the way the SuperNeuroMAT *software* does
(create neurons, connect synapses, configure STDP) and programs it onto the FPGA
without ever touching the chip's sparse memory map by hand. It auto-computes the
synapse ordering and the src_ptr boundaries (the easy thing to get wrong), the
signed encodings, and the STDP tables.

    from snm_network import SNN
    net = SNN(port="COM5")
    a = net.neuron(threshold=10, leak=1, reset=-3, refractory=2, inp=True)
    b = net.neuron(threshold=5)
    net.synapse(a, b, weight=10, stdp=True)
    net.stdp(window=1, apos=[2], aneg=[-1])
    net.build()
    train = net.run({0: {a: 12}}, steps=5)      # inputs at t=0; run 5 ticks
    print(train)                                # [[a],[a,b],[a,b],...]
    print(net.weight(a, b), net.vmem(b), net.refrac(a))

Feature testing maps directly:
  * spikes/threshold -> net.run(...) returns the per-tick spike raster
  * reset            -> set reset=...; net.vmem(n) after a spike
  * refractory       -> set refractory=...; net.refrac(n) counts down
  * leak             -> set leak=...; net.vmem(n) decays toward reset
  * STDP             -> synapse(stdp=True)+net.stdp(...); net.weight(a,b) changes

Cross-check against the cycle-accurate software model (if `superneuromat` is
installed):  net.cross_check({...}, steps)  builds the same network in software,
simulates it, and diffs the spike trains + learned weights.

Constraints (FPGA core): integer params; synapse delay is fixed at 1; the spike
from tick T reaches its destination on tick T+1.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import snm_driver as snm
from .snm_driver import SNMDriver


def spike_table(train, n_neurons=None, mark="*", blank=".", cols=None,
                max_cols=32) -> str:
    """Format a spike train as a raster table: rows = timesteps (from 0),
    columns = neuron indices, '*' = fired, '.' = silent.

    `train` is a list (per tick) of spiking-neuron-index lists (what run()
    returns). Handles LARGE networks gracefully:
      * cols=<iterable>  -> show exactly those neuron columns
      * otherwise, if the network has <= max_cols neurons, show them all
      * if it is wider, show ONLY neurons that fired at least once (sparse view)
        and, if still too many, the first max_cols of them.
    The header row lists the actual neuron index of each column.
    """
    fired = sorted({i for s in train for i in s})
    if cols is not None:
        cols = list(cols)
    else:
        n = n_neurons if n_neurons is not None else (1 + (max(fired) if fired else -1))
        n = max(n, 1)
        cols = list(range(n)) if n <= max_cols else (fired or [0])
    if not cols:
        cols = [0]
    note = ""
    if len(cols) > max_cols:
        note = f"\n(+{len(cols) - max_cols} more active neurons; pass cols=... or use raster_plot)"
        cols = cols[:max_cols]

    w = max(2, max((len(str(c)) for c in cols), default=1))
    label = "time/neuron"
    pre = len(label)
    header = f"{label} | " + " ".join(f"{c:>{w}}" for c in cols)
    rows = [header, "-" * len(header)]
    for t, s in enumerate(train):
        sset = set(s)
        cells = " ".join(f"{(mark if c in sset else blank):>{w}}" for c in cols)
        rows.append(f"{t:>{pre}} | {cells}")
    return "\n".join(rows) + note


def spike_raster_plot(train, n_neurons=None, title="spike raster", show=True):
    """Matplotlib raster plot of a spike train (best for large/dense networks).
    Each '|' mark is a spike; only fired points are drawn, so it scales to any N.
    """
    import matplotlib.pyplot as plt
    xs = [t for t, s in enumerate(train) for _ in s]
    ys = [i for s in train for i in s]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(xs, ys, marker="|", s=200)
    ax.set_xlabel("timestep"); ax.set_ylabel("neuron"); ax.set_title(title)
    format_raster_axes(ax, len(train), n_neurons)
    if show:
        plt.show()
    return fig


def compare_trace_outputs(observed_train, reference_train,
                          observed_vmem=None, reference_vmem=None,
                          observed_refrac=None, reference_refrac=None, *,
                          value_tol=1e-9):
    """Compare two timestep-by-timestep execution traces."""
    spike_mismatches = []
    steps = max(len(observed_train or []), len(reference_train or []))
    for t in range(steps):
        obs = sorted(observed_train[t]) if t < len(observed_train or []) else []
        ref = sorted(reference_train[t]) if t < len(reference_train or []) else []
        if obs != ref:
            spike_mismatches.append({
                "timestep": t,
                "observed": obs,
                "reference": ref,
            })

    vmem_mismatches = []
    if observed_vmem is not None and reference_vmem is not None:
        steps = max(len(observed_vmem), len(reference_vmem))
        for t in range(steps):
            obs_row = observed_vmem[t] if t < len(observed_vmem) else []
            ref_row = reference_vmem[t] if t < len(reference_vmem) else []
            width = max(len(obs_row), len(ref_row))
            for neuron in range(width):
                obs = obs_row[neuron] if neuron < len(obs_row) else None
                ref = ref_row[neuron] if neuron < len(ref_row) else None
                if obs is None or ref is None or abs(float(obs) - float(ref)) > value_tol:
                    vmem_mismatches.append({
                        "timestep": t,
                        "neuron": neuron,
                        "observed": obs,
                        "reference": ref,
                    })

    refrac_mismatches = []
    if observed_refrac is not None and reference_refrac is not None:
        steps = max(len(observed_refrac), len(reference_refrac))
        for t in range(steps):
            obs_row = observed_refrac[t] if t < len(observed_refrac) else []
            ref_row = reference_refrac[t] if t < len(reference_refrac) else []
            width = max(len(obs_row), len(ref_row))
            for neuron in range(width):
                obs = obs_row[neuron] if neuron < len(obs_row) else None
                ref = ref_row[neuron] if neuron < len(ref_row) else None
                if obs != ref:
                    refrac_mismatches.append({
                        "timestep": t,
                        "neuron": neuron,
                        "observed": obs,
                        "reference": ref,
                    })

    return {
        "ok": not spike_mismatches and not vmem_mismatches and not refrac_mismatches,
        "spike_mismatches": spike_mismatches,
        "vmem_mismatches": vmem_mismatches,
        "refrac_mismatches": refrac_mismatches,
    }


def compare_weight_results(net, learned, spike_train, *, value_tol=1e-9):
    """Compare learned decimal weights against the SuperNeuroMAT STDP rule."""
    predicted_raw = net.predict_weights(spike_train)
    mismatches = []
    for s in net.synapses:
        observed = float(learned.get(s.idx, s.weight))
        reference = float(from_raw(predicted_raw[s.idx], net.frac_bits))
        if abs(observed - reference) > value_tol:
            mismatches.append({
                "pre": s.pre,
                "post": s.post,
                "observed": observed,
                "reference": reference,
                "delta": observed - reference,
            })
    return {
        "ok": not mismatches,
        "weight_mismatches": mismatches,
    }


def compare_weight_results_reference(net, learned, reference_weights, *, value_tol=1e-9):
    """Compare learned decimal weights against explicit software-simulated weights."""
    mismatches = []
    for s in net.synapses:
        observed = float(learned.get(s.idx, s.weight))
        reference = float(reference_weights.get(s.idx, s.weight))
        if abs(observed - reference) > value_tol:
            mismatches.append({
                "pre": s.pre,
                "post": s.post,
                "observed": observed,
                "reference": reference,
                "delta": observed - reference,
            })
    return {
        "ok": not mismatches,
        "weight_mismatches": mismatches,
    }


def format_raster_axes(ax, steps, n_neurons=None):
    """Force raster axes to show discrete integer timestep/neuron ticks."""
    from matplotlib.ticker import MaxNLocator
    ax.set_xlim(-0.5, max(steps - 0.5, 0.5))
    if steps <= 64:
        ax.set_xticks(range(max(steps, 1)))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if n_neurons:
        ax.set_ylim(-0.5, n_neurons - 0.5)
        if n_neurons <= 64:
            ax.set_yticks(range(n_neurons))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


# =====================================================================
# Hardware spec + fixed-point number layer.
#
# The chip does INTEGER arithmetic with fixed bit widths (RTL parameters N_MAX,
# DATA_W, WEIGHT_W, REF_W). HardwareSpec mirrors those on the host, sourced from
# config/snm_config.yaml via the generated snm_config module -- the SAME source
# the RTL builds from. The current packaged Basys-3 profile uses N_MAX=1400 and
# SYN_MAX=32768; alternate board profiles must be regenerated and packaged with
# matching manifest metadata.
#
# To support decimals we scale every "analog" quantity by 2**frac_bits (a global
# Q-format): the SNN math is linear, so scaling threshold/leak/reset/weight/input
# together is exact, just with resolution 2**-frac_bits and a reduced range. Tick
# counts (refractory, STDP window, timesteps, indices) stay integers.
# =====================================================================

from . import (
    snm_config as _cfg,  # generated from config/snm_config.yaml (gen_config.py)
)


@dataclass(frozen=True)
class HardwareSpec:
    """Mirror of the chip's RTL parameters. Defaults come from snm_config, which
    tools/gen_config.py generates from config/snm_config.yaml -- the SAME source
    the RTL is built from, so host and silicon cannot silently disagree."""
    n_max: int = _cfg.N_MAX            # neurons (RTL N_MAX)
    syn_max: int = _cfg.SYN_MAX        # synapse-entry capacity (RTL SYN_MAX)
    data_bits: int = _cfg.DATA_W       # threshold/leak/reset/vmem/input (RTL DATA_W)
    weight_bits: int = _cfg.WEIGHT_W   # weight, STDP Apos/Aneg (RTL WEIGHT_W)
    ref_bits: int = _cfg.REF_W         # refractory period/count (RTL REF_W)
    stdp_window_max: int = _cfg.STDP_T_MAX        # RTL STDP_T_MAX
    event_fifo_depth: int = _cfg.EVENT_FIFO_DEPTH # sparse input events per queued frame
    npu: bool = False                  # True = NPU array (K-lane dense-capacity engine):
    num_lanes: int = 0                 # persistent inputs, no async classic-core semantics
    npu_stdp: bool = False             # 2026-07-27: True = STDP-CAPABLE NPU variant (this
                                        # board_variants/npu_stdp_dev GUI copy only) -- same
                                        # K-lane architecture/capacity as npu=True, but WITH
                                        # on-chip STDP training and weight readback. Narrows
                                        # (does not replace) the "npu implies inference-only"
                                        # restrictions below -- see is_npu/_validate_npu/
                                        # weight_raw, each gated `and not npu_stdp`.

    @property
    def s_max(self):
        """Dense logical upper bound (n_max^2). The real capacity is syn_max,
        which the wire protocol + SRAM depth cap below n_max^2 at large n_max."""
        return self.n_max * self.n_max

    @property
    def spike_words(self):
        """32-bit words per output spike frame (one bit per neuron)."""
        return (self.n_max + 31) // 32


HW = HardwareSpec()             # default = SuperNeuroMAT3 on Basys-3

# fixed-point "analog" fields -> (HardwareSpec width attribute, signed?)
_FP_FIELDS = {
    "threshold": ("data_bits", True),
    "leak":      ("data_bits", False),   # leak is a non-negative magnitude
    "reset":     ("data_bits", True),
    "vmem":      ("data_bits", True),
    "input":     ("data_bits", True),
    "weight":    ("weight_bits", True),
    "apos":      ("weight_bits", True),
    "aneg":      ("weight_bits", True),
}

# convenience aliases (default spec) — prefer a HardwareSpec for scaled hardware
N_MAX = HW.n_max
S_MAX = HW.s_max
SYN_MAX = HW.syn_max


def fixed_range(field, frac_bits, hw=HW):
    """(lo, hi, step) decimal range for a fixed-point `field` at `frac_bits`."""
    bits = getattr(hw, _FP_FIELDS[field][0]); signed = _FP_FIELDS[field][1]
    scale = 1 << frac_bits
    lo = (-(1 << (bits - 1)) if signed else 0) / scale
    hi = ((1 << (bits - 1)) - 1 if signed else (1 << bits) - 1) / scale
    return lo, hi, 1.0 / scale


def to_raw(value, field, frac_bits, hw=HW, name=None):
    """Decimal value -> the chip's raw integer for `field`. Raises if out of range."""
    bits = getattr(hw, _FP_FIELDS[field][0]); signed = _FP_FIELDS[field][1]
    raw = round(float(value) * (1 << frac_bits))
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    if not (lo <= raw <= hi):
        rl, rh, _ = fixed_range(field, frac_bits, hw)
        raise ValueError(f"{name or field}={value} is out of range "
                         f"[{rl}, {rh}] at frac_bits={frac_bits}")
    return raw


def from_raw(raw, frac_bits):
    """Chip raw integer -> decimal value."""
    return raw / (1 << frac_bits)


@dataclass
class Neuron:
    idx: int
    threshold: float = 0.0
    leak: float = 0.0
    reset: float = 0.0
    refractory: int = 0
    refractory_state: int = 0
    initial_state: float = 0.0
    inp: bool = False
    def __index__(self):  # so a Neuron can be used as a dict key / int
        return self.idx
    def __hash__(self):
        return hash(("n", self.idx))


@dataclass
class Synapse:
    idx: int            # logical creation order (chip index assigned at build)
    pre: int
    post: int
    weight: float = 1.0
    stdp: bool = False
    chip_idx: int = -1


class SNN:
    """Declarative SNN that programs/drives the SuperNeuroMAT3 FPGA."""

    def __init__(self, port: str | None = None, baud: int = _cfg.BAUD,
                 driver: SNMDriver | None = None, hw: HardwareSpec = HW,
                 frac_bits: int = 0):
        self.neurons: list[Neuron] = []
        self.synapses: list[Synapse] = []
        self._stdp = None          # dict(window, apos, aneg, enable) or None
        self.hw = hw               # hardware spec (RTL parameters)
        self.n_max = hw.n_max
        self.frac_bits = int(frac_bits)   # fixed-point fractional bits (0 = integer)
        if driver is not None:
            self.drv = driver
        elif port is not None:
            self.drv = snm.connect(port, baud, n_max=hw.n_max)
        else:
            self.drv = None        # build-only / software-only

    # ---------------- construction ----------------
    def neuron(self, threshold=0, leak=0, reset=0, refractory=0,
               refractory_state=0, initial_state=0, inp=False) -> Neuron:
        n = Neuron(len(self.neurons), threshold, leak, reset, refractory,
                   refractory_state, initial_state, inp)
        self.neurons.append(n)
        return n

    def synapse(self, pre, post, weight=1, stdp=False) -> Synapse:
        s = Synapse(len(self.synapses), int(pre), int(post), float(weight), bool(stdp))
        self.synapses.append(s)
        return s

    def stdp(self, window=1, apos=None, aneg=None, enable=True):
        """Configure global STDP. apos/aneg are per-delay lists of length `window`."""
        apos = list(apos or [])
        aneg = list(aneg or [])
        self._stdp = {"window": window, "apos": apos, "aneg": aneg, "enable": enable}

    # ---------------- NPU array (inference-only lane engine) ----------------
    @property
    def is_npu(self) -> bool:
        return bool(getattr(self.hw, "npu", False))

    @property
    def is_npu_stdp(self) -> bool:
        """True for the STDP-capable NPU variant (this GUI copy only) -- same
        K-lane architecture as is_npu, but with on-chip training + readback."""
        return self.is_npu and bool(getattr(self.hw, "npu_stdp", False))

    @property
    def supports_state_readback(self) -> bool:
        """True when run_trace()'s per-tick vmem/refractory readback will
        succeed on this device -- its only caller (snm_gui.build_run) uses it
        purely to decide whether to call run_trace() vs. run()+software_trace().

        Was previously `(not self.is_npu) or self.is_npu_stdp`, i.e. True for
        the STDP-capable NPU variant. That was wrong: run_trace() itself blocks
        on the broader `self.is_npu` with no STDP exception (no on-chip
        vmem/refractory readback exists for ANY NPU variant -- STDP only adds
        weight readback via a completely different path, read_synapse(), which
        this property does not gate). The mismatch meant build_run() called
        run_trace() for STDP NPU boards expecting it to work, and it always
        raised RuntimeError -- reproduced live via the GUI's digits classifier
        example on npu-stdp:basys3 (2026-07-31)."""
        return not self.is_npu

    def _validate_npu(self):
        """NPU-specific constraints, checked on top of validate()'s ranges."""
        if self.is_npu_stdp:
            return   # STDP-capable variant: on-chip STDP is supported, skip the block below
        if self._stdp and self._stdp.get("enable"):
            raise ValueError("the NPU array is inference-only: on-chip STDP is not "
                             "supported (weights are fixed once loaded). Uncheck STDP, "
                             "or select a classic SuperNeuroMAT3 board for STDP runs.")
        for s in self.synapses:
            if s.stdp:
                raise ValueError(f"synapse {s.idx} has STDP active, but the NPU array "
                                 "is inference-only -- clear the synapse's STDP flag.")
        for nn in self.neurons:
            if float(nn.initial_state) != 0.0:
                raise ValueError(f"neuron {nn.idx}: initial_state={nn.initial_state} -- "
                                 "the NPU array always cold-starts at vmem=0 (no state "
                                 "write path); set initial_state=0.")
            if int(nn.refractory_state) != 0:
                raise ValueError(f"neuron {nn.idx}: refractory_state must be 0 on the "
                                 "NPU array (no counter write path).")
        # per-lane synapse capacity: entries land in the DESTINATION's lane;
        # each lane holds LOCAL_D * N_MAX entries (LOCAL_D = ceil(N_MAX/K)).
        k = int(getattr(self.hw, "num_lanes", 0)) or 1
        local_d = (self.hw.n_max + k - 1) // k
        cap = local_d * self.hw.n_max
        per_lane: dict[int, int] = {}
        for s in self.synapses:
            lane = int(s.post) % k
            per_lane[lane] = per_lane.get(lane, 0) + 1
        for lane, cnt in per_lane.items():
            if cnt > cap:
                raise ValueError(f"NPU lane {lane} needs {cnt} synapse entries but "
                                 f"holds {cap} (destinations are striped dst%K; "
                                 "rebalance the network's fan-ins).")

    # ---------------- layout + programming ----------------
    def _assign_chip_indices(self):
        """Order synapses grouped by source neuron and compute src_ptr."""
        n = len(self.neurons)
        by_src = {k: [] for k in range(n)}
        for s in self.synapses:
            by_src[s.pre].append(s)
        ordered, src_ptr = [], [0] * (n + 1)
        for k in range(n):
            for s in by_src[k]:
                s.chip_idx = len(ordered)
                ordered.append(s)
            src_ptr[k + 1] = len(ordered)   # end boundary for source k
        return src_ptr

    def validate(self):
        """Check the network against the chip's capacity + fixed-point ranges.
        Raises ValueError on the first violation. Called by build()."""
        fb, hw = self.frac_bits, self.hw
        ref_max = (1 << hw.ref_bits) - 1
        if len(self.neurons) > hw.n_max:
            raise ValueError(f"{len(self.neurons)} neurons exceeds N_MAX={hw.n_max}")
        if len(self.synapses) > hw.syn_max:
            raise ValueError(f"{len(self.synapses)} synapses exceeds synapse capacity "
                             f"SYN_MAX={hw.syn_max}")
        for nn in self.neurons:
            to_raw(nn.threshold, "threshold", fb, hw, f"neuron {nn.idx} threshold")
            to_raw(nn.leak, "leak", fb, hw, f"neuron {nn.idx} leak")
            to_raw(nn.reset, "reset", fb, hw, f"neuron {nn.idx} reset")
            to_raw(nn.initial_state, "vmem", fb, hw, f"neuron {nn.idx} vmem")
            if not (0 <= int(nn.refractory) <= ref_max):
                raise ValueError(f"neuron {nn.idx} refractory={nn.refractory} out of 0..{ref_max}")
        for s in self.synapses:
            to_raw(s.weight, "weight", fb, hw, f"synapse {s.idx} weight")
            for who in (s.pre, s.post):
                if not (0 <= int(who) < len(self.neurons)):
                    raise ValueError(f"synapse {s.idx} references neuron {who} (only "
                                     f"{len(self.neurons)} neurons exist)")
        if self._stdp:
            w = int(self._stdp["window"])
            if not (0 <= w <= hw.stdp_window_max):
                raise ValueError(f"STDP window={w} out of 0..{hw.stdp_window_max}")
            for a in self._stdp["apos"]:
                to_raw(a, "apos", fb, hw, "STDP apos")
            for a in self._stdp["aneg"]:
                to_raw(a, "aneg", fb, hw, "STDP aneg")
        if self.is_npu:
            self._validate_npu()

    def build(self):
        """Program the whole network onto the FPGA (validates ranges first)."""
        if self.drv is None:
            raise RuntimeError("no driver/port: SNN was created build-only")
        self.validate()
        fb, hw = self.frac_bits, self.hw
        d = self.drv
        if self.is_npu_stdp:
            return self._build_npu_stdp()
        if self.is_npu:
            return self._build_npu()
        src_ptr = self._assign_chip_indices()
        d.flush()                       # drain leftover events/frames/outputs (clean start)
        if hasattr(d, "probe_n_max"):
            try:
                built_n_max = int(d.probe_n_max())
            except Exception:  # noqa: BLE001 - optional device capability probe
                built_n_max = None
            if built_n_max is not None:
                if len(self.neurons) > built_n_max:
                    raise ValueError(
                        f"the programmed bitstream supports N_MAX={built_n_max}, "
                        f"but this network needs {len(self.neurons)} neurons. "
                        "Program the selected board's max-capacity SpikeEngine "
                        "bitstream before loading."
                    )
                if built_n_max != hw.n_max:
                    raise ValueError(
                        f"the programmed bitstream reports N_MAX={built_n_max}, "
                        f"but the selected board profile expects N_MAX={hw.n_max}. "
                        "Select the matching board or reprogram the board before "
                        "running from the GUI."
                    )
        d.set_n_active(len(self.neurons))
        d.set_s_active(len(self.synapses))
        has_clear_enables = hasattr(d, "clear_enables")
        if has_clear_enables:
            d.clear_enables()
        bulk = hasattr(d, "begin_bulk") and hasattr(d, "end_bulk")
        if bulk:
            d.begin_bulk()
        for nn in self.neurons:
            d.set_neuron(nn.idx,
                         threshold=to_raw(nn.threshold, "threshold", fb, hw),
                         leak=to_raw(nn.leak, "leak", fb, hw),
                         reset_state=to_raw(nn.reset, "reset", fb, hw),
                         refrac_period=int(nn.refractory),
                         input_enable=nn.inp,
                         vmem=to_raw(nn.initial_state, "vmem", fb, hw))
            # Always write the refractory countdown (default 0) so a leftover
            # counter from a previous run cannot suppress spikes in this one.
            d.write_config(snm.SEL_REFRAC_COUNT, nn.idx, int(nn.refractory_state))
        # src_ptr boundaries: one entry per source plus the final end boundary
        for k, p in enumerate(src_ptr):
            d.set_src_ptr(k, p)
        for s in self.synapses:
            stdp_arg = (True if s.stdp else None) if has_clear_enables else s.stdp
            d.set_synapse(s.chip_idx, weight=to_raw(s.weight, "weight", fb, hw),
                          dst=s.post, enable=True, stdp=stdp_arg)
        if self._stdp:
            st = self._stdp
            d.write_config(snm.SEL_STDP_WINDOW, 0, int(st["window"]))
            for j, a in enumerate(st["apos"]):
                d.write_config(snm.SEL_STDP_APOS, j, to_raw(a, "apos", fb, hw))
            for j, a in enumerate(st["aneg"]):
                d.write_config(snm.SEL_STDP_ANEG, j, to_raw(a, "aneg", fb, hw))
            d.write_config(snm.SEL_STDP_GLOBAL, 0, 1 if st["enable"] else 0)
        else:
            # Always drive the global STDP enable low when the GUI-level STDP
            # block is not configured. Otherwise a previous run can leave the
            # FPGA's sticky global STDP bit enabled, and merely marking a
            # synapse as STDP-active would mutate weights on hardware while the
            # software reference still assumes STDP is off.
            d.write_config(snm.SEL_STDP_GLOBAL, 0, 0)
        if bulk:
            d.end_bulk()
        return self

    def check_schedule(self, schedule: dict | None = None):
        """Validate an input schedule against the network before running.

        Catches a silent FPGA-vs-software divergence: the chip only applies an
        external input event if the target neuron has input_enable=1 (inp=True);
        a non-input neuron's events are dropped by the core. The software model,
        however, applies add_spike to ANY neuron -- so a software preview would
        show a spike the real hardware never produces. Reject it up front with a
        clear message instead of letting the two paths disagree.
        """
        for t, ins in (schedule or {}).items():
            if len(ins) > self.hw.event_fifo_depth:
                raise ValueError(
                    f"timestep {t} has {len(ins)} input events, but the hardware "
                    f"event FIFO holds only {self.hw.event_fifo_depth} per tick; "
                    f"spread them across multiple timesteps (e.g. drive all "
                    f"{len(self.neurons)} neurons in ceil(N/{self.hw.event_fifo_depth}) "
                    f"waves)")
            for n in ins:
                i = int(n)
                if not (0 <= i < len(self.neurons)):
                    raise ValueError(f"schedule t={t}: neuron {i} does not exist "
                                     f"(only {len(self.neurons)} neurons)")
                if not self.neurons[i].inp:
                    raise ValueError(
                        f"schedule t={t}: neuron {i} receives an input but was not "
                        f"created with inp=True; the FPGA ignores inputs to "
                        f"non-input neurons (set inp=True on neuron {i})")

    def _build_npu(self):
        """Program the network onto the NPU array via LaneEngineDevice.load_network()
        -- the destination-partitioned load path (incl. the dst_ptr sentinel writes)
        that was hardware-validated bit-exact against superneuromat.SNN. Raw
        fixed-point values are converted here; load_network passes ints through."""
        from types import SimpleNamespace
        fb, hw, d = self.frac_bits, self.hw, self.drv
        proxy = SimpleNamespace(
            neurons=[SimpleNamespace(
                idx=nn.idx,
                threshold=to_raw(nn.threshold, "threshold", fb, hw),
                leak=to_raw(nn.leak, "leak", fb, hw),
                reset_state=to_raw(nn.reset, "reset", fb, hw),
                refractory_period=int(nn.refractory),
            ) for nn in self.neurons],
            synapses=[SimpleNamespace(
                pre_id=s.pre, post_id=s.post,
                weight=to_raw(s.weight, "weight", fb, hw),
            ) for s in self.synapses],
        )
        d.clear_error()
        d.load_network(proxy)
        self._npu_injected = set()      # neurons holding a nonzero input register
        return self

    def _build_npu_stdp(self):
        """Program the network onto the STDP-capable NPU array via
        LaneEngineStdpDevice.load_network() (snm_npu_stdp.py) -- same
        destination-partitioned load as _build_npu(), plus the STDP table
        (Apos/Aneg) and global-enable config. Hardware-validated bit-exact
        against SuperNeuroMAT this session (640/640 trained weights)."""
        from types import SimpleNamespace
        fb, hw, d = self.frac_bits, self.hw, self.drv
        proxy = SimpleNamespace(
            neurons=[SimpleNamespace(
                idx=nn.idx,
                threshold=to_raw(nn.threshold, "threshold", fb, hw),
                leak=to_raw(nn.leak, "leak", fb, hw),
                reset_state=to_raw(nn.reset, "reset", fb, hw),
                refractory_period=int(nn.refractory),
            ) for nn in self.neurons],
            synapses=[SimpleNamespace(
                pre_id=s.pre, post_id=s.post,
                weight=to_raw(s.weight, "weight", fb, hw),
            ) for s in self.synapses],
            apos=[to_raw(a, "apos", fb, hw) for a in (self._stdp or {}).get("apos", [])],
            aneg=[to_raw(a, "aneg", fb, hw) for a in (self._stdp or {}).get("aneg", [])],
            stdp_window=hw.stdp_window_max,
            stdp_enable=bool(self._stdp and self._stdp.get("enable")),
        )
        d.clear_error()
        d.load_network(proxy)
        self._npu_injected = set()      # neurons holding a nonzero input register
        return self

    # ---------------- run ----------------
    def _step_npu(self, inputs: dict) -> list[int]:
        """One NPU tick. The NPU's input registers are PERSISTENT (with a global
        per-tick apply gate), while the software model's add_spike is one-shot --
        so stale values from the previous tick are zeroed before each run. Same
        semantics as fpga_src/tools/test_infer_vs_superneuromat.py, where this
        difference was first found on hardware."""
        d = self.drv
        pairs = {int(n): to_raw(v, "input", self.frac_bits, self.hw)
                 for n, v in (inputs or {}).items()}
        injected = getattr(self, "_npu_injected", set())
        for n in injected - set(pairs):
            d.write_input_value(n, 0)
        for n, v in pairs.items():
            d.write_input_value(n, v)
        self._npu_injected = set(pairs)
        d.run_tick()
        spikes = d.read_spikes()
        return [i for i in range(len(self.neurons)) if spikes[i]]

    def step(self, inputs: dict | None = None) -> list[int]:
        """Inject one frame of inputs ({neuron: value}) and run one tick.
        Input values are decimals in the same fixed-point format as the network."""
        d = self.drv
        inputs = inputs or {}
        if self.is_npu:
            return self._step_npu(inputs)
        if len(inputs) > self.hw.event_fifo_depth:
            raise ValueError(f"{len(inputs)} input events exceeds hardware "
                             f"event FIFO depth {self.hw.event_fifo_depth}")
        pairs = [(int(n), to_raw(v, "input", self.frac_bits, self.hw)) for n, v in inputs.items()]
        if hasattr(d, "input_events"):
            d.input_events(pairs)
        else:
            for neuron, value in pairs:
                d.input_event(neuron, value)
        if hasattr(d, "commit_run_read"):
            frame = d.commit_run_read()
        else:
            d.commit_frame()
            d.run_step()
            frame = d.read_output_frame()
        return frame[1] if frame else []

    def run(self, schedule: dict | None = None, steps: int = 1,
            delay: float = 0.0) -> list[list[int]]:
        """Run `steps` ticks. `schedule` maps timestep -> {neuron: value}.

        Returns the spike train: a list (per tick) of spiking neuron indices.
        `delay` (seconds) pauses between ticks so the board LEDs are watchable.
        """
        import time
        schedule = schedule or {}
        self.check_schedule(schedule)
        train = []
        for t in range(steps):
            train.append(self.step(schedule.get(t, {})))
            if delay:
                time.sleep(delay)
        return train

    def run_trace(self, schedule: dict | None = None, steps: int = 1,
                  delay: float = 0.0):
        """Like run(), but ALSO reads each neuron's internal state back from the
        chip after every tick -- for debugging *why* a neuron did or didn't fire.

        Returns (train, vmem, refrac):
          train[t]     = list of neuron indices that fired at tick t
          vmem[t][i]   = neuron i's membrane voltage after tick t (decimal)
          refrac[t][i] = neuron i's refractory countdown after tick t (ticks)

        Costs N read-backs per tick over UART, so it is slower than run(); use it
        when you need the per-step trace, not for bulk runs. Hardware only.
        """
        import time
        if self.drv is None:
            raise RuntimeError("run_trace needs a hardware driver (use software_trace for preview)")
        if self.is_npu:
            raise RuntimeError("the NPU array has no on-chip state readback (spikes only) "
                               "-- use run() for hardware spikes and software_trace() for "
                               "the per-tick vmem/refractory view")
        schedule = schedule or {}
        self.check_schedule(schedule)
        n = len(self.neurons)
        train, vmem, refrac = [], [], []
        self.tick_cycles = []                 # per-tick on-chip latency (clock cycles)
        perf_ok = True                        # bitstream may predate the perf counter
        for t in range(steps):
            train.append(self.step(schedule.get(t, {})))
            vmem.append([self.vmem(i) for i in range(n)])
            refrac.append([self.refrac(i) for i in range(n)])
            if perf_ok:
                try:
                    self.tick_cycles.append(self.drv.read_perf())  # cycles for THIS timestep
                except snm.SNMError:
                    # old bitstream without the CFG_PERF counter -> skip timing,
                    # clear the sticky error it set, and keep running.
                    perf_ok = False
                    self.tick_cycles = []
                    try:
                        self.drv.clear_error()
                    except snm.SNMError:
                        pass
            if delay:
                time.sleep(delay)
        return train, vmem, refrac

    def cycles_to_seconds(self, cycles: int) -> float:
        """Convert on-chip clock cycles to wall-clock seconds at the build clock."""
        return cycles / float(_cfg.CORE_CLK_HZ)

    # tick-latency model fitted to the cycle-accurate sim (tb_perf_tick_latency.v):
    #   cycles ~= base + per_neuron*N + per_synapse*S_walked  (STDP off)
    _PERF_BASE = 132
    _PERF_PER_NEURON = 15
    _PERF_PER_SYNAPSE = 24

    def estimate_tick_cycles(self, n=None, s=None):
        """Estimated on-chip cycles per timestep as (floor, worst):
          floor = no synaptic activity   = base + per_neuron*N
          worst = every synapse walked each tick = floor + per_synapse*S
        Real latency lies between, depending on spike density. Coefficients come
        from the cycle-accurate sim characterization of this build."""
        n = len(self.neurons) if n is None else n
        s = len(self.synapses) if s is None else s
        floor = self._PERF_BASE + self._PERF_PER_NEURON * n
        return floor, floor + self._PERF_PER_SYNAPSE * s

    def software_trace(self, schedule: dict | None = None, steps: int = 1):
        """Software-model equivalent of run_trace (no board): per-tick spikes +
        Vmem + refractory, by stepping the superneuromat model one tick at a time
        and reading its neuron states. Same (train, vmem, refrac) shape."""
        schedule = schedule or {}
        self.check_schedule(schedule)
        sw, nmap, _ = self.to_software()
        for t, ins in schedule.items():
            for nn, vv in ins.items():
                sw.add_spike(int(t), nmap[int(nn)], float(vv))
        n = len(self.neurons)
        train, vmem, refrac = [], [], []
        for t in range(steps):
            sw.simulate(1)
            train.append([i for i in range(n) if bool(sw.ispikes[t, i])])
            vmem.append([float(sw.neuron_states[i]) for i in range(n)])
            try:
                refrac.append([int(sw.neuron_refractory_periods_state[i]) for i in range(n)])
            except (AttributeError, IndexError, TypeError):
                refrac.append([0] * n)
        return train, vmem, refrac

    def software_reference(self, schedule: dict | None = None, steps: int = 1):
        """Independent SuperNeuroMAT software reference trace + final weights."""
        schedule = schedule or {}
        self.check_schedule(schedule)
        sw, nmap, smap = self.to_software()
        for t, ins in schedule.items():
            for nn, vv in ins.items():
                sw.add_spike(int(t), nmap[int(nn)], float(vv))
        n = len(self.neurons)
        train, vmem, refrac = [], [], []
        for t in range(steps):
            sw.simulate(1)
            train.append([i for i in range(n) if bool(sw.ispikes[t, i])])
            vmem.append([float(sw.neuron_states[i]) for i in range(n)])
            try:
                refrac.append([int(sw.neuron_refractory_periods_state[i]) for i in range(n)])
            except (AttributeError, IndexError, TypeError):
                refrac.append([0] * n)
        weights = {}
        for s in self.synapses:
            sw_syn = smap[(s.pre, s.post)]
            syn_idx = getattr(sw_syn, "idx", int(sw_syn))
            weights[s.idx] = float(sw.synaptic_weights[syn_idx])
        return train, vmem, refrac, weights

    def raster(self, train, **kw) -> str:
        """Spike train as a raster table (rows=timesteps, cols=this net's neurons).
        Large nets auto-collapse to only-fired columns; pass cols=[...] to choose,
        or max_cols=N to change the width threshold. See spike_table()."""
        kw.setdefault("n_neurons", len(self.neurons))
        return spike_table(train, **kw)

    def raster_plot(self, train, **kw):
        """Matplotlib raster plot (best for many neurons). See spike_raster_plot()."""
        kw.setdefault("n_neurons", len(self.neurons))
        return spike_raster_plot(train, **kw)

    # ---------------- read-back ----------------
    def weight_raw(self, a, b=None) -> int:
        """Learned weight of a synapse as the chip's raw int (no fixed-point scaling).

        NPU array (inference-only): there is no weight readback and no on-chip
        STDP, so the loaded weight IS the final weight -- return it from the
        host-side network. NPU-STDP variant: weights DO change on-chip, so
        this reads them back for real, same as the classic core."""
        s = self._find_synapse(a, b)
        if self.is_npu and not self.is_npu_stdp:
            return to_raw(s.weight, "weight", self.frac_bits, self.hw)
        if self.is_npu_stdp:
            # per-lane addressing: chip_idx (a flat classic-core index) does not
            # apply here -- look the entry up by (dst,src), recorded by the last
            # load_network()/_build_npu_stdp() call (snm_npu_stdp.py).
            return self.drv.read_synapse_weight_by_pair(s.post, s.pre, signed=True)
        return self.drv.read_synapse_weight(s.chip_idx, signed=True)

    def weight(self, a, b=None) -> float:
        """Learned weight of synapse a (or a->b) as a decimal (fixed-point scaled)."""
        return from_raw(self.weight_raw(a, b), self.frac_bits)

    def vmem(self, n) -> float:
        """Membrane voltage of neuron n as a decimal (fixed-point scaled)."""
        raw = self.drv.read_config(snm.SEL_VMEM_STATE, int(n), signed=True)
        return from_raw(raw, self.frac_bits)

    def refrac(self, n) -> int:
        """Refractory countdown of neuron n (integer ticks)."""
        return self.drv.read_config(snm.SEL_REFRAC_COUNT, int(n))

    def _find_synapse(self, a, b=None) -> Synapse:
        if isinstance(a, Synapse):
            return a
        if b is not None:
            for s in self.synapses:
                if s.pre == int(a) and s.post == int(b):
                    return s
        raise KeyError(f"no synapse {a}->{b}")

    def close(self):
        if self.drv is not None:
            self.drv.close()

    # ---------------- STDP reference model ----------------
    def predict_weights(self, spike_train) -> dict:
        """Predict final synapse weights using the SuperNeuroMAT STDP rule.

        For each valid history slot j, every STDP-enabled synapse gets either
        +Apos[j] when its pre neuron fired j+1 ticks ago and its post neuron
        fires now, or +Aneg[j] otherwise. This now matches the updated FPGA RTL
        and the SuperNeuroMAT software model.

        Returns {synapse.idx: predicted RAW integer weight} (chip representation;
        compare against weight_raw()). Saturation uses the weight bit width.
        """
        fb, hw = self.frac_bits, self.hw
        w = {s.idx: to_raw(s.weight, "weight", fb, hw) for s in self.synapses}
        if not self._stdp or not self._stdp["enable"]:
            return w
        win = self._stdp["window"]
        apos = [to_raw(a, "apos", fb, hw) for a in self._stdp["apos"]]
        aneg = [to_raw(a, "aneg", fb, hw) for a in self._stdp["aneg"]]
        wlo = -(1 << (hw.weight_bits - 1)); whi = (1 << (hw.weight_bits - 1)) - 1
        sat = lambda x: max(wlo, min(whi, x))
        hist: list[set] = []                      # hist[j] = spikes j+1 ticks ago
        for spikes in spike_train:
            now = set(spikes)
            for s in self.synapses:
                if not s.stdp:
                    continue
                delta = 0
                for j in range(win):
                    if j < len(hist):
                        if (s.pre in hist[j]) and (s.post in now):
                            delta += apos[j] if j < len(apos) else 0
                        else:
                            delta += aneg[j] if j < len(aneg) else 0
                w[s.idx] = sat(w[s.idx] + delta)
            hist.insert(0, now)
            hist = hist[:max(win, 1)]
        return w

    # ---------------- software cross-check ----------------
    def to_software(self):
        """Build the equivalent SNN in the superneuromat software model."""
        try:
            from superneuromat import SNN as SNM
        except ImportError as e:
            raise RuntimeError(
                "the software model is required for Mock preview / cross-check but "
                "'superneuromat' is not installed. Install it with:  pip install "
                "superneuromat   (or: pip install -e ../../superneuromat-main)") from e
        sw = SNM()
        nmap = []
        for nn in self.neurons:
            sn = sw.create_neuron(
                threshold=float(nn.threshold), leak=float(nn.leak),
                reset_state=float(nn.reset), refractory_period=int(nn.refractory),
                refractory_state=int(nn.refractory_state),
                initial_state=float(nn.initial_state))
            nmap.append(sn)
        smap = {}
        for s in self.synapses:
            ss = sw.create_synapse(nmap[s.pre], nmap[s.post], weight=float(s.weight),
                                   delay=1, stdp_enabled=s.stdp)
            smap[(s.pre, s.post)] = ss
        if self._stdp:
            st = self._stdp
            sw.stdp_setup(Apos=[float(x) for x in st["apos"]] or None,
                          Aneg=[float(x) for x in st["aneg"]] or None,
                          positive_update=bool(st["apos"]) and st["enable"],
                          negative_update=bool(st["aneg"]) and st["enable"])
        return sw, nmap, smap

    def cross_check(self, schedule: dict | None = None, steps: int = 1,
                    verbose=True, check_weights=True):
        """Run the SAME network on FPGA and in software; diff spikes + weights.

        Call this on a FRESHLY built network (it runs the FPGA, which mutates
        STDP weights). Neuron/synapse DYNAMICS match the software model exactly.

        Spike DYNAMICS and learned STDP weights are compared against the
        SuperNeuroMAT software model. Pass check_weights=False to skip the
        weight check entirely.
        """
        schedule = schedule or {}
        fpga_train = self.run(schedule, steps)

        sw, nmap, _smap = self.to_software()
        for t, ins in schedule.items():
            for n, v in ins.items():
                sw.add_spike(int(t), nmap[int(n)], float(v))
        sw.simulate(steps)

        n = len(self.neurons)
        mism = []
        for t in range(steps):
            fpga_set = set(fpga_train[t])
            sw_set = {i for i in range(n) if bool(sw.ispikes[t, i])}
            if fpga_set != sw_set:
                mism.append((t, sorted(fpga_set), sorted(sw_set)))

        wdiff = []
        if check_weights:
            predicted = self.predict_weights(fpga_train)
            for s in self.synapses:
                fw = self.weight_raw(s)
                if fw != predicted[s.idx]:
                    wdiff.append((s.pre, s.post, fw, predicted[s.idx]))

        ok = not mism and not wdiff
        if verbose:
            print(f"cross_check: {'MATCH' if ok else 'MISMATCH'} "
                  f"({steps} steps, {n} neurons, {len(self.synapses)} synapses)"
                  f"  [spikes + weights vs software model]")
            for t, f, s in mism:
                print(f"  t={t}: FPGA spikes {f} vs software {s}")
            for pre, post, fw, pw in wdiff:
                print(f"  weight {pre}->{post}: FPGA {fw} vs FPGA-model {pw}")
        return ok, mism, wdiff
