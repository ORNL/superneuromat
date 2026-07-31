"""NPU-STDP array support for the SNN Console GUI (2026-07-27, rewired 2026-07-30).

Sibling of `snm_npu.py` (the inference-only NPU array support module, which
this file deliberately does NOT modify) for the STDP-CAPABLE variant of the
same destination-partitioned K-lane engine: on-chip STDP training is
supported here, plus weight readback -- the two capabilities `snm_npu.py`'s
own module docstring documents as missing ("NO on-chip STDP", "no state
readback").

REWIRED 2026-07-30: previously subclassed `superneuromat.spikeengine.
fpga_runtime.LaneEngineDevice` (a separate, independently-maintained package)
and re-implemented width-parametric write_synapse/read_synapse/write_stdp_*
on top of it. Now subclasses `superneuromat.spikeengine.runtime.InferEngineStdpDevice`
directly -- the class this session hardware-validated on Basys3/SP701/ZCU104
(citation-GNN datasets, 100% SW==HW agreement) -- which already provides
every one of those methods natively (verified byte-identical wire packing
against the old overrides before removing them: same OP/IL constants, same
`(src << weight_w) | weight` packing, same STDP-table `(aneg << weight_w) |
apos` packing). This removes the GUI's hard dependency on the other package
entirely -- `superneuromat.spikeengine` alone is now sufficient to import and run this
module.

Board specs (all three: basys3, sp701, zcu104) are sourced directly from the
installed `superneuromat.spikeengine` package's `boards.py` -- the SAME catalogue
`superneuromat.spikeengine.connect()`/`superneuromat.spikeengine.program.program()` use -- rather than a
second, hand-duplicated board list here.

Hardware-validated this session: on-chip STDP training of the 64-input/
10-output digits classifier is bit-exact against SuperNeuroMAT (640/640
weights) on Basys3; the citation-GNN datasets (microseer/miniseer/cora/
citeseer) additionally validate this same wire protocol end-to-end on
Basys3, SP701, and ZCU104 with 100% SW==HW per-paper agreement (see
superneuromat/spikeengine/docs/ARCHITECTURE.md Sec.4).
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from superneuromat.spikeengine.boards import StdpBoard, get_board, list_boards
from superneuromat.spikeengine.datasets import get_dataset
from superneuromat.spikeengine.runtime import InferEngineStdpDevice

KEY_PREFIX = "npu-stdp:"

# Empty as of 2026-07-31: SP701's default image was rebuilt (the previous one
# had confirmed-wrong N_MAX/NUM_LANES) and is now hardware-validated itself --
# 6/6 synapses bit-exact, correct addressing. Every board's own default is
# validated, so no substitution is needed; this stays as the mechanism for the
# next time a board default doesn't hold up under real hardware testing.
_VALIDATED_GUI_DATASET: dict[str, str] = {}


def is_npu_stdp(name) -> bool:
    return isinstance(name, str) and name.startswith(KEY_PREFIX)


def base_board(key: str) -> str:
    """'npu-stdp:basys3' -> 'basys3'."""
    return key[len(KEY_PREFIX):] if is_npu_stdp(key) else key


def keys() -> list[str]:
    return [KEY_PREFIX + b for b in list_boards()]


def _spec(key: str) -> StdpBoard:
    b = base_board(key)
    try:
        return get_board(b)
    except KeyError:
        raise KeyError(f"no NPU-STDP board named {b!r}; available: {list_boards()}") from None


def dataset_profile(key: str) -> str | None:
    """Dataset image selected by the GUI for this board, if any."""
    return _VALIDATED_GUI_DATASET.get(base_board(key))


def _resolved(key: str):
    board = _spec(key)
    dataset_name = dataset_profile(key)
    dataset = get_dataset(dataset_name) if dataset_name else None
    return board, dataset


def label(key: str) -> str:
    s, ds = _resolved(key)
    if ds is not None:
        return (f"NPU array (STDP) — {s.label} — validated {ds.key} image "
                f"(N={ds.neurons}, K={ds.num_lanes})")
    return f"NPU array (STDP) — {s.label} (N={s.n_max}, K={s.num_lanes})"


def baud_for(key: str) -> int:
    return int(_spec(key).baud)


def bitstream_path(key: str):
    s, ds = _resolved(key)
    if ds is not None:
        path = ds.bitstream_path(s.key)
        if path is None:
            raise RuntimeError(
                f"GUI profile {s.key}/{ds.key} has no board-specific bitstream")
        return path
    return s.bitstream_path()


def board_dict(key: str) -> dict:
    """Board descriptor dict shaped like snm_boards.get()'s / snm_npu.board_dict()'s."""
    s, ds = _resolved(key)
    n_max = ds.neurons if ds is not None else s.n_max
    num_lanes = ds.num_lanes if ds is not None else s.num_lanes
    validated = ds.hardware_validated if ds is not None else s.hardware_validated
    return {
        "name": key,
        "label": label(key),
        "part": s.part,
        "core_clk_hz": s.core_clk_hz,
        "status": "supported" if validated else "built_unverified",
        "validation_profile": (f"dataset {ds.key}" if ds is not None
                            else "board default"),
        "npu": {"n_max": n_max, "num_lanes": num_lanes,
                 "hardware_validated": validated,
                 "hardware_validated_date": s.hardware_validated_date,
                 "dataset_profile": ds.key if ds is not None else None},
    }


def hw_for_board(key: str):
    """HardwareSpec for an NPU-STDP board (lazy import, same pattern as snm_npu)."""
    from . import snm_network as nm
    s, ds = _resolved(key)
    n_max = ds.neurons if ds is not None else s.n_max
    num_lanes = ds.num_lanes if ds is not None else s.num_lanes
    syn_cap = (ds.hardware_syn_cap_per_lane(s.key)
               if ds is not None else s.syn_cap_per_lane)
    data_w = ds.data_w if ds is not None else s.data_w
    weight_w = ds.weight_w if ds is not None else s.weight_w
    return nm.HardwareSpec(
        n_max=n_max,
        syn_max=syn_cap * num_lanes,
        data_bits=data_w,
        weight_bits=weight_w,
        ref_bits=8,
        stdp_window_max=s.stdp_window,   # ON-CHIP STDP: nonzero, unlike inference-only NPU
        event_fifo_depth=n_max,
        npu=True,
        npu_stdp=True,
        num_lanes=num_lanes,
    )


def frac_bits_for(key: str) -> int:
    # Dataset-specific wide images were validated at Q*.13. Board-default
    # images retain their catalogue fixed-point setting.
    return 13 if dataset_profile(key) is not None else int(_spec(key).frac_bits_default)


class LaneEngineStdpDevice(InferEngineStdpDevice):
    """`superneuromat.spikeengine.runtime.InferEngineStdpDevice` plus the GUI-specific
    (dst,src)->(lane,idx) bookkeeping `read_synapse_weight_by_pair` needs, and
    a `load_network()` DEVICE METHOD (the GUI's own network representation --
    `net.neurons`/`net.synapses` with `.post_id`/`.pre_id` -- is not a
    `superneuromat.SNN`, so this does not reuse `superneuromat.spikeengine.network.
    load_network()`, which expects that specific shape; it calls the same
    underlying device primitives directly instead). Every wire-protocol
    method (write_synapse, read_synapse, write_threshold/leak/reset_state/
    input_value, write_stdp_table_all_lanes, set_stdp_enable, soft_reset,
    run_tick_streaming, read_spikes_streaming, configure_neuron, lane_of,
    local_of, write_dptr/write_dptr_raw, begin_bulk/end_bulk) is inherited
    unmodified from the base class -- no overrides needed here."""

    def __init__(self, port, baud, n_max=1024, num_lanes=16, timeout=1.0,
                 weight_w: int = 16, data_w: int = 24,
                 syn_cap_per_lane: int | None = None,
                 prefer_interface: int | None = None,
                 stdp_window: int = 5):
        super().__init__(port, baud, n_max=n_max, num_lanes=num_lanes,
                         timeout=timeout, weight_w=weight_w, data_w=data_w,
                         syn_cap_per_lane=syn_cap_per_lane,
                         prefer_interface=prefer_interface,
                         stdp_window=stdp_window)
        self._entry_index: dict[tuple[int, int], tuple[int, int]] = {}

    def read_synapse_weight_by_pair(self, dst_neuron: int, src_neuron: int, signed: bool = True) -> int:
        """Weight readback keyed by (dst, src) instead of a flat chip_idx --
        this engine addresses synapses per-lane, so `load_network` below
        records the (dst,src)->(lane,idx) map needed to look one back up.
        `signed` is accepted for call-site compatibility with older code
        paths; this method always returns a signed value (weight_w
        two's-complement, which is what `read_synapse` already returns)."""
        key = (int(dst_neuron), int(src_neuron))
        if key not in self._entry_index:
            raise KeyError(f"no synapse {src_neuron}->{dst_neuron} in the last load_network() call")
        _lane, idx = self._entry_index[key]
        _got_src, weight = self.read_synapse(dst_neuron, idx)
        return weight

    def load_network(self, net: Any, *, use_bulk: bool = True) -> None:
        """Destination-partitioned load (dst_ptr sentinel writes included) of
        the GUI's own network representation, PLUS: (1) records
        `self._entry_index[(dst,src)] = (lane, idx)` so weights can be read
        back after training, and (2) loads the STDP table + global enable if
        `net` carries `.apos`/`.aneg`/`.stdp_enable` (a GUI-built proxy sets
        these from the SNN's STDP block; see snm_network._build_npu_stdp)."""
        by_dst: dict[int, list[tuple[int, int]]] = {}
        for syn in net.synapses:
            by_dst.setdefault(int(syn.post_id), []).append((int(syn.pre_id), int(syn.weight)))

        if use_bulk:
            self.clear_error()
            self.begin_bulk()
        try:
            for neuron in net.neurons:
                self.configure_neuron(
                    int(neuron.idx), threshold=int(neuron.threshold), leak=int(neuron.leak),
                    reset_state=int(neuron.reset_state), refrac_period=int(neuron.refractory_period),
                    input_enable=True,
                )
            # Build the COMPLETE per-lane dst_ptr table -- every lane, every
            # local index 0..local_d inclusive (2026-07-31 fix). The previous
            # version wrote pointers only for destinations that had incoming
            # synapses, and a sentinel only after the last populated
            # destination on each USED lane. Neurons with no inputs, index
            # gaps, and entirely unused lanes therefore kept whatever values a
            # previously-loaded network had left in that memory, which defines
            # a bogus synapse range for them: the gather stage walks
            # dst_ptr[d]..dst_ptr[d+1], so a stale pair makes a neuron
            # accumulate another network's synapses. This mirrors
            # spikeengine.network.load_network, which always writes the full
            # table.
            num_lanes = self.num_lanes
            local_d = self.local_d
            lane_dptr = [[0] * (local_d + 1) for _ in range(num_lanes)]
            lane_entries: list[list[tuple[int, int, int]]] = [[] for _ in range(num_lanes)]

            self._entry_index = {}
            for lane in range(num_lanes):
                cursor = 0
                for local_dst in range(local_d):
                    d = local_dst * num_lanes + lane
                    lane_dptr[lane][local_dst] = cursor
                    for (pre, w) in by_dst.get(d, []):
                        self._entry_index[(d, pre)] = (lane, cursor)
                        lane_entries[lane].append((d, pre, w))
                        cursor += 1
                lane_dptr[lane][local_d] = cursor      # per-lane sentinel

            for lane in range(num_lanes):
                for local_idx, off in enumerate(lane_dptr[lane]):
                    self.write_dptr_raw(lane, local_idx, off)
            # Each physical entry is written from its own record, so duplicate
            # (dst, pre) pairs keep their individual weights.
            for lane in range(num_lanes):
                for idx, (d, pre, w) in enumerate(lane_entries[lane]):
                    self.write_synapse(d, idx, pre, w)

            apos = list(getattr(net, "apos", []) or [])
            aneg = list(getattr(net, "aneg", []) or [])
            stdp_window = int(getattr(net, "stdp_window", 5) or 5)
            if apos or aneg:
                for j in range(stdp_window):
                    a = apos[j] if j < len(apos) else 0
                    n = aneg[j] if j < len(aneg) else 0
                    self.write_stdp_table_all_lanes(j, int(a), int(n))
            self.set_stdp_enable(bool(getattr(net, "stdp_enable", False)))
        finally:
            if use_bulk:
                self.end_bulk()


def connect(port: str, key: str, baud: int | None = None,
            timeout: float = 2.0) -> LaneEngineStdpDevice:
    """Open a LaneEngineStdpDevice for the NPU-STDP bitstream and health-check it."""
    s, ds = _resolved(key)
    n_max = ds.neurons if ds is not None else s.n_max
    num_lanes = ds.num_lanes if ds is not None else s.num_lanes
    syn_cap = (ds.hardware_syn_cap_per_lane(s.key)
               if ds is not None else s.syn_cap_per_lane)
    weight_w = ds.weight_w if ds is not None else s.weight_w
    data_w = ds.data_w if ds is not None else s.data_w
    p = None if (not port or str(port).strip().lower() == "auto") else port
    dev = LaneEngineStdpDevice(
        p, int(baud or s.baud), n_max=n_max, num_lanes=num_lanes,
        timeout=timeout, weight_w=weight_w, data_w=data_w,
        syn_cap_per_lane=syn_cap, prefer_interface=s.uart_interface_index,
        stdp_window=s.stdp_window)
    try:
        dev.clear_error()
        dev.read_status()
        # Verify the loaded image before the GUI can emit configuration writes.
        # The previous GUI tooltip promised this check but the connector only
        # performed a generic status read.
        from superneuromat.spikeengine.validate_hardware import probe_spike_words
        actual_words = probe_spike_words(dev)
    except Exception as exc:
        with suppress(Exception):
            dev.close()
        raise RuntimeError(
            f"no response from the NPU-STDP bitstream on {port!r}. Program "
            f"{bitstream_path(key)} first (Vivado JTAG; see this package's "
            f"program.py). Underlying error: {exc}"
        ) from exc
    expected_words = (n_max + 31) // 32
    if actual_words != expected_words:
        dev.close()
        profile = f"{s.key}/{ds.key}" if ds is not None else f"{s.key} default"
        raise RuntimeError(
            f"loaded bitstream reports SPIKE_WORDS={actual_words}, but GUI "
            f"profile {profile} requires {expected_words} (N={n_max}). "
            f"Program {bitstream_path(key)} before connecting.")
    return dev
