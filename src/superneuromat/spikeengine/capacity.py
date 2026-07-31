"""Capacity estimation + board recommendation for custom (sparse) STDP builds.

Answers, BEFORE any Vivado time: given a network's real size (neurons +
per-destination-neuron in-synapse counts), which supported board can a
bitstream be BUILT for, and with what `N_MAX` / `NUM_LANES` /
`SYN_CAP_PER_LANE`?

Calibrated against the THREE real builds this project has measured (see
`_CALIBRATION` and `tests/test_capacity.py`), NOT pure theory. Two hard facts
baked in from the sparse-capacity pilot + the A2 dataset analysis
(board_variants/npu_stdp_dev/{pilot_sparse_capacity,sgnn_port}):

1. Xilinx BRAM/URAM primitives are natively dual-port -- STDP's extra read
   port does NOT double the tile count (an early assumption did; the real
   builds disproved it).

2. All of a destination neuron's in-synapses live on ONE lane
   (`lane = dst % NUM_LANES`), so `SYN_CAP_PER_LANE >= max_in_degree` is a
   HARD FLOOR no lane count can lower. `required_syn_cap_per_lane()` therefore
   computes the REAL max-lane total from the actual network, not a heuristic.

LUT is modeled only DIRECTIONALLY (3 confounded calibration points; does not
capture LOCAL_D) and is never used as a hard fit gate here -- it is reported
so a caller can see it, with the explicit caveat that a real build is the
final arbiter. Memory (BRAM/URAM) is the gate.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

RAMB36_BITS = 36864       # 7-series RAMB36 / UltraScale+ RAMB36 (2x RAMB18)
URAM_BITS = 4096 * 72     # UltraScale+ URAM288: 4096 rows x 72 bits

# URAM packs ENTRY_W-wide words more efficiently than a flat bits/URAM_BITS
# division -- calibrated so zcu104's real dense build (64 tiles) is reproduced.
_URAM_PACKING_EFFICIENCY = 4 / 6

# Real designs use a few BRAM tiles for the command/spike FIFOs that the pure
# synapse-memory model doesn't count. Measured on basys3 (synapse model 48,
# real 50). Applied ONLY in the fit decision (conservative), never in the
# regression-tested core estimate.
_FIXED_BRAM_OVERHEAD = 2
# Extra safety margin on memory before declaring a fit, since these are
# estimates outside the calibration range for large sparse builds.
_FIT_MARGIN = 0.05


@dataclass(frozen=True)
class BoardResources:
    key: str
    label: str
    lut_total: int
    bram36_total: int
    uram_total: int = 0


# LUT/BRAM/URAM capacities of each board's actual part.
BOARD_RESOURCES = {
    "basys3": BoardResources("basys3", "Digilent Basys3 (xc7a35t)",
                             lut_total=20800, bram36_total=50, uram_total=0),
    "sp701": BoardResources("sp701", "Xilinx SP701 (xc7s100)",
                            lut_total=64000, bram36_total=120, uram_total=0),
    "zcu104": BoardResources("zcu104", "Xilinx ZCU104 (xczu7ev)",
                             lut_total=230400, bram36_total=312, uram_total=96),
}

# (n_max, num_lanes, syn_cap_per_lane, weight_w, data_w) -> measured tiles.
# From the actual post_route_util.rpt files (re-read 2026-07-29). These are
# what tests/test_capacity.py asserts the model reproduces.
_CALIBRATION = {
    "basys3": {"n_max": 256, "num_lanes": 8, "weight_w": 16, "data_w": 24,
                   "syn_cap_per_lane": (256 // 8) * 256,        # dense
                   "real_bram": 50, "real_uram": 0, "real_lut": 20437},
    "sp701": {"n_max": 352, "num_lanes": 8, "weight_w": 16, "data_w": 24,
                  "syn_cap_per_lane": math.ceil(352 / 8) * 352,  # dense
                  "real_bram": 86, "real_uram": 0, "real_lut": 21894},
    "zcu104": {"n_max": 1024, "num_lanes": 16, "weight_w": 16, "data_w": 24,
                   "syn_cap_per_lane": (1024 // 16) * 1024,     # dense
                   "real_bram": 18, "real_uram": 64, "real_lut": 82392},
}


def entry_width(weight_w: int, n_max: int) -> int:
    """Bits per stored synapse entry: signed weight + source-neuron index."""
    src_w = max(1, math.ceil(math.log2(max(n_max, 2))))
    return weight_w + src_w


def synapse_bram36_tiles(syn_cap_per_lane: int, num_lanes: int,
                         weight_w: int, n_max: int) -> int:
    """BRAM36 tiles for the synapse tables only (no FIFO overhead). This is the
    regression-tested core estimate."""
    bits_per_lane = syn_cap_per_lane * entry_width(weight_w, n_max)
    return math.ceil(bits_per_lane / RAMB36_BITS) * num_lanes


def synapse_uram_tiles(syn_cap_per_lane: int, num_lanes: int,
                       weight_w: int, n_max: int) -> int:
    bits_per_lane = syn_cap_per_lane * entry_width(weight_w, n_max)
    return math.ceil(bits_per_lane / URAM_BITS * _URAM_PACKING_EFFICIENCY) * num_lanes


def estimate_lut(n_max: int, num_lanes: int) -> int:
    """N-driven LUT estimate. Recalibrated 2026-07-30 against the real ZCU104
    citation-dataset builds (Option A, wide fixed-point, K=8): miniseer N=2116 ->
    89,350 LUT, cora N=2715 -> 112,873. The dominant term is O(N*NUM_LANES) --
    the spike/history state is replicated per lane (spike_frozen[NUM_LANES], each
    N-wide), so LUT/neuron roughly doubles from K=8 (~40) to K=16 (~80). This
    matches all five calibration points within +/-25%:
        basys3 N=256/K8  20,437 (est ~20.4k)   sp701 N=352/K8 21,894 (est ~24.3k)
        zcu104 N=1024/K16 82,392 (est ~100k)   miniseer/cora as above.
    Reliable enough that `recommend_board` uses it to reject boards whose neuron
    count blows the LUT budget (the real ceiling -- neuron count is LUT-bound,
    not synapse-bound)."""
    return int(2000 + 5 * num_lanes * n_max + 1025 * num_lanes)


class CapacityError(ValueError):
    """A network would exceed a board's per-lane synapse capacity. Raised BEFORE
    any wire write, because the RTL has no write-side bounds check -- an
    out-of-range synapse write silently corrupts a DIFFERENT neuron's data on
    hardware (proven in the sparse-capacity pilot). This is the mandatory host
    guard for that."""


def validate_network_fits(post_neuron_ids: Iterable[int], num_lanes: int,
                          syn_cap_per_lane: int) -> None:
    """Raise CapacityError if any lane's total in-synapse count would exceed
    ``syn_cap_per_lane``. Pure host arithmetic against the exact lane mapping
    the RTL loader uses (``lane = dst % num_lanes``) -- needs no hardware.

    The error names EVERY overflowing lane, the top contributing destination
    neuron on the worst lane, and the required capacity, so a caller can either
    rebuild with a larger ``SYN_CAP_PER_LANE`` or reduce the network."""
    post = [int(d) for d in post_neuron_ids]
    in_deg = Counter(post)
    lane_tot = [0] * num_lanes
    lane_hub = [(0, -1)] * num_lanes   # (max single-neuron in-deg on lane, that neuron id)
    for d, c in in_deg.items():
        L = d % num_lanes
        lane_tot[L] += c
        if c > lane_hub[L][0]:
            lane_hub[L] = (c, d)
    over = [(L, lane_tot[L]) for L in range(num_lanes) if lane_tot[L] > syn_cap_per_lane]
    if not over:
        return
    worst_L, worst_tot = max(over, key=lambda x: x[1])
    hub_deg, hub_id = lane_hub[worst_L]
    required = max(lane_tot)
    raise CapacityError(
        f"network exceeds SYN_CAP_PER_LANE={syn_cap_per_lane} on {len(over)} "
        f"lane(s). Worst: lane {worst_L} needs {worst_tot} synapse entries "
        f"(over by {worst_tot - syn_cap_per_lane}); its largest single "
        f"destination neuron is id {hub_id} with {hub_deg} in-synapses. "
        f"Required SYN_CAP_PER_LANE for this network at NUM_LANES={num_lanes} "
        f"is {required}. Rebuild the bitstream with SYN_CAP_PER_LANE>={required}, "
        f"or reduce the network. (No wire writes were issued.)")


def required_syn_cap_per_lane(post_neuron_ids: Iterable[int], num_lanes: int) -> int:
    """The REAL minimum SYN_CAP_PER_LANE for a network, from its actual
    synapse list: group synapses by destination neuron -> lane (dst %
    num_lanes) -> the max over lanes of summed in-degree. This is the number a
    build MUST use (or exceed); anything less silently corrupts on hardware
    (the RTL has no write-side bounds check -- see A4/the host guard)."""
    in_deg = Counter(int(d) for d in post_neuron_ids)
    lane_tot = [0] * num_lanes
    for d, c in in_deg.items():
        lane_tot[d % num_lanes] += c
    return max(lane_tot) if lane_tot else 0


def estimate(board: str, n_max: int, num_lanes: int, syn_cap_per_lane: int,
             *, weight_w: int = 16, data_w: int = 24) -> dict:
    """Estimate resources for a (custom) build config on `board`. Memory is the
    fit gate; LUT is reported directionally only."""
    b = BOARD_RESOURCES[board]
    bram_syn = synapse_bram36_tiles(syn_cap_per_lane, num_lanes, weight_w, n_max)
    lut = estimate_lut(n_max, num_lanes)

    # Prefer BRAM; fall back to URAM (if the board has it) only when BRAM alone
    # would overflow -- matches the real zcu104 dense build's behavior.
    bram_with_oh = bram_syn + _FIXED_BRAM_OVERHEAD
    use_uram = b.uram_total > 0 and bram_with_oh > b.bram36_total
    if use_uram:
        mem_used = synapse_uram_tiles(syn_cap_per_lane, num_lanes, weight_w, n_max)
        mem_total, mem_kind = b.uram_total, "URAM"
    else:
        mem_used, mem_total, mem_kind = bram_with_oh, b.bram36_total, "BRAM"

    # `fits`: does it fit at all (estimated used <= total)? A config at exactly
    #   100% (like basys3's real dense build, 50/50 BRAM) reports fits=True.
    # `fits_with_margin`: is there `_FIT_MARGIN` headroom? recommend_board
    #   prefers this, since large SPARSE builds are estimates outside the
    #   calibration range and shouldn't be planned to the exact tile.
    mem_fits = mem_used <= mem_total
    mem_fits_margin = mem_used <= math.floor(mem_total * (1 - _FIT_MARGIN))

    lut_fits = lut <= b.lut_total   # advisory only
    return {
        "board": board, "n_max": n_max, "num_lanes": num_lanes, "syn_cap_per_lane": syn_cap_per_lane,
        "mem_kind": mem_kind, "mem_used": mem_used, "mem_total": mem_total,
        "lut_est": lut, "lut_total": b.lut_total, "lut_advisory_fits": lut_fits,
        "fits": bool(mem_fits),               # MEMORY is the gate; LUT is advisory
        "fits_with_margin": bool(mem_fits_margin),
    }


def recommend_board(n_neurons: int, post_neuron_ids: Iterable[int], *,
                    weight_w: int = 16, data_w: int = 24,
                    num_lanes_options: tuple[int, ...] = (8, 16),
                    boards: tuple[str, ...] = ("basys3", "sp701", "zcu104"),
                    cap_headroom: float = 0.1) -> dict:
    """Given a real network (neuron count + its synapse destination-id list),
    return the required per-lane capacity and a ranked list of feasible build
    configs, plus a chosen recommendation.

    Each option reports BOTH `required_syn_cap_per_lane` (the exact max-lane
    total of THIS network instance) and `syn_cap_per_lane` = the value to
    actually BUILD with, which is `required * (1 + cap_headroom)` rounded up.

    Why the headroom: the sgnn model's `make_network` iterates over Python
    `set`s of string paper-IDs, so the neuron->lane assignment (dst % NUM_LANES)
    varies by a few synapses per lane across process runs (hash randomization;
    ~2% for microseer, confirmed 2026-07-29). Building with headroom means a
    later rebuild of the same model won't exceed the bitstream's capacity. For
    fully reproducible sizing, also fix PYTHONHASHSEED. The load-time host guard
    (`validate_network_fits`) remains the hard safety net for the EXACT instance
    being loaded, regardless of headroom.
    """
    post_ids = list(post_neuron_ids)
    options = []
    for board in boards:
        for k in num_lanes_options:
            req = required_syn_cap_per_lane(post_ids, k)
            build_cap = math.ceil(req * (1 + cap_headroom))
            est = estimate(board, n_neurons, k, build_cap, weight_w=weight_w, data_w=data_w)
            est["required_syn_cap_per_lane"] = req   # exact, for THIS instance
            est["syn_cap_per_lane"] = build_cap      # value to BUILD with (has headroom)
            options.append(est)
    board_order = {b: i for i, b in enumerate(boards)}
    # Ranking, most-conservative first:
    #   1. memory fits WITH margin, AND LUT advisory is clean (safest)
    #   2. memory fits with margin, LUT advisory tight (LUT model is unreliable
    #      -- see estimate_lut docstring -- so this is "probably fine, confirm
    #      with a real build", not a rejection)
    #   3. memory fits with zero margin (e.g. a basys3-style 100% build)
    # then by fewest lanes, then smallest board.
    def key(o):
        return (o["num_lanes"], board_order[o["board"]])
    # LUT feasibility gates every tier (2026-07-31). Previously only tier1
    # considered it, so whenever no margin-clean option existed, tier2/tier3
    # could return a board whose LUT estimate exceeds the device -- a config
    # that cannot be built at all, while estimate_lut()'s own docstring says
    # this estimate is used "to reject boards whose neuron count blows the LUT
    # budget". A LUT-infeasible option is never preferred over a feasible one
    # now. It stays reachable as a last resort rather than returning nothing,
    # because the estimate carries +/-25% error and memory is the calibrated
    # quantity; `recommended_lut_advisory_clean` reports which case applies.
    tier1 = sorted((o for o in options if o["fits_with_margin"] and o["lut_advisory_fits"]), key=key)
    tier2 = sorted((o for o in options if o["fits"] and o["lut_advisory_fits"]), key=key)
    tier3 = sorted((o for o in options if o["fits_with_margin"]), key=key)
    tier4 = sorted((o for o in options if o["fits"]), key=key)
    recommended = (tier1 or tier2 or tier3 or tier4 or [None])[0]
    return {
        "n_neurons": n_neurons,
        "required_cap_by_lanes": {k: required_syn_cap_per_lane(post_ids, k) for k in num_lanes_options},
        "options": options,
        "recommended": recommended,
        "recommended_has_margin": bool(recommended and recommended["fits_with_margin"]),
        "recommended_lut_advisory_clean": bool(recommended and recommended["lut_advisory_fits"]),
    }
