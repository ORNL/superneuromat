"""High-level helpers to map a SuperNeuroMAT ``SNN`` onto a programmed STDP
board and replay a training/inference schedule -- the app-layer API the
example notebook uses, extracted from the validated hardware cross-check.

The functions here deliberately mirror the fixed-point contract the RTL
implements (signed WEIGHT_W-bit weights on a 1/2**frac_bits grid, per-tick
saturating clamp) so a host-side SuperNeuroMAT reference stays bit-exact
against the device.
"""
from __future__ import annotations

import math

from .capacity import CapacityError
from .runtime import InferEngineStdpDevice


def quantize_raw(value_real: float, frac_bits: int, *, width: int | None = None) -> int:
    """Real value -> raw integer on the 1/2**frac_bits grid (round-half-to-even
    matches numpy.round, which the SuperNeuroMAT reference uses).

    Infinities are saturated to the widest representable signed value rather
    than raising (2026-07-31). SuperNeuroMAT's DEFAULT neuron leak is ``inf``,
    meaning "leak fully to reset_state every tick"; the RTL's
    ``leak_toward_reset`` clamps at reset_state, so a maximal leak is exactly
    equivalent. Without this, a network built with library defaults crashed
    here with OverflowError. ``width`` gives the target field width for that
    saturation; when omitted a 64-bit bound is used and the caller's own mask
    or clamp applies afterwards.
    """
    if math.isnan(value_real):
        raise ValueError("cannot quantize NaN")
    bits = 64 if width is None else int(width)
    hi, lo = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    if value_real == float("inf"):
        return hi
    if value_real == float("-inf"):
        return lo
    return max(lo, min(hi, round(value_real * (1 << frac_bits))))


def _clamp_weight(raw: int, weight_w: int) -> int:
    hi = (1 << (weight_w - 1)) - 1
    lo = -(1 << (weight_w - 1))
    return max(lo, min(hi, raw))


def load_network(device: InferEngineStdpDevice, snn, *, frac_bits: int,
                 weight_w: int | None = None, data_w: int | None = None,
                 use_bulk: bool = True,
                 stdp_window: int = 5, syn_cap_per_lane: int | None = None):
    """Configure ``device`` with the topology, weights, per-neuron params and
    STDP table of a SuperNeuroMAT ``snn``. Loads the SNN's CURRENT weights
    (so a freshly-built network loads its initial weights, and the device then
    learns on-chip as ticks run). Returns a dict of bookkeeping useful for
    reading results back:  ``{'entry_index': {(post,pre):(lane,idx)},
    'n_neurons': int, 'n_synapses': int}``.

    Assumes synapses were created pre-ascending (all of source i's synapses
    before i+1's), which makes the on-chip source ordering a no-op -- the same
    assumption the reference flow enforces. dst_ptr sentinel slots (one per
    lane) are written so ragged/short lanes have well-formed empty ranges.
    """
    # Datapath widths MUST default to the device's, not to fixed 8/16
    # (2026-07-31 fix). The old defaults silently truncated on a wide device:
    # weight 2.0 at frac_bits=11 stored raw 127 instead of 4096, with no error
    # anywhere. An explicit value that disagrees with the device is also
    # rejected -- it can only mean the caller has the wrong bitstream in mind.
    if weight_w is None:
        weight_w = device.weight_w
    elif int(weight_w) != int(device.weight_w):
        raise ValueError(
            f"weight_w={weight_w} disagrees with the connected device's "
            f"weight_w={device.weight_w}. The device value comes from the "
            "board/dataset catalogue and must match the loaded bitstream; "
            "writing at a different width corrupts every weight. Omit "
            "weight_w to use the device's, or reconnect with the right "
            "board=/dataset=.")
    if data_w is None:
        data_w = device.data_w
    elif int(data_w) != int(device.data_w):
        raise ValueError(
            f"data_w={data_w} disagrees with the connected device's "
            f"data_w={device.data_w} (see weight_w note above).")

    n_neurons = len(snn.neuron_thresholds)
    pre = [int(x) for x in snn.pre_synaptic_neuron_ids]
    post = [int(x) for x in snn.post_synaptic_neuron_ids]
    wts = [float(x) for x in snn.synaptic_weights]

    num_lanes = device.num_lanes
    local_d = device.local_d

    # Bounds validation BEFORE any wire write (2026-07-31 fix). The RTL has no
    # write-side bounds check, so an out-of-range neuron id does not error --
    # it writes into another neuron's slot, or into the per-lane dst_ptr
    # sentinel at local index local_d, which silently redefines that lane's
    # synapse ranges. Reproduced: a 3-neuron network on an n_max=2 device
    # configured neuron 2 at local index 1, the sentinel slot.
    if n_neurons > device.n_max:
        raise CapacityError(
            f"network has {n_neurons} neurons but the loaded bitstream "
            f"supports n_max={device.n_max}. Nothing was written. Use a board "
            "or dataset bitstream built for this size (see capacity.py's "
            "recommend_board()).")
    # The three synapse arrays are zipped together below; unequal lengths make
    # zip() silently drop the tail, so a network reporting N synapses would
    # write fewer with no error at all (2026-07-31).
    if not (len(pre) == len(post) == len(wts)):
        raise CapacityError(
            f"synapse arrays have unequal lengths: pre={len(pre)}, "
            f"post={len(post)}, weights={len(wts)}. Nothing was written "
            "(zipping these would silently discard the unmatched entries).")
    # IDs must be inside the NETWORK, not merely inside the device
    # (2026-07-31). Validating against device.n_max let a 2-neuron network
    # reference source id 3 on a 4-neuron device: physically written, pointing
    # at a neuron the network never configured. A post-id in
    # [n_neurons, n_max) additionally raised a bare KeyError from the by_dst
    # lookup rather than a clear capacity error.
    for name, ids in (("pre", pre), ("post", post)):
        for i in ids:
            if not (0 <= i < n_neurons):
                raise CapacityError(
                    f"synapse {name} neuron id {i} is outside this network's "
                    f"neuron range [0, {n_neurons}). Nothing was written.")
    for arr_name in ("neuron_thresholds", "neuron_leaks", "neuron_reset_states",
                     "neuron_refractory_periods"):
        arr = getattr(snn, arr_name, None)
        if arr is not None and len(arr) != n_neurons:
            raise CapacityError(
                f"{arr_name} has {len(arr)} entries but the network has "
                f"{n_neurons} neurons. Nothing was written.")

    # MANDATORY host-side capacity guard (2026-07-29): the RTL has NO write-side
    # bounds check -- an out-of-range synapse write silently corrupts a
    # different neuron's data on hardware (proven in the sparse-capacity pilot).
    # Refuse to write anything if any lane's total in-synapse count would exceed
    # the board's per-lane capacity. `syn_cap_per_lane=None` uses the device's
    # own (for the dense packaged boards this is local_d*n_max, so this never
    # false-trips them); pass an explicit value when talking to a custom sparse
    # bitstream built with a smaller SYN_CAP_PER_LANE.
    from .capacity import validate_network_fits
    cap = int(syn_cap_per_lane) if syn_cap_per_lane is not None else device.syn_cap_per_lane
    validate_network_fits(post, num_lanes, cap)

    # group synapses by destination (post) neuron, preserving creation order
    by_dst: dict[int, list[tuple[int, float]]] = {d: [] for d in range(n_neurons)}
    for s, d, w in zip(pre, post, wts):
        by_dst[d].append((s, w))

    # per-lane dst_ptr table + ascending synapse-entry list
    entry_index: dict[tuple[int, int], tuple[int, int]] = {}
    lane_dptr = [[0] * (local_d + 1) for _ in range(num_lanes)]
    # (dst, src, raw_weight) per physical entry, in on-chip order. Carrying
    # dst here is what makes duplicate (dst,src) edges safe: each physical
    # entry is written from its OWN record instead of being looked up by
    # (dst,src) -- that lookup collapsed parallel edges and wrote the last
    # edge's weight to every one of them (2026-07-31 fix).
    lane_entries: list[list[tuple[int, int, int]]] = [[] for _ in range(num_lanes)]
    duplicates: set[tuple[int, int]] = set()
    for lane in range(num_lanes):
        cursor = 0
        for local_dst in range(local_d):
            d = local_dst * num_lanes + lane
            lane_dptr[lane][local_dst] = cursor
            if d < n_neurons:
                for (s, w) in by_dst[d]:
                    if (d, s) in entry_index:
                        duplicates.add((d, s))
                    # entry_index maps a PAIR to one entry; with duplicates it
                    # necessarily names only the last. Readback helpers keyed
                    # by (dst,src) therefore cannot address parallel edges --
                    # see the warning raised below.
                    entry_index[(d, s)] = (lane, cursor)
                    lane_entries[lane].append(
                        (d, s, _clamp_weight(quantize_raw(w, frac_bits), weight_w)))
                    cursor += 1
        lane_dptr[lane][local_d] = cursor

    if duplicates:
        import warnings
        sample = sorted(duplicates)[:5]
        warnings.warn(
            f"{len(duplicates)} duplicate (post, pre) synapse pair(s) present, "
            f"e.g. {sample}. All physical entries are written with their own "
            "correct weights, but entry_index -- and therefore read_weights() "
            "and any (post,pre) readback -- can only address the LAST entry "
            "of each duplicated pair.",
            RuntimeWarning, stacklevel=2)

    if use_bulk:
        device.begin_bulk()
    try:
        # dst_ptr (incl. per-lane sentinel slot local_d)
        for lane in range(num_lanes):
            for local_idx, off in enumerate(lane_dptr[lane]):
                device.write_dptr_raw(lane, local_idx, off)
        # Synapses, strict ascending entry index within each lane. Written
        # straight from lane_entries so every physical entry carries its own
        # weight; the previous (d,s) lookup mis-wrote duplicate edges.
        for lane in range(num_lanes):
            for idx, (d, s, w_raw) in enumerate(lane_entries[lane]):
                device.write_synapse(d, idx, s, w_raw)
        # per-neuron params
        thr = [float(x) for x in snn.neuron_thresholds]
        lk = [float(x) for x in snn.neuron_leaks]
        rst = [float(x) for x in snn.neuron_reset_states]
        rp = [int(x) for x in snn.neuron_refractory_periods]
        for d in range(n_neurons):
            device.configure_neuron(
                d,
                # width= saturates infinities into the field (SuperNeuroMAT's
                # default leak is inf); the mask then applies as before.
                threshold=quantize_raw(thr[d], frac_bits, width=data_w) & ((1 << data_w) - 1),
                leak=quantize_raw(lk[d], frac_bits, width=data_w) & ((1 << data_w) - 1),
                reset_state=quantize_raw(rst[d], frac_bits, width=data_w) & ((1 << data_w) - 1),
                refrac_period=rp[d] & 0xFF,
                input_enable=True,
            )
        # STDP tables (same Apos/Aneg on every lane, one global rule).
        # CRITICAL (2026-07-27 fix): write EVERY one of the hardware's
        # stdp_window slots, explicitly zeroing the ones beyond the provided
        # apos/aneg -- the tap table is CONFIG that soft_reset() does NOT clear,
        # so a shorter rule loaded after a longer one would otherwise leave
        # stale nonzero taps in the higher slots and silently over-apply STDP
        # (found on hardware: a 1-tap rule loaded after a 3-tap rule kept
        # potentiating from the stale slots 1-2, ~3.5x over-potentiation).
        # Also: pass the raw signed clamped tap straight through -- the old
        # `& 0xFF` truncated to 8 bits, which at weight_w=16 turned a negative
        # tap (e.g. -5) into a large POSITIVE value (251); write_stdp_table_
        # all_lanes already masks to the device's weight_w two's-complement.
        apos = list(getattr(snn, "apos", []) or [])
        aneg = list(getattr(snn, "aneg", []) or [])
        for j in range(stdp_window):
            ap = _clamp_weight(quantize_raw(apos[j] if j < len(apos) else 0.0, frac_bits), weight_w)
            an = _clamp_weight(quantize_raw(aneg[j] if j < len(aneg) else 0.0, frac_bits), weight_w)
            device.write_stdp_table_all_lanes(j, ap, an)
        device.set_stdp_enable(bool(getattr(snn, "stdp", False)))
    except BaseException:
        # Discard the batch instead of flushing it (2026-07-31). Flushing from
        # a `finally` transmitted a PARTIALLY BUILT configuration whenever an
        # exception occurred mid-construction, leaving the board looking
        # configured while holding an incomplete network -- worse than writing
        # nothing at all.
        if use_bulk:
            device.cancel_bulk()
        raise
    else:
        if use_bulk:
            device.end_bulk()

    return {"entry_index": entry_index, "n_neurons": n_neurons, "n_synapses": len(pre)}


def read_weights(device: InferEngineStdpDevice, entry_index: dict[tuple[int, int], tuple[int, int]],
                 *, frac_bits: int) -> dict[tuple[int, int], float]:
    """Read back every synapse's CURRENT (e.g. post-training) weight directly
    off the chip, using the ``entry_index`` map ``load_network()`` returned
    (2026-07-25, needs OP_READ_CONFIG -- see ``InferEngineStdpDevice.
    read_synapse``). Returns real-valued weights (raw / 2**frac_bits, the same
    grid ``quantize_raw`` uses), keyed the same way as ``entry_index``:
    ``{(post, pre): weight}``. This is the direct-measurement counterpart to
    inferring hardware accuracy only via "bit-exact spikes => bit-exact
    weights" determinism -- use it to compare hardware-learned weights
    against a software reference's weight matrix explicitly."""
    scale = float(1 << frac_bits)
    out: dict[tuple[int, int], float] = {}
    for (d, s), (lane, idx) in entry_index.items():
        got_src, raw_w = device.read_synapse(d, idx)
        if got_src != s:
            raise RuntimeError(
                f"read_weights: synapse at (dst={d}, lane={lane}, idx={idx}) "
                f"has src={got_src} on-chip, expected src={s} -- entry_index "
                "is stale (network changed since load_network()?)")
        out[(d, s)] = raw_w / scale
    return out


def run_schedule(device: InferEngineStdpDevice, schedule: dict[int, dict[int, float]],
                 total_ticks: int, *, frac_bits: int, n_neurons: int,
                 progress=None, streaming: bool = True) -> list[list[bool]]:
    """Replay a per-tick input schedule (``{tick: {neuron_id: value_real}}``)
    on the device, one tick at a time, returning the per-tick spike vector
    (list of ``n_neurons`` booleans) read back after each tick.

    CRUCIAL: input_value[] is a PERSISTENT per-neuron register and input_valid
    is a single global per-tick gate, so any neuron injected on an earlier
    tick and NOT part of this tick's frame is explicitly re-zeroed first --
    otherwise its stale value would be re-applied whenever any other neuron's
    write asserts input_valid (the exact hardware inter-run divergence).
    Call ``device.soft_reset()`` + reload config before this to guarantee a
    clean starting state across repeated runs.

    ``streaming=True`` (default, 2026-07-27): streams each tick's input-value
    writes over the UART command FIFO (``begin_bulk``/``end_bulk``, the SAME
    mechanism ``load_network()`` already uses for the initial bulk config
    load) instead of one lock-step round-trip per write, and uses
    ``run_tick_streaming()``/``read_spikes_streaming()`` for the tick itself and the
    8-word spike readback instead of ``run_tick()``/``read_spikes()`` --
    identical wire commands and identical semantics, just batched. This is
    the reason a training run previously took ~1.5s/tick (~10+ separate
    lock-step UART round-trips per tick, each paying Windows' serial
    round-trip overhead) -- NONE of run_schedule's own per-tick calls used
    the streaming primitives that already existed for exactly this purpose.
    Falls back to the original one-command-at-a-time path automatically if
    ``streaming=False`` or the transport can't stream
    (``device._can_stream()`` is False, e.g. a non-streaming transport).
    Still only ever issues ONE RUN_START per burst (never batches multiple
    ticks' RUN_START together -- see ``run_tick_streaming``'s own docstring for
    why that specifically is unsafe: a queued-too-early second RUN_START gets
    silently dropped, not delayed, since the bridge's per-frame cycle time is
    shorter than a tick's ~0.657ms hardware compute time).

    NOT YET HARDWARE-VERIFIED in this combined form as of 2026-07-27 (the
    individual primitives -- begin_bulk/end_bulk, run_tick_streaming,
    read_spikes_streaming -- were each hardware-tested standalone previously, but
    this is the first time they're wired together into the tick loop). Pass
    ``streaming=False`` to reproduce the original, extensively hardware-
    verified lock-step behavior for an A/B comparison before trusting
    ``streaming=True``'s results on a new board/bitstream.
    """
    # Preflight validation before ANY wire write (2026-07-31). Schedule
    # neuron ids were previously passed straight to write_input_value(): an id
    # of 7 on a 4-neuron device was accepted and physically written, landing
    # in another neuron's storage because the RTL has no write-side bounds
    # check. Negative tick counts silently produced an empty run that looked
    # like success.
    n_neurons = int(n_neurons)
    total_ticks = int(total_ticks)
    if n_neurons <= 0:
        raise CapacityError(f"n_neurons must be positive, got {n_neurons}.")
    if n_neurons > device.n_max:
        raise CapacityError(
            f"n_neurons={n_neurons} exceeds the loaded bitstream's "
            f"n_max={device.n_max}. Nothing was written.")
    if total_ticks < 0:
        raise ValueError(f"total_ticks must be >= 0, got {total_ticks}.")
    for tick, frame in (schedule or {}).items():
        # Reject BOTH ends of the range (2026-07-31). A tick >= total_ticks was
        # accepted and then never executed, because the run loop iterates
        # range(total_ticks) -- the caller's input was silently dropped and the
        # run looked successful. Non-integral keys are rejected for the same
        # reason: `schedule.get(t)` only ever matches exact ints, so 1.0 or "1"
        # would never fire.
        if isinstance(tick, bool) or not isinstance(tick, int):
            raise TypeError(
                f"schedule tick key {tick!r} is {type(tick).__name__}, not int; "
                "ticks are integer indices and a non-int key would never be "
                "matched by the run loop.")
        if not (0 <= tick < total_ticks):
            raise ValueError(
                f"schedule contains tick {tick}, outside [0, {total_ticks}); "
                "it would never execute. Nothing was written.")
        for nid in (frame or {}):
            if not (0 <= int(nid) < n_neurons):
                raise CapacityError(
                    f"schedule tick {tick} targets neuron id {nid}, outside "
                    f"[0, {n_neurons}). Nothing was written.")

    spikes: list[list[bool]] = []
    dirty: set[int] = set()
    ticks = range(total_ticks)
    if progress is not None:
        ticks = progress(ticks)
    use_streaming = streaming and device._can_stream()
    for t in ticks:
        frame = schedule.get(t, {})
        frame_ids = {int(k) for k in frame}
        if use_streaming:
            device.begin_bulk()
        try:
            for nid in (dirty - frame_ids):
                device.write_input_value(int(nid), 0)
            for nid, val in frame.items():
                # Pass the full quantized value; write_input_value masks it to the
                # device's data_w. (2026-07-27 fix: this used a hardcoded & 0xFFFF,
                # which at data_w>16 / large frac_bits TRUNCATED big inputs -- e.g.
                # a teacher current of 100 at frac_bits=10 is raw 102400, and
                # 102400 & 0xFFFF = 36864, silently injecting 36 instead of 100 and
                # failing to fire a threshold-99 output. Found on the 16-bit build.)
                device.write_input_value(int(nid), quantize_raw(val, frac_bits))
        except BaseException:
            # Never flush a half-built input frame; leaving bulk mode active
            # would also break every later call (2026-07-31).
            if use_streaming:
                device.cancel_bulk()
            raise
        if use_streaming:
            device.end_bulk()
        dirty = frame_ids
        if use_streaming:
            device.run_tick_streaming()
            allbits = device.read_spikes_streaming()
        else:
            device.run_tick()
            allbits = device.read_spikes()
        spikes.append([bool(allbits[i]) for i in range(n_neurons)])
    return spikes
