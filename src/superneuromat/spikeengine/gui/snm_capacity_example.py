"""Dedicated ZCU104 full-capacity example (generated, not table-editable).

This exercises the whole chip at once: every neuron and every synapse the packaged
ZCU104 bitstream supports (N_MAX = 4096 neurons, SYN_DEPTH = 1,179,648 synapses), then
drives an input into *all* neurons for 10 timesteps and reads the spike output each step.

It is intentionally NOT a normal GUI preset: 1.18M synapses cannot be rendered as editable
table rows. Instead it is *generated* and streamed straight to the connected board via the
runtime's bulk command-FIFO path, and the GUI only offers it when the ZCU104 board is
selected (see snm_presets.names_for_board / the GUI's generated-example dispatch).

The synapse map is a dense CSR layout: source neuron ``i`` owns the contiguous slot range
``[i*per_src, (i+1)*per_src)`` (per_src = SYN_DEPTH // N_MAX = 288 for the ZCU104 image),
each slot driving a spread destination so activity propagates across the whole array.
"""

from __future__ import annotations

import time

# The example's identity as shown in the GUI example picker. `board` gates visibility.
KEY = "zcu104_full_capacity"
BOARD = "zcu104"
STEPS = 10
LABEL = "ZCU104 full-capacity stress (all neurons + all synapses, 10 steps)"


def spec_for(board_spec) -> dict:
    """Human-facing summary numbers derived from a board manifest spec (FPGABoard)."""
    n = int(board_spec.n_max)
    s = int(board_spec.syn_depth)
    per_src = s // n
    return {
        "key": KEY, "board": BOARD, "label": LABEL, "steps": STEPS,
        "n_max": n, "syn_depth": s, "per_src": per_src,
        "desc": (f"Full-capacity hardware stress: configure all {n} neurons and all {s:,} "
              f"synapses ({per_src} per source neuron, dense CSR), then drive an input into "
              f"EVERY neuron for {STEPS} timesteps and read the spike output each step. "
              "Generated and streamed straight to the board -- it is not shown as editable "
              "tables. Requires a live ZCU104 connection."),
    }


def run_full_capacity(dev, steps: int = STEPS, progress=None,
                      syn_chunk: int = 16384) -> dict:
    """Configure the full-capacity network on ``dev`` and run ``steps`` timesteps.

    ``dev`` is a connected :class:`fpga_runtime.FPGADevice`. ``progress`` (optional) is
    called as ``progress(phase:str, done:int, total:int)`` so a UI can show a bar.
    Returns a dict of timings and per-step spike counts. Read-heavy and long-running
    (a fully dense 1.18M-synapse load streams a few MB over UART -- expect a couple of
    minutes at 4 Mbaud); it does not touch the packaged bitstream or flash.
    """
    spec = dev.board
    n = int(spec.n_max)
    s = int(spec.syn_depth)
    if n <= 0 or s <= 0:
        raise ValueError(f"board {getattr(spec, 'key', '?')} has no N_MAX/SYN_DEPTH capacity")
    per_src = s // n

    def _tick(phase, done, total):
        if progress is not None:
            progress(phase, done, total)

    t0 = time.perf_counter()
    dev.flush()
    dev.set_n_active(n)
    dev.set_s_active(s)

    # 1) neurons: every neuron input-enabled, low threshold so the whole array is exercised.
    dev.begin_bulk()
    for i in range(n):
        dev.set_neuron(i, threshold=1, leak=0, reset_state=0,
                       refrac_period=0, input_enable=True, vmem=0)
    dev.end_bulk()
    _tick("neurons", n, n)

    # 2) CSR source pointers: src_ptr[i] = i*per_src, plus the end sentinel src_ptr[n] = s.
    dev.begin_bulk()
    for i in range(n):
        dev.set_src_ptr(i, i * per_src)
    dev.set_src_ptr(n, s)
    dev.end_bulk()
    _tick("pointers", n + 1, n + 1)

    # 3) synapses: dense, streamed in chunks to bound host memory. Ascending index order
    #    keeps the protocol-v2 page latch (SEL_SYN_ADDR_HI) monotonic -- minimal page writes.
    t_syn0 = time.perf_counter()
    idx = 0
    while idx < s:
        hi = min(idx + syn_chunk, s)
        dev.begin_bulk()
        for j in range(idx, hi):
            src = j // per_src
            if src >= n:              # trailing remainder (if s not a multiple of n) -> last source
                src = n - 1
            dst = (src + 1 + (j - src * per_src)) % n
            dev.set_synapse(j, weight=1, dst=dst, enable=True, stdp=False)
        dev.end_bulk()
        idx = hi
        _tick("synapses", idx, s)
    t_config = time.perf_counter() - t0
    t_syn = time.perf_counter() - t_syn0

    # 4) run: the input event FIFO caps events-per-frame (measured, not the manifest's
    #    nominal depth -- see event_cap). Drive a rotating slice of `event_cap` FRESH neurons
    #    each step so direct input coverage advances across the array; the dense enabled
    #    synapse fabric (avg fan-in per_src, threshold 1) then propagates so that within a
    #    couple of steps every neuron fires every step. Read the full output frame each step.
    event_cap = _probe_event_cap(dev, n)
    per_step_spikes = []
    driven = 0
    t_run0 = time.perf_counter()
    for st in range(steps):
        batch = [((driven + k) % n, 2) for k in range(min(event_cap, n))]
        driven += len(batch)
        dev.input_events(batch)
        dev.commit_frame()
        dev.run_step()
        frame = dev.read_output_frame()
        per_step_spikes.append(len(frame[1]) if frame else 0)
        _tick("run", st + 1, steps)
    t_run = time.perf_counter() - t_run0

    return {
        "n_max": n, "syn_depth": s, "per_src": per_src, "steps": steps,
        "event_cap": event_cap, "direct_driven": min(driven, n),
        "config_s": t_config, "synapse_s": t_syn, "run_s": t_run,
        "total_s": time.perf_counter() - t0,
        "spikes_per_step": per_step_spikes,
        "syn_writes": s * 3,                       # weight+dst+enable per synapse
        "config_write_rate": (s * 3) / t_syn if t_syn > 0 else 0.0,
    }


def _probe_event_cap(dev, n: int, ceiling: int = 4096) -> int:
    """How many input events this bitstream accepts per committed frame.

    Protocol-v3 images (manifest supports_input_vector) size the BRAM event FIFO to
    N_MAX by construction, so the manifest value is trusted -- no probe needed (and a
    lock-step probe of N events would cost minutes at USB latency). Pre-v3 images had
    a hardwired 64-deep queue that CONTRADICTED the manifest (measured 2026-07-10), so
    for those the cap is measured: fill a throwaway frame until QUEUE_FULL, then clear.
    """
    if getattr(dev.board, "supports_input_vector", False):
        return int(getattr(dev.board, "event_fifo_depth", None) or n)
    from superneuromat.spikeengine._transport import SNMError
    dev.flush()
    cap = 0
    try:
        for i in range(min(n, ceiling)):
            dev.input_event(i, 0)     # status-checked; raises QUEUE_FULL at the limit
            cap += 1
    except SNMError:
        pass
    dev.flush()                        # drop the probe frame's events
    return max(1, cap)
