"""Python host runtime for the parallel-lane STDP engine.

Defines two device classes:

  * ``InferEngineDevice`` -- the inference datapath (config load, per-tick
    input injection, tick execution, spike/weight readback).
  * ``InferEngineStdpDevice`` -- subclasses the above and adds the on-chip
    STDP controls (tap table, global enable, ``soft_reset``, learned-weight
    readback). This is what ``connect()`` returns and what every example uses.

Wire protocol: an 8-byte command word (opcode<<56 | sel<<48 | addr<<32 | data,
big-endian, request/response over UART->SPI). The board top reuses
``snm_spi_slave`` verbatim, so the framing is shared with the serial-core
boards; only the config *map* differs (see ``snm_infer_cmd_ctrl_stdp.v``'s
IL_* selectors), because storage here is destination-partitioned (neuron n ->
lane n%K, local index n/K) rather than source-partitioned.

``SerialTransport``/``autodetect_port``/``SNMError`` come from the vendored
``._transport`` submodule (generic 8-byte-frame transport, no board-specific
assumptions), keeping this package self-contained.
"""
from __future__ import annotations

import math
from numbers import Integral

# Vendored transport (a copy of the shipping SpikeEngine fpga_runtime.py, which
# has no package-relative imports of its own). Kept as a private submodule so
# this package is fully standalone and does not depend on an installed
# superneuromat.spikeengine tree at runtime.
from ._transport import (
    OP_CLEAR_ERROR,
    OP_INPUT_VECTOR_WRITE,
    OP_READ_OUTPUT,
    OP_READ_STATUS,
    OP_RUN_START,
    OP_WRITE_CONFIG,
    SerialTransport,
)

# STDP soft-reset opcode (snm_infer_cmd_ctrl_stdp.v OP_ENGINE_RESET = 0x11).
OP_ENGINE_RESET = 0x11

# Read counterpart of OP_WRITE_CONFIG (2026-07-25, snm_infer_cmd_ctrl_stdp.v
# only). Only IL_SYN is a supported selector so far: reads back a
# hardware-learned synapse weight directly, instead of only being able to
# infer it via the "bit-exact spikes => bit-exact weights" determinism
# argument.
OP_READ_CONFIG = 0x02

# ---- inference-specific config-map selectors (must match snm_infer_cmd_ctrl.v IL_*) ----
IL_THRESHOLD     = 0x06
IL_LEAK          = 0x07
IL_RESET_STATE   = 0x08
IL_REFRAC        = 0x09
IL_INPUT_ENABLE  = 0x0A
IL_DPTR          = 0x0D
IL_SYN           = 0x0E
IL_INPUT_VALUE   = 0x0F
IL_SYN_ADDR_HI   = 0x19
IL_INPUT_VEC_VAL = 0x1A
IL_SET_LANE      = 0x1B

ST_OK = 0x00

STATUS_NAME = {
    0x00: "OK", 0x01: "BUSY", 0x02: "INVALID_OP", 0x03: "INVALID_SEL",
    0x04: "ADDR_RANGE",
}


class InferEngineError(RuntimeError):
    pass


class InferEngineDevice:
    """High-level runtime connection to a programmed snm_infer_top bitstream.

    Parameters mirror the RTL instantiation (``N_MAX``/``NUM_LANES``) so the
    host can compute the same lane/local-index mapping and synapse-capacity
    bounds the hardware uses.
    """

    def __init__(self, port: str | None, baud: int, n_max: int = 1024,
                 num_lanes: int = 16, timeout: float = 1.0,
                 weight_w: int = 8, data_w: int = 16,
                 prefer_interface: int | None = None,
                 syn_cap_per_lane: int | None = None,
                 stdp_window: int = 5):
        self.n_max = int(n_max)
        self.num_lanes = int(num_lanes)
        # 2026-07-27: weight_w / data_w must match the bitstream's WEIGHT_W /
        # DATA_W generics -- they set how synapse weight+src and neuron params
        # are packed into the 32-bit command word (the field offsets are
        # WEIGHT_W-parametric in snm_infer_cmd_ctrl_stdp.v, so an 8-bit-weight
        # bitstream and a 16-bit-weight bitstream pack differently and are NOT
        # wire-compatible). Defaults keep the original 8-bit/16-bit packing.
        self.weight_w = int(weight_w)
        self.data_w = int(data_w)
        self._weight_mask = (1 << self.weight_w) - 1
        self._data_mask = (1 << self.data_w) - 1
        self.local_d = (self.n_max + self.num_lanes - 1) // self.num_lanes
        # Per-lane synapse capacity of the LOADED bitstream. The dense default
        # (local_d * n_max) is only right for the dense packaged builds; a
        # sparse dataset build is far smaller (citeseer: 7,500 vs a dense
        # 1,376,970 -- 183x). Because the RTL has NO write-side bounds check,
        # an over-capacity write silently corrupts a different neuron's data,
        # so load_network's guard is only meaningful when this is the real
        # figure. connect(dataset=...) now supplies it from the catalogue
        # (2026-07-31); previously the guard fell back to the dense value and
        # was effectively inert for exactly the sparse builds it exists to
        # protect.
        self.syn_cap_per_lane = (int(syn_cap_per_lane)
                                 if syn_cap_per_lane is not None
                                 else self.local_d * self.n_max)
        self.stdp_window = int(stdp_window)
        self.spike_words = (self.n_max + 31) // 32
        self.t = SerialTransport(port, baud, timeout=timeout,
                                 prefer_interface=prefer_interface)
        self._cur_lane = None
        self._syn_addr_hi = None
        self._bulk = None          # when a list, write commands are recorded instead of sent

    def close(self):
        self.t.close()

    # Context-manager support (2026-07-30): without it, any exception between
    # connect() and close() leaks the serial port, and on Windows the port
    # stays locked until the interpreter exits -- so the next run fails to
    # open it. `with se.connect(...) as dev:` always closes, error or not.
    # close() remains public and unchanged for existing try/finally callers.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False        # never swallow the exception

    # ---- wire-level primitives (identical framing to fpga_runtime.FPGADevice) ----
    @staticmethod
    def _cmd_word(op, sel=0, addr=0, data=0) -> int:
        return ((op & 0xFF) << 56) | ((sel & 0xFF) << 48) | ((addr & 0xFFFF) << 32) | (data & 0xFFFFFFFF)

    def _raw(self, word: int) -> int:
        rx = self.t.xfer8(word.to_bytes(8, "big"))
        return int.from_bytes(rx, "big")

    def command(self, op, sel=0, addr=0, data=0):
        self._raw(self._cmd_word(op, sel, addr, data))
        resp = self._raw(0)
        return (resp >> 56) & 0xFF, resp & 0xFFFFFFFF

    def command_checked(self, op, sel=0, addr=0, data=0, ok=(ST_OK,)):
        self._reject_if_bulk_active("command_checked")
        status, rdata = self.command(op, sel, addr, data)
        if status not in ok:
            name = STATUS_NAME.get(status, hex(status))
            raise InferEngineError(f"op=0x{op:02x} sel=0x{sel:02x} addr={addr} -> {name}")
        return rdata

    def _reject_if_bulk_active(self, what: str):
        """Guard against calling an immediate (non-bulk) command while
        begin_bulk() is recording: such a call would fire out of order,
        ahead of the still-queued bulk writes, silently corrupting command
        ordering on the wire. Every read/run entry point routes through
        here (or through _emit, which handles writes)."""
        if self._bulk is not None:
            raise InferEngineError(
                f"{what}() called while begin_bulk() is active -- call end_bulk() "
                "first. Reading or running a tick before queued bulk writes are "
                "actually sent would silently reorder them on the wire.")

    def _emit(self, op, sel=0, addr=0, data=0, ok=(ST_OK,)):
        """Send a config write immediately, OR (while begin_bulk() is active) append
        it to the bulk buffer to be streamed later via end_bulk() -- used by every
        write_* method so callers don't need two code paths. Config writes have no
        return value the caller needs in bulk mode (bulk is a fire-and-forget batch
        of writes; use lock-step command_checked() for anything needing rdata)."""
        if self._bulk is not None:
            self._bulk.append(self._cmd_word(op, sel, addr, data))
            return None
        return self.command_checked(op, sel, addr, data, ok=ok)

    # ---- streaming (2026-07-13): pipeline many commands over the UART bridge's
    # command-byte FIFO instead of paying a full blocking round trip (and its
    # Windows FTDI latency-timer stall) per command. This reuses the SAME
    # mechanism as the shipping fpga_runtime.FPGADevice._stream_write_words --
    # the underlying hardware FIFO (uart_to_spi_master.v's u_cmd_fifo,
    # CMD_FIFO_DEPTH=8192 bytes = 1024 frames) is inherited unmodified by this
    # board top (it reuses uart_to_spi_master/snm_spi_slave verbatim), so the
    # same technique applies even though this engine isn't manifest-registered
    # (no board.cmd_fifo_frames to read the depth from -- hardcoded here instead,
    # matching the known RTL parameter).
    #
    # SAFETY NOTE (found via RTL trace, 2026-07-13): this is only safe for FAST
    # commands (config writes, reads -- snm_infer_cmd_ctrl resolves these in ~2-3
    # clock cycles, far shorter than the bridge's own ~50-100us per-frame cycle
    # time, so the core is always back at idle before the next frame arrives).
    # OP_RUN_START is NOT safe to blindly burst: it blocks for the whole tick
    # (up to ~0.657ms at full N=1024/K=16/S=1.05M capacity), which DOES exceed
    # the bridge's frame cycle time -- if a second RUN_START's bytes fully
    # arrive while the first tick is still running, snm_spi_slave answers
    # ST_BUSY and silently NEVER dispatches it to the core (the tick is lost,
    # not delayed; see snm_spi_slave.v's `cmd_state != C_IDLE` branch). Do not
    # stream multiple RUN_START commands back to back without adding
    # retry-on-BUSY logic first.
    _CMD_FIFO_FRAMES = 1024  # uart_to_spi_master.v CMD_FIFO_DEPTH=8192 bytes / 8

    def _can_stream(self) -> bool:
        return all(hasattr(self.t, m) for m in
                   ("write_raw", "read_avail", "read_blocking", "reset_input"))

    def _stream_window(self) -> int:
        """Max frames kept in flight, with headroom under the FIFO depth so it
        never actually fills (same margin formula as the shipping protocol)."""
        fifo = self._CMD_FIFO_FRAMES
        return max(1, fifo - max(16, fifo // 8))

    def _stream_commands(self, words, window=None, timeout=8.0):
        """Pipeline many command words over the FIFO; returns a (status, rdata)
        tuple per word, in order. Coalesces sends into large write_raw() bursts
        (few USB transactions instead of one per command) and drains responses
        via non-blocking read_avail() polling instead of a blocking read per
        command -- this is what lets the Windows FTDI latency-timer stall be
        amortized across many commands' responses instead of paid once each."""
        import time as _t
        if not self._can_stream():
            raise InferEngineError("this transport cannot stream (missing raw I/O methods)")
        if window is None:
            window = self._stream_window()
        if window < 1:
            raise InferEngineError("stream window < 1 -- nothing would ever be sent")
        self.t.reset_input()
        frames = [w.to_bytes(8, "big") for w in words]
        frames.append((0).to_bytes(8, "big"))   # trailing NOP flushes the last real response
        n = len(frames)
        results: list[tuple[int, int]] = []
        sent = 0
        recv = 0
        rbuf = bytearray()
        last = _t.time()
        while recv < n:
            hi = sent
            while hi < n and (hi - recv) < window:
                hi += 1
            if hi > sent:
                self.t.write_raw(b"".join(frames[sent:hi]))
                sent = hi
            got = self.t.read_avail()
            if got:
                rbuf += got
                last = _t.time()
            elif sent >= n:
                tail = self.t.read_blocking(8)
                if tail:
                    rbuf += tail
                    last = _t.time()
            while len(rbuf) >= 8:
                resp = rbuf[:8]
                del rbuf[:8]
                idx = recv
                recv += 1
                if idx >= 1:               # response idx>=1 is the reply to words[idx-1]
                    status = resp[0]
                    rdata = int.from_bytes(resp[4:8], "big")
                    results.append((status, rdata))
            if _t.time() - last > timeout:
                raise InferEngineError(f"stream stalled: sent {sent}/{n}, recv {recv}/{n}")
        return results

    def begin_bulk(self):
        """Start recording write_* calls to stream as one batch over the UART
        command FIFO instead of a blocking round trip each. Wrap a bulk config
        load in begin_bulk() ... end_bulk(). If the transport can't stream,
        this is a no-op and writes stay immediate (end_bulk has nothing to
        flush), so callers can use the same begin/end pattern unconditionally."""
        if self._bulk is not None:
            raise InferEngineError(
                "begin_bulk() called again while already recording -- the "
                f"previous batch ({len(self._bulk)} queued commands) would be "
                "silently discarded. Call end_bulk() first.")
        self._bulk = [] if self._can_stream() else None

    def cancel_bulk(self):
        """Discard a recording batch WITHOUT transmitting it (2026-07-31).

        ``end_bulk()`` in a ``finally`` is wrong on the error path: if an
        exception happened while the batch was being built, the batch is
        partially constructed, and flushing it writes an incomplete
        configuration to the board -- worse than writing nothing, because the
        device then looks configured. Use this on failure and ``end_bulk()``
        only on success. Safe to call when no batch is active.
        """
        self._bulk = None

    def end_bulk(self, timeout=None):
        """Stream all recorded writes, then resume immediate (lock-step) mode.

        ``timeout`` is the stall-detection window passed to _stream_commands
        (how long to wait for the NEXT chunk of bytes before giving up, not a
        total-operation deadline). Defaults to a size-scaled value -- 2026-07-13:
        a real full-scale (1,048,576-write) load twice hit the old flat 8.0s
        default's stall window on a run where the same operation had completed
        cleanly in ~30s total moments earlier and completed cleanly again with
        a longer timeout (0 non-OK responses both times) -- i.e. genuine USB/OS
        scheduling jitter under sustained multi-second transfers, not lost or
        corrupted data. A flat 8s window is fine for small/medium batches but
        was too tight to reliably absorb that jitter on a batch this large."""
        words = self._bulk
        self._bulk = None
        if words:
            if timeout is None:
                timeout = max(8.0, len(words) * 0.00006)   # ~63s floor at 1.05M words
            results = self._stream_commands(words, timeout=timeout)
            for idx, (status, _) in enumerate(results):
                if status != ST_OK:
                    name = STATUS_NAME.get(status, hex(status))
                    raise InferEngineError(f"bulk write cmd#{idx} -> {name}")

    # ---- bounds checks (2026-07-31) --------------------------------------
    # The RTL has NO write-side bounds check: an out-of-range id does not
    # error, it writes into another neuron's storage or into the per-lane
    # dst_ptr sentinel, silently corrupting results. Callers above validate
    # too, but these are the last gate before a frame reaches the wire, and
    # they protect direct users of the low-level API.
    @staticmethod
    def _check_integral(value: int, what: str) -> int:
        """Return an integer address, rejecting lossy coercions.

        ``int(1.9)`` and ``int("1")`` both succeed, but accepting either for a
        hardware address silently targets a different location than the caller
        supplied.  NumPy integer scalars implement ``numbers.Integral`` and are
        intentionally accepted; booleans are not addresses.
        """
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise InferEngineError(
                f"{what}={value!r} must be an integer (bool/float/string values "
                "are not valid hardware addresses).")
        return int(value)

    def _check_neuron(self, neuron_id: int, what: str = "neuron_id") -> int:
        n = self._check_integral(neuron_id, what)
        if not (0 <= n < self.n_max):
            raise InferEngineError(
                f"{what}={n} outside [0, {self.n_max}) for the loaded "
                "bitstream; refusing to write (the RTL would not catch this).")
        return n

    def _check_lane(self, lane: int) -> int:
        v = self._check_integral(lane, "lane")
        if not (0 <= v < self.num_lanes):
            raise InferEngineError(
                f"lane={v} outside [0, {self.num_lanes}); refusing to write.")
        return v

    def _check_entry_idx(self, idx: int) -> int:
        v = self._check_integral(idx, "synapse entry index")
        if not (0 <= v < self.syn_cap_per_lane):
            raise InferEngineError(
                f"synapse entry index {v} outside [0, {self.syn_cap_per_lane}) "
                "-- the loaded bitstream's per-lane capacity. Refusing to "
                "write; an over-capacity write corrupts another neuron's row.")
        return v

    def _check_dptr_offset(self, offset: int) -> int:
        v = self._check_integral(offset, "dst_ptr entry offset")
        if not (0 <= v <= self.syn_cap_per_lane):
            raise InferEngineError(
                f"dst_ptr entry offset {v} outside [0, {self.syn_cap_per_lane}] "
                "for the loaded bitstream (the upper bound is valid only for "
                "the final sentinel).")
        return v

    def _check_local_idx(self, local_idx: int) -> int:
        v = self._check_integral(local_idx, "dst_ptr local index")
        if not (0 <= v <= self.local_d):
            raise InferEngineError(
                f"dst_ptr local index {v} outside [0, {self.local_d}] "
                "(local_d itself is the valid sentinel slot).")
        return v

    def _check_window_idx(self, window_idx: int) -> int:
        v = self._check_integral(window_idx, "STDP window index")
        if not (0 <= v < self.stdp_window):
            raise InferEngineError(
                f"STDP window index {v} outside [0, {self.stdp_window}) for "
                "the loaded bitstream.")
        return v

    # ---- neuron id -> (lane, local index), matching the RTL's n%K / n/K split ----
    def lane_of(self, neuron_id: int) -> int:
        return neuron_id % self.num_lanes

    def local_of(self, neuron_id: int) -> int:
        return neuron_id // self.num_lanes

    def _set_lane(self, lane: int):
        if lane == self._cur_lane:
            return
        self._emit(OP_WRITE_CONFIG, IL_SET_LANE, 0, lane)
        self._cur_lane = lane

    # ---- network topology load ----
    def write_dptr(self, neuron_id: int, entry_offset: int):
        """dst_ptr[local_idx] = entry_offset (the running synapse-entry cursor for
        that lane; write LOCAL_D+1 entries per lane including the sentinel)."""
        neuron_id = self._check_neuron(neuron_id)
        entry_offset = self._check_dptr_offset(entry_offset)
        self._set_lane(self.lane_of(neuron_id))
        self._emit(OP_WRITE_CONFIG, IL_DPTR, self.local_of(neuron_id),
                   entry_offset & 0xFFFFFFFF)

    def write_dptr_raw(self, lane: int, local_idx: int, entry_offset: int):
        """Same as write_dptr but for the sentinel slot (local_idx == LOCAL_D),
        which has no corresponding neuron id."""
        lane = self._check_lane(lane)
        local_idx = self._check_local_idx(local_idx)
        entry_offset = self._check_dptr_offset(entry_offset)
        self._set_lane(lane)
        self._emit(OP_WRITE_CONFIG, IL_DPTR, local_idx, entry_offset & 0xFFFFFFFF)

    def write_synapse(self, dst_neuron: int, entry_idx: int, src_neuron: int, weight: int):
        """Write one synapse entry into destination neuron's lane, at the given
        per-lane entry index (entries must be loaded in strict ascending order
        starting at 0 within each lane -- see snm_gather_lane.v's packed-row
        write path). weight is a signed ``weight_w``-bit int (e.g. -128..127 at
        weight_w=8, -32768..32767 at weight_w=16). src is packed ABOVE the
        weight field (offset = weight_w), matching the WEIGHT_W-parametric
        IL_SYN packing in snm_infer_cmd_ctrl_stdp.v."""
        # 2026-07-31: this is the highest-volume write path and it was the one
        # NOT calling the bounds helpers added alongside it. An out-of-range
        # dst/src/entry_idx does not error on the wire -- it aliases into
        # another neuron's row.
        dst_neuron = self._check_neuron(dst_neuron, "dst_neuron")
        src_neuron = self._check_neuron(src_neuron, "src_neuron")
        entry_idx = self._check_entry_idx(entry_idx)
        if entry_idx > 0xFFFF:
            hi = (entry_idx >> 16) & 0xFFFF
            if hi != self._syn_addr_hi:
                self._emit(OP_WRITE_CONFIG, IL_SYN_ADDR_HI, 0, hi)
                self._syn_addr_hi = hi
        elif self._syn_addr_hi not in (None, 0):
            self._emit(OP_WRITE_CONFIG, IL_SYN_ADDR_HI, 0, 0)
            self._syn_addr_hi = 0
        self._set_lane(self.lane_of(dst_neuron))
        src_w = max(1, math.ceil(math.log2(max(self.n_max, 2))))
        data = ((int(src_neuron) & ((1 << src_w) - 1)) << self.weight_w) | (int(weight) & self._weight_mask)
        self._emit(OP_WRITE_CONFIG, IL_SYN, entry_idx & 0xFFFF, data)

    def read_synapse(self, dst_neuron: int, entry_idx: int) -> tuple[int, int]:
        """Read back one synapse entry (src_neuron, weight) directly off the
        chip, at the given per-lane entry index -- same addressing as
        write_synapse (IL_SYN_ADDR_HI + lane select for entry_idx > 0xFFFF).
        Requires OP_READ_CONFIG (2026-07-25); raises InferEngineError on
        boards whose bitstream predates it. Not valid mid-tick (same "config
        commands only while idle" rule as every write_* method here)."""
        self._reject_if_bulk_active("read_synapse")
        dst_neuron = self._check_neuron(dst_neuron, "dst_neuron")
        entry_idx = self._check_entry_idx(entry_idx)
        if entry_idx > 0xFFFF:
            hi = (entry_idx >> 16) & 0xFFFF
            if hi != self._syn_addr_hi:
                self._emit(OP_WRITE_CONFIG, IL_SYN_ADDR_HI, 0, hi)
                self._syn_addr_hi = hi
        elif self._syn_addr_hi not in (None, 0):
            self._emit(OP_WRITE_CONFIG, IL_SYN_ADDR_HI, 0, 0)
            self._syn_addr_hi = 0
        self._set_lane(self.lane_of(dst_neuron))
        rdata = self.command_checked(OP_READ_CONFIG, IL_SYN, entry_idx & 0xFFFF, 0)
        weight = rdata & self._weight_mask
        if weight >= (1 << (self.weight_w - 1)):   # sign-extend from weight_w bits
            weight -= (1 << self.weight_w)
        src_w = max(1, math.ceil(math.log2(max(self.n_max, 2))))
        src = (rdata >> self.weight_w) & ((1 << src_w) - 1)
        return src, weight

    # ---- neuron parameters ----
    # threshold / leak / reset_state / input_value are DATA_W-wide signed
    # fields; mask to data_w (was a hardcoded 16-bit 0xFFFF) so a DATA_W=24
    # bitstream receives the full-width value.
    def write_threshold(self, neuron_id: int, value: int):
        neuron_id = self._check_neuron(neuron_id)
        self._set_lane(self.lane_of(neuron_id))
        self._emit(OP_WRITE_CONFIG, IL_THRESHOLD, self.local_of(neuron_id), value & self._data_mask)

    def write_leak(self, neuron_id: int, value: int):
        neuron_id = self._check_neuron(neuron_id)
        self._set_lane(self.lane_of(neuron_id))
        self._emit(OP_WRITE_CONFIG, IL_LEAK, self.local_of(neuron_id), value & self._data_mask)

    def write_reset_state(self, neuron_id: int, value: int):
        neuron_id = self._check_neuron(neuron_id)
        self._set_lane(self.lane_of(neuron_id))
        self._emit(OP_WRITE_CONFIG, IL_RESET_STATE, self.local_of(neuron_id), value & self._data_mask)

    def write_refrac_period(self, neuron_id: int, value: int):
        neuron_id = self._check_neuron(neuron_id)
        self._set_lane(self.lane_of(neuron_id))
        self._emit(OP_WRITE_CONFIG, IL_REFRAC, self.local_of(neuron_id), value & 0xFF)

    def write_input_enable(self, neuron_id: int, enable: bool):
        neuron_id = self._check_neuron(neuron_id)
        self._set_lane(self.lane_of(neuron_id))
        self._emit(OP_WRITE_CONFIG, IL_INPUT_ENABLE, self.local_of(neuron_id), 1 if enable else 0)

    def configure_neuron(self, neuron_id: int, threshold: int, leak: int,
                          reset_state: int, refrac_period: int, input_enable: bool = False):
        """Convenience: write all 5 per-neuron fields (5 wire frames -- there is
        no bulk-all-fields opcode in this config map, unlike the serial core's
        L_CFG_PARAM_ROW)."""
        self.write_threshold(neuron_id, threshold)
        self.write_leak(neuron_id, leak)
        self.write_reset_state(neuron_id, reset_state)
        self.write_refrac_period(neuron_id, refrac_period)
        self.write_input_enable(neuron_id, input_enable)

    # ---- external input ----
    def write_input_value(self, neuron_id: int, value: int):
        """Per-neuron persistent input CURRENT (added to membrane when this
        neuron's input_enable is set and a tick applies input)."""
        neuron_id = self._check_neuron(neuron_id)
        self._set_lane(self.lane_of(neuron_id))
        self._emit(OP_WRITE_CONFIG, IL_INPUT_VALUE, self.local_of(neuron_id), value & self._data_mask)

    def set_input_vec_value(self, value: int):
        """The current injected by every subsequent OP_INPUT_VECTOR_WRITE frame's
        set bits (matches the shipping protocol's SEL_INPUT_VEC_VALUE)."""
        self._emit(OP_WRITE_CONFIG, IL_INPUT_VEC_VAL, 0, value & self._data_mask)

    def input_vector(self, neuron_ids, value: int | None = None):
        """Inject external input current into a set of neurons via dense
        32-neuron mask frames (OP_INPUT_VECTOR_WRITE) -- the same wire opcode
        the shipping protocol uses for spike-mask input, reinterpreted here as
        current injection. ``neuron_ids`` is any iterable of global ids in
        [0, n_max)."""
        if value is not None:
            self.set_input_vec_value(value)
        ids = sorted({int(n) for n in neuron_ids})
        by_word: dict[int, int] = {}
        for n in ids:
            if not (0 <= n < self.n_max):
                raise InferEngineError(f"neuron id {n} out of range [0,{self.n_max})")
            w, b = divmod(n, 32)
            by_word[w] = by_word.get(w, 0) | (1 << b)
        for w, mask in by_word.items():
            self._emit(OP_INPUT_VECTOR_WRITE, 0, w, mask)

    # ---- run + readback ----
    def run_tick(self):
        """Advance one inference tick. The wire transaction blocks (the
        controller only responds once tick_done fires), so this call returns
        only after the hardware tick has completed."""
        return self.command_checked(OP_RUN_START)

    def read_output_word(self, word_idx: int) -> int:
        if not (0 <= word_idx < self.spike_words):
            raise InferEngineError(f"word index {word_idx} out of range [0,{self.spike_words})")
        return self.command_checked(OP_READ_OUTPUT, 0, word_idx, 0)

    def read_spikes(self) -> list[bool]:
        """Read this tick's full spike vector (n_max booleans, global neuron order)."""
        bits: list[bool] = []
        for w in range(self.spike_words):
            mask = self.read_output_word(w)
            for b in range(32):
                if len(bits) >= self.n_max:
                    break
                bits.append(bool((mask >> b) & 1))
        return bits

    def read_status(self) -> int:
        return self.command_checked(OP_READ_STATUS)

    def clear_error(self):
        return self.command_checked(OP_CLEAR_ERROR)

    # ---- streaming tick + readback, 2026-07-13 ----
    # NOT cross-tick pipelined (see the SAFETY NOTE above _stream_commands): each
    # RUN_START is still issued alone, one at a time -- only the WITHIN-tick
    # overhead is cut, by (a) using the same single-flush streaming primitive for
    # RUN_START instead of command()'s two separate blocking reads (one of which
    # reads a stale, discarded response), and (b) bursting the spike_words
    # OP_READ_OUTPUT reads together instead of one lock-step read per word. Real
    # per-tick latency is still bounded below by at least one FTDI driver flush
    # event even with this change; getting to the hardware-bound ~0.657ms/tick
    # needs either the registry latency-timer fix or retry-on-BUSY cross-tick
    # pipelining (deferred -- see NOTES.md).
    def run_tick_streaming(self):
        """Same effect as run_tick(), issued via the streaming primitive (one
        coalesced burst + poll-based drain) instead of two separate blocking
        reads. Safe as a STANDALONE call (not for back-to-back bursting)."""
        self._reject_if_bulk_active("run_tick_streaming")
        results = self._stream_commands([self._cmd_word(OP_RUN_START)])
        status, rdata = results[0]
        if status != ST_OK:
            raise InferEngineError(f"RUN_START -> {STATUS_NAME.get(status, hex(status))}")
        return rdata

    def read_spikes_streaming(self) -> list[bool]:
        """Same result as read_spikes(), but bursts all spike_words READ_OUTPUT
        commands in one streamed batch instead of one lock-step read per word."""
        self._reject_if_bulk_active("read_spikes_streaming")
        words = [self._cmd_word(OP_READ_OUTPUT, 0, w, 0) for w in range(self.spike_words)]
        results = self._stream_commands(words)
        bits: list[bool] = []
        for w, (status, rdata) in enumerate(results):
            if status != ST_OK:
                raise InferEngineError(f"READ_OUTPUT word {w} -> {STATUS_NAME.get(status, hex(status))}")
            for b in range(32):
                if len(bits) >= self.n_max:
                    break
                bits.append(bool((rdata >> b) & 1))
        return bits


# ---- STDP-specific config-map selectors (2026-07-21, npu_stdp_dev experiment;
# must match snm_infer_cmd_ctrl_stdp.v's IL_* additions, distinct codes above
# the inference-only controller's highest selector 0x1B) ----
IL_STDP_TABLE  = 0x1C  # addr=window slot idx, wdata[7:0]=Apos, wdata[15:8]=Aneg
IL_STDP_ENABLE = 0x1D  # wdata[0] = stdp_global_enable


class InferEngineStdpDevice(InferEngineDevice):
    """Host runtime for the STDP-capable engine (``snm_infer_engine_stdp`` /
    ``snm_infer_cmd_ctrl_stdp``). Subclasses ``InferEngineDevice`` -- every
    inference-mode method (config load, run_tick, read_spikes, ...) is
    identical over the wire, since ``snm_infer_multilane_stdp``'s single
    ``tick_start`` already runs inference+STDP as one hardware-side unit (see
    NOTES.md progress update 4/6); this class only ADDS the STDP-parameter-
    table and global-enable config writes the inference-only class has no
    selectors for.

    Not connectable to an inference-only bitstream: IL_STDP_TABLE/IL_STDP_ENABLE
    are selectors this class writes unconditionally, and an inference-only
    ``snm_infer_cmd_ctrl`` would correctly reject them with ST_INVALID_SEL
    (a real, intentional safety property of the protocol, not a bug to work
    around) -- callers must confirm the connected bitstream is STDP-capable
    before using this class, e.g. via a board/manifest capability flag once
    one exists (not implemented yet; TODO alongside the fold-back to the
    shipping package).
    """

    def write_stdp_table(self, neuron_id: int, window_idx: int, apos: int, aneg: int):
        """Write one Apos/Aneg slot of the per-lane STDP window table.
        ``neuron_id`` is used only to select the lane (STDP tables are
        per-lane, shared by every destination that lane owns -- matching
        ``cfg_stdp_we``'s wiring in snm_gather_lane_stdp.v, not per-neuron
        storage)."""
        neuron_id = self._check_neuron(neuron_id)
        window_idx = self._check_window_idx(window_idx)
        self._set_lane(self.lane_of(neuron_id))
        data = ((int(aneg) & self._weight_mask) << self.weight_w) | (int(apos) & self._weight_mask)
        self._emit(OP_WRITE_CONFIG, IL_STDP_TABLE, window_idx & 0xFFFF, data)

    def write_stdp_table_all_lanes(self, window_idx: int, apos: int, aneg: int):
        """Convenience: write the same Apos/Aneg slot to every lane's table
        (the common case -- one global STDP rule, matching
        superneuromat.SNN.stdp_setup()'s single Apos/Aneg list applying to
        every STDP-enabled synapse network-wide)."""
        window_idx = self._check_window_idx(window_idx)
        for lane in range(self.num_lanes):
            self._set_lane(lane)
            data = ((int(aneg) & self._weight_mask) << self.weight_w) | (int(apos) & self._weight_mask)
            self._emit(OP_WRITE_CONFIG, IL_STDP_TABLE, window_idx & 0xFFFF, data)

    def set_stdp_enable(self, enable: bool):
        """Global STDP on/off (``stdp_global_enable`` in the RTL) -- when
        False, the engine still runs the STDP FSM's bookkeeping every tick
        but skips all weight writes (see snm_gather_lane_stdp.v's
        T_NEXT_DST state), so toggling this does not affect tick timing."""
        self._emit(OP_WRITE_CONFIG, IL_STDP_ENABLE, 0, 1 if enable else 0)

    def soft_reset(self):
        """Clear ALL engine runtime state (per-neuron input_value, membrane
        potentials, refractory counters, the STDP warm-up window counter, the
        spike-feedback register, and every pipeline/FSM) WITHOUT reprogramming
        the bitstream -- OP_ENGINE_RESET (snm_infer_cmd_ctrl_stdp.v).

        This is the supported way to run the SAME network multiple times (or a
        fresh network) back-to-back on one programmed board: config-only
        reloads canNOT clear this runtime state, so a stale value (e.g. a
        neuron's last-written input current) would otherwise persist across
        runs and corrupt the next one. Call this first, then reload the full
        config (weights/dptr/params/STDP tables) for the run. Config memories
        themselves are preserved by the soft-reset; the host reloads them
        explicitly. Not a wire-round-trip you need in bulk mode -- issue it in
        lock-step (outside begin_bulk/end_bulk)."""
        self.command_checked(OP_ENGINE_RESET)

    def read_stdp_cycles(self) -> int:
        """Last tick's STDP-phase cycle count, truncated to 16 bits (rdata[31:16]
        of OP_READ_STATUS sel=0). For the FULL 32-bit value use
        read_stdp_cycles_full() -- this method is kept for backward compat with
        earlier scripts in this experiment."""
        rdata = self.command_checked(OP_READ_STATUS)
        return (rdata >> 16) & 0xFFFF

    def read_tick_cycles(self) -> int:
        """Last tick's FULL 32-bit real inference-phase cycle count
        (OP_READ_STATUS sel=1, 2026-07-22 addition). Before this existed, a
        complete per-tick timing report needed the inference-phase count from
        a matching Icarus Verilog simulation (cross-validated as exact, since
        the inference walk is a deterministic FSM with no data-dependent
        branching -- but still not a direct hardware read). This reads it
        straight off the chip's own counter."""
        return self.command_checked(OP_READ_STATUS, sel=1)

    def read_stdp_cycles_full(self) -> int:
        """Last tick's FULL 32-bit STDP-phase cycle count (OP_READ_STATUS
        sel=2, 2026-07-22 addition) -- same data as read_stdp_cycles() but not
        truncated to 16 bits (matters once STDP cycles exceed 65535, e.g. the
        real dense-all-to-all measurement of 40,869 cycles)."""
        return self.command_checked(OP_READ_STATUS, sel=2)
