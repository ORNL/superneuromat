"""Serial runtime for a programmed SuperNeuroMAT FPGA board."""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass

OP_WRITE_CONFIG = 0x01
OP_READ_CONFIG = 0x02
OP_INPUT_EVENT_WRITE = 0x03
OP_INPUT_FRAME_COMMIT = 0x04
OP_RUN_START = 0x05
OP_WRITE_SYNAPSE = 0x06       # protocol v3: packed {stdp[25],enable[24],dst[23:8],weight[7:0]}
OP_RUN_STEP = 0x07
OP_READ_STATUS = 0x09
OP_READ_OUTPUT = 0x0A
OP_OUTPUT_POP = 0x0B
OP_INPUT_VECTOR_WRITE = 0x0C  # protocol v3: addr = 32-neuron word index, data = spike mask
OP_CLEAR_ERROR = 0x10

SEL_N_ACTIVE = 0x00
SEL_S_ACTIVE = 0x01
SEL_THRESHOLD = 0x06
SEL_LEAK = 0x07
SEL_RESET_STATE = 0x08
SEL_REFRAC_PERIOD = 0x09
SEL_INPUT_ENABLE = 0x0A
SEL_VMEM_STATE = 0x0B
SEL_REFRAC_COUNT = 0x0C
SEL_SRC_PTR = 0x0D
SEL_SYN_WEIGHT = 0x0E
SEL_SYN_DST = 0x0F
SEL_SYN_ENABLE = 0x10
SEL_STDP_ENABLE = 0x11
SEL_STDP_GLOBAL = 0x12
SEL_STDP_WINDOW = 0x13
SEL_STDP_APOS = 0x14
SEL_STDP_ANEG = 0x15
SEL_CLEAR_SPIKES = 0x16
SEL_CLEAR_ENABLES = 0x17
SEL_PERF = 0x18
SEL_SYN_ADDR_HI = 0x19
SEL_INPUT_VEC_VALUE = 0x1A    # protocol v3: value applied by vector-injected events (resets to 1)

ST_OK = 0x00
ST_BUSY = 0x01
ST_INVALID_OP = 0x02
ST_INVALID_SEL = 0x03
ST_ADDR_RANGE = 0x04
ST_QUEUE_FULL = 0x06
ST_QUEUE_EMPTY = 0x07
ST_OUT_NOT_READY = 0x08

STATUS_NAME = {
    ST_OK: "OK",
    ST_BUSY: "BUSY",
    ST_INVALID_OP: "INVALID_OP",
    ST_INVALID_SEL: "INVALID_SEL",
    ST_ADDR_RANGE: "ADDR_RANGE",
    ST_QUEUE_FULL: "QUEUE_FULL",
    ST_QUEUE_EMPTY: "QUEUE_EMPTY",
    ST_OUT_NOT_READY: "OUT_NOT_READY",
}

SYN_ADDR_SELECTORS = {SEL_SYN_WEIGHT, SEL_SYN_DST, SEL_SYN_ENABLE, SEL_STDP_ENABLE}
NOP_WORD = 0


class SNMError(RuntimeError):
    """Raised when the FPGA returns a non-OK status."""


def _to_u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def _to_s32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


@dataclass
class StatusWord:
    """Decoded 32-bit status word from READ_STATUS (see snm_cmd_ctrl.v)."""
    raw: int

    @property
    def core_busy(self):
        return bool(self.raw & (1 << 0))

    @property
    def input_event_q_full(self):
        return bool(self.raw & (1 << 4))

    @property
    def input_frame_full(self):
        return bool(self.raw & (1 << 5))

    @property
    def input_frame_empty(self):
        return bool(self.raw & (1 << 6))

    @property
    def output_full(self):
        return bool(self.raw & (1 << 7))

    @property
    def output_empty(self):
        return bool(self.raw & (1 << 8))

    @property
    def error(self):
        return bool(self.raw & (1 << 11))

    @property
    def output_count(self):
        return (self.raw >> 16) & 0xFF

    def __str__(self):
        flags = [n for n, v in [
            ("busy", self.core_busy), ("err", self.error),
            ("out_empty", self.output_empty), ("out_full", self.output_full),
            ("in_empty", self.input_frame_empty),
        ] if v]
        return f"Status(raw=0x{self.raw:08x}, out_count={self.output_count}, {','.join(flags) or 'idle'})"


# USB VID:PID pairs seen on supported development boards. Basys3 presents an FT2232
# UART/JTAG bridge (0403:6010). Some boards use FT4232H-class FTDI bridges
# (0403:6011). SP701 and some ZCU104 bring-up variants expose a Silicon Labs CP210x
# USB-UART bridge for the runtime serial link, while JTAG stays on a separate cable /
# interface, so auto-detect must not assume "FTDI only".
SNM_BOARD_USB_IDS = [
    (0x0403, 0x6010),  # FT2232
    (0x0403, 0x6011),  # FT4232H-class
    (0x10C4, 0xEA60),  # Silicon Labs CP2102/CP2103/CP2104 single UART bridge (SP701)
    (0x10C4, 0xEA70),  # Silicon Labs CP2105 dual UART bridge
    (0x10C4, 0xEA71),  # Silicon Labs CP2108 QUAD UART bridge (ZCU104 onboard USB-UART)
]  # (VID, PID)


def _all_ports():
    """[(device, description)] of every serial port currently visible."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [(p.device, p.description or "") for p in list_ports.comports()]


def _board_port_infos():
    """Raw pyserial ListPortInfo objects whose VID:PID matches an SNM board."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [p for p in list_ports.comports() if (p.vid, p.pid) in SNM_BOARD_USB_IDS]


def list_board_ports():
    """[(device, description)] of serial ports that look like an SNM FPGA board."""
    return [(p.device, p.description or "") for p in _board_port_infos()]


def board_ports_for_spec(spec) -> list:
    """Candidate serial device names for a specific board, best match first.

    Ports whose USB VID:PID matches the board's declared runtime UART bridge
    (``FPGABoard.uart_usb_ids``) are tried first -- and among those, the node on the
    board's declared UART interface (``uart_interface_index``, e.g. UART2 = interface 2
    of the ZCU104's quad CP2108) leads. Any remaining generic SNM-looking ports follow,
    so a board with incomplete metadata still connects the way it did before.
    """
    infos = _board_port_infos()
    pairs = []
    try:
        pairs = spec.uart_usb_id_pairs()
    except AttributeError:
        pass
    want_iface = getattr(spec, "uart_interface_index", None)
    preferred, matching, rest = [], [], []
    for p in infos:
        if pairs and (p.vid, p.pid) in pairs:
            iface = _usb_interface_index(p)
            if want_iface is not None and iface == want_iface:
                preferred.append(p.device)
            else:
                matching.append(p.device)
        else:
            rest.append(p.device)
    return preferred + matching + rest


def _usb_interface_index(p):
    """Best-effort USB bInterfaceNumber for a serial port node.

    Some supported boards expose a dual-/multi-interface FTDI bridge where interface 0
    is JTAG and interface 1 is the runtime UART. Others (for example SP701's CP2103
    UART bridge) expose only a plain serial node. Returns the interface index when the
    OS reports one, or ``None`` otherwise.
    """
    loc = getattr(p, "location", None) or ""
    m = re.search(r":\d+\.(\d+)\b", loc)            # Linux location, e.g. '1-1.2:1.1'
    if m:
        return int(m.group(1))
    dev = getattr(p, "device", "") or ""
    m = re.search(r"usbserial-.*?([A-Z])$", dev)    # macOS cu.usbserial-<serial>A/B
    if m:
        return ord(m.group(1)) - ord("A")
    # Windows: COM names carry no interface number and `location` is None, but a
    # multi-channel FTDI (FT2232/FT4232) appends the channel letter to the USB serial
    # (e.g. SER=19713D -> channel D = interface 3). Mirror the macOS suffix logic.
    sn = getattr(p, "serial_number", None) or ""
    m = re.search(r"([A-D])$", sn)
    if m and re.search(r"\d", sn):                   # base serial is numeric -> trailing A-D is the channel
        return ord(m.group(1)) - ord("A")
    m = re.search(r"ttyUSB(\d+)$", dev)             # Linux fallback: interfaces enumerate in order
    if m:
        return int(m.group(1))
    return None


def _pick_uart_interface(infos, prefer_interface: int | None = None):
    """Given several same-board nodes, return the UART one, or None if it can't be
    identified.

    Most FTDI-based boards wire the host UART to interface 1 (channel B), which
    is the default. That is NOT universal -- a multi-channel part can expose the
    runtime UART elsewhere -- so ``prefer_interface`` lets a board override it
    (``StdpBoard.uart_interface_index``, plumbed through by ``connect()``).
    Boards presenting a single serial node bypass this logic entirely, and an
    unresolved choice falls through to the round-trip probe in
    ``autodetect_port`` rather than guessing (2026-07-31).
    """
    want = 1 if prefer_interface is None else int(prefer_interface)
    scored = [(_usb_interface_index(p), p) for p in infos]
    known = [(i, p) for (i, p) in scored if i is not None]
    if known and len(known) == len(infos):
        matches = [port.device for idx, port in known if idx == want]
        # The interface number identifies a channel within one USB bridge, not
        # a physical board. With two FT4232-based boards attached there can be
        # several interface-3 nodes; returning the first one can select an
        # inactive channel on the wrong board. Let autodetect_port probe when
        # the preferred interface is duplicated.
        if len(matches) == 1:
            return matches[0]
    return None


def autodetect_port(required: bool = False, prefer_interface: int | None = None,
                    baud: int | None = None):
    """Find the FPGA board's UART port automatically (by USB VID:PID).

    Returns the device string (e.g. ``'COM5'`` / ``'/dev/ttyUSB1'`` /
    ``'/dev/cu.usbserial-...B'``), or ``None`` if not found and not ``required``.
    Raises :class:`SNMError` with actionable guidance when ``required`` and the board
    is missing or ambiguous.

    Resolution order:

    1. ``SNM_FPGA_PORT`` in the environment, if set -- an explicit override that
       skips detection entirely. (Until 2026-07-31 this variable was named in
       three different error messages but never actually read; it works now.)
    2. The sole matching port, when only one board-like device is present.
    3. The board's declared UART interface index (``prefer_interface``, from
       ``StdpBoard.uart_interface_index``), defaulting to 1 -- correct for the
       usual FTDI channel-B wiring but overridable per board.
    4. A harmless READ_STATUS round-trip probe of each remaining candidate at
       ``baud`` when supplied (the transport always supplies its real baud).
    """
    override = os.environ.get("SNM_FPGA_PORT", "").strip()
    if override:
        return override

    infos = _board_port_infos()
    if not infos:
        if required:
            raise SNMError(_no_board_help())
        return None
    if len(infos) == 1:
        return infos[0].device
    uart = _pick_uart_interface(infos, prefer_interface)
    if uart is not None:
        return uart
    # If there are several candidate nodes, try a harmless READ_STATUS handshake on
    # each one. This disambiguates machines where Windows numbers the UART ports
    # differently (COM5 vs COM7 vs COM9) even though the board looks identical.
    if required:
        want = 1 if prefer_interface is None else int(prefer_interface)
        preferred = [p for p in infos if _usb_interface_index(p) == want]
        # Probe duplicated preferred-interface nodes before unrelated channels.
        # This keeps board metadata meaningful while still allowing the
        # round-trip handshake to identify the live UART among multiple boards.
        # When the requested interface exists, do not fall through to other
        # interfaces: on a host with multiple boards that can silently select
        # a healthy UART belonging to a different board. Only use all ports as
        # a compatibility fallback when interface metadata found no candidate.
        probe_order = preferred if preferred else infos
        for port in (p.device for p in probe_order):
            probe_bauds = (int(baud),) if baud is not None else (115200, 1000000, 4000000)
            for candidate_baud in probe_bauds:
                if _probe_port(port, candidate_baud):
                    return port
    if required:
        opts = ", ".join(f"{p.device} ({p.description or ''})" for p in infos)
        raise SNMError(
            f"Multiple FPGA-like serial ports found ({opts}) and the UART channel could "
            "not be identified. Pass the port explicitly (port='/dev/ttyUSB1' or "
            "'COM5'), or set SNM_FPGA_PORT."
        )
    return infos[0].device


def _port_listing():
    ports = _all_ports()
    return "\n".join(f"    {d}  ({desc})" for d, desc in ports) or "    (none)"


def _no_board_help():
    return (
        "Could not auto-detect the FPGA board's UART port (looking for supported USB UART "
        "VID:PID pairs such as 0403:6010, 0403:6011, 10C4:EA60, 10C4:EA70, or 10C4:EA71).\n"
        f"Serial ports currently visible:\n{_port_listing()}\n"
        "Check that the board is plugged in and powered, the USB cable is a data "
        "cable, and the correct USB-UART driver is installed. Then pass port='COM5' explicitly "
        "or set SNM_FPGA_PORT."
    )


def _port_open_help(port, err):
    return (
        f"Could not open serial port {port!r}: {err}\n"
        f"Is the COM port correct and the board connected? Ports currently visible:\n"
        f"{_port_listing()}\n"
        "Update the port (port='COMx' / set SNM_FPGA_PORT), or use port='auto' to "
        "detect the board automatically."
    )


def _probe_port(port: str, baud: int, timeout: float = 0.35) -> bool:
    """Return True if ``port`` answers a harmless read-status probe at ``baud``."""
    try:
        import serial
    except ImportError:
        return False
    ser = None
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        time.sleep(0.03)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        cmd = ((OP_READ_STATUS & 0xFF) << 56).to_bytes(8, "big")
        nop = NOP_WORD.to_bytes(8, "big")
        ser.write(cmd)
        ser.flush()
        _ = ser.read(8)
        ser.reset_input_buffer()
        ser.write(nop)
        ser.flush()
        resp = ser.read(8)
        return len(resp) == 8 and resp[0] in (ST_OK, ST_BUSY)
    except (OSError, ValueError, serial.SerialException):
        return False
    finally:
        try:
            if ser is not None:
                ser.close()
        except (OSError, serial.SerialException):
            pass


def _maybe_lower_latency_timer(port, target_ms: int = 1):
    """Best-effort: drop the FTDI ``latency_timer`` toward ``target_ms`` for this device.

    Lowering it from the 16 ms default cuts the per-command round-trip latency
    (measured 2026-07-10 on Windows: the 16 ms timer + a 64-frame window throttled
    bulk config to 2,245 of the wire's 50,000 frames/s). Per-device, best-effort,
    and silent on any failure. Linux: sysfs (may need root). Windows: the FTDI VCP
    driver reads the persistent ``LatencyTimer`` registry value at port OPEN, so a
    successful write (needs admin) applies from the NEXT open; without admin this
    silently does nothing -- the protocol-v3 deep command FIFO (window >= the
    bandwidth-delay product) is the primary fix and needs no registry change.
    macOS has no equivalent knob.
    """
    if sys.platform == "linux":
        try:
            name = os.path.basename(str(port))
            path = f"/sys/bus/usb-serial/devices/{name}/latency_timer"
            if os.path.exists(path):
                with open(path, "r") as fh:
                    if fh.read().strip() == str(target_ms):
                        return
                with open(path, "w") as fh:
                    fh.write(str(target_ms))
        except OSError:
            pass   # not settable (e.g. needs root) -- streaming already amortizes it
        return
    if sys.platform == "win32":
        try:
            import winreg
            base = r"SYSTEM\CurrentControlSet\Enum\FTDIBUS"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
                i = 0
                while True:
                    try:
                        dev = winreg.EnumKey(root, i)
                    except OSError:
                        break
                    i += 1
                    try:
                        params = base + "\\" + dev + r"\0000\Device Parameters"
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, params, 0,
                                            winreg.KEY_READ | winreg.KEY_SET_VALUE) as k:
                            try:
                                port_name, _ = winreg.QueryValueEx(k, "PortName")
                            except OSError:
                                continue
                            if str(port_name).upper() != str(port).upper():
                                continue
                            cur, _ = winreg.QueryValueEx(k, "LatencyTimer")
                            if int(cur) > target_ms:
                                winreg.SetValueEx(k, "LatencyTimer", 0,
                                                  winreg.REG_DWORD, int(target_ms))
                            return
                    except OSError:
                        continue   # no admin / key shape differs: leave it alone
        except OSError:
            pass
        return


class SerialTransport:
    """8-byte frame transport over pyserial."""

    def __init__(self, port: str | None, baud: int, timeout: float = 1.0,
                 prefer_interface: int | None = None):
        try:
            import serial
        except ImportError as err:
            raise ImportError(
                "Serial (pyserial) support is required to talk to the board; "
                "reinstall with `pip install superneuromat`.") from err
        if port is None or str(port).strip().lower() == "auto":
            # prefer_interface comes from the board catalogue via connect(), so
            # autodetection is board-aware instead of always assuming FTDI
            # interface 1.
            port = autodetect_port(required=True, prefer_interface=prefer_interface,
                                   baud=baud)
        try:
            self._ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        except serial.SerialException as err:
            raise SNMError(_port_open_help(port, err)) from err
        self.port = str(port)
        self.baud = int(baud)
        time.sleep(0.05)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        _maybe_lower_latency_timer(port)

    def xfer8(self, tx: bytes) -> bytes:
        if len(tx) != 8:
            raise ValueError("FPGA transport frames must be exactly 8 bytes")
        self._ser.reset_input_buffer()
        self._ser.write(tx)
        self._ser.flush()
        rx = self._ser.read(8)
        if len(rx) != 8:
            raise SNMError(f"UART read timeout: got {len(rx)}/8 bytes")
        return rx

    # --- raw streaming helpers (for the batched config load) ---
    def write_raw(self, data: bytes):
        self._ser.write(data)

    def read_avail(self) -> bytes:
        n = self._ser.in_waiting
        return self._ser.read(n) if n else b""

    def read_blocking(self, n: int) -> bytes:
        return self._ser.read(n)

    def reset_input(self):
        self._ser.reset_input_buffer()

    def close(self):
        self._ser.close()


# NOTE (2026-07-31): FPGADevice and LaneEngineDevice were removed from this
# module. Both drove the pre-STDP engines -- the classic single-core design
# and the non-STDP lane engine -- which this package does not implement.
# Neither was referenced anywhere here, and FPGADevice additionally did
# `from .fpga import get_fpga_board` at call time: a module that does not
# exist in this package, so any use raised ModuleNotFoundError. Being
# public-looking dead code, it invited exactly that mistake.
#
# What remains is the reusable transport layer only: SerialTransport,
# SNMError, StatusWord, the port-discovery helpers, and the shared opcode /
# selector / status constants. The STDP device classes live in runtime.py.
# The removed implementations are preserved in
# legacy/spikeengine_v0.1.0/fpga_runtime.py.
