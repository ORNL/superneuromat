"""Stub for the "classic" (non-lane, single-core SuperNeuroMAT3) driver
module (2026-07-30).

The real `snm_driver.py` (see `board_variants/npu_stdp_dev/gui/fpga_gui_app/`)
depends entirely on `superneuromat.spikeengine.fpga_runtime` for the classic
engine's wire protocol -- a different, non-lane wire protocol this package
(`spikeengine`) has no equivalent for (it implements only the NPU/NPU-STDP
lane engine). Classic boards are hidden from this GUI copy's board dropdown
(see `snm_gui.py`'s `_connection_bar`), so every name below is unreachable in
practice; this stub exists only so `snm_network.py`'s module-level
`from . import snm_driver as snm` / `from .snm_driver import SNMDriver`
succeed without requiring the other package. Every constant is a harmless
placeholder (never actually compared against real wire data, since
`SNMDriver.connect`/`snm.connect` always raises before any code path that
would read one); every callable raises a clear error if actually invoked --
which nothing should ever do, given classic is unreachable.
"""

from __future__ import annotations


def _unavailable(*_a, **_k):
    raise RuntimeError(
        "The 'classic' (non-lane SuperNeuroMAT3) engine is not available in "
        "the spikeengine package's bundled GUI -- only the NPU-STDP lane "
        "engine is. Install the separate superneuromat.spikeengine package "
        "for the board_variants/npu_stdp_dev GUI copy, which still supports "
        "the classic engine.")


class SNMError(RuntimeError):
    """Placeholder -- never actually raised in this copy (classic engine is
    unreachable), kept so `except snm.SNMError` clauses remain valid."""


class StatusWord:
    pass


class SNMDriver:
    """Placeholder for the classic engine's device class. `connect()` (the
    only way `snm_network.py` obtains one) always raises before this would
    ever be instantiated for real."""

    def __init__(self, *_a, **_k):
        _unavailable()


# ---- wire opcodes / selectors / status codes: placeholders, never read for
# real (see module docstring) ----
(OP_WRITE_CONFIG, OP_READ_CONFIG, OP_INPUT_EVENT_WRITE, OP_INPUT_FRAME_COMMIT,
 OP_RUN_START, OP_RUN_STEP, OP_READ_STATUS, OP_READ_OUTPUT, OP_OUTPUT_POP,
 OP_CLEAR_ERROR, SEL_N_ACTIVE, SEL_S_ACTIVE, SEL_THRESHOLD, SEL_LEAK,
 SEL_RESET_STATE, SEL_REFRAC_PERIOD, SEL_INPUT_ENABLE, SEL_VMEM_STATE,
 SEL_REFRAC_COUNT, SEL_SRC_PTR, SEL_SYN_WEIGHT, SEL_SYN_DST, SEL_SYN_ENABLE,
 SEL_STDP_ENABLE, SEL_STDP_GLOBAL, SEL_STDP_WINDOW, SEL_STDP_APOS,
 SEL_STDP_ANEG, SEL_CLEAR_SPIKES, SEL_CLEAR_ENABLES, SEL_PERF,
 SEL_SYN_ADDR_HI, ST_OK, ST_BUSY, ST_INVALID_OP, ST_INVALID_SEL,
 ST_ADDR_RANGE, ST_QUEUE_FULL, ST_QUEUE_EMPTY, ST_OUT_NOT_READY,
 ) = range(40)

STATUS_NAME: dict = {}
NOP_WORD = 0
SNM_BOARD_USB_IDS: list = []
_SYN_ADDR_SELECTORS: set = set()

autodetect_port = _unavailable
list_board_ports = _unavailable
_to_s32 = _unavailable
_to_u32 = _unavailable
_RuntimeSerialTransport = None

connect = _unavailable
