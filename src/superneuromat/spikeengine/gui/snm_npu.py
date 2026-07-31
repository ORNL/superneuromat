"""Stub for the inference-only NPU (non-STDP) board module (2026-07-30).

The real `snm_npu.py` (see `board_variants/npu_stdp_dev/gui/fpga_gui_app/`)
depends on `superneuromat.spikeengine.fpga`/`fpga_runtime.LaneEngineDevice` --
a separate package this one (`spikeengine`) does not depend on and does not
bundle a non-STDP board catalogue equivalent to. Rather than pull in that
dependency, non-STDP NPU support is dropped from THIS copy of the GUI (NPU-
STDP is a strict superset of what non-STDP NPU offers, so nothing is lost
functionally -- see `snm_npu_stdp.py`).

`is_npu()` is kept real (it has no external dependency) so every existing
call site in `snm_gui.py` that checks `snm_npu.is_npu(name)` keeps working
correctly -- it always returns False here because `snm_gui.py`'s board
dropdown never adds any `"npu:..."` keys in this copy, so no caller can ever
reach a name this would need to say True for. Every other function is a
stub that raises clearly if a code path ever calls it (which none should,
given the above) rather than failing at import time -- keeping `from . import
snm_npu` safe so the rest of the GUI (including the classic driver path,
which is independent of this module) is unaffected.
"""

from __future__ import annotations

KEY_PREFIX = "npu:"


def is_npu(name) -> bool:
    return isinstance(name, str) and name.startswith(KEY_PREFIX)


def _unavailable(*_a, **_k):
    raise RuntimeError(
        "Non-STDP NPU support is not available in the spikeengine package's "
        "bundled GUI (no 'npu:' board is ever offered by this copy, so this "
        "should be unreachable). Use NPU-STDP (snm_npu_stdp.py) instead, or "
        "install the separate superneuromat.spikeengine package for the "
        "board_variants/npu_stdp_dev GUI copy, which still supports it.")


base_board = _unavailable
keys = _unavailable
label = _unavailable
baud_for = _unavailable
board_dict = _unavailable
hw_for_board = _unavailable
connect = _unavailable
