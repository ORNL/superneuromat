"""Direct low-level address-safety tests.

The RTL intentionally has no write-side range checks, so every public host
method must reject a bad address before emitting any command frame.
"""

import numpy as np
import pytest

from superneuromat.spikeengine.runtime import InferEngineError, InferEngineStdpDevice


def _device():
    dev = object.__new__(InferEngineStdpDevice)
    dev.n_max = 4
    dev.num_lanes = 2
    dev.local_d = 2
    dev.syn_cap_per_lane = 4
    dev.stdp_window = 5
    dev.weight_w = 8
    dev.data_w = 16
    dev._weight_mask = 0xFF
    dev._data_mask = 0xFFFF
    dev._cur_lane = None
    dev._syn_addr_hi = None
    dev._bulk = None
    dev.emitted = []
    dev._emit = lambda *args: dev.emitted.append(args)
    dev.command_checked = lambda *args: (_ for _ in ()).throw(
        AssertionError("an invalid read reached the transport"))
    return dev


@pytest.mark.parametrize("call", [
    lambda d: d.write_threshold(4, 1),
    lambda d: d.write_leak(-1, 1),
    lambda d: d.write_reset_state(99, 1),
    lambda d: d.write_refrac_period(-1, 1),
    lambda d: d.write_input_enable(4, True),
    lambda d: d.write_input_value(1.9, 1),
    lambda d: d.write_dptr(1, 5),
    lambda d: d.write_dptr_raw(2, 0, 0),
    lambda d: d.write_dptr_raw(0, 3, 0),
    lambda d: d.write_dptr_raw(0, 0, -1),
    lambda d: d.write_stdp_table(4, 0, 1, -1),
    lambda d: d.write_stdp_table(0, 5, 1, -1),
    lambda d: d.write_stdp_table_all_lanes(-1, 1, -1),
    lambda d: d.read_synapse(4, 0),
    lambda d: d.read_synapse(0, 4),
])
def test_invalid_address_is_rejected_before_any_frame(call):
    dev = _device()
    with pytest.raises(InferEngineError):
        call(dev)
    assert dev.emitted == []


@pytest.mark.parametrize("bad", [True, 1.0, 1.9, "1", None])
def test_lossy_address_coercions_are_rejected(bad):
    dev = _device()
    with pytest.raises(InferEngineError):
        dev.write_threshold(bad, 1)
    assert dev.emitted == []


def test_numpy_integral_addresses_are_supported():
    dev = _device()
    dev.write_threshold(np.int64(1), 1)
    assert dev.emitted


def test_dptr_capacity_is_valid_for_the_final_sentinel():
    dev = _device()
    dev.write_dptr_raw(0, dev.local_d, dev.syn_cap_per_lane)
    assert dev.emitted

