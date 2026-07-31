"""Package tests for spikeengine.

The import/API/board tests need no hardware. The end-to-end hardware test is
skipped unless SE_STDP_PORT is set in the environment (e.g. SE_STDP_PORT=COM17),
so CI without a board still exercises everything reachable offline.
"""
import os

import numpy as np
import pytest


def test_imports_and_api():
    from superneuromat import spikeengine as se
    assert se.__version__
    assert "basys3" in se.list_boards()
    b = se.get_board("basys3")
    assert b.n_max == 256 and b.num_lanes == 8 and b.local_d == 32
    # wide fixed-point build: 16-bit weights, 24-bit membrane, frac_bits=11
    assert b.weight_w == 16 and b.data_w == 24 and b.frac_bits_default == 11
    assert b.bitstream_path().exists(), "packaged Basys3 bitstream missing"
    # runtime + soft-reset + weight-readback opcodes present
    from superneuromat.spikeengine.runtime import (
        OP_ENGINE_RESET,
        OP_READ_CONFIG,
        InferEngineStdpDevice,
    )
    assert OP_ENGINE_RESET == 0x11
    assert OP_READ_CONFIG == 0x02
    assert hasattr(InferEngineStdpDevice, "soft_reset")
    assert hasattr(InferEngineStdpDevice, "read_synapse")
    assert hasattr(InferEngineStdpDevice, "run_tick_streaming")   # renamed from *_fast
    assert hasattr(InferEngineStdpDevice, "read_spikes_streaming")
    assert hasattr(InferEngineStdpDevice, "load_network") is False  # load_network is a module fn
    from superneuromat.spikeengine import (
        load_network,
        quantize_raw,
        read_weights,
        run_schedule,
    )
    # These three are the module-level public API. The import itself is the
    # assertion (a rename or accidental removal fails here), but assert on
    # them explicitly so a linter cannot "helpfully" delete the import and
    # silently drop the check with it.
    assert all(callable(f) for f in (load_network, run_schedule, read_weights))
    assert quantize_raw(0.2, 3) == 2
    assert quantize_raw(-0.125, 3) == -1


def test_all_boards_have_bitstreams():
    from superneuromat import spikeengine as se
    for key in se.list_boards():
        b = se.get_board(key)
        assert b.bitstream_path().exists(), f"{key} bitstream missing at {b.bitstream_path()}"


def _build_digits(num_train=20, num_epochs=1, frac_bits=11):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    from superneuromat import SNN
    digits = load_digits()
    npix = digits.data.shape[1]; ncls = len(set(digits.target))
    binx = np.round(digits.data / digits.data.max())
    Xtr, _, Ytr, _ = train_test_split(binx, digits.target, test_size=0.2, random_state=0)
    net = SNN(); net.stdp = True
    net.apos = [round(0.2 * (1 << frac_bits)) / (1 << frac_bits)]
    net.aneg = [0.0]
    ins = [net.create_neuron(threshold=0.0, leak=4000.0).idx for _ in range(npix)]
    outs = [net.create_neuron(threshold=20.0, leak=4000.0).idx for _ in range(ncls)]
    for i in ins:
        for o in outs:
            net.create_synapse(i, o, weight=0.0, stdp_enabled=True)
    schedule = {}; t = 0
    for _ in range(num_epochs):
        for k in range(num_train):
            frame = {ins[j]: 1.0 for j in range(npix) if Xtr[k][j] == 1}
            if frame:
                schedule[t] = frame
            schedule[t + 1] = {outs[Ytr[k]]: 100.0}
            t += 2
    return net, schedule, t


def _reference(schedule, total, frac_bits=11):
    net, _, _ = _build_digits()
    lo, hi = -(1 << 15), (1 << 15) - 1
    ref = []
    for tick in range(total):
        net.clear_input_spikes()
        for nid, val in schedule.get(tick, {}).items():
            net.add_spike(0, nid, val)
        net.simulate(1)
        raw = np.clip(np.round(net.weight_mat() * (1 << frac_bits)), lo, hi)
        net.set_weights_from_mat(raw / (1 << frac_bits))
        net.shorten_spike_train()
        ref.append({i for i, v in enumerate(net.spike_train[-1]) if v})
    return ref


def test_software_reference_runs():
    _net, schedule, total = _build_digits(num_train=20)
    ref = _reference(schedule, total)
    assert len(ref) == total
    assert sum(len(s) for s in ref) > 0     # something spikes


def _reference_net(schedule, total, frac_bits=11):
    """Same fixed-point-faithful loop as _reference(), but returns the net at
    its final post-training weights instead of the per-tick spike trains --
    used to directly compare hardware-read-back weights against."""
    net, _, _ = _build_digits()
    lo, hi = -(1 << 15), (1 << 15) - 1
    for tick in range(total):
        net.clear_input_spikes()
        for nid, val in schedule.get(tick, {}).items():
            net.add_spike(0, nid, val)
        net.simulate(1)
        raw = np.clip(np.round(net.weight_mat() * (1 << frac_bits)), lo, hi)
        net.set_weights_from_mat(raw / (1 << frac_bits))
    return net


@pytest.mark.skipif(not os.environ.get("SE_STDP_PORT"),
                    reason="set SE_STDP_PORT (e.g. COM17) to run the on-board bit-exact test")
def test_hardware_bit_exact():
    from superneuromat import spikeengine as se
    net, schedule, total = _build_digits(num_train=20)
    ref = _reference(schedule, total)
    b = se.get_board("basys3")
    fb = b.frac_bits_default   # 11 on the wide 16-bit build
    dev = se.connect(port=os.environ["SE_STDP_PORT"], board="basys3")
    try:
        dev.soft_reset()
        info = se.load_network(dev, net, frac_bits=fb, weight_w=b.weight_w,
                               data_w=b.data_w, stdp_window=b.stdp_window)
        hw = se.run_schedule(dev, schedule, total, frac_bits=fb, n_neurons=info["n_neurons"])
        hw_weights = se.read_weights(dev, info["entry_index"], frac_bits=fb)
    finally:
        dev.close()
    for t in range(total):
        got = {i for i, v in enumerate(hw[t]) if v}
        assert got == ref[t], f"tick {t}: hw {sorted(got)} != ref {sorted(ref[t])}"

    # Direct hardware weight readback (2026-07-25): a second, independent
    # confirmation alongside the spike-train check above -- not inferred via
    # "bit-exact spikes therefore bit-exact weights" determinism.
    ref_net = _reference_net(schedule, total)
    sw_w = ref_net.weight_mat()
    for (d, s), hw_w in hw_weights.items():
        assert hw_w == sw_w[s, d], f"dst={d} src={s}: hw={hw_w} sw={sw_w[s, d]}"
