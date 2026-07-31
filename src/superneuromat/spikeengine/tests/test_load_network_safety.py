"""Regression tests for the load_network defects raised in the 2026-07-31
sign-off audit. Each test reproduces the reported failure, so a fix that
regresses is caught immediately.

All are software-only: a recording fake device stands in for hardware, which
is what makes the corruption visible -- on real hardware these bugs are
SILENT (the RTL has no write-side bounds check, so a bad write lands in some
other neuron's storage and nothing reports an error).
"""
import pytest

from superneuromat.spikeengine import load_network
from superneuromat.spikeengine.capacity import CapacityError


class RecordingDevice:
    """Records every wire write instead of performing one."""

    def __init__(self, n_max=64, num_lanes=8, weight_w=16, data_w=24,
                 syn_cap_per_lane=4096):
        self.n_max = n_max
        self.num_lanes = num_lanes
        self.local_d = (n_max + num_lanes - 1) // num_lanes
        self.weight_w = weight_w
        self.data_w = data_w
        self.syn_cap_per_lane = syn_cap_per_lane
        self.synapses = []      # (dst, idx, src, raw_weight)
        self.dptr = []          # (lane, local_idx, offset)
        self.neurons = []       # (idx, kwargs)

    def clear_error(self): pass
    def begin_bulk(self): pass
    def end_bulk(self, *a, **k): pass
    def set_stdp_enable(self, *a, **k): pass
    def write_stdp_table_all_lanes(self, *a, **k): pass

    def write_synapse(self, dst, idx, src, w):
        self.synapses.append((dst, idx, src, w))

    def write_dptr_raw(self, lane, local_idx, off):
        self.dptr.append((lane, local_idx, off))

    def write_dptr(self, *a, **k):
        self.dptr.append(a)

    def configure_neuron(self, idx, **kw):
        self.neurons.append((idx, kw))


def _snn(n_neurons, edges):
    from superneuromat import SNN
    snn = SNN()
    ids = [snn.create_neuron(threshold=1.0) for _ in range(n_neurons)]
    for src, dst, w in edges:
        snn.create_synapse(ids[src], ids[dst], weight=w)
    return snn


# --- audit finding: default widths ignored the device ------------------------

def test_widths_default_to_the_device_not_to_8_16():
    """Reported: loading weight 2.0 at frac_bits=11 onto a 16-bit device
    stored raw 127 (8-bit clamp) instead of 4096."""
    dev = RecordingDevice(weight_w=16, data_w=24)
    load_network(dev, _snn(2, [(0, 1, 2.0)]), frac_bits=11)
    assert dev.synapses, "no synapse written"
    raw = dev.synapses[0][3]
    assert raw == 4096, f"weight truncated to {raw}; expected 4096"


def test_explicit_width_disagreeing_with_device_is_rejected():
    """Silently honouring a mismatched width corrupts every weight, so it must
    raise instead."""
    dev = RecordingDevice(weight_w=16)
    with pytest.raises(ValueError, match="disagrees with the connected device"):
        load_network(dev, _snn(2, [(0, 1, 2.0)]), frac_bits=11, weight_w=8)


def test_matching_explicit_width_still_allowed():
    dev = RecordingDevice(weight_w=16, data_w=24)
    load_network(dev, _snn(2, [(0, 1, 2.0)]), frac_bits=11,
                 weight_w=16, data_w=24)
    assert dev.synapses[0][3] == 4096


# --- audit finding: oversized networks overwrote the dst_ptr sentinel --------

def test_network_larger_than_n_max_is_refused_before_any_write():
    """Reported: a 3-neuron model on an n_max=2 device configured neuron 2 at
    local index 1 -- the per-lane sentinel slot, which redefines that lane's
    synapse ranges."""
    dev = RecordingDevice(n_max=2, num_lanes=2)
    with pytest.raises(CapacityError, match="n_max"):
        load_network(dev, _snn(3, []), frac_bits=0)
    assert not dev.neurons and not dev.synapses and not dev.dptr, \
        "writes were issued despite the network not fitting"


def test_neuron_ids_are_never_written_into_the_sentinel_slot():
    """No configure_neuron may target local index local_d, which is the
    sentinel entry of the dst_ptr table."""
    dev = RecordingDevice(n_max=16, num_lanes=4)
    load_network(dev, _snn(16, [(0, 1, 1.0)]), frac_bits=0)
    for idx, _kw in dev.neurons:
        assert 0 <= idx < dev.n_max
        assert idx // dev.num_lanes < dev.local_d, (
            f"neuron {idx} maps to local index {idx // dev.num_lanes}, "
            f"the sentinel slot (local_d={dev.local_d})")


# --- audit finding: duplicate (post, pre) edges were corrupted ---------------

class _RawNet:
    """Minimal stand-in exposing exactly the attributes load_network reads.

    Needed because superneuromat.SNN REFUSES duplicate synapses
    ("Synapse already exists"), so parallel edges cannot be built through its
    API. load_network nevertheless consumes the raw arrays, so anything that
    populates them directly -- an importer, a generated model, a future SNN
    option -- can still present duplicates. This reproduces that input.
    """

    def __init__(self, n, pre, post, wts):
        self.neuron_thresholds = [1.0] * n
        self.neuron_leaks = [0.0] * n
        self.neuron_reset_states = [0.0] * n
        self.neuron_refractory_periods = [0] * n
        self.pre_synaptic_neuron_ids = list(pre)
        self.post_synaptic_neuron_ids = list(post)
        self.synaptic_weights = list(wts)
        self.enable_stdp = [False] * len(pre)
        self.apos = []
        self.aneg = []


def test_duplicate_edges_keep_their_own_weights():
    """Reported: parallel edges with raw weights 8 and 16 produced two
    physical writes of 16. Each physical entry must carry its own weight."""
    dev = RecordingDevice()
    net = _RawNet(2, pre=[0, 0], post=[1, 1], wts=[1.0, 2.0])
    with pytest.warns(RuntimeWarning, match="duplicate"):
        load_network(dev, net, frac_bits=3)

    written = sorted(w for (_d, _i, _s, w) in dev.synapses)
    assert written == [8, 16], (
        f"duplicate edges corrupted: wrote {written}, expected [8, 16]")


def test_duplicate_edges_warn_about_readback_ambiguity():
    """entry_index is keyed by (post, pre) and cannot address both parallel
    edges -- the caller must be told rather than silently getting one."""
    dev = RecordingDevice()
    net = _RawNet(2, pre=[0, 0], post=[1, 1], wts=[1.0, 2.0])
    with pytest.warns(RuntimeWarning, match="read_weights"):
        load_network(dev, net, frac_bits=3)


def test_superneuromat_itself_rejects_duplicate_synapses():
    """Documents why the duplicate path is defensive rather than routine: the
    normal SNN API refuses to create parallel edges at all."""
    from superneuromat import SNN
    s = SNN()
    a = s.create_neuron(threshold=1.0)
    b = s.create_neuron(threshold=1.0)
    s.create_synapse(a, b, weight=1.0)
    with pytest.raises(RuntimeError, match="already exists"):
        s.create_synapse(a, b, weight=2.0)


def test_no_warning_when_edges_are_unique():
    import warnings
    dev = RecordingDevice()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        load_network(dev, _snn(3, [(0, 2, 1.0), (1, 2, 2.0)]), frac_bits=3)


# --- ordering invariant the RTL depends on ----------------------------------

def test_synapse_entry_indices_are_contiguous_and_ascending_per_lane():
    """The gather stage walks dst_ptr[d]..dst_ptr[d+1], so entries for a lane
    must be written at strictly ascending, gap-free indices."""
    dev = RecordingDevice(n_max=32, num_lanes=4)
    edges = [(s, d, 1.0) for d in range(8) for s in range(3)]
    load_network(dev, _snn(32, edges), frac_bits=0)

    per_lane = {}
    for (dst, idx, _src, _w) in dev.synapses:
        per_lane.setdefault(dst % dev.num_lanes, []).append(idx)
    for lane, idxs in per_lane.items():
        assert idxs == list(range(len(idxs))), (
            f"lane {lane}: entry indices {idxs} are not contiguous ascending")


# --- audit round 2 (2026-07-31) ---------------------------------------------

def test_unequal_synapse_arrays_are_rejected():
    """zip() silently drops unmatched entries, so a network reporting N
    synapses would write fewer with no error at all."""
    dev = RecordingDevice()
    net = _RawNet(2, pre=[0, 0], post=[1], wts=[1.0, 2.0])   # post is short
    with pytest.raises(CapacityError, match="unequal lengths"):
        load_network(dev, net, frac_bits=3)
    assert not dev.synapses and not dev.dptr


def test_synapse_ids_must_be_inside_the_network_not_just_the_device():
    """Reported: a 2-neuron network with source id 3 on a 4-neuron device was
    accepted and physically written -- pointing at a neuron the network never
    configured. Validation must use n_neurons, not device.n_max."""
    dev = RecordingDevice(n_max=4, num_lanes=2)
    net = _RawNet(2, pre=[3], post=[1], wts=[1.0])
    with pytest.raises(CapacityError, match=r"\[0, 2\)"):
        load_network(dev, net, frac_bits=3)
    assert not dev.synapses


def test_out_of_network_post_id_raises_capacity_not_keyerror():
    dev = RecordingDevice(n_max=8, num_lanes=2)
    net = _RawNet(2, pre=[0], post=[5], wts=[1.0])
    with pytest.raises(CapacityError):
        load_network(dev, net, frac_bits=3)


def test_bulk_batch_is_cancelled_not_flushed_on_error():
    """Flushing a partially built batch leaves the board looking configured
    while holding an incomplete network -- worse than writing nothing."""
    dev = RecordingDevice()
    dev.ended = False
    dev.cancelled = False

    def _end(*a, **k):
        dev.ended = True

    def _cancel():
        dev.cancelled = True

    dev.end_bulk = _end
    dev.cancel_bulk = _cancel

    def _boom(*a, **k):
        raise RuntimeError("mid-load failure")

    dev.configure_neuron = _boom

    with pytest.raises(RuntimeError, match="mid-load failure"):
        load_network(dev, _snn(2, [(0, 1, 1.0)]), frac_bits=3)

    assert dev.cancelled, "batch was not cancelled on error"
    assert not dev.ended, "partial batch was flushed to the board"


# --- run_schedule preflight --------------------------------------------------

def test_run_schedule_rejects_out_of_range_neuron_ids():
    """Reported: schedule neuron id 7 was accepted on a 4-neuron device and
    passed straight to write_input_value()."""
    from superneuromat.spikeengine import run_schedule
    dev = RecordingDevice(n_max=8, num_lanes=2)
    dev.write_input_value = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a write was issued despite an invalid schedule id"))
    with pytest.raises(CapacityError, match=r"neuron id 7"):
        run_schedule(dev, {0: {7: 1.0}}, total_ticks=1, frac_bits=0, n_neurons=4)


def test_run_schedule_rejects_network_larger_than_device():
    from superneuromat.spikeengine import run_schedule
    dev = RecordingDevice(n_max=4, num_lanes=2)
    with pytest.raises(CapacityError, match="n_max"):
        run_schedule(dev, {}, total_ticks=1, frac_bits=0, n_neurons=99)


def test_run_schedule_rejects_negative_ticks():
    from superneuromat.spikeengine import run_schedule
    dev = RecordingDevice()
    with pytest.raises(ValueError, match="total_ticks"):
        run_schedule(dev, {}, total_ticks=-1, frac_bits=0, n_neurons=2)
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        run_schedule(dev, {-3: {0: 1.0}}, total_ticks=2, frac_bits=0, n_neurons=2)


def test_run_schedule_rejects_ticks_at_or_past_total():
    """A tick >= total_ticks was accepted and then silently never executed --
    the run loop iterates range(total_ticks), so the caller's input was
    dropped while the run reported success."""
    from superneuromat.spikeengine import run_schedule
    dev = RecordingDevice()
    dev.write_input_value = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a write was issued for an unreachable tick"))
    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        run_schedule(dev, {3: {0: 1.0}}, total_ticks=3, frac_bits=0, n_neurons=2)
    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        run_schedule(dev, {99: {0: 1.0}}, total_ticks=3, frac_bits=0, n_neurons=2)


def test_run_schedule_rejects_non_integer_tick_keys():
    """schedule.get(t) only matches exact ints, so a float or str key would
    never fire -- silently dropping the caller's input."""
    from superneuromat.spikeengine import run_schedule
    dev = RecordingDevice()
    for bad in (1.0, "1"):
        with pytest.raises(TypeError, match="not int"):
            run_schedule(dev, {bad: {0: 1.0}}, total_ticks=3,
                         frac_bits=0, n_neurons=2)
