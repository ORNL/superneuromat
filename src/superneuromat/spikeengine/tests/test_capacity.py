"""Regression test for spikeengine.capacity, calibrated against the THREE real
Vivado builds this project has measured (numbers re-read from the actual
post_route_util.rpt files 2026-07-29). If the estimator's tile math drifts,
these fail.

Scope caveat: all 3 calibration points are DENSE builds, so this validates the
estimator only in the dense regime. A real NON-dense (sparse) build must be
added as a 4th point (plan task C2) to validate the sparse regime -- until then
sparse "fits" verdicts are estimates, not guarantees.
"""

import pytest

from superneuromat.spikeengine import capacity as cap

# tolerance in tiles: observed model-vs-real error is +/-2 on BRAM, 0 on URAM;
# allow +/-3 BRAM / +/-1 URAM so the test is meaningful (catches a real drift)
# without being brittle to rounding-boundary changes.
BRAM_TOL = 3
URAM_TOL = 1


@pytest.mark.parametrize("board", ["basys3", "sp701"])
def test_bram_matches_real_dense_build(board):
    c = cap._CALIBRATION[board]
    est = cap.synapse_bram36_tiles(
        c["syn_cap_per_lane"], c["num_lanes"], c["weight_w"], c["n_max"])
    assert abs(est - c["real_bram"]) <= BRAM_TOL, (
        f"{board}: BRAM estimate {est} vs real {c['real_bram']} exceeds "
        f"+/-{BRAM_TOL} -- estimator drifted from calibrated silicon")


def test_uram_matches_real_dense_build_zcu104():
    c = cap._CALIBRATION["zcu104"]
    est = cap.synapse_uram_tiles(
        c["syn_cap_per_lane"], c["num_lanes"], c["weight_w"], c["n_max"])
    assert abs(est - c["real_uram"]) <= URAM_TOL, (
        f"zcu104: URAM estimate {est} vs real {c['real_uram']} exceeds "
        f"+/-{URAM_TOL}")


def test_lut_is_directional_only():
    # LUT is NOT a hard gate; assert only a loose sanity bound so a wildly-off
    # LUT model is caught, but normal error (up to ~15%) does not fail.
    for board, c in cap._CALIBRATION.items():
        est = cap.estimate_lut(c["n_max"], c["num_lanes"])
        assert 0.7 * c["real_lut"] <= est <= 1.4 * c["real_lut"], (
            f"{board}: LUT estimate {est} wildly off real {c['real_lut']}")


def test_required_syn_cap_is_max_lane_indegree():
    # neuron 0: 5 in-synapses, neuron 8: 3, neuron 1: 2; K=8 -> lane0 holds
    # neurons 0 and 8 (both == lane 0 under dst%8) = 5+3 = 8; lane1 = 2.
    post_ids = [0] * 5 + [8] * 3 + [1] * 2
    assert cap.required_syn_cap_per_lane(post_ids, num_lanes=8) == 8
    # more lanes cannot lower a single hub's contribution: a neuron with 5
    # in-synapses forces cap >= 5 regardless of lane count.
    hub = [7] * 5
    assert cap.required_syn_cap_per_lane(hub, num_lanes=16) == 5
    assert cap.required_syn_cap_per_lane(hub, num_lanes=8) == 5


def test_dense_configs_are_reported_fitting():
    # each board's OWN dense config must come back as fitting on that board
    for board, c in cap._CALIBRATION.items():
        est = cap.estimate(board, c["n_max"], c["num_lanes"], c["syn_cap_per_lane"],
                           weight_w=c["weight_w"], data_w=c["data_w"])
        assert est["fits"], f"{board} dense config should fit its own board: {est}"


def test_recommend_board_smoke():
    # a tiny sparse network: 90 neurons, a couple of modest hubs
    post_ids = list(range(90)) * 2 + [5] * 40  # neuron 5 is a small hub
    rec = cap.recommend_board(90, post_ids)
    assert "required_cap_by_lanes" in rec
    assert rec["recommended"] is not None  # something small should fit
    assert rec["recommended"]["fits"]


# ---- A4: host-side capacity guard ----

def test_validate_fits_passes_within_capacity():
    # lane 0 holds neurons {0, 8} under dst%8; in-degrees 5 and 3 -> total 8
    post_ids = [0] * 5 + [8] * 3 + [1] * 2
    cap.validate_network_fits(post_ids, num_lanes=8, syn_cap_per_lane=8)   # exactly fits
    cap.validate_network_fits(post_ids, num_lanes=8, syn_cap_per_lane=100)  # ample


def test_validate_raises_over_by_one_naming_lane_and_overage():
    # neuron 0 has 9 in-synapses; on lane 0 (dst%8). cap=8 -> over by exactly 1.
    post_ids = [0] * 9
    with pytest.raises(cap.CapacityError) as ei:
        cap.validate_network_fits(post_ids, num_lanes=8, syn_cap_per_lane=8)
    msg = str(ei.value)
    assert "lane 0" in msg          # names the offending lane
    assert "over by 1" in msg       # names the exact overage
    assert "id 0" in msg            # names the hub neuron
    assert "9" in msg               # its in-synapse count / required cap


def test_load_network_guard_fires_before_any_wire_write():
    """load_network must raise CapacityError from the guard BEFORE issuing any
    device write. A fake device whose every write method raises proves no wire
    activity happened -- if the guard were missing or ran after writes, we'd get
    the fake's AssertionError instead of CapacityError."""
    from superneuromat import SNN
    from superneuromat.spikeengine import load_network

    class _NoWriteDevice:
        num_lanes = 8
        local_d = 2
        n_max = 10
        syn_cap_per_lane = 8   # small on purpose
        weight_w = 16          # load_network takes widths from the device
        data_w = 24

        def _boom(self, *a, **k):
            raise AssertionError("a wire write was issued before the capacity guard!")

        clear_error = begin_bulk = end_bulk = configure_neuron = _boom
        write_dptr = write_dptr_raw = write_synapse = _boom
        write_stdp_table_all_lanes = set_stdp_enable = _boom

    snn = SNN()
    ns = [snn.create_neuron(threshold=1.0) for _ in range(10)]
    for i in range(1, 10):                     # 9 synapses into neuron 0
        snn.create_synapse(ns[i], ns[0], weight=1.0)

    with pytest.raises(cap.CapacityError):
        load_network(_NoWriteDevice(), snn, frac_bits=0)   # cap=8 < needed 9 -> raise

    # and it SUCCEEDS (reaches the writes, then the fake boom) when capacity is ample
    with pytest.raises(AssertionError, match="wire write was issued"):
        load_network(_NoWriteDevice(), snn, frac_bits=0, syn_cap_per_lane=100)


def test_recommend_never_prefers_a_lut_infeasible_board():
    """estimate_lut()'s docstring says the estimate is used to reject boards
    whose neuron count blows the LUT budget, but only the first tier honoured
    it: with no margin-clean option, recommend_board() could return a config
    that cannot be built. A LUT-feasible option must always win (2026-07-31).
    """
    for n in (256, 1024, 2116, 2715, 3318):
        post = list(range(n))
        rec = cap.recommend_board(n, post)
        chosen = rec["recommended"]
        if chosen is None:
            continue
        if not chosen["lut_advisory_fits"]:
            # only acceptable if NOTHING that fits memory is also LUT-clean
            assert not any(o["fits"] and o["lut_advisory_fits"]
                           for o in rec["options"]), (
                f"n={n}: recommended LUT-infeasible {chosen['board']} while a "
                "LUT-feasible option existed")
            assert rec["recommended_lut_advisory_clean"] is False


def test_recommend_reports_lut_status_of_its_choice():
    rec = cap.recommend_board(256, list(range(256)))
    chosen = rec["recommended"]
    assert chosen is not None
    assert rec["recommended_lut_advisory_clean"] == chosen["lut_advisory_fits"]
