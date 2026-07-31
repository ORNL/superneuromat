"""Tests for the hardware validation harness itself.

The harness only does something useful with a board attached, but its
scaffolding -- argument handling, board/dataset defaults, geometry
comparison, report shape, exit codes -- is testable without one, and is worth
testing precisely because it is the thing that will be trusted to say
"hardware passed".
"""
import json

import pytest

from superneuromat.spikeengine import validate_hardware as vh


def test_default_datasets_are_real_and_validated_on_their_board():
    """Each board's default sweep must name real datasets that are actually
    validated on that board -- otherwise the sweep silently validates nothing,
    or fails on a dataset that could never fit. Note a dataset may be
    validated on more than one board (microseer: basys3 and sp701), so this
    checks `boards()`, not the single `board` field."""
    from superneuromat.spikeengine import get_dataset, list_datasets
    for board, names in vh._DEFAULT_DATASETS.items():
        assert names, f"{board}: no default datasets"
        for n in names:
            assert n in list_datasets(), f"{board}: unknown dataset {n!r}"
            ds = get_dataset(n)
            assert board in ds.boards(), (
                f"{board}: dataset {n!r} is only validated on {ds.boards()}")


def test_every_validated_board_has_a_reachable_bitstream():
    """A bitstream that ships but cannot be reached through the API is
    invisible to users -- the SP701 microseer build was in exactly that state
    until 2026-07-30, mentioned only in a prose note on boards.py. Every board
    a dataset claims must resolve to either a real file or an explicit None
    (meaning: use the board's own packaged default)."""
    from superneuromat.spikeengine import DATASETS
    for ds in DATASETS.values():
        for board in ds.boards():
            p = ds.bitstream_path(board)
            if p is not None:
                assert p.exists(), f"{ds.key}/{board}: missing bitstream {p}"


def test_bitstream_path_rejects_a_board_with_no_build():
    """Falling back to the primary board's bitstream would program the wrong
    geometry, so an unknown board must raise instead."""
    from superneuromat.spikeengine import get_dataset
    with pytest.raises(KeyError, match="no bitstream for board"):
        get_dataset("miniseer").bitstream_path("basys3")


def test_microseer_sp701_bitstream_is_not_the_legacy_default():
    """Microseer retains its smaller, separately validated N90 image even
    though SP701's rebuilt default is now also a valid wide-fixed-point build."""
    from superneuromat.spikeengine import get_board, get_dataset
    ds_path = get_dataset("microseer").bitstream_path("sp701")
    assert ds_path is not None and ds_path.exists()
    assert ds_path != get_board("sp701").bitstream_path()


class _FakeDev:
    """Answers OP_READ_OUTPUT like the RTL does: OK below SPIKE_WORDS,
    ADDR_RANGE at or above it. Lets the probe be tested without a board."""

    def __init__(self, n_max: int):
        self.spike_words = (n_max + 31) // 32
        self.reads = 0

    def command(self, op, sel=0, addr=0, data=0):
        from superneuromat.spikeengine._transport import ST_ADDR_RANGE, ST_OK
        self.reads += 1
        return (ST_OK, 0) if addr < self.spike_words else (ST_ADDR_RANGE, 0)


@pytest.mark.parametrize("n_max", [90, 256, 352, 1024, 2116, 2715, 3318])
def test_probe_measures_real_spike_words(n_max):
    """The probe must recover SPIKE_WORDS from the device's own answers, for
    every geometry this project ships."""
    dev = _FakeDev(n_max)
    assert vh.probe_spike_words(dev) == (n_max + 31) // 32
    # binary search, not a linear scan: must stay well under the word count
    assert dev.reads < 64


def test_geometry_check_detects_a_wrong_bitstream():
    """The critical check: a board running a DIFFERENT build must be caught.

    This is the scenario that silently corrupts results -- e.g. zcu104 still
    holding its packaged N=1024 default while the host loads miniseer
    (N=2116). Nothing on the wire raises; only this probe catches it.
    """
    # correct bitstream -> passes
    check = vh.Check("geometry")
    vh._check_geometry(check, _FakeDev(2116), "zcu104", "miniseer")
    assert check.detail["measured_spike_words"] == check.detail["expected_spike_words"]

    # board still running its packaged N=1024 default -> must fail loudly
    with pytest.raises(AssertionError, match="BITSTREAM MISMATCH"):
        vh._check_geometry(vh.Check("geometry"), _FakeDev(1024), "zcu104", "miniseer")


def test_geometry_check_is_not_tautological():
    """Regression guard: an earlier version compared dev.n_max against the
    catalogue, but connect() SETS dev.n_max from that same catalogue -- so it
    could never fail. The check must consult the device, not the catalogue.
    A device object carrying the 'right' attributes but the wrong hardware
    geometry must still be rejected.
    """
    from superneuromat.spikeengine import get_dataset
    ds = get_dataset("miniseer")

    dev = _FakeDev(1024)              # hardware is WRONG ...
    dev.n_max = ds.neurons            # ... but host attributes look right
    dev.num_lanes = ds.num_lanes
    dev.weight_w, dev.data_w = ds.weight_w, ds.data_w

    with pytest.raises(AssertionError, match="BITSTREAM MISMATCH"):
        vh._check_geometry(vh.Check("geometry"), dev, "zcu104", "miniseer")


def test_probe_rejects_a_device_that_is_not_a_spike_engine():
    class _Dead:
        def command(self, op, sel=0, addr=0, data=0):
            from superneuromat.spikeengine._transport import ST_ADDR_RANGE
            return (ST_ADDR_RANGE, 0)      # even word 0 rejected

    from superneuromat.spikeengine.runtime import InferEngineError
    with pytest.raises(InferEngineError, match="word 0 rejected"):
        vh.probe_spike_words(_Dead())


def test_probe_identifies_a_classic_non_stdp_bitstream():
    """Observed on real hardware (COM17, 2026-07-31): a board answered
    OUT_NOT_READY (0x08). The STDP controller only ever emits 0x00..0x04, so
    0x08 proves a different design is loaded. The error must name that cause
    rather than printing a bare status byte."""
    from superneuromat.spikeengine.runtime import InferEngineError

    class _Classic:
        def command(self, op, sel=0, addr=0, data=0):
            return (0x08, 0)              # ST_OUT_NOT_READY, classic engine

    with pytest.raises(InferEngineError, match="CLASSIC"):
        vh.probe_spike_words(_Classic())


def test_run_records_failure_without_raising():
    """_run must never propagate: one failed check has to leave the sweep able
    to continue and still write a report."""
    def boom(_check):
        raise RuntimeError("simulated")

    c = vh.Check("x")
    ok = vh._run(c, boom)
    assert ok is False
    assert c.status == "FAIL"
    assert "simulated" in c.message
    assert "traceback" in c.detail


def test_run_marks_success():
    c = vh.Check("y")
    ok = vh._run(c, lambda ch: ch.detail.update(v=1))
    assert ok is True and c.status == "PASS" and c.detail["v"] == 1


def test_no_board_exits_nonzero_and_writes_a_report(tmp_path, monkeypatch):
    """With no board attached the sweep must fail loudly (non-zero exit) and
    still leave a machine-readable report -- never exit 0 having silently
    checked nothing."""
    report = tmp_path / "r.json"
    monkeypatch.chdir(tmp_path)

    def no_board(*a, **k):
        from superneuromat.spikeengine._transport import SNMError
        raise SNMError("no board")

    monkeypatch.setattr(vh, "_check_port", no_board)

    rc = vh.main(["--board", "basys3", "--quick", "--report", str(report)])
    assert rc == 1, "must exit non-zero when the board is missing"

    data = json.loads(report.read_text())
    assert data["summary"]["failed"] >= 1
    assert data["summary"]["passed"] == 0
    assert data["board"] == "basys3"
    assert data["checks"][0]["check"] == "port-discovery"
    assert data["checks"][0]["status"] == "FAIL"
    # downstream checks must not be reported as passed when the link is dead
    assert all(c["status"] != "PASS" for c in data["checks"])


def test_dataset_checks_are_skipped_when_a_prerequisite_fails(tmp_path, monkeypatch):
    """Audit finding (2026-07-31): dataset checks gated only on the LINK
    result, so they could run -- and be reported -- after soft-reset or the
    geometry check had failed. A failed prerequisite must skip everything
    downstream, and a skipped check must never count as passed."""
    report = tmp_path / "r.json"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(vh, "_check_port", lambda ch, port, *a, **k: "COM_TEST")

    calls = {"dataset": 0}

    def _never(*a, **k):
        calls["dataset"] += 1

    monkeypatch.setattr(vh, "_validate_dataset", _never)

    # link passes, soft-reset fails
    class _Dev:
        n_max = num_lanes = weight_w = data_w = 1
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def clear_error(self): pass
        def read_status(self): return 0
        def soft_reset(self): raise RuntimeError("reset failed")

    monkeypatch.setattr(vh, "connect", lambda **k: _Dev())

    rc = vh.main(["--board", "zcu104", "--datasets", "miniseer",
                  "--report", str(report)])

    assert rc == 1
    assert calls["dataset"] == 0, "dataset ran despite soft-reset failing"

    data = json.loads(report.read_text())
    by_name = {c["check"]: c for c in data["checks"]}
    assert by_name["soft-reset"]["status"] == "FAIL"
    assert by_name["identify-loaded-build"]["status"] == "SKIP"
    assert by_name["dataset-miniseer"]["status"] == "SKIP"
    assert "prerequisite failed" in by_name["dataset-miniseer"]["message"]
    assert data["summary"]["validated"] is False


def test_incomplete_sweep_is_not_reported_as_a_pass(tmp_path, monkeypatch):
    """Zero failures but some checks skipped is NOT a validated board. Exit
    code must stay non-zero so a caller cannot read it as success."""
    report = tmp_path / "r.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vh, "_check_port", lambda ch, port, *a, **k: (_ for _ in ()).throw(
        RuntimeError("no board")))

    rc = vh.main(["--board", "basys3", "--report", str(report)])
    data = json.loads(report.read_text())

    assert rc == 1
    assert data["summary"]["skipped"] > 0
    assert data["summary"]["validated"] is False
    assert all(c["status"] != "PASS" for c in data["checks"])


def test_unknown_dataset_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        vh.main(["--board", "zcu104", "--datasets", "not_a_dataset"])


def test_quick_mode_runs_no_dataset_checks(tmp_path, monkeypatch):
    report = tmp_path / "q.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vh, "_check_port", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no board")))
    vh.main(["--board", "zcu104", "--quick", "--report", str(report)])
    data = json.loads(report.read_text())
    assert data["datasets"] == []
    assert not any(c["check"].startswith("dataset-") for c in data["checks"])
