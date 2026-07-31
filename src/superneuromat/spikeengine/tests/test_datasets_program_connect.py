"""Unit tests for datasets.py, program.py's script lookup, and connect()'s
geometry-override logic. All software-only -- no hardware, no Vivado, no
serial port required. This closes the coverage gap noted in SIGNOFF_PLAN.md
Item 4: these three were previously validated only by manual/hardware
testing, with no automated regression check.
"""
import pytest

# ---------------------------------------------------------------------------
# datasets.py
# ---------------------------------------------------------------------------

def test_list_datasets_matches_catalogue():
    from superneuromat.spikeengine.datasets import DATASETS, list_datasets
    assert list_datasets() == sorted(DATASETS)
    assert set(list_datasets()) == {
        "microseer", "miniseer", "cora", "citeseer", "pubmed"}


def test_get_dataset_known_keys_round_trip():
    from superneuromat.spikeengine.datasets import get_dataset
    for key in ("microseer", "miniseer", "cora", "citeseer", "pubmed"):
        ds = get_dataset(key)
        assert ds.key == key
        assert ds.neurons > 0
        assert ds.synapses > 0
        assert ds.num_lanes > 0


def test_get_dataset_unknown_key_raises_with_available_list():
    from superneuromat.spikeengine.datasets import get_dataset
    with pytest.raises(KeyError) as exc:
        get_dataset("not_a_real_dataset")
    msg = str(exc.value)
    assert "not_a_real_dataset" in msg
    assert "microseer" in msg   # the "available: [...]" listing


def test_bitstream_path_none_when_no_custom_bitstream():
    from superneuromat.spikeengine.datasets import get_dataset
    # microseer runs on the board's own packaged bitstream, not a dataset-
    # specific one -- Dataset.bitstream is None, bitstream_path() must be too.
    assert get_dataset("microseer").bitstream is None
    assert get_dataset("microseer").bitstream_path() is None
    assert get_dataset("pubmed").bitstream_path() is None


def test_bitstream_path_resolves_under_package_bitstreams_dir():
    from superneuromat.spikeengine.datasets import _PKG_ROOT, get_dataset
    for key in ("miniseer", "cora", "citeseer"):
        ds = get_dataset(key)
        p = ds.bitstream_path()
        assert p is not None
        assert p == _PKG_ROOT / "bitstreams" / ds.bitstream
        assert p.exists(), f"{key} bitstream missing at {p}"


def test_hardware_validated_datasets_record_both_accuracy_metrics():
    """One-vs-rest and top-1 are DIFFERENT metrics (see Dataset.sw_accuracy /
    sw_top1 docstrings). Both must be on record for any dataset claimed as
    hardware-validated, so a one-vs-rest number is never mistaken for, or
    compared against, a top-1 figure. Values below are the measured
    software-reference results (citation_gnn_fpga --no-hardware, full test
    sets, 2026-07-30); the FPGA agrees with this reference on every paper."""
    from superneuromat.spikeengine.datasets import DATASETS
    expected = {
        "microseer": (0.6424, (20, 48)),
        "miniseer": (0.7667, (66, 120)),
        "cora": (0.8173, (88, 140)),
        "citeseer": (0.5806, (50, 120)),
    }
    for key, (ovr, top1) in expected.items():
        ds = DATASETS[key]
        assert ds.sw_accuracy == ovr, f"{key}: one-vs-rest drifted"
        assert ds.sw_top1 == top1, f"{key}: top-1 drifted"
        # top-1 is the stricter metric; it must never exceed one-vs-rest, which
        # counts true negatives on every non-target topic.
        assert ds.sw_top1_accuracy <= ds.sw_accuracy, (
            f"{key}: top-1 ({ds.sw_top1_accuracy}) > one-vs-rest "
            f"({ds.sw_accuracy}) -- metrics likely swapped")

    for ds in DATASETS.values():
        if ds.hardware_validated:
            assert ds.sw_top1 is not None, (
                f"{ds.key}: hardware_validated but top-1 not recorded")


def test_hardware_validated_datasets_have_a_real_accuracy_and_note():
    # A dataset marked hardware_validated=True must report an actual measured
    # accuracy (not the 0.0 placeholder pubmed uses for "not run at all") and
    # a note that isn't still saying a run is pending.
    from superneuromat.spikeengine.datasets import DATASETS
    for ds in DATASETS.values():
        if ds.hardware_validated:
            assert ds.sw_accuracy > 0.0, f"{ds.key}: validated but accuracy is 0.0"
            assert "pending" not in ds.note.lower(), (
                f"{ds.key}: marked hardware_validated=True but note still says "
                f"pending: {ds.note!r}")


# ---------------------------------------------------------------------------
# program.py -- board-specific Tcl script lookup
# ---------------------------------------------------------------------------

def test_program_picks_the_matching_board_script(monkeypatch, tmp_path):
    import superneuromat.spikeengine.program as program_mod

    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))

        class _Proc:
            stdout = "PROGRAM_DONE"
            stderr = ""
        return _Proc()

    fake_bit = tmp_path / "fake.bit"
    fake_bit.write_bytes(b"\x00")

    monkeypatch.setattr(program_mod, "find_vivado", lambda: "vivado.bat")
    monkeypatch.setattr(program_mod.subprocess, "run", fake_run)

    expected_script = {
        "basys3": "program_infer_basys3_stdp_cap256x8.tcl",
        "sp701": "program_infer_sp701_stdp_maxcap352x8.tcl",
        "zcu104": "program_infer_zcu104_stdp_maxcap1024x16.tcl",
    }
    for board, script_name in expected_script.items():
        calls.clear()
        program_mod.program(board, bitstream=fake_bit)
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert "-source" in args
        script_arg = args[args.index("-source") + 1]
        assert script_arg.endswith(script_name), (
            f"{board}: expected script ending in {script_name!r}, got {script_arg!r}")

        # Vivado must run in a scratch dir, never the caller's cwd or the
        # installed package: it writes .Xil/ and journal files into its working
        # directory, which pollutes the user's project and on Windows can blow
        # the path-length limit (2026-07-30 fix).
        import pathlib

        import superneuromat.spikeengine as se_pkg
        assert kwargs.get("cwd"), f"{board}: Vivado invoked without an explicit cwd"
        cwd = pathlib.Path(kwargs["cwd"]).resolve()
        pkg_root = pathlib.Path(se_pkg.__file__).parent.resolve()
        assert pkg_root not in cwd.parents and cwd != pkg_root, (
            f"{board}: Vivado cwd {cwd} is inside the installed package")
        assert "-tempDir" in args, f"{board}: -tempDir not passed to Vivado"


def test_program_missing_bitstream_raises_before_touching_vivado(monkeypatch, tmp_path):
    import superneuromat.spikeengine.program as program_mod

    def fail_if_called(*a, **k):
        raise AssertionError("subprocess.run should not be called when the "
                              "bitstream is missing")

    monkeypatch.setattr(program_mod.subprocess, "run", fail_if_called)
    missing = tmp_path / "does_not_exist.bit"
    with pytest.raises(FileNotFoundError):
        program_mod.program("basys3", bitstream=missing)


def test_program_no_vivado_raises_runtime_error(monkeypatch, tmp_path):
    import superneuromat.spikeengine.program as program_mod

    fake_bit = tmp_path / "fake.bit"
    fake_bit.write_bytes(b"\x00")
    monkeypatch.setattr(program_mod, "find_vivado", lambda: None)
    with pytest.raises(RuntimeError, match="Vivado not found"):
        program_mod.program("basys3", bitstream=fake_bit)


def test_program_no_program_done_raises_runtime_error(monkeypatch, tmp_path):
    import superneuromat.spikeengine.program as program_mod

    fake_bit = tmp_path / "fake.bit"
    fake_bit.write_bytes(b"\x00")

    def fake_run(args, **kwargs):
        class _Proc:
            stdout = "some other Vivado output, no success marker"
            stderr = ""
        return _Proc()

    monkeypatch.setattr(program_mod, "find_vivado", lambda: "vivado.bat")
    monkeypatch.setattr(program_mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="PROGRAM_DONE"):
        program_mod.program("basys3", bitstream=fake_bit)


def test_device_closes_on_exception_via_context_manager():
    """A device used as a context manager must close even when the body
    raises -- otherwise the serial port leaks and, on Windows, stays locked
    so the NEXT run cannot open it."""
    from superneuromat.spikeengine.runtime import InferEngineStdpDevice

    closed = []

    class _Dev(InferEngineStdpDevice):
        def __init__(self):            # bypass __init__: no real serial port
            self._bulk = None
        def close(self):
            closed.append(True)

    with pytest.raises(ValueError), _Dev():
        raise ValueError("boom")
    assert closed == [True], "device was not closed when the block raised"

    closed.clear()
    with _Dev() as d:
        assert d is not None
    assert closed == [True], "device was not closed on normal exit"


# ---------------------------------------------------------------------------
# build.py -- scratch-directory handling
# ---------------------------------------------------------------------------

def test_build_runs_vivado_outside_the_installed_package(monkeypatch, tmp_path):
    """build_bitstream() must not run Vivado inside site-packages.

    Vivado writes `.Xil/` and journal files into its working directory. Doing
    that in an installed package pollutes site-packages, breaks on a read-only
    install, and -- how this was actually found, via the build_vs_packaged
    notebook -- exceeds the Windows path limit:
    "ERROR: [Common 17-1373] The path length of the Xilinx temporary directory
    ... is too long for Windows".
    """
    import pathlib

    import superneuromat.spikeengine as se_pkg
    import superneuromat.spikeengine.build as build_mod

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")

        class _Proc:
            stdout = "BUILD_FAILED"     # stop early; we only inspect invocation
            stderr = ""
            returncode = 1
        return _Proc()

    monkeypatch.setattr(build_mod, "find_vivado", lambda: "vivado.bat")
    monkeypatch.setattr(build_mod.subprocess, "run", fake_run)
    monkeypatch.setenv("SPIKEENGINE_BUILD_DIR", str(tmp_path / "scratch"))

    with pytest.raises(build_mod.VivadoError):
        build_mod.build_bitstream("basys3")

    assert seen.get("cwd"), "Vivado invoked without an explicit cwd"
    cwd = pathlib.Path(seen["cwd"]).resolve()
    pkg_root = pathlib.Path(se_pkg.__file__).parent.resolve()
    assert pkg_root not in cwd.parents and cwd != pkg_root, (
        f"Vivado cwd {cwd} is inside the installed package")
    assert cwd.is_relative_to((tmp_path / "scratch").resolve()), (
        "SPIKEENGINE_BUILD_DIR was not honoured")
    assert "-tempDir" in seen["cmd"], "-tempDir not passed to Vivado"


# ---------------------------------------------------------------------------
# connect() -- geometry/weight-width resolution (dataset= override logic)
# ---------------------------------------------------------------------------

class _FakeDevice:
    """Stands in for InferEngineStdpDevice: records the resolved geometry
    without opening a real serial port (the real class's __init__ calls
    SerialTransport(port, baud, ...) immediately, which needs hardware)."""

    def __init__(self, port, baud, n_max=1024, num_lanes=16, timeout=1.0,
                 weight_w=8, data_w=16, prefer_interface=None,
                 syn_cap_per_lane=None, stdp_window=5):
        self.port = port
        self.baud = baud
        self.n_max = n_max
        self.num_lanes = num_lanes
        self.timeout = timeout
        self.weight_w = weight_w
        self.data_w = data_w
        self.prefer_interface = prefer_interface
        self.syn_cap_per_lane = syn_cap_per_lane
        self.stdp_window = stdp_window


def test_connect_no_dataset_uses_board_defaults(monkeypatch):
    from superneuromat import spikeengine as se
    monkeypatch.setattr(se, "InferEngineStdpDevice", _FakeDevice)
    dev = se.connect(port="COM_TEST", board="zcu104")
    b = se.get_board("zcu104")
    assert dev.n_max == b.n_max == 1024
    assert dev.num_lanes == b.num_lanes == 16
    assert dev.weight_w == b.weight_w == 8    # legacy packaged bitstream
    assert dev.data_w == b.data_w == 16
    assert dev.baud == b.baud


def test_connect_dataset_overrides_geometry_and_weight_width(monkeypatch):
    from superneuromat import spikeengine as se
    monkeypatch.setattr(se, "InferEngineStdpDevice", _FakeDevice)
    dev = se.connect(port="COM_TEST", board="zcu104", dataset="miniseer")
    ds = se.get_dataset("miniseer")
    # dataset geometry, NOT the board's legacy packaged-bitstream defaults
    assert dev.n_max == ds.neurons == 2116
    assert dev.num_lanes == ds.num_lanes == 8
    # dataset's wide datapath (W16/D24), not the board's legacy W8/D16 --
    # this is exactly the weight-truncation bug this parameter was added to
    # fix (a structural weight of 2.0 would silently read back as 0 under
    # the board's legacy weight_w=8).
    assert dev.weight_w == ds.weight_w == 16
    assert dev.data_w == ds.data_w == 24


def test_connect_explicit_kwargs_take_precedence_over_dataset(monkeypatch):
    from superneuromat import spikeengine as se
    monkeypatch.setattr(se, "InferEngineStdpDevice", _FakeDevice)
    dev = se.connect(port="COM_TEST", board="zcu104", dataset="miniseer",
                     n_max=42, num_lanes=2, weight_w=12, data_w=20)
    assert dev.n_max == 42
    assert dev.num_lanes == 2
    assert dev.weight_w == 12
    assert dev.data_w == 20


def test_connect_unknown_dataset_raises_before_constructing_device(monkeypatch):
    from superneuromat import spikeengine as se

    def fail_if_called(*a, **k):
        raise AssertionError("device should not be constructed when the "
                              "dataset lookup fails")

    monkeypatch.setattr(se, "InferEngineStdpDevice", fail_if_called)
    with pytest.raises(KeyError):
        se.connect(port="COM_TEST", board="zcu104", dataset="not_a_dataset")


def test_connect_rejects_dataset_not_validated_on_this_board(monkeypatch):
    """connect(board="basys3", dataset="cora") silently built a 2,715-neuron
    device on a 256-neuron board. Contradictory pairings must raise."""
    import superneuromat.spikeengine as se
    monkeypatch.setattr(se, "InferEngineStdpDevice", _FakeDevice)
    with pytest.raises(ValueError, match="not validated on board"):
        se.connect(port="COM_TEST", board="basys3", dataset="cora")


def test_connect_rejects_software_only_dataset(monkeypatch):
    """pubmed has no hardware build at all."""
    import superneuromat.spikeengine as se
    monkeypatch.setattr(se, "InferEngineStdpDevice", _FakeDevice)
    with pytest.raises(ValueError, match="software-only"):
        se.connect(port="COM_TEST", board="zcu104", dataset="pubmed")


def test_connect_passes_real_capacity_for_a_dataset_with_its_own_build(monkeypatch):
    """The capacity guard is only meaningful with the bitstream's ACTUAL
    per-lane capacity; the dense default is 183x too loose for citeseer."""
    import superneuromat.spikeengine as se
    monkeypatch.setattr(se, "InferEngineStdpDevice", _FakeDevice)
    dev = se.connect(port="COM_TEST", board="zcu104", dataset="citeseer")
    assert dev.syn_cap_per_lane == se.get_dataset("citeseer").syn_cap_per_lane


def test_connect_uses_dense_default_for_a_dataset_on_the_board_bitstream(monkeypatch):
    """microseer runs on Basys3's PACKAGED dense build. Its catalogue
    syn_cap_per_lane is the network REQUIREMENT (144), not a hardware
    capacity -- passing it as the capacity rejected a network that fits
    easily (the dense build provides local_d * n_max = 8192)."""
    import superneuromat.spikeengine as se
    monkeypatch.setattr(se, "InferEngineStdpDevice", _FakeDevice)
    dev = se.connect(port="COM_TEST", board="basys3", dataset="microseer")
    assert dev.syn_cap_per_lane is None, (
        "microseer must not pass its requirement as the device capacity")
