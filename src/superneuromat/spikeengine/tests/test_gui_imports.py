"""Import-level tests for the bundled GUI.

These verify the module wiring only -- that every GUI module imports, that the
entry point exists, and that the two board kinds this copy does NOT support
fail loudly rather than silently. They deliberately do not construct any Qt
widget: PyVista/VTK needs a real OpenGL context, which is unavailable in a
headless environment, so widget construction is not testable here and must be
checked by running the GUI on a machine with a display.

Skipped entirely when PyQt5 is absent.
"""
import pytest

pytest.importorskip("PyQt5", reason="GUI tests need PyQt5")


def test_gui_package_and_entry_point_import():
    """The whole GUI module tree must import, including the rewired NPU-STDP
    device path. This is what catches a broken import after the package was
    moved to superneuromat.spikeengine."""
    from superneuromat.spikeengine.gui import snm_gui
    assert callable(snm_gui.main), "spikeengine-gui entry point must be callable"


def test_every_gui_module_imports():
    import importlib
    for name in (
        "snm_npu_stdp", "snm_npu", "snm_driver", "snm_network", "snm_boards",
        "snm_config", "snm_presets", "snm_snn_io", "snm_digits_example",
        "snm_capacity_example", "network_view", "pyvista_network_view",
        "graph_layout",
    ):
        importlib.import_module(f"superneuromat.spikeengine.gui.{name}")


def test_stdp_device_subclasses_this_packages_runtime():
    """The GUI device must derive from THIS package's runtime, not the
    archived legacy package's LaneEngineDevice (the dependency removed during
    the rewire). Guards against the old dependency creeping back in."""
    from superneuromat.spikeengine.gui.snm_npu_stdp import LaneEngineStdpDevice
    from superneuromat.spikeengine.runtime import InferEngineStdpDevice
    assert issubclass(LaneEngineStdpDevice, InferEngineStdpDevice)


def test_unsupported_board_kinds_fail_loudly():
    """Classic and non-STDP NPU are stubbed out. They must raise a clear error
    if ever reached -- never return fabricated data."""
    from superneuromat.spikeengine.gui import snm_driver, snm_npu
    with pytest.raises(RuntimeError):
        snm_npu.connect()
    with pytest.raises(RuntimeError):
        snm_driver.connect()
    # is_npu() has no dependency and stays real; no "npu:" board is ever
    # offered by this copy, so it must report False for the STDP keys.
    assert snm_npu.is_npu("npu-stdp:basys3") is False
    assert snm_npu.is_npu("basys3") is False


def test_pyvista_availability_is_reported_not_assumed():
    """The 3D view is optional. pyvista_available() must answer without
    raising either way, so the GUI can fall back to the 2D view."""
    from superneuromat.spikeengine.gui.pyvista_network_view import (
        pyvista_available,
        pyvista_import_error_message,
    )
    assert isinstance(pyvista_available(), bool)
    if not pyvista_available():
        assert pyvista_import_error_message()


def test_no_unguarded_imports_of_archived_legacy_modules():
    """The pre-STDP spikeengine package (fpga.py / fpga_runtime.py /
    fpga_vivado.py, now archived under legacy/) is NOT shipped. Nothing here
    may import it at module level or from inside a function, because such
    lazy imports pass an import test and then fail at runtime -- which is
    exactly what happened after the package move (2026-07-30) and is why this
    test exists.

    The single permitted exception is snm_snn_io's manifest lookup, which is
    wrapped in try/except and documented as a graceful no-op for classic
    boards this GUI copy never offers.
    """
    import pathlib
    import re

    import superneuromat.spikeengine as pkg

    root = pathlib.Path(pkg.__file__).parent
    banned = re.compile(
        r"^\s*(from\s+superneuromat\.spikeengine\.(fpga|fpga_runtime|fpga_vivado|readiness|setup_loader)\b"
        r"|from\s+superneuromat\.spikeengine\s+import\s+(fpga|fpga_runtime|fpga_vivado)\b"
        r"|from\s+superneuromat\.fpga\b"
        r"|from\s+superneuromat\s+import\s+fpga\b)")
    allowed = {("snm_snn_io.py", "load_fpga_manifest")}

    offenders = []
    for path in root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if banned.match(line):
                if any(path.name == f and tok in line for f, tok in allowed):
                    continue
                offenders.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "imports of archived legacy modules found (these fail at runtime):\n  "
        + "\n  ".join(offenders))


def test_preset_runner_modules_are_importable():
    """Generated presets name their runner by dotted module path. A stale path
    (e.g. the pre-move `fpga_gui_app.*`) only fails when a user clicks Run, so
    resolve every one of them here instead."""
    import importlib

    from superneuromat.spikeengine.gui.snm_presets import GENERATED
    for name, preset in GENERATED.items():
        mod = preset.get("runner_module")
        if not mod:
            continue
        importlib.import_module(mod)   # raises if the path is stale
        entry = preset.get("entry_point")
        if entry:
            assert hasattr(importlib.import_module(mod), entry), (
                f"preset {name!r}: {mod} has no attribute {entry!r}")


def test_gui_board_list_offers_only_stdp_boards():
    """Only NPU-STDP boards are supported here; the keys must all carry the
    npu-stdp: prefix and resolve to real boards in the catalogue."""
    from superneuromat.spikeengine.boards import BOARDS
    from superneuromat.spikeengine.gui import snm_npu_stdp
    keys = snm_npu_stdp.keys()
    assert keys, "no NPU-STDP boards offered"
    for k in keys:
        assert snm_npu_stdp.is_npu_stdp(k)
        assert snm_npu_stdp.base_board(k) in BOARDS


def test_gui_uses_exact_validated_profiles_for_sp701_and_zcu104():
    from superneuromat.spikeengine.gui import snm_npu_stdp

    # SP701's own default was rebuilt 2026-07-31 after its packaged bitstream
    # was found to have confirmed-wrong N_MAX/NUM_LANES (see boards.py) --
    # no dataset substitution is needed any more, same as basys3/zcu104.
    expected = {
        "npu-stdp:sp701": (None, 352, 8, 352 * 44),
        "npu-stdp:zcu104": (None, 1024, 16, 65536),
    }
    for key, (dataset, n_max, lanes, cap) in expected.items():
        desc = snm_npu_stdp.board_dict(key)
        hw = snm_npu_stdp.hw_for_board(key)
        assert desc["status"] == "supported"
        assert desc["npu"]["hardware_validated"] is True
        assert desc["npu"]["dataset_profile"] == dataset
        assert hw.n_max == n_max
        assert hw.num_lanes == lanes
        assert hw.syn_max == lanes * cap
        if dataset is not None:
            assert dataset in snm_npu_stdp.bitstream_path(key).name
        elif key == "npu-stdp:zcu104":
            assert snm_npu_stdp.bitstream_path(key).name.endswith(
                "stdp_maxcap1024x16.bit")
        else:
            assert snm_npu_stdp.bitstream_path(key).name.endswith(
                "stdp_maxcap352x8.bit")


def test_gui_zcu104_profile_uses_observed_runtime_interface():
    from superneuromat.spikeengine.boards import get_board
    assert get_board("zcu104").uart_interface_index == 3


def test_gui_loader_writes_the_complete_dptr_table():
    """Audit finding (2026-07-31): the GUI loader wrote dst_ptr entries only
    for destinations that had synapses, and a sentinel only on used lanes.
    Empty neurons, gaps and unused lanes kept stale pointers from whatever
    network was loaded before, which makes them accumulate another network's
    synapses (the gather stage walks dst_ptr[d]..dst_ptr[d+1]).

    Every lane must receive all local_d + 1 entries.
    """
    from superneuromat.spikeengine.gui.snm_npu_stdp import LaneEngineStdpDevice

    class _Rec(LaneEngineStdpDevice):
        def __init__(self, n_max=16, num_lanes=4):
            self.n_max = n_max
            self.num_lanes = num_lanes
            self.local_d = (n_max + num_lanes - 1) // num_lanes
            self._entry_index = {}
            self.dptr = []
            self.syn = []
        def clear_error(self): pass
        def begin_bulk(self): pass
        def end_bulk(self, *a, **k): pass
        def configure_neuron(self, *a, **k): pass
        def set_stdp_enable(self, *a, **k): pass
        def write_stdp_table_all_lanes(self, *a, **k): pass
        def write_dptr_raw(self, lane, idx, off): self.dptr.append((lane, idx, off))
        def write_synapse(self, dst, idx, src, w): self.syn.append((dst, idx, src, w))

    class _N:
        def __init__(self, idx):
            self.idx = idx
            self.threshold = self.leak = self.reset_state = 0
            self.refractory_period = 0

    class _S:
        def __init__(self, pre, post, w):
            self.pre_id, self.post_id, self.weight = pre, post, w

    class _Net:
        # only ONE destination has synapses, on ONE lane -- the exact case
        # that previously left every other lane's table untouched
        def __init__(self):
            self.neurons = [_N(i) for i in range(16)]
            self.synapses = [_S(1, 4, 7)]

    dev = _Rec()
    dev.load_network(_Net())

    written = {(lane, idx) for (lane, idx, _off) in dev.dptr}
    expected = {(lane, idx)
                for lane in range(dev.num_lanes)
                for idx in range(dev.local_d + 1)}
    missing = expected - written
    assert not missing, f"dst_ptr entries never written: {sorted(missing)}"

    # the one real synapse still lands correctly
    assert dev.syn == [(4, 0, 1, 7)], dev.syn
