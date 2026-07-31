"""SuperNeuroMAT3 FPGA <-> superneuromat SNN interchange (import/export).

Defines the portable JSON format ("superneuromat-fpga-snn", version 1) used to
move a spiking network -- and optionally its per-timestep inputs -- between the
superneuromat software framework and this FPGA GUI. Provides:

  * the schema + load/save (save goes to ./snn_exports/ in the run directory),
  * converters to/from the host SNN builder (snm_network.SNN),
  * guard_snn(): a single conformance guard that checks a network against the
    FPGA/GUI requirements (capacity, value ranges at the chosen fixed-point,
    inputs only to input-enabled neurons, <= event-FIFO events per tick) and
    reports fixed-point snaps -- reused by BOTH the exporter and the importer.

Schema (all "analog" params are decimals; tick counts/indices are integers):

  {
    "format": "superneuromat-fpga-snn", "version": 1,
    "frac_bits": 1, "steps": 6,
    "neurons":  [{"threshold":20.0,"leak":0.0,"reset":0.0,"refractory":0,
                  "input":true,"initial_state":0.0}, ...],
    "synapses": [{"pre":0,"post":1,"weight":10.0,"stdp":false}, ...],
    "stdp":     null | {"window":1,"apos":[2.0],"aneg":[-1.0]},
    "inputs":   null | {"0":{"0":12.0}, "2":{"1":-3.0}},   # timestep -> {neuron:value}
    "meta":     { ... }
  }
"""

from __future__ import annotations

import json
import os

from . import snm_network as nm

FORMAT = "superneuromat-fpga-snn"
VERSION = 1
DEFAULT_FRAC_BITS = 1            # 1-decimal-place default precision
EXPORT_DIRNAME = "snn_exports"   # auto-created in the run directory


def _r1(x):
    """Round an analog value to 1 decimal place (the export precision default)."""
    return round(float(x), 1)


def _manifest_board_entry(board: str | None):
    if not board:
        return None
    manifest = None
    try:
        # The bitstream manifest.json is a CLASSIC-engine concept: it belonged to
        # the pre-STDP spikeengine package (now archived under legacy/). This
        # package describes its boards in `boards.py` instead and ships no
        # manifest, so this import normally fails and we return None -- the
        # caller (`snm_gui._active_hw`) then falls back to its default HW spec.
        # Only classic boards reach here, and this GUI copy never offers them
        # (see snm_driver.py), so this path is effectively dead; it is kept so
        # an installation that DOES still provide the legacy module keeps
        # working rather than breaking.
        from superneuromat.spikeengine.fpga import load_fpga_manifest  # type: ignore[import-not-found]  # noqa: I001
        manifest = load_fpga_manifest()
    except (ImportError, AttributeError, OSError):
        path = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            "..", "superneuromat-main", "src", "superneuromat",
            "fpga_bitstreams", "manifest.json",
        ))
        try:
            with open(path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError):
            return None
    return (manifest.get("boards") or {}).get(str(board))


def manifest_hw_for_board(board: str | None):
    """Return a HardwareSpec from the packaged bitstream manifest when possible."""

    raw = _manifest_board_entry(board)
    if not raw or raw.get("n_max") is None or raw.get("syn_depth") is None:
        return None
    return nm.HardwareSpec(
        n_max=int(raw["n_max"]),
        syn_max=int(raw["syn_depth"]),
        data_bits=int(raw.get("data_w", nm.HW.data_bits)),
        weight_bits=int(raw.get("weight_w", nm.HW.weight_bits)),
        ref_bits=int(raw.get("ref_w", nm.HW.ref_bits)),
        stdp_window_max=int(raw.get("stdp_t_max", nm.HW.stdp_window_max)),
        event_fifo_depth=int(raw.get("event_fifo_depth") or nm.HW.event_fifo_depth),
    )


def manifest_baud_for_board(board: str | None) -> int:
    """UART baud the packaged bitstream for ``board`` was built for.

    Reads it straight from ``fpga_bitstreams/manifest.json`` so the GUI opens the port at the
    same rate the Python API does (the packaged Basys3 image runs at 4 Mbaud). Falls back to
    the generated ``snm_config.BAUD`` only if the board has no manifest entry.
    """
    raw = _manifest_board_entry(board)
    try:
        if raw and raw.get("baud"):
            return int(raw["baud"])
    except (TypeError, ValueError):
        pass
    from . import snm_config as _cfg
    return int(_cfg.BAUD)


def _manifest_hw_for_schema(data: dict):
    """Return a HardwareSpec from the packaged bitstream manifest when possible.

    The GUI may be run from the FPGA development tree while SuperNeuroMAT exports
    target the packaged max-capacity bitstream. Prefer the schema's board manifest
    capacity so the GUI does not reject a valid packaged-bitstream network because
    `host/snm_config.py` was generated for a smaller development case.
    """

    board = (data.get("meta") or {}).get("board")
    return manifest_hw_for_board(board)


# --------------------------------------------------------------------------
# run-directory export folder + load/save
# --------------------------------------------------------------------------
def export_dir():
    """Path to ./snn_exports/ in the current run directory (created if missing)."""
    d = os.path.join(os.getcwd(), EXPORT_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def save_snn(data: dict, name: str | None = None, path: str | None = None) -> str:
    """Write a schema dict as JSON. With no explicit path, saves into the run
    directory's snn_exports/ folder. Returns the written path."""
    if path is None:
        name = (name or "snn").strip() or "snn"
        if not name.endswith(".json"):
            name += ".json"
        path = os.path.join(export_dir(), name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_snn(path: str) -> dict:
    """Read + structurally validate a schema JSON file."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{path!r} is not a {FORMAT} file "
                         f"(format={data.get('format')!r} if dict)")
    if int(data.get("version", 0)) > VERSION:
        raise ValueError(f"{path!r} is version {data.get('version')}, newer than "
                         f"this tool supports (v{VERSION})")
    for key in ("neurons", "synapses"):
        if not isinstance(data.get(key), list):
            raise TypeError(f"{path!r}: missing/invalid '{key}' list")
    return data


# --------------------------------------------------------------------------
# converters: snm_network.SNN <-> schema
# --------------------------------------------------------------------------
def snn_to_schema(net, inputs: dict | None = None, steps: int | None = None,
                  frac_bits: int | None = None, meta: dict | None = None) -> dict:
    """Build a schema dict from a host SNN (snm_network.SNN). Analog values are
    rounded to 1 decimal place."""
    fb = int(net.frac_bits if frac_bits is None else frac_bits)
    neurons = [{"threshold": _r1(n.threshold), "leak": _r1(n.leak), "reset": _r1(n.reset),
                    "refractory": int(n.refractory), "input": bool(n.inp),
                    "initial_state": _r1(n.initial_state)} for n in net.neurons]
    synapses = [{"pre": int(s.pre), "post": int(s.post), "weight": _r1(s.weight),
                     "stdp": bool(s.stdp)} for s in net.synapses]
    stdp = None
    if net._stdp:
        st = net._stdp
        stdp = {"window": int(st["window"]),
                    "apos": [_r1(a) for a in st["apos"]],
                    "aneg": [_r1(a) for a in st["aneg"]]}
    sch = {"format": FORMAT, "version": VERSION, "frac_bits": fb,
               "steps": (int(steps) if steps else None),
               "neurons": neurons, "synapses": synapses, "stdp": stdp,
               "inputs": None, "meta": dict(meta or {})}
    if inputs:
        sch["inputs"] = {str(int(t)): {str(int(n)): _r1(v) for n, v in ins.items()}
                         for t, ins in inputs.items()}
    return sch


def schema_inputs(data: dict) -> dict:
    """Extract the input schedule {timestep: {neuron: value}} (ints/floats)."""
    inp = data.get("inputs")
    if not inp:
        return {}
    return {int(t): {int(n): float(v) for n, v in ins.items()}
            for t, ins in inp.items()}


def schema_to_snn(data: dict, driver=None, hw=None, frac_bits: int | None = None):
    """Build a host SNN (snm_network.SNN) from a schema dict (build-only by default)."""
    hw = nm.HW if hw is None else hw
    fb = int(data.get("frac_bits", DEFAULT_FRAC_BITS)) if frac_bits is None else int(frac_bits)
    net = nm.SNN(driver=driver, hw=hw, frac_bits=fb)
    for nd in data["neurons"]:
        net.neuron(threshold=nd.get("threshold", 0), leak=nd.get("leak", 0),
                   reset=nd.get("reset", 0), refractory=int(nd.get("refractory", 0)),
                   initial_state=nd.get("initial_state", 0), inp=bool(nd.get("input", False)))
    for sd in data["synapses"]:
        net.synapse(sd["pre"], sd["post"], weight=sd.get("weight", 1),
                    stdp=bool(sd.get("stdp", False)))
    st = data.get("stdp")
    if st:
        net.stdp(window=int(st.get("window", 1)), apos=st.get("apos"), aneg=st.get("aneg"))
    return net


# --------------------------------------------------------------------------
# the conformance guard (shared by exporter + importer)
# --------------------------------------------------------------------------
def guard_snn(data: dict, hw=None, frac_bits: int | None = None):
    """Validate a schema against the FPGA/GUI requirements.

    Returns (errors, warnings): errors block import/run; warnings are advisory
    (notably fixed-point SNAP of an analog value that isn't exactly representable
    at the chosen frac_bits). Mirrors what SNN.validate()/check_schedule() enforce
    on the real chip, so a passing network is guaranteed loadable by the GUI.
    """
    hw = hw or _manifest_hw_for_schema(data) or nm.HW
    fb = int(data.get("frac_bits", DEFAULT_FRAC_BITS)) if frac_bits is None else int(frac_bits)
    errors, warnings = [], []
    neurons = data.get("neurons") or []
    synapses = data.get("synapses") or []
    n = len(neurons)
    ref_max = (1 << hw.ref_bits) - 1

    if n == 0:
        errors.append("network has no neurons")
    if n > hw.n_max:
        errors.append(f"{n} neurons exceeds N_MAX={hw.n_max}")
    if len(synapses) > hw.syn_max:
        errors.append(f"{len(synapses)} synapses exceeds SYN_MAX={hw.syn_max}")

    def _chk(value, field, where):
        try:
            raw = nm.to_raw(value, field, fb, hw)
        except (ValueError, TypeError) as e:
            errors.append(f"{where}: {e}")
            return
        back = nm.from_raw(raw, fb)
        if back != float(value):
            if raw == 0 and float(value) != 0.0:
                warnings.append(f"{where}={value} rounds to 0 at frac_bits={fb} "
                                f"(no effect on the chip; increase frac_bits or rescale)")
            else:
                warnings.append(f"{where}={value} snaps to {back:g} at frac_bits={fb}")

    for i, nd in enumerate(neurons):
        _chk(nd.get("threshold", 0), "threshold", f"neuron {i} threshold")
        _chk(nd.get("leak", 0), "leak", f"neuron {i} leak")
        _chk(nd.get("reset", 0), "reset", f"neuron {i} reset")
        _chk(nd.get("initial_state", 0), "vmem", f"neuron {i} initial_state")
        ref = int(nd.get("refractory", 0))
        if not (0 <= ref <= ref_max):
            errors.append(f"neuron {i} refractory={ref} out of 0..{ref_max}")

    for i, sd in enumerate(synapses):
        _chk(sd.get("weight", 1), "weight", f"synapse {i} weight")
        for who in ("pre", "post"):
            idx = sd.get(who)
            if idx is None or not (0 <= int(idx) < n):
                errors.append(f"synapse {i} {who}={idx} references a non-existent neuron "
                              f"(only {n} neurons)")

    st = data.get("stdp")
    if st:
        w = int(st.get("window", 0))
        if not (0 <= w <= hw.stdp_window_max):
            errors.append(f"STDP window={w} out of 0..{hw.stdp_window_max}")
        for j, a in enumerate(st.get("apos") or []):
            _chk(a, "apos", f"STDP apos[{j}]")
        for j, a in enumerate(st.get("aneg") or []):
            _chk(a, "aneg", f"STDP aneg[{j}]")

    input_enabled = {i for i, nd in enumerate(neurons) if nd.get("input")}
    for t, ins in (data.get("inputs") or {}).items():
        if len(ins) > hw.event_fifo_depth:
            errors.append(f"timestep {t}: {len(ins)} input events exceeds the hardware "
                          f"event FIFO depth {hw.event_fifo_depth}")
        for nidx, v in ins.items():
            i = int(nidx)
            if not (0 <= i < n):
                errors.append(f"timestep {t}: input for neuron {i} which does not exist")
            elif i not in input_enabled:
                errors.append(f"timestep {t}: neuron {i} gets an input but is not "
                              f"input-enabled (the FPGA ignores it)")
            _chk(v, "input", f"timestep {t} neuron {nidx} input")

    return errors, warnings
