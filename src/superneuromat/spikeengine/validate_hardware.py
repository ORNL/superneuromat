"""One-command hardware validation sweep.

Runs the full board-validation checklist and writes a timestamped report, so a
validation session is a single command with logged evidence rather than a
sequence of ad-hoc invocations whose results live only in a terminal.

Usage (board programmed and connected)::

    python -m superneuromat.spikeengine.validate_hardware --board basys3
    python -m superneuromat.spikeengine.validate_hardware --board zcu104 --datasets miniseer,cora
    python -m superneuromat.spikeengine.validate_hardware --board sp701 --quick

What it checks, in order (later steps are skipped if an earlier one fails,
since they cannot be meaningful):

  1. PORT      -- the board's UART is discoverable (USB VID:PID, then probe).
  2. LINK      -- a real command round-trip succeeds and the status word reads
                  back cleanly.
  3. RESET     -- soft_reset() completes (clears runtime state, no reprogram).
  4. GEOMETRY  -- probes SPIKE_WORDS, identifying N_MAX to a 32-neuron band.
                  NUM_LANES/weight_w/data_w are not exposed by the current
                  protocol and are reported as assumptions rather than facts.
  5. DATASETS  -- for each requested dataset: run every test paper on-chip and
                  compare against the fixed-point-faithful software reference
                  paper-by-paper. Reports one-vs-rest and top-1 for BOTH, and
                  the per-paper agreement count.
  6. TIMING    -- reads the on-chip cycle counters for the last tick and
                  reports microseconds per timestep against the 1 ms budget.

Exit code is 0 only if every executed check passed. Nothing here reprograms
the board or writes a bitstream: it is read/compare only, apart from loading
networks into volatile device state.

The citation datasets need the external sgnn-superneuro checkout; point at it
with SGNN_REPO or --sgnn-repo (see USER_MANUAL section 3).
"""
# A validation sweep must turn every unexpected board/dataset failure into a
# recorded FAIL and still close the serial device. Broad catches below are the
# command's deliberate fault-containment boundary.
# ruff: noqa: BLE001, S110
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, connect, get_board, get_dataset, list_datasets
from .runtime import OP_READ_OUTPUT, InferEngineError


def _distribution_version() -> str:
    try:
        return importlib.metadata.version("superneuromat")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"

# datasets that actually fit each board, in ascending size. Used when
# --datasets is not given.
_DEFAULT_DATASETS = {
    "basys3": ["microseer"],
    "sp701": ["microseer"],
    "zcu104": ["miniseer", "cora", "citeseer"],
}


class Check:
    """One named check plus its outcome, so the report is uniform."""

    def __init__(self, name: str):
        self.name = name
        self.status = "SKIP"        # PASS | FAIL | SKIP
        self.detail: dict = {}
        self.message = ""
        self.seconds = 0.0

    def as_dict(self) -> dict:
        return {"check": self.name, "status": self.status,
                "message": self.message, "seconds": round(self.seconds, 2),
                **({"detail": self.detail} if self.detail else {})}


def _run(check: Check, fn):
    """Execute fn(check); record PASS/FAIL and timing. Never raises."""
    t0 = time.time()
    try:
        fn(check)
        if check.status == "SKIP":
            check.status = "PASS"
    except Exception as exc:
        check.status = "FAIL"
        check.message = f"{type(exc).__name__}: {exc}"
        check.detail.setdefault("traceback", traceback.format_exc(limit=6))
    check.seconds = time.time() - t0
    icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[check.status]
    print(f"  [{icon}] {check.name}"
          + (f" -- {check.message}" if check.message else "")
          + f"  ({check.seconds:.1f}s)", flush=True)
    return check.status == "PASS"


def _port_geometry(port: str, baud: int) -> int | None:
    """SPIKE_WORDS of whatever is on ``port``, or None if it isn't a live
    STDP engine. Read-only and never raises."""
    from .runtime import InferEngineStdpDevice
    try:
        dev = InferEngineStdpDevice(port, baud, n_max=4096, num_lanes=8, timeout=0.4)
    except Exception:
        return None
    try:
        return probe_spike_words(dev)
    except Exception:
        return None
    finally:
        try:
            dev.close()
        except Exception:
            pass


def _check_port(check: Check, port: str, board: str | None = None,
                want_words: int | None = None) -> str:
    """Resolve the serial port, preferring one that is actually running the
    geometry ``board`` expects.

    Plain VID:PID autodetection cannot tell WHICH board a responding port
    belongs to. With two boards attached (observed 2026-07-31: SP701 on COM5,
    ZCU104 on COM13), ``--board zcu104`` connected to the SP701 -- caught by
    the geometry check, but only after the fact. When the expected SPIKE_WORDS
    is known, probe the candidates and pick the matching one; fall back to
    ordinary autodetection when nothing matches, so the geometry check still
    reports the mismatch rather than this silently guessing.
    """
    from ._transport import autodetect_port, list_board_ports
    if port != "auto":
        check.message = f"using {port} (explicit)"
        check.detail["port"] = port
        return port

    found = list_board_ports()
    check.detail["candidates"] = [f"{d} ({desc})" for d, desc in found]

    if want_words is not None and len(found) > 1:
        from .boards import get_board
        baud = get_board(board).baud if board else 4_000_000
        seen = {}
        for dev_name, _desc in found:
            w = _port_geometry(dev_name, baud)
            if w is not None:
                seen[dev_name] = w
            if w == want_words:
                check.detail["probed_geometry"] = seen
                check.message = (f"resolved {dev_name} by geometry "
                                 f"(SPIKE_WORDS={w}, matches {board})")
                check.detail["port"] = dev_name
                return dev_name
        if seen:
            check.detail["probed_geometry"] = seen
            check.detail["note"] = (
                f"no candidate reported SPIKE_WORDS={want_words}; another "
                "board may be attached, or this one needs reprogramming")

    resolved = autodetect_port(required=True)
    check.message = f"resolved {resolved}"
    check.detail["port"] = resolved
    return resolved


def probe_spike_words(dev, limit: int = 4096) -> int:
    """Measure SPIKE_WORDS from the RUNNING bitstream, read-only.

    ``OP_READ_OUTPUT`` answers ST_ADDR_RANGE for any word index >=
    SPIKE_WORDS, and SPIKE_WORDS = ceil(N_MAX/32) is fixed at synthesis
    (``snm_infer_cmd_ctrl_stdp.v``). Binary-searching that boundary therefore
    reports the geometry the HARDWARE actually has, independent of anything
    the host believes. Reads only -- no device state is modified.
    """
    from ._transport import ST_ADDR_RANGE, ST_OK

    def ok(word: int) -> bool:
        status, _ = dev.command(OP_READ_OUTPUT, 0, word, 0)
        if status == ST_OK:
            return True
        if status == ST_ADDR_RANGE:
            return False
        # The STDP controller emits only 0x00..0x04 (snm_infer_cmd_ctrl_stdp.v).
        # Anything else means the board is not running an STDP lane-engine
        # bitstream at all -- most often the classic single-core design, whose
        # status set includes OUT_NOT_READY (0x08) and QUEUE_* (0x06/0x07).
        # Say so plainly instead of reporting a bare hex code.
        extra = ""
        if status in (0x06, 0x07, 0x08):
            extra = (" -- that status belongs to the CLASSIC (non-STDP) engine, "
                     "which this package does not drive. The board is running a "
                     "different bitstream; program an STDP build first "
                     "(spikeengine.program.program(board)).")
        raise InferEngineError(
            f"READ_OUTPUT word {word} returned status 0x{status:02x}{extra}")

    if not ok(0):
        raise InferEngineError("READ_OUTPUT word 0 rejected -- device is not "
                               "responding as a spike-engine bitstream")
    lo, hi = 0, 1
    while hi < limit and ok(hi):
        lo, hi = hi, hi * 2
    while lo + 1 < hi:                  # largest index that still answers OK
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo + 1                        # count = last valid index + 1


def expected_n_max(board: str, dataset: str | None) -> tuple[int, str]:
    """The N_MAX the LOADED BITSTREAM should have, and where that came from.

    A dataset does not always imply a dataset-sized build. When
    ``Dataset.bitstream_path(board)`` is None the dataset runs on the board's
    own packaged bitstream -- microseer on Basys3 is exactly this: a
    90-neuron network running on the N=256 default build. Expecting N=90 there
    would fail correct usage. Only a dataset with its OWN bitstream implies a
    dataset-sized N_MAX.
    """
    if dataset is None:
        return get_board(board).n_max, f"board {board!r} packaged default"
    ds = get_dataset(dataset)
    try:
        own = ds.bitstream_path(board)
    except KeyError:
        own = None
    if own is None:
        return (get_board(board).n_max,
                f"board {board!r} packaged default (dataset {dataset!r} runs on it)")
    return ds.neurons, f"dataset {dataset!r} bitstream"


def _check_geometry(check: Check, dev, board: str, dataset: str | None):
    """Verify the LOADED BITSTREAM matches what the host is about to assume.

    This must probe the hardware, not re-read the catalogue: `connect()` sets
    dev.n_max/num_lanes FROM the catalogue, so comparing those against the
    catalogue is tautological and can never fail (an earlier version of this
    function did exactly that and was worthless). Instead measure SPIKE_WORDS
    off the device and compare against ceil(expected_n_max/32).

    A wrong bitstream does not raise on the wire -- it silently misaddresses
    synapses and truncates weights -- so this is the one check standing
    between a mismatched board and hours of corrupt results.

    Granularity note: SPIKE_WORDS pins N_MAX only to a 32-neuron band, so
    this detects a wrong BUILD (e.g. N=1024 default vs N=2116 miniseer), not
    a discrepancy smaller than 32 neurons. num_lanes/weight_w/data_w are not
    reported by the wire protocol at all and remain unverifiable from the
    host -- recorded here as assumed, not confirmed.
    """
    want_n, source = expected_n_max(board, dataset)
    if dataset:
        ds = get_dataset(dataset)
        want_k, want_w, want_d = ds.num_lanes, ds.weight_w, ds.data_w
    else:
        b = get_board(board)
        want_k, want_w, want_d = b.num_lanes, b.weight_w, b.data_w

    expected_words = (want_n + 31) // 32
    actual_words = probe_spike_words(dev)
    lo_n, hi_n = (actual_words - 1) * 32 + 1, actual_words * 32

    check.detail = {
        "source": source,
        "expected_n_max": want_n,
        "expected_spike_words": expected_words,
        "measured_spike_words": actual_words,
        "measured_n_max_range": [lo_n, hi_n],
        "assumed_not_probed": {"num_lanes": want_k,
                               "weight_w": want_w, "data_w": want_d},
    }
    if actual_words != expected_words:
        raise AssertionError(
            f"BITSTREAM MISMATCH vs {source}: hardware reports SPIKE_WORDS="
            f"{actual_words} (N_MAX in {lo_n}..{hi_n}), but {want_n} neurons "
            f"were expected (SPIKE_WORDS={expected_words}). The board is "
            "running a different build -- reprogram it before trusting any "
            "result.")
    check.message = (f"hardware SPIKE_WORDS={actual_words} "
                     f"(N_MAX {lo_n}..{hi_n}) matches {source} (N={want_n})")


def _validate_dataset(check: Check, board: str, dataset: str, port: str,
                      sgnn_repo: str | None, num_test: int | None):
    """Full on-chip run for one dataset, compared per paper against the
    fixed-point-faithful software reference."""
    from .examples import citation_gnn_fpga as cg

    repo = cg._find_sgnn_repo(sgnn_repo)
    sys.path.insert(0, str(repo))
    import gnn_citation_networks as G
    import xyaml as yaml

    graph, _cfg = cg.build_graph(G, yaml, repo, dataset, cg.GRAPH_WEIGHT)
    papers = graph.selected_papers[:num_test] if num_test else graph.selected_papers
    ds = get_dataset(dataset)

    sw_res, hw_res, agree, mismatches = [], [], 0, []
    with connect(port=port, board=board, dataset=dataset) as dev:
        dev.clear_error()
        for i, pid in enumerate(papers):
            sw = cg.software_infer(G, graph, pid)
            hw = cg.hardware_infer(_se_module(), dev, graph, pid,
                                   syn_cap_per_lane=ds.syn_cap_per_lane)
            sw_res.append((sw, 0))
            hw_res.append((hw, 0))
            if sw[1] == hw[1]:
                agree += 1
            else:
                mismatches.append({"paper": str(pid), "sw": list(map(str, sw[1])),
                                   "hw": list(map(str, hw[1]))})
            if (i + 1) % 20 == 0:
                print(f"      {i + 1}/{len(papers)} papers ...", flush=True)
        timing = cg.measure_timestep(dev)

    sw_acc = G.calculate_accuracy(sw_res, graph.resolution_order, "SW")
    hw_acc = G.calculate_accuracy(hw_res, graph.resolution_order, "HW")
    n = len(papers)
    # Top-1 is the PRIMARY metric (2026-07-31). These tasks are multiclass
    # (a paper gets one topic out of k), and one-vs-rest treats each topic as
    # an independent binary problem, so every topic the model correctly did
    # NOT predict counts as a true negative. With k topics the denominator is
    # n_papers * k and true negatives dominate: a classifier that is right
    # 42% of the time still scores 0.64 one-vs-rest. It is retained only
    # because it is the figure the SGNN benchmark itself persists, so results
    # stay comparable with that repo -- not because it describes accuracy.
    check.detail = {
        "papers": n,
        "sw_top1_accuracy": round(int(sw_acc.legacy) / n, 4),
        "hw_top1_accuracy": round(int(hw_acc.legacy) / n, 4),
        "sw_top1": f"{int(sw_acc.legacy)}/{n}",
        "hw_top1": f"{int(hw_acc.legacy)}/{n}",
        "per_paper_agreement": f"{agree}/{n}",
        "catalogue_top1": f"{ds.sw_top1[0]}/{ds.sw_top1[1]}" if ds.sw_top1 else None,
        # secondary, for comparability with the SGNN benchmark only
        "sw_one_vs_rest": round(float(sw_acc.accuracy), 4),
        "hw_one_vs_rest": round(float(hw_acc.accuracy), 4),
        "catalogue_one_vs_rest": ds.sw_accuracy,
        "metric_note": (
            "top-1 is the accuracy figure for this multiclass task; "
            "one-vs-rest is inflated by true negatives and is reported only "
            "to match the SGNN benchmark's own metric"),
        "us_per_timestep": round(timing["total_us"], 1),
        "within_1ms": timing["within_1ms"],
    }
    if mismatches:
        check.detail["mismatches"] = mismatches[:10]
    check.message = (f"{agree}/{n} SW==HW, "
                     f"top-1 {int(hw_acc.legacy)}/{n} "
                     f"({int(hw_acc.legacy) / n:.1%}), "
                     f"one-vs-rest {hw_acc.accuracy:.4f} (inflated, see note), "
                     f"{timing['total_us']:.0f} us/tick")
    if agree != n:
        raise AssertionError(
            f"{n - agree}/{n} papers disagree between software and hardware")
    # Regression against the recorded catalogue value, when there is one and the
    # full test set was run (a truncated run legitimately differs).
    if num_test is None and abs(float(hw_acc.accuracy) - ds.sw_accuracy) > 1e-4:
        raise AssertionError(
            f"one-vs-rest {hw_acc.accuracy:.4f} != catalogue {ds.sw_accuracy} "
            "-- results drifted from the recorded value")


def _se_module():
    """`citation_gnn_fpga.hardware_infer` takes the package module itself as
    its first argument (it calls se.load_network / se.run_schedule)."""
    import superneuromat.spikeengine as se
    return se


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Hardware validation sweep for a programmed SpikeEngine board.")
    ap.add_argument("--board", required=True, choices=["basys3", "sp701", "zcu104"])
    ap.add_argument("--port", default="auto")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated; default = every dataset that fits this board")
    ap.add_argument("--num-test", type=int, default=None,
                    help="limit papers per dataset (a truncated run skips the "
                         "catalogue-regression assertion)")
    ap.add_argument("--quick", action="store_true",
                    help="link/reset/geometry only -- no dataset runs")
    ap.add_argument("--sgnn-repo", default=None)
    ap.add_argument("--report", default=None,
                    help="report path (default: hw_validation_<board>_<UTC>.json)")
    args = ap.parse_args(argv)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        for d in datasets:
            if d not in list_datasets():
                ap.error(f"unknown dataset {d!r}; known: {list_datasets()}")
    else:
        datasets = list(_DEFAULT_DATASETS.get(args.board, []))
    if args.quick:
        datasets = []

    started = datetime.now(timezone.utc)
    print(f"SpikeEngine hardware validation -- board={args.board} "
          f"port={args.port} datasets={datasets or '(none, quick mode)'}")
    dist_version = _distribution_version()
    print(f"superneuromat {dist_version} | spikeengine {__version__} | "
          f"{platform.platform()} | {started.isoformat()}\n")

    checks: list[Check] = []
    resolved = {"port": args.port}
    measured: dict[str, int] = {}      # SPIKE_WORDS probed off the loaded build

    def skipped(name: str, because: str) -> Check:
        """Record a check that was NOT executed because a prerequisite failed.

        Every stage below gates on the ones it depends on (2026-07-31 fix):
        previously only the LINK result was propagated, so dataset checks
        could run -- and be reported -- after soft-reset or the geometry
        check had failed, i.e. against a device in an unknown state or
        running the wrong bitstream. A skipped check is never counted as
        passed, and carries the reason so the report explains itself.
        """
        c = Check(name)
        c.status = "SKIP"
        c.message = f"prerequisite failed: {because}"
        checks.append(c)
        print(f"  [SKIP] {name} -- {c.message}", flush=True)
        return c

    c = Check("port-discovery")
    checks.append(c)
    # What geometry should the target board be running? Used to pick the right
    # port when several boards are attached. For a sweep with datasets, the
    # first dataset's build is the expectation; otherwise the board default.
    _want_n, _ = expected_n_max(args.board, datasets[0] if datasets else None)
    _want_words = (_want_n + 31) // 32
    port_ok = _run(c, lambda ch: resolved.__setitem__(
        "port", _check_port(ch, args.port, args.board, _want_words)))
    resolved_port = resolved["port"]

    link_ok = reset_ok = geo_ok = False

    if not port_ok:
        for name in ("link-round-trip", "soft-reset", "geometry-board-default"):
            skipped(name, "port-discovery")
    else:
        c = Check("link-round-trip")
        checks.append(c)

        def _link(ch):
            with connect(port=resolved_port, board=args.board) as dev:
                dev.clear_error()
                status = dev.read_status()      # returns the status word (int)
                ch.detail["status_word"] = int(status)
                ch.message = f"status=0x{int(status):02x}"
        link_ok = _run(c, _link)

        if not link_ok:
            for name in ("soft-reset", "geometry-board-default"):
                skipped(name, "link-round-trip")
        else:
            c = Check("soft-reset")
            checks.append(c)

            def _reset(ch):
                with connect(port=resolved_port, board=args.board) as dev:
                    dev.soft_reset()
                    ch.message = "runtime state cleared"
            reset_ok = _run(c, _reset)

            if not reset_ok:
                skipped("identify-loaded-build", "soft-reset")
            else:
                # Probe the loaded build ONCE (2026-07-31 fix). The previous
                # version ran a board-default geometry check AND a per-dataset
                # geometry check, which are mutually exclusive: only one
                # bitstream is loaded at a time, so on a dataset build the
                # board-default check failed and skipped everything, while on
                # the default build every dataset check failed. Identify what
                # is actually loaded, then judge each dataset against it.
                c = Check("identify-loaded-build")
                checks.append(c)

                def _identify(ch):
                    with connect(port=resolved_port, board=args.board) as dev:
                        words = probe_spike_words(dev)
                    measured["words"] = words
                    lo, hi = (words - 1) * 32 + 1, words * 32
                    ch.detail["measured_spike_words"] = words
                    ch.detail["measured_n_max_range"] = [lo, hi]

                    # name the build, if it matches something we ship
                    # Only datasets actually validated on THIS board are
                    # candidates. Without the boards() filter, expected_n_max()
                    # falls back to the board default for every other dataset,
                    # so a Basys3 probe listed citeseer/cora/pubmed as "N=256"
                    # -- pure noise that reads like they are loadable here.
                    cands = {f"board default (N={get_board(args.board).n_max})":
                             (get_board(args.board).n_max + 31) // 32}
                    for d in list_datasets():
                        if args.board not in get_dataset(d).boards():
                            continue
                        n, _src = expected_n_max(args.board, d)
                        cands[f"dataset {d} (N={n})"] = (n + 31) // 32
                    matches = [k for k, w in cands.items() if w == words]
                    ch.detail["matching_known_builds"] = matches
                    ch.message = (f"SPIKE_WORDS={words} (N_MAX {lo}..{hi})"
                                  + (f" -> {', '.join(matches)}" if matches
                                     else " -> no shipped build has this geometry"))
                    if not matches:
                        raise AssertionError(
                            f"loaded build (SPIKE_WORDS={words}, N_MAX {lo}..{hi}) "
                            "matches no bitstream this package ships. The board is "
                            "running something else entirely.")
                geo_ok = _run(c, _identify)

    prereq_failure = (None if (port_ok and link_ok and reset_ok and geo_ok)
                      else "port-discovery" if not port_ok
                      else "link-round-trip" if not link_ok
                      else "soft-reset" if not reset_ok
                      else "identify-loaded-build")

    for ds_name in datasets:
        if prereq_failure:
            skipped(f"dataset-{ds_name}", prereq_failure)
            continue

        # Does the CURRENTLY loaded build match what this dataset needs? If
        # not, that is not a failure -- it means the board is programmed for a
        # different dataset and must be reprogrammed to validate this one.
        # Reporting it as SKIP (which still blocks an overall pass) is honest;
        # reporting FAIL would imply a defect that does not exist.
        want_n, src = expected_n_max(args.board, ds_name)
        want_words = (want_n + 31) // 32
        got_words = measured.get("words")
        if got_words != want_words:
            skipped(f"dataset-{ds_name}",
                    f"loaded build has SPIKE_WORDS={got_words}, but {ds_name} "
                    f"needs {want_words} (N={want_n}, from {src}) -- reprogram "
                    f"the board with this dataset's bitstream to validate it")
            continue

        c = Check(f"dataset-{ds_name}")
        checks.append(c)
        print(f"    running {ds_name} on hardware (this takes a while) ...", flush=True)
        _run(c, lambda ch, _d=ds_name: _validate_dataset(
            ch, args.board, _d, resolved_port, args.sgnn_repo, args.num_test))

    passed = sum(1 for c in checks if c.status == "PASS")
    failed = sum(1 for c in checks if c.status == "FAIL")
    skipped_n = sum(1 for c in checks if c.status == "SKIP")
    finished = datetime.now(timezone.utc)

    report = {
        "package_version": dist_version,
        "distribution_version": dist_version,
        "spikeengine_version": __version__,
        "board": args.board,
        "port": resolved_port,
        "datasets": datasets,
        "num_test": args.num_test,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "summary": {"passed": passed, "failed": failed,
                    "skipped": skipped_n, "total": len(checks),
                    # A sweep is only a pass if everything ran AND passed.
                    # Skipped checks mean the board was not fully exercised,
                    # so the overall verdict cannot be "validated".
                    "validated": failed == 0 and skipped_n == 0},
        "checks": [c.as_dict() for c in checks],
    }
    path = Path(args.report or
                f"hw_validation_{args.board}_{started.strftime('%Y%m%dT%H%M%SZ')}.json")
    path.write_text(json.dumps(report, indent=2))

    print(f"\n{passed}/{len(checks)} checks passed"
          + (f", {failed} FAILED" if failed else "")
          + (f", {skipped_n} SKIPPED (not run)" if skipped_n else ""))
    if failed == 0 and skipped_n:
        print("VERDICT: INCOMPLETE -- some checks never ran; this is NOT a pass.")
    elif failed == 0:
        print("VERDICT: VALIDATED")
    else:
        print("VERDICT: FAILED")
    print(f"report: {path.resolve()}")
    # Non-zero for failures AND for an incomplete sweep: a caller must not be
    # able to read exit 0 as "hardware validated" when checks were skipped.
    return 0 if (failed == 0 and skipped_n == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
