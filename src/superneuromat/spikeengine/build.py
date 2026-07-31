"""Vivado RTL-to-bitstream build backend for the STDP SpikeEngine.

Unlike ``program.py`` (loads an already-built ``.bit``), this runs a REAL
synth -> place -> route -> write_bitstream flow via each board's bundled Tcl
build script (``scripts/build_infer_<board>*.tcl``). This is the OPT-IN,
heavy (10-40+ minutes, real Vivado synthesis) path -- never invoked silently
-- as an alternative to ``program.program(board)``, which just loads the
packaged bitstream in seconds.

Ported from the proven pattern already used for the shipping (inference-only)
lane engine in ``superneuromat.spikeengine.fpga_vivado``
(``build_lane_engine_bitstream`` / ``build_fpga_bitstream``): the report
regex parsers, and trusting the build script's own ``BUILD_DONE`` sentinel +
a written ``.bit`` file over the raw subprocess exit code (Vivado has been
observed to crash during its OWN post-completion teardown -- e.g. an
``EXCEPTION_ACCESS_VIOLATION`` on exit -- after the real work already
succeeded; treating that as a failure would be wrong).

Each board's Tcl script here is a FIXED single configuration (not a
parametrized sweep template like the lane engine's), so this module is
simpler: no RTL-source scratch-copy or config-file generation step, just
one Tcl script per board, already bundled under ``scripts/``.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

from .program import find_vivado

_PKG_ROOT = Path(__file__).parent

_BUILD_SCRIPT_BY_BOARD = {
    "basys3": "build_infer_basys3_stdp_cap256x8.tcl",
    "sp701": "build_infer_sp701_stdp_maxcap352x8.tcl",
    "zcu104": "build_infer_zcu104_stdp_maxcap1024x16.tcl",
}


def rtl_source_dir() -> Path:
    """Directory holding the RTL sources every build script reads from --
    available even without running a build (e.g. to just read/inspect the
    Verilog, independent of whether you use the packaged bitstream or build
    a fresh one). Same tree ``build_bitstream()`` synthesizes from; nothing
    is generated at build time that isn't already here."""
    return _PKG_ROOT / "rtl"


def build_script_path(board: str) -> Path:
    """Path to the Tcl build script `build_bitstream(board)` runs -- useful
    if you want to invoke Vivado yourself, or just read the script to see
    exactly what it does (RTL file list, -generic overrides, pipeline
    directives) before running it."""
    if board not in _BUILD_SCRIPT_BY_BOARD:
        raise KeyError(f"unknown board {board!r}; known boards: {sorted(_BUILD_SCRIPT_BY_BOARD)}")
    return _PKG_ROOT / "scripts" / _BUILD_SCRIPT_BY_BOARD[board]


class VivadoError(RuntimeError):
    """Vivado build failure."""


class VivadoNotFoundError(VivadoError):
    """Vivado is required to build from RTL but was not found."""


class BuildResult:
    """Structured result of a :func:`build_bitstream` run.

    ``outdir`` is the Vivado build script's own output directory -- it holds
    everything the flow generated, not just the bitstream: ``post_synth.dcp``
    / ``post_route.dcp`` checkpoints and the full ``post_route_*.rpt`` report
    set (timing/utilization/DRC/power/worst-paths). All of that is real
    output from THIS build (not just the final .bit), and stays on disk after
    the call returns -- nothing here is deleted or hidden.
    """

    def __init__(self, *, success: bool, bitstream_path: Path | None, outdir: Path | None,
                 rtl_dir: Path, wns_ns: float | None,
                 lut: int | None, ff: int | None, bram: int | None, uram: int | None,
                 drc_errors: int | None, drc_warnings: int | None,
                 log_path: Path, log_tail: str):
        self.success = success
        self.bitstream_path = bitstream_path
        self.outdir = outdir
        self.rtl_dir = rtl_dir
        self.wns_ns = wns_ns
        self.lut = lut
        self.ff = ff
        self.bram = bram
        self.uram = uram
        self.drc_errors = drc_errors
        self.drc_warnings = drc_warnings
        self.log_path = log_path
        self.log_tail = log_tail

    def __repr__(self) -> str:
        return (f"BuildResult(success={self.success}, wns_ns={self.wns_ns}, "
                f"lut={self.lut}, ff={self.ff}, bram={self.bram}, uram={self.uram}, "
                f"drc_errors={self.drc_errors}, bitstream_path={self.bitstream_path}, "
                f"outdir={self.outdir})")


# ---- report parsers (ported verbatim from superneuromat.spikeengine.fpga_vivado,
# validated against real Vivado reports this session) ----

def _parse_wns(timing_rpt: Path) -> float | None:
    text = timing_rpt.read_text(errors="replace")
    m = re.search(r"WNS\(ns\).*?\n\s*-+.*?\n\s*(-?\d+\.\d+)", text, re.DOTALL)
    return float(m.group(1)) if m else None


def _util_row(text: str, site_name: str) -> tuple[int | float, float] | None:
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7:
            continue
        if parts[1] == site_name:
            try:
                used = float(parts[2])
                return (int(used) if used.is_integer() else used), float(parts[6])
            except ValueError:
                continue
    return None


def _parse_util(util_rpt: Path) -> dict:
    text = util_rpt.read_text(errors="replace")
    out = {"lut": None, "ff": None, "bram": None, "uram": None}
    for site_names, key in [(("Slice LUTs", "CLB LUTs"), "lut"),
                             (("Slice Registers", "CLB Registers"), "ff"),
                             (("Block RAM Tile",), "bram"),
                             (("URAM",), "uram")]:
        for site in site_names:
            row = _util_row(text, site)
            if row:
                out[key] = row[0]
                break
    return out


def _parse_drc(drc_rpt: Path) -> dict:
    text = drc_rpt.read_text(errors="replace")
    out = {"drc_errors": 0, "drc_warnings": 0}
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5 or parts[1] in ("Rule", "----"):
            continue
        severity, checks = parts[2], parts[4]
        if not checks.isdigit():
            continue
        n = int(checks)
        if severity == "Error":
            out["drc_errors"] += n
        elif severity in ("Warning", "Critical Warning"):
            out["drc_warnings"] += n
    return out


_GENERIC_RE = re.compile(r"-generic\s+(\w+)=(\d+)")


def script_generics(board: str) -> dict[str, int]:
    """RTL parameters the board's build script actually passes to Vivado.

    Parsed from the Tcl rather than assumed, because the packaged bitstream
    and a fresh rebuild can disagree -- see ``check_rebuild_matches_catalogue``.
    """
    text = build_script_path(board).read_text(errors="replace")
    return {k: int(v) for k, v in _GENERIC_RE.findall(text)}


def check_rebuild_matches_catalogue(board: str) -> dict:
    """Compare what a rebuild WOULD produce against what ``boards.py`` says
    the packaged bitstream for ``board`` is.

    They can differ. The sp701/zcu104 packaged bitstreams are 8-bit builds:
    their ``-generic WEIGHT_W=16 DATA_W=24`` was silently dropped because the
    board tops did not declare or forward those parameters. That was fixed in
    the RTL on 2026-07-29, so a rebuild TODAY is wide (W16/D24) while the
    catalogue still describes the packaged 8-bit build.

    This matters because the widths set the wire packing offsets. Building,
    programming the result, then connecting with ``connect(board=...)`` would
    give an 8-bit host driving a 16-bit device -- every weight and source id
    packed at the wrong offset, corrupted silently, which is exactly what
    ``boards.py``'s own note warns about in the other direction.

    Returns a dict with ``matches`` plus the two parameter sets.
    """
    from .boards import get_board

    g = script_generics(board)
    b = get_board(board)
    catalogue = {"WEIGHT_W": b.weight_w, "DATA_W": b.data_w,
                 "SPIKE_MON_BASE": b.spike_mon_base}
    rebuild = {k: g[k] for k in catalogue if k in g}
    differing = {k: (catalogue[k], rebuild[k])
                 for k in rebuild if catalogue[k] != rebuild[k]}
    return {
        "board": board,
        "matches": not differing,
        "catalogue": catalogue,
        "rebuild_would_produce": rebuild,
        "differing": differing,     # {param: (catalogue, rebuild)}
    }


def build_bitstream(board: str = "basys3", *, vivado_path: str | None = None,
                    timeout: int = 2400) -> BuildResult:
    """Run a real Vivado synth/place/route/write_bitstream flow for ``board``.

    Raises :class:`VivadoNotFoundError` if Vivado isn't found (pointing at the
    pre-packaged-bitstream fallback, ``program.program(board)``), or
    :class:`VivadoError` on any build failure. This is OPT-IN and heavy
    (10-40+ minutes real synthesis) -- never invoked silently.

    Does NOT overwrite the package's own packaged bitstream (``boards.py``'s
    ``bitstream_path()``) -- the whole point of this function is to produce a
    SEPARATE, freshly-built bitstream so a caller can compare "build from
    RTL" against "use the packaged one" (e.g. ``program.program(board,
    bitstream=result.bitstream_path)`` to load the fresh build instead).

    Vivado runs in a scratch directory outside the installed package
    (``$SPIKEENGINE_BUILD_DIR``, else ``<tempdir>/spikeengine_build``), so
    nothing is written into site-packages and the working path stays short
    enough for Windows. The build log is written there as
    ``run_<board>/_build_<board>.log``; the returned
    :class:`BuildResult` carries the paths.

    WARNS if the rebuild would not reproduce the packaged bitstream (see
    :func:`check_rebuild_matches_catalogue`) -- true today for sp701 and
    zcu104, whose packaged builds are 8-bit.
    """
    if board not in _BUILD_SCRIPT_BY_BOARD:
        raise KeyError(f"unknown board {board!r}; known boards: {sorted(_BUILD_SCRIPT_BY_BOARD)}")

    # Warn BEFORE spending 10-40 minutes on a build whose result the host
    # catalogue would then mis-describe (2026-07-31). A silent width mismatch
    # corrupts every weight, so this is loud and it comes first.
    _chk = check_rebuild_matches_catalogue(board)
    if not _chk["matches"]:
        diffs = ", ".join(f"{k}: catalogue={c} rebuild={r}"
                          for k, (c, r) in _chk["differing"].items())
        warnings.warn(
            f"build_bitstream({board!r}) will NOT reproduce the packaged "
            f"bitstream: {diffs}. The packaged {board} build predates the "
            "2026-07-29 board-top parameter-forwarding fix, so it is an 8-bit "
            f"build; a rebuild today is wide. boards.py still describes the "
            f"PACKAGED one, so connect(board={board!r}) after programming this "
            "result would drive a wide device with an 8-bit host and corrupt "
            "every weight. Pass the real widths explicitly to connect() "
            "(weight_w=/data_w=) when using this build.",
            RuntimeWarning, stacklevel=2)

    vivado = vivado_path or find_vivado()
    if vivado is None:
        raise VivadoNotFoundError(
            "Vivado was not found. Install Vivado 2025.2, pass vivado_path=..., "
            "or use the packaged bitstream instead: spikeengine.program.program(board).")

    scripts_dir = _PKG_ROOT / "scripts"
    script_path = scripts_dir / _BUILD_SCRIPT_BY_BOARD[board]
    if not script_path.exists():
        raise VivadoError(f"build script not found: {script_path}")

    # Run Vivado from a WRITABLE scratch directory, never from the package's
    # own scripts/ dir (2026-07-30 fix). Vivado drops `.Xil/` and journal
    # files into its working directory: doing that inside an installed package
    # pollutes site-packages, fails outright if site-packages is read-only,
    # and -- the way this was actually caught -- blew Windows' path-length
    # limit, because a venv site-packages path plus Vivado's own nested temp
    # dir name exceeds the maximum ("ERROR: [Common 17-1373] The path length
    # of the Xilinx temporary directory ... is too long for Windows").
    # SPIKEENGINE_BUILD_DIR overrides the location; the Tcl build scripts read
    # the same variable, so both halves of the flow stay together.
    build_root = Path(os.environ.get("SPIKEENGINE_BUILD_DIR")
                      or Path(tempfile.gettempdir()) / "spikeengine_build")
    run_dir = build_root / f"run_{board}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [vivado, "-mode", "batch", "-nojournal",
           "-tempDir", str(run_dir),
           "-source", str(script_path)]
    try:
        proc = subprocess.run(
            cmd, cwd=run_dir, capture_output=True, text=True,
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise VivadoError(
            f"Vivado build timed out after {timeout}s building {board}."
        ) from e

    log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = run_dir / f"_build_{board}.log"
    log_path.write_text(log_text, errors="replace")

    build_done = "BUILD_DONE" in log_text
    # Anchored to line-start: Vivado batch mode echoes the Tcl source line
    # (`# puts "BUILD_DONE ... outdir=$outdir"`) before executing it, which
    # contains the literal, unexpanded text "BUILD_DONE ... outdir=$outdir"
    # -- an earlier, unanchored version of this regex matched THAT line
    # instead of the real output line below it, extracting a bogus literal
    # "$outdir" path (found via basys3/sp701 builds that had actually
    # succeeded -- .bit on disk, DRC clean -- but this bug made them look
    # like build failures with bitstream_written=False).
    m = re.search(r"^BUILD_DONE.*?bitstream=(.*?) outdir=(.*)$", log_text, re.MULTILINE)
    bitstream_path = Path(m.group(1).strip()) if m else None
    outdir = Path(m.group(2).strip()) if m else None
    # Retry .exists(): OneDrive cloud-sync has a documented history in this project
    # of the on-disk file lagging a moment behind the writing process reporting
    # success (see program.py's docstring on the same issue) -- a bare single
    # check here produced a false-negative "build failed" for a bitstream that
    # Vivado had already written successfully (confirmed on disk seconds later).
    bitstream_written = False
    if bitstream_path is not None:
        for _ in range(10):
            if bitstream_path.exists():
                bitstream_written = True
                break
            time.sleep(1)

    # Trust the script's own completion sentinel + a written bitstream over the
    # raw exit code -- see module docstring (Vivado can crash on its way out
    # after the real work already succeeded).
    if not (build_done and bitstream_written):
        raise VivadoError(
            f"Vivado build failed for {board} (exit {proc.returncode}, "
            f"build_done={build_done}, bitstream_written={bitstream_written}).\n\n"
            f"Command:\n{' '.join(cmd)}\n\n"
            f"Log (tail):\n{log_text[-2000:]}\n\nFull log: {log_path}"
        )

    timing_rpt = outdir / "post_route_timing.rpt" if outdir else None
    util_rpt = outdir / "post_route_util.rpt" if outdir else None
    drc_rpt = outdir / "post_route_drc.rpt" if outdir else None
    wns = _parse_wns(timing_rpt) if timing_rpt and timing_rpt.exists() else None
    util = _parse_util(util_rpt) if util_rpt and util_rpt.exists() else {}
    drc = _parse_drc(drc_rpt) if drc_rpt and drc_rpt.exists() else {}

    return BuildResult(
        success=True, bitstream_path=bitstream_path, outdir=outdir, rtl_dir=_PKG_ROOT / "rtl",
        wns_ns=wns,
        lut=util.get("lut"), ff=util.get("ff"), bram=util.get("bram"), uram=util.get("uram"),
        drc_errors=drc.get("drc_errors"), drc_warnings=drc.get("drc_warnings"),
        log_path=log_path, log_tail=log_text[-2000:],
    )
