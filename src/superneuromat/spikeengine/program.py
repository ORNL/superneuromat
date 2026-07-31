"""Vivado-based board programming for the STDP SpikeEngine (default flow).

Volatile JTAG load (lost on power-cycle, does not touch flash) of a packaged
bitstream via ``vivado -mode batch``. Vivado is the project's chosen default
programmer for these boards. The bitstream is copied to a short local path
first (a documented workaround for OneDrive cloud-sync intermittently failing
Vivado's own file-existence check on synced paths).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .boards import get_board

_PKG_ROOT = Path(__file__).parent


def find_vivado() -> str | None:
    """Best-effort locate a vivado(.bat) launcher; returns a path or None."""
    for cand in (
        r"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat",
        r"C:\Xilinx\Vivado\2025.2\bin\vivado.bat",
    ):
        if Path(cand).exists():
            return cand
    exe = shutil.which("vivado") or shutil.which("vivado.bat")
    return exe


def program(board: str = "basys3", *, bitstream: str | Path | None = None,
            vivado_path: str | None = None, timeout: int = 300) -> None:
    """Volatile-program ``board`` with its packaged bitstream (or an explicit
    ``bitstream`` path) over Vivado JTAG. Raises RuntimeError on any failure.
    """
    spec = get_board(board)
    bit = Path(bitstream) if bitstream is not None else spec.bitstream_path()
    if not bit.exists():
        raise FileNotFoundError(f"bitstream not found: {bit}")
    vivado = vivado_path or find_vivado()
    if vivado is None:
        raise RuntimeError(
            "Vivado not found. Install Vivado 2025.2 or pass vivado_path=..., "
            "or program the packaged bitstream manually.")

    # find the board's programmer Tcl (the config-specific name varies per board,
    # e.g. program_infer_zcu104_stdp_maxcap1024x16.tcl). Prefer a board-specific
    # script; only fall back to the basys3 generic body if none exists.
    candidates = sorted(_PKG_ROOT.glob(f"scripts/program_infer_{board}_*.tcl"))
    if candidates:
        script = candidates[0]
    else:
        script = _PKG_ROOT / "scripts" / "program_infer_basys3_stdp_cap256x8.tcl"

    # Scratch dir for the Vivado run: short, non-OneDrive path (the documented
    # cloud-sync file-lock workaround) AND the working directory, so Vivado's
    # `.Xil/` and journal files land here instead of the caller's project or
    # the installed package (2026-07-30).
    #
    # NOT a TemporaryDirectory (2026-07-31): making the temp dir the process
    # cwd meant Windows still had a handle on it at cleanup, so the context
    # manager's rmtree raised
    # "PermissionError: [WinError 32] ... used by another process" AFTER the
    # board had already been programmed -- turning a successful program into
    # an exception. A persistent, reused scratch dir has no such teardown, and
    # matches build.py's SPIKEENGINE_BUILD_DIR convention.
    build_root = Path(os.environ.get("SPIKEENGINE_BUILD_DIR")
                      or Path(tempfile.gettempdir()) / "spikeengine_build")
    run_dir = build_root / f"program_{board}"
    run_dir.mkdir(parents=True, exist_ok=True)
    local_bit = run_dir / bit.name
    shutil.copy2(bit, local_bit)
    proc = subprocess.run(
        [vivado, "-mode", "batch", "-nojournal", "-tempDir", str(run_dir),
         "-source", str(script), "-tclargs", str(local_bit)],
        cwd=run_dir, capture_output=True, text=True, timeout=timeout, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    if "PROGRAM_DONE" not in out:
        tail = "\n".join(out.splitlines()[-25:])
        raise RuntimeError(f"Vivado programming did not report PROGRAM_DONE.\n{tail}")
