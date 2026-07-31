"""Provenance for the packaged bitstreams.

A shipped ``.bit`` is an opaque binary. Without a record tying it to the
sources in this package, a reviewer cannot tell whether the binary was built
from the RTL they are reading, and a user cannot tell whether their copy is
intact. This module provides both:

* ``bitstreams_manifest.json`` (alongside the bitstreams) records, per file,
  its SHA-256, size, target board, the RTL parameters it was built with, the
  Vivado build results (WNS, LUT/FF/BRAM/URAM), and the SHA-256 of every RTL
  source that went into it.
* ``verify_bitstreams()`` re-hashes the installed files and reports any that
  are missing, altered, or absent from the manifest.

Regenerate after building a new bitstream::

    python -m superneuromat.spikeengine.provenance --write

Honesty note: the manifest records what is *verifiable from this machine*.
Where a field could not be established (for example a build predating this
record, whose Vivado report directory no longer exists), it is written as
null with a ``notes`` entry rather than guessed. A null is a real statement --
"not established" -- and is preferable to a plausible-looking value nobody
checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_PKG_ROOT = Path(__file__).parent
MANIFEST_PATH = _PKG_ROOT / "bitstreams" / "bitstreams_manifest.json"

MANIFEST_VERSION = 1


def sha256_of(path: Path, _chunk: int = 1 << 20) -> str:
    """Byte-exact SHA-256. Used for bitstreams, which are binary."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of_text(path: Path) -> str:
    """SHA-256 of a text source with line endings normalised to LF.

    Git converts LF to CRLF when checking out on Windows, so the same RTL
    file hashes differently in a clone than in the tree it was committed
    from. A byte-exact digest would therefore report "RTL changed" on every
    Windows clone -- a false alarm that trains people to ignore the check.
    Verified 2026-07-31: snm_spi_slave.v is 223 LF in the source tree and 223
    CRLF in a fresh Windows clone; identical once normalised.

    Normalising means the digest attests to CONTENT, not to bytes on disk,
    which is the property that matters for "is this the RTL the bitstream was
    built from".
    """
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rtl_digest() -> dict:
    """SHA-256 of every RTL source in the package, plus a single combined
    digest over them in sorted path order.

    The combined digest is what a reviewer compares: if it differs from the
    value recorded for a bitstream, the RTL in this package is not the RTL
    that bitstream was built from.
    """
    rtl_root = _PKG_ROOT / "rtl"
    files = sorted(p for p in rtl_root.rglob("*")
                   if p.suffix in (".v", ".vh") and p.is_file())
    # Text digest (LF-normalised): survives Git's CRLF checkout conversion.
    per_file = {str(p.relative_to(rtl_root)).replace("\\", "/"): sha256_of_text(p)
                for p in files}
    combined = hashlib.sha256()
    for rel in sorted(per_file):
        combined.update(rel.encode())
        combined.update(per_file[rel].encode())
    return {"files": per_file, "combined_sha256": combined.hexdigest()}


def _build_records() -> dict:
    """Build parameters and measured results per shipped bitstream.

    Values come from this project's recorded Vivado runs (the numbers in
    boards.py / datasets.py, each re-read from the corresponding
    post_route_*.rpt). Fields that could not be established are null.
    """
    from .boards import BOARDS
    from .datasets import DATASETS

    rec: dict[str, dict] = {}

    for key, b in BOARDS.items():
        rec[b.bitstream] = {
            "board": key,
            "role": "board packaged default",
            "params": {"N_MAX": b.n_max, "NUM_LANES": b.num_lanes,
                       "WEIGHT_W": b.weight_w, "DATA_W": b.data_w,
                       "SPIKE_MON_BASE": b.spike_mon_base},
            "hardware_validated": b.hardware_validated,
            "hardware_validated_date": b.hardware_validated_date or None,
        }

    for key, ds in DATASETS.items():
        for board in ds.boards():
            try:
                p = ds.bitstream_path(board)
            except KeyError:
                continue
            if p is None:
                continue                     # runs on the board default
            rel = str(p.relative_to(_PKG_ROOT / "bitstreams")).replace("\\", "/")
            rec[rel] = {
                "board": board,
                "role": f"dataset build ({key})",
                "dataset": key,
                "params": {"N_MAX": ds.neurons, "NUM_LANES": ds.num_lanes,
                           "SYN_CAP_PER_LANE": ds.hardware_syn_cap_per_lane(board),
                           "WEIGHT_W": ds.weight_w, "DATA_W": ds.data_w},
                "build": {"wns_ns": ds.wns_ns},
                "hardware_validated": ds.hardware_validated,
            }
    return rec


def build_manifest() -> dict:
    """Assemble the manifest from the bitstreams currently in the package."""
    bit_root = _PKG_ROOT / "bitstreams"
    records = _build_records()
    entries = []
    for p in sorted(bit_root.rglob("*.bit")):
        rel = str(p.relative_to(bit_root)).replace("\\", "/")
        meta = records.get(rel, {})
        notes = []
        if not meta:
            notes.append(
                "no catalogue entry references this file; it is shipped but "
                "not reachable through boards.py or datasets.py")
        if meta.get("build", {}).get("wns_ns") is None and "dataset" in meta:
            notes.append("timing margin not recorded for this build")
        entries.append({
            "path": rel,
            "sha256": sha256_of(p),
            "size_bytes": p.stat().st_size,
            **meta,
            "notes": notes or None,
        })

    return {
        "manifest_version": MANIFEST_VERSION,
        "purpose": (
            "Ties each packaged .bit to the RTL in this package and to its "
            "recorded build results, so a reviewer can verify the binaries "
            "correspond to the shipped sources and a user can verify their "
            "copy is intact."),
        "rtl": rtl_digest(),
        "bitstreams": entries,
        "limitations": [
            ("Vivado version and the exact Tcl invocation are not recorded per "
            "bitstream: the builds predate this manifest and their scratch "
            "directories are not part of the repository. Rebuilding with "
            "scripts/build_*.tcl regenerates both, and a future manifest "
            "written by that flow can record them."),
            ("The RTL digest proves the shipped RTL is unchanged since the "
            "manifest was written. It does NOT by itself prove a given .bit "
            "was produced from exactly that RTL -- only a reproducible "
            "rebuild does that. Use build.build_bitstream() to reproduce a "
            "packaged board build and compare results."),
        ],
    }


def write_manifest(path: Path | None = None) -> Path:
    path = Path(path or MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(), indent=2) + "\n")
    return path


def load_manifest(path: Path | None = None) -> dict:
    path = Path(path or MANIFEST_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"no bitstream manifest at {path}. Generate it with "
            "`python -m superneuromat.spikeengine.provenance --write`.")
    return json.loads(path.read_text())


def verify_bitstreams(path: Path | None = None) -> dict:
    """Re-hash the installed bitstreams against the manifest.

    Returns a dict with ``ok`` plus lists of ``missing`` (in the manifest but
    not on disk), ``altered`` (hash differs), ``unlisted`` (on disk but not in
    the manifest) and ``verified``. Also re-checks the RTL digest.
    """
    manifest = load_manifest(path)
    bit_root = _PKG_ROOT / "bitstreams"

    listed = {e["path"]: e for e in manifest["bitstreams"]}
    on_disk = {str(p.relative_to(bit_root)).replace("\\", "/")
               for p in bit_root.rglob("*.bit")}

    missing, altered, verified = [], [], []
    for rel, entry in listed.items():
        p = bit_root / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_of(p) != entry["sha256"]:
            altered.append(rel)
        else:
            verified.append(rel)

    rtl_now = rtl_digest()["combined_sha256"]
    rtl_recorded = manifest.get("rtl", {}).get("combined_sha256")

    return {
        # `unlisted` counts as failure (2026-07-31): a .bit present but
        # absent from the manifest is unattested -- reporting ok=True with
        # an unknown binary in the package defeats the purpose.
        "ok": (not missing and not altered and not (on_disk - set(listed))
               and rtl_now == rtl_recorded),
        "verified": sorted(verified),
        "missing": sorted(missing),
        "altered": sorted(altered),
        "unlisted": sorted(on_disk - set(listed)),
        "rtl_matches_manifest": rtl_now == rtl_recorded,
        "rtl_combined_sha256_now": rtl_now,
        "rtl_combined_sha256_recorded": rtl_recorded,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="regenerate the manifest from the packaged bitstreams")
    ap.add_argument("--verify", action="store_true",
                    help="check installed bitstreams against the manifest (default)")
    args = ap.parse_args(argv)

    if args.write:
        p = write_manifest()
        m = load_manifest(p)
        print(f"wrote {p}")
        print(f"  {len(m['bitstreams'])} bitstreams, "
              f"RTL combined sha256 {m['rtl']['combined_sha256'][:16]}...")
        for e in m["bitstreams"]:
            flag = "" if e.get("hardware_validated") else "  (not hardware-validated)"
            print(f"  {e['path']}  {e['sha256'][:16]}...{flag}")
            for n in (e.get("notes") or []):
                print(f"      note: {n}")
        return 0

    res = verify_bitstreams()
    print(f"verified : {len(res['verified'])}")
    for k in ("missing", "altered", "unlisted"):
        if res[k]:
            print(f"{k:9}: {res[k]}")
    if not res["rtl_matches_manifest"]:
        print("RTL DIGEST MISMATCH -- the RTL in this package has changed "
              "since the manifest was written; the bitstreams may no longer "
              "correspond to these sources.")
    print("OK" if res["ok"] else "FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
