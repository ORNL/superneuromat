"""Does a rebuild reproduce the packaged bitstream? (audit N1, 2026-07-31)

README presents build_bitstream() as "reproduce the packaged build from
source". For sp701/zcu104 it does not: their packaged bitstreams predate the
2026-07-29 board-top parameter-forwarding fix and are 8-bit, while a rebuild
today is wide (W16/D24). The widths set wire packing offsets, so

    build_bitstream(b) -> program(result) -> connect(board=b)

would drive a wide device with an 8-bit host and corrupt every weight. These
tests pin the known state and guarantee the mismatch is announced.
"""
import warnings

import pytest

from superneuromat.spikeengine import build as B


def test_every_board_script_declares_its_widths():
    """The comparison is only meaningful if the generics are actually parsed."""
    for board in ("basys3", "sp701", "zcu104"):
        g = B.script_generics(board)
        for key in ("WEIGHT_W", "DATA_W", "N_MAX", "NUM_LANES"):
            assert key in g, f"{board}: build script declares no {key}"


@pytest.mark.parametrize("board", ["basys3", "sp701"])
def test_rebuild_reproduces_the_packaged_bitstream(board):
    """basys3 and sp701 were both rebuilt after the forwarding fix (sp701 on
    2026-07-31, after its packaged default was found to have confirmed-wrong
    N_MAX/NUM_LANES -- see boards.py's sp701 note), so both must agree."""
    chk = B.check_rebuild_matches_catalogue(board)
    assert chk["matches"], f"{board} drifted from its catalogue entry: {chk['differing']}"


def test_known_mismatched_board_is_reported_not_hidden():
    """zcu104's packaged default is STILL the legacy narrow build (only its
    N_MAX/NUM_LANES were correct; WEIGHT_W/DATA_W/SPIKE_MON_BASE were not
    rebuilt wide -- left as-is deliberately, see boards.py). The check must
    say so rather than quietly claiming agreement -- silence here is the
    corruption path."""
    chk = B.check_rebuild_matches_catalogue("zcu104")
    assert not chk["matches"], (
        "zcu104 now reports agreement. If it was genuinely rebuilt and "
        "repackaged as wide, update boards.py and delete this expectation; "
        "do not weaken the check.")
    assert "WEIGHT_W" in chk["differing"]
    cat, rebuild = chk["differing"]["WEIGHT_W"]
    assert cat == 8 and rebuild == 16


@pytest.mark.parametrize("board", ["zcu104"])
def test_build_warns_before_producing_a_mismatched_bitstream(board, monkeypatch):
    """The warning must fire BEFORE the build runs -- a 10-40 minute build
    that ends in a mis-described artifact is worse than one that never starts.
    """
    monkeypatch.setattr(B, "find_vivado", lambda: None)   # stop right after the check
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(B.VivadoNotFoundError):
            B.build_bitstream(board)
        msgs = [str(x.message) for x in w]
    assert any("will NOT reproduce the packaged bitstream" in m for m in msgs), msgs
    assert any("corrupt every weight" in m for m in msgs), msgs


def test_no_warning_for_the_board_that_does_reproduce(monkeypatch):
    monkeypatch.setattr(B, "find_vivado", lambda: None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(B.VivadoNotFoundError):
            B.build_bitstream("basys3")
        assert not [x for x in w if "NOT reproduce" in str(x.message)]


def test_sdist_excludes_local_distribution_archives():
    """A source-adjacent release folder must not be embedded recursively in
    the next sdist. This previously tripled the archive from 23.6 to 70.5 MB."""
    import pathlib

    import tomllib

    pyproject = pathlib.Path(__file__).resolve().parents[4] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml is not installed in wheels")
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    excludes = config["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "dist*" in excludes


def test_build_scripts_never_write_into_the_installed_package():
    """audit N2: basys3's script wrote its outdir to $src/../vivado_build/,
    which is inside site-packages once installed -- failing on a read-only
    install and re-opening the Windows path-length problem. basys3 is the
    default board, so it was the most-hit path."""
    import pathlib

    import superneuromat.spikeengine as pkg

    scripts = pathlib.Path(pkg.__file__).parent / "scripts"
    offenders = []
    for tcl in sorted(scripts.glob("build_*.tcl")):
        text = tcl.read_text(errors="replace")
        if "SPIKEENGINE_BUILD_DIR" not in text:
            offenders.append(f"{tcl.name}: does not honour SPIKEENGINE_BUILD_DIR")
        for line in text.splitlines():
            s = line.strip()
            if (s.startswith("set ") and "outdir" in s.split()[1]
                    and "vivado_build" in s and "$src" in s):
                offenders.append(f"{tcl.name}: output dir inside the package -> {s}")
    assert not offenders, "build scripts write into the installed package:\n  " + \
        "\n  ".join(offenders)


def test_copy_destinations_are_created_before_use():
    import pathlib

    import superneuromat.spikeengine as pkg

    scripts = pathlib.Path(pkg.__file__).parent / "scripts"
    for tcl in sorted(scripts.glob("build_*.tcl")):
        text = tcl.read_text(errors="replace")
        if "file copy" in text and "$final_outdir" in text:
            assert "file mkdir $final_outdir" in text, (
                f"{tcl.name}: copies to $final_outdir without creating it")


def test_direct_program_script_defaults_are_packaged_bitstreams():
    import pathlib

    import superneuromat.spikeengine as pkg

    scripts = pathlib.Path(pkg.__file__).parent / "scripts"
    for tcl in sorted(scripts.glob("program_*.tcl")):
        text = tcl.read_text(errors="replace")
        assert "$src/../bitstreams/" in text, (
            f"{tcl.name}: default does not point at a packaged bitstream")
        assert "$src/../vivado_build/" not in text, (
            f"{tcl.name}: default points at an unshipped build tree")
