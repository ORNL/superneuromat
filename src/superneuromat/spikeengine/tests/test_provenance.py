"""Tests for bitstream provenance.

Sign-off requires being able to show that the shipped binaries correspond to
the shipped sources, and that a user's copy is intact. These verify the
manifest exists, covers every bitstream, and actually detects tampering.
"""
import json

import pytest

from superneuromat.spikeengine import provenance as prov


def test_manifest_exists_and_is_current():
    """A stale or missing manifest is worse than none: it implies a
    verification that did not happen."""
    res = prov.verify_bitstreams()
    assert res["ok"], (
        f"provenance verification failed: missing={res['missing']} "
        f"altered={res['altered']} rtl_matches={res['rtl_matches_manifest']}. "
        "Regenerate with `python -m superneuromat.spikeengine.provenance --write`.")


def test_every_shipped_bitstream_is_listed():
    res = prov.verify_bitstreams()
    assert not res["unlisted"], (
        f"bitstreams ship but are absent from the manifest: {res['unlisted']}")


def test_manifest_records_hash_size_and_params_for_each_entry():
    m = prov.load_manifest()
    assert m["bitstreams"], "manifest lists no bitstreams"
    for e in m["bitstreams"]:
        assert len(e["sha256"]) == 64, f"{e['path']}: bad sha256"
        assert e["size_bytes"] > 0
        assert e.get("board"), f"{e['path']}: no board recorded"
        assert e.get("params"), f"{e['path']}: no build parameters recorded"


def test_rtl_digest_covers_the_shipped_rtl():
    """The digest must span every .v/.vh actually shipped, or it proves
    nothing about the parts it skipped."""
    import pathlib

    import superneuromat.spikeengine as pkg

    rtl_root = pathlib.Path(pkg.__file__).parent / "rtl"
    on_disk = {str(p.relative_to(rtl_root)).replace("\\", "/")
               for p in rtl_root.rglob("*") if p.suffix in (".v", ".vh")}
    listed = set(prov.load_manifest()["rtl"]["files"])
    assert on_disk == listed, f"RTL digest misses: {on_disk - listed}"


def test_verification_detects_an_altered_bitstream(tmp_path, monkeypatch):
    """The whole point: a corrupted or swapped binary must be caught."""
    m = prov.load_manifest()
    tampered = json.loads(json.dumps(m))
    tampered["bitstreams"][0]["sha256"] = "0" * 64

    p = tmp_path / "m.json"
    p.write_text(json.dumps(tampered))
    res = prov.verify_bitstreams(p)

    assert not res["ok"]
    assert res["altered"] == [tampered["bitstreams"][0]["path"]]


def test_verification_detects_changed_rtl(tmp_path):
    m = prov.load_manifest()
    tampered = json.loads(json.dumps(m))
    tampered["rtl"]["combined_sha256"] = "f" * 64

    p = tmp_path / "m.json"
    p.write_text(json.dumps(tampered))
    res = prov.verify_bitstreams(p)

    assert not res["ok"]
    assert res["rtl_matches_manifest"] is False


def test_manifest_states_its_limitations():
    """Provenance that overstates itself is worse than none. The manifest must
    say plainly what it does NOT establish."""
    m = prov.load_manifest()
    assert m.get("limitations"), "manifest claims provenance without caveats"
    joined = " ".join(m["limitations"]).lower()
    assert "vivado" in joined      # build-tool version not recorded per bitstream
    assert "rebuild" in joined     # only a rebuild proves bit-for-bit origin


def test_missing_manifest_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--write"):
        prov.load_manifest(tmp_path / "nope.json")
