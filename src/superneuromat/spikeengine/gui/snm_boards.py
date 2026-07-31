"""SuperNeuroMAT3 FPGA board registry access.

Loads config/boards.yaml -- the list of supported FPGA boards and their
device-specific facts (part, BRAM budget, core clock, constraints, programming,
SRAM sizes, tick-latency coefficients). The host/GUI use this so capacity, timing,
and the BRAM budget follow the selected board, and so adding a new FPGA is a data
edit (plus its board wrapper + .xdc) rather than code changes scattered around.

    import snm_boards
    b = snm_boards.get(snm_boards.default_name())   # the default board (Artix-7)
    b["bram_kb"], b["core_clk_hz"], b["part"]
"""

from __future__ import annotations

import os

try:
    import yaml
except ImportError:                                  # pragma: no cover
    yaml = None

_HERE = os.path.dirname(os.path.abspath(__file__))
# The board registry is bundled with the package (standalone). Fall back to the
# repo-level config/boards.yaml when running from the FPGA development checkout.
_BUNDLED = os.path.join(_HERE, "boards.yaml")
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "config", "boards.yaml"))
BOARDS_YAML = _BUNDLED if os.path.isfile(_BUNDLED) else _REPO

_cache = None


def _load():
    global _cache
    if _cache is not None:
        return _cache
    if yaml is None:
        raise RuntimeError("PyYAML is required to read boards.yaml: pip install pyyaml")
    with open(BOARDS_YAML) as f:
        doc = yaml.safe_load(f)
    boards = doc.get("boards", {})
    for name, b in boards.items():
        b.setdefault("name", name)
    _cache = {"default": doc.get("default") or next(iter(boards)), "boards": boards}
    return _cache


def names() -> list:
    """All board ids, default first."""
    doc = _load()
    d = doc["default"]
    rest = [n for n in doc["boards"] if n != d]
    return ([d] + rest) if d in doc["boards"] else list(doc["boards"])


def default_name() -> str:
    return _load()["default"]


def get(name: str) -> dict:
    """The descriptor dict for a board id (raises KeyError if unknown)."""
    boards = _load()["boards"]
    if name not in boards:
        raise KeyError(f"unknown board {name!r}; known: {list(boards)}")
    return boards[name]


def label(name: str) -> str:
    return get(name).get("label", name)


def current() -> dict:
    """The board this host's loaded bitstream was generated for (snm_config.BOARD),
    falling back to the registry default."""
    try:
        from . import snm_config
        return get(getattr(snm_config, "BOARD", default_name()))
    except (ImportError, AttributeError, KeyError, ValueError):
        return get(default_name())
