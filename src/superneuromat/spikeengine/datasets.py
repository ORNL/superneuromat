"""Catalogue of the citation-graph (SGNN) datasets implemented on hardware.

Each entry records network size, board + bitstream, build timing, and the
software-reference accuracy hardware is verified against. Bitstreams use the
wide fixed-point datapath (16-bit weights / 24-bit membrane) plus Option A
neuron-state-in-BRAM, which fits the 2-3k-neuron graphs on the ZCU104.

microseer runs on the packaged Basys3 STDP bitstream; miniseer/cora/citeseer
have their own ZCU104 bitstreams; pubmed is software-only (over LUT budget).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PKG_ROOT = Path(__file__).parent


@dataclass(frozen=True)
class Dataset:
    key: str
    board: str
    neurons: int
    synapses: int
    num_lanes: int
    # Meaning depends on `bitstream`: with a custom build, the hardware
    # SYN_CAP_PER_LANE it was compiled with (a real bound, checked by
    # load_network's guard). With no custom build, just the measured
    # REQUIREMENT for sizing -- the board's dense default build is used
    # instead, which is far larger (conflating the two once rejected
    # microseer on Basys3: requirement 144 vs. dense capacity 8192).
    syn_cap_per_lane: int
    bitstream: str | None      # path relative to bitstreams/; None -> board default
    # One-vs-rest (TP+TN)/(TP+TN+FP+FN) macro-averaged over every (paper,
    # topic) pair -- the SGNN repo's Results.accuracy (gnn_citation_networks.py
    # :346-348), the only metric it persists. Not comparable to top-1: correctly
    # rejecting wrong topics inflates the denominator.
    sw_accuracy: float
    wns_ns: float | None       # build timing margin (None if not a custom build)
    hardware_validated: bool = False
    note: str = ""
    weight_w: int = 16         # bitstream datapath (all datasets: W16/D24)
    data_w: int = 24
    # Strict top-1: highest-weight predicted topic vs. true label, as
    # (correct, n_papers). The repo computes this as Results.legacy
    # (gnn_citation_networks.py:383) but only prints it, never saves it.
    sw_top1: tuple[int, int] | None = None
    # Extra boards this dataset is validated on -> {board: bitstream path}.
    # Without this a two-board dataset could reference only one bitstream,
    # leaving the other unreachable (SP701's microseer build, until 2026-07-30).
    extra_bitstreams: dict[str, str] | None = None
    # Per-lane capacity for an extra board's image (one syn_cap_per_lane can't
    # describe both microseer's Basys3 requirement and its SP701 image).
    extra_syn_cap_per_lane: dict[str, int] | None = None

    @property
    def sw_top1_accuracy(self) -> float | None:
        """Top-1 accuracy as a fraction, or None if top-1 was not measured."""
        if self.sw_top1 is None:
            return None
        correct, n = self.sw_top1
        return correct / n if n else 0.0

    def boards(self) -> list[str]:
        """Every board this dataset is validated on, primary first."""
        out = [self.board] if self.board else []
        out += [b for b in (self.extra_bitstreams or {}) if b not in out]
        return out

    def bitstream_path(self, board: str | None = None) -> Path | None:
        """Path to the bitstream for ``board`` (default: the primary board).

        Returns None when this dataset runs on the board's own packaged
        default rather than a dataset-specific build. Raises KeyError for a
        board this dataset has no build for -- silently falling back to the
        primary board's bitstream would program the wrong device geometry.
        """
        if board is None or board == self.board:
            rel = self.bitstream
        else:
            extra = self.extra_bitstreams or {}
            if board not in extra:
                raise KeyError(
                    f"dataset {self.key!r} has no bitstream for board {board!r}; "
                    f"validated boards: {self.boards()}")
            rel = extra[board]
        return None if rel is None else _PKG_ROOT / "bitstreams" / rel

    def hardware_syn_cap_per_lane(self, board: str | None = None) -> int | None:
        """Compiled capacity for this board's custom image, or ``None`` when
        the dataset uses the board's dense default image."""
        board = board or self.board
        if self.bitstream_path(board) is None:
            return None
        if board == self.board:
            return self.syn_cap_per_lane
        try:
            return int((self.extra_syn_cap_per_lane or {})[board])
        except KeyError:
            raise KeyError(
                f"dataset {self.key!r} has a custom bitstream for {board!r} "
                "but no per-lane capacity for it") from None


DATASETS: dict[str, Dataset] = {
    "microseer": Dataset(
        key="microseer", board="basys3", neurons=90, synapses=996,
        # 144 not 143: hash-seed graph-construction variance (lane 5 needs
        # 144). Requirement only -- runs on the dense packaged build (8192/lane).
        num_lanes=8, syn_cap_per_lane=144, bitstream=None, sw_accuracy=0.6424,
        sw_top1=(20, 48), wns_ns=None, hardware_validated=True,
        # Also validated on SP701 via its own wide-fixed-point Option A build --
        # SP701's packaged default is a legacy 8-bit bitstream, wrong for this.
        extra_bitstreams={
            "sp701": "sp701/snm_infer_sp701_microseer_N90_K8_cap200_w16.bit"},
        extra_syn_cap_per_lane={"sp701": 200},
        note="Packaged Basys3 STDP bitstream. Bit-exact on silicon: 48/48 "
             "papers SW==HW, one-vs-rest 0.6424, top-1 20/48."),
    "miniseer": Dataset(
        key="miniseer", board="zcu104", neurons=2116, synapses=31456,
        # Rebuilt 5200 (was 4532): required capacity varies with the hash
        # seed (measured 4477..4598), so 4532 failed on ~half of runs. 5200
        # clears the max by ~13% at unchanged URAM usage.
        num_lanes=8, syn_cap_per_lane=5200,
        bitstream="zcu104/snm_infer_zcu104_miniseer_N2116_K8_bramA.bit",
        sw_accuracy=0.7667, sw_top1=(66, 120), wns_ns=0.382, hardware_validated=True,
        note="ZCU104 sparse build (Option A). WNS +0.382ns, LUT 39%, FF 30%, "
             "URAM 8%. 120/120 papers exact SW==HW, one-vs-rest 0.7667."),
    "cora": Dataset(
        key="cora", board="zcu104", neurons=2715, synapses=46788,
        # Rebuilt 7200 (was 6436), same hash-seed capacity variance as
        # miniseer (measured 6280..6451). Resource usage unchanged -- both
        # give $clog2=13 and fit one URAM tile/lane -- so WNS +0.498ns repeats.
        num_lanes=8, syn_cap_per_lane=7200,
        bitstream="zcu104/snm_infer_zcu104_cora_N2715_K8_bramA.bit",
        sw_accuracy=0.8173, sw_top1=(88, 140), wns_ns=0.498, hardware_validated=True,
        note="ZCU104 sparse build (Option A). WNS +0.498ns, LUT 49%, FF 39%. "
             "140/140 papers exact SW==HW, one-vs-rest 0.8173."),
    "citeseer": Dataset(
        key="citeseer", board="zcu104", neurons=3318, synapses=47616,
        num_lanes=8, syn_cap_per_lane=7500,
        bitstream="zcu104/snm_infer_zcu104_citeseer_N3318_K8_bramA.bit",
        sw_accuracy=0.5806, sw_top1=(50, 120), wns_ns=0.889, hardware_validated=True,
        note="ZCU104 sparse build (Option A), rebuilt at cap 7500 (an earlier "
             "6842 build was undersized -- real requirement 6881, caught by "
             "the capacity guard before any wire write; see ARCHITECTURE.md "
             "Sec.13). WNS +0.889ns, LUT 58.1%, FF 46.5%, BRAM 9.6%, URAM "
             "8.3%, DRC clean. 120/120 papers exact SW==HW, one-vs-rest "
             "0.5806, top-1 50/120."),
    "pubmed": Dataset(
        key="pubmed", board="", neurons=19720, synapses=206710,
        num_lanes=16, syn_cap_per_lane=38134, bitstream=None,
        sw_accuracy=0.0, wns_ns=None, hardware_validated=False,
        note="SOFTWARE ONLY: 19,720 neurons is ~4x over every board's LUT "
             "budget. Needs event-driven spikes or DRAM-backed state (see "
             "USER_MANUAL section 8, Roadmap)."),
}


def get_dataset(key: str) -> Dataset:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; available: {sorted(DATASETS)}")
    return DATASETS[key]


def list_datasets() -> list[str]:
    return sorted(DATASETS)
