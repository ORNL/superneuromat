"""superneuromat.spikeengine -- host runtime + packaged bitstreams for the
STDP-capable parallel-lane SpikeEngine FPGA accelerator.

Adds on-chip STDP learning to the parallel-lane engine (destination-
partitioned K-lane engine, source-indexed history BRAM, II=1 STDP pipeline).
Replaced the earlier, non-STDP `superneuromat.spikeengine` v0.1.0 at this
import path on 2026-07-30 (the old package -- classic single-core engine +
non-STDP lane engine -- is kept for reference only, outside the installed
package, at `legacy/spikeengine_v0.1.0/` in the repo). It ships:

  * a UART host runtime (``InferEngineStdpDevice``) speaking the engine's
    8-byte command protocol, including the ``soft_reset()`` that clears runtime
    state without a bitstream reload;
  * packaged, timing-closed bitstreams for Basys3 / SP701 / ZCU104 (the Basys3
    one hardware-validated bit-exact vs SuperNeuroMAT);
  * high-level helpers to map a ``superneuromat.SNN`` onto the board and replay
    a training schedule;
  * the RTL sources and Vivado build/program scripts.

Quick start (board programmed & connected)::

    from superneuromat import spikeengine as se
    dev = se.connect(port="auto", board="basys3")
    dev.soft_reset()                       # clean state for a fresh run
    info = se.load_network(dev, snn, frac_bits=11, weight_w=16, data_w=24)
    spikes = se.run_schedule(dev, schedule, total_ticks, frac_bits=11,
                             n_neurons=info["n_neurons"])
    dev.close()
"""
from __future__ import annotations

from .boards import BOARDS, StdpBoard, get_board, list_boards
from .capacity import estimate as estimate_capacity
from .capacity import recommend_board, required_syn_cap_per_lane
from .datasets import DATASETS, Dataset, get_dataset, list_datasets
from .network import load_network, quantize_raw, read_weights, run_schedule
from .runtime import InferEngineError, InferEngineStdpDevice

# Component version of the spikeengine subpackage. Deliberately DISTINCT from
# the distribution version in pyproject.toml (`superneuromat`): the two track
# different things -- the simulator's release line and
# this FPGA subsystem's own maturity. `pip show superneuromat` reports the
# distribution version; this reports the engine's. Flagged as a "mismatch" in
# review, so stated explicitly rather than left to inference.
__version__ = "0.3.0"

__all__ = [
    "BOARDS",
    "DATASETS",
    "Dataset",
    "InferEngineError",
    "InferEngineStdpDevice",
    "StdpBoard",
    "bitstream_path",
    "connect",
    "estimate_capacity",
    "get_board",
    "get_dataset",
    "list_boards",
    "list_datasets",
    "load_network",
    "quantize_raw",
    "read_weights",
    "recommend_board",
    "required_syn_cap_per_lane",
    "run_schedule",
]


def connect(port: str | None = "auto", board: str = "basys3", *,
            baud: int | None = None, timeout: float = 2.0,
            n_max: int | None = None, num_lanes: int | None = None,
            weight_w: int | None = None, data_w: int | None = None,
            dataset: str | None = None) -> InferEngineStdpDevice:
    """Open a serial connection to a programmed STDP board, sized for it.

    ``port='auto'`` autodetects the board's UART by USB VID:PID. ``board``
    selects the default neuron->lane geometry (N_MAX/NUM_LANES) + baud rate the
    board's PACKAGED bitstream was built for. These MUST match the RUNNING
    bitstream or the lane/index mapping will be wrong.

    When you have programmed a NON-default bitstream on a board -- e.g. one of
    the citation-dataset builds whose N_MAX/NUM_LANES differ from the board's
    packaged build -- override the geometry: pass ``dataset="miniseer"`` to pull
    it from the dataset catalogue, or set ``n_max``/``num_lanes`` explicitly.
    """
    spec = get_board(board)
    syn_cap_per_lane = None
    if dataset is not None:
        from .datasets import get_dataset
        ds = get_dataset(dataset)
        # Reject a dataset that is not validated on this board (2026-07-31).
        # Previously connect(board="basys3", dataset="cora") silently built a
        # 2,715-neuron device on a 256-neuron board, and dataset="pubmed"
        # (software-only, no board) was accepted too -- both produce a device
        # whose addressing cannot match any bitstream that board can run.
        valid = ds.boards()
        if not valid:
            raise ValueError(
                f"dataset {dataset!r} has no hardware build (it is "
                "software-only) and cannot be used with connect().")
        if board not in valid:
            raise ValueError(
                f"dataset {dataset!r} is not validated on board {board!r}; "
                f"it runs on {valid}. Connecting anyway would size the device "
                "for a bitstream this board cannot be running.")
        if n_max is None:
            n_max = ds.neurons
        if num_lanes is None:
            num_lanes = ds.num_lanes
        # Per-lane synapse capacity of the bitstream that will actually be
        # loaded (2026-07-31). Only meaningful when the dataset has its OWN
        # build: `Dataset.syn_cap_per_lane` then records the capacity that
        # build was compiled with. When the dataset runs on the board's
        # PACKAGED bitstream (bitstream_path(board) is None -- microseer on
        # Basys3), that field is the network's *requirement*, not the
        # hardware's capacity: the packaged builds are dense, with
        # local_d * n_max entries per lane. Passing the requirement as the
        # capacity made the guard reject a network that fits comfortably
        # (microseer needs 144, the dense Basys3 build provides 8192).
        # Leaving it None lets the device use its dense default, which is
        # correct for those builds.
        syn_cap_per_lane = ds.hardware_syn_cap_per_lane(board)
        # CRITICAL: the dataset bitstream's datapath width (W16/D24), NOT the
        # board's default. The board's packaged zcu104 bitstream is a legacy
        # 8-bit build, so board.weight_w=8 -- using that would truncate the
        # dataset's 16-bit weights on the wire (2.0 -> raw 16384 -> low byte 0).
        if weight_w is None:
            weight_w = ds.weight_w
        if data_w is None:
            data_w = ds.data_w
    return InferEngineStdpDevice(
        port, baud if baud is not None else spec.baud,
        n_max=n_max if n_max is not None else spec.n_max,
        num_lanes=num_lanes if num_lanes is not None else spec.num_lanes,
        timeout=timeout,
        weight_w=weight_w if weight_w is not None else spec.weight_w,
        data_w=data_w if data_w is not None else spec.data_w,
        # Board-aware UART discovery: boards exposing several serial nodes can
        # declare which interface carries the runtime UART instead of the
        # autodetector assuming FTDI interface 1 for everything.
        prefer_interface=spec.uart_interface_index,
        syn_cap_per_lane=syn_cap_per_lane,
        stdp_window=spec.stdp_window,
    )


def bitstream_path(board: str = "basys3"):
    """Filesystem path to the packaged bitstream for ``board``."""
    return get_board(board).bitstream_path()
