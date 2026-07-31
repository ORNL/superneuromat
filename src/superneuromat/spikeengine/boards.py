"""Board catalogue for the STDP parallel-lane SpikeEngine.

Each entry records the hardware facts a host needs to talk to a programmed
board (N_MAX/NUM_LANES define the neuron->lane mapping the RTL uses, baud is
the UART rate the bitstream was built for) plus the packaged bitstream that
implements that configuration. All three bitstreams were produced by real
Vivado 2025.2 synth+place+route runs (timing-closed, DRC-clean); the Basys3
one is additionally hardware-validated bit-exact against SuperNeuroMAT on a
physical board.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PKG_ROOT = Path(__file__).parent


@dataclass(frozen=True)
class StdpBoard:
    key: str
    label: str
    part: str
    n_max: int
    num_lanes: int
    baud: int
    bitstream: str            # path relative to the package's bitstreams/ dir
    stdp_window: int = 5
    weight_w: int = 8
    data_w: int = 16
    frac_bits_default: int = 3
    spike_mon_base: int = 0    # LED[0] neuron id in the packaged bitstream
    # USB interface index carrying the RUNTIME UART, when the board exposes
    # several serial nodes (multi-channel FTDI/CP210x parts). None = use the
    # common FTDI channel-B convention (interface 1). Consumed by
    # connect() -> SerialTransport -> autodetect_port so detection is
    # board-aware rather than assuming interface 1 for every board.
    uart_interface_index: int | None = None
    core_clk_hz: int = 100_000_000
    hardware_validated: bool = False
    hardware_validated_date: str = ""
    note: str = ""

    @property
    def local_d(self) -> int:
        return (self.n_max + self.num_lanes - 1) // self.num_lanes

    @property
    def syn_cap_per_lane(self) -> int:
        return self.local_d * self.n_max

    def bitstream_path(self) -> Path:
        return _PKG_ROOT / "bitstreams" / self.bitstream


BOARDS: dict[str, StdpBoard] = {
    "basys3": StdpBoard(
        key="basys3", label="Digilent Basys3 (Artix-7 xc7a35t)",
        part="xc7a35tcpg236-1", n_max=256, num_lanes=8, baud=4_000_000,
        bitstream="basys3/snm_infer_basys3_stdp_cap256x8.bit",
        spike_mon_base=64, weight_w=16, data_w=24, frac_bits_default=11,
        hardware_validated=True, hardware_validated_date="2026-07-27",
        note="N=256/K=8 STDP engine, WIDE fixed-point (WEIGHT_W=16, DATA_W=24). "
             "WNS +0.174ns, BRAM 50, DRC clean. Hardware-validated bit-exact vs "
             "SuperNeuroMAT on a physical board: on-chip STDP training (640/640 "
             "weights bit-exact) and on-chip rate-readout inference (bit-exact, "
             "tick-by-tick). Digits classifier on the full 899-image test set: "
             "one-vs-rest accuracy 0.909 (recall 0.776, exact match to the "
             "software reference), strict single-answer top-1 58.4%, at "
             "frac_bits=11; the wide fixed-point lets the fine 3-tap STDP taps "
             "(0.002, -0.0005) survive per-update quantization. LEDs show "
             "output neurons 64+ (class 0..9 on LED[0..9]). Includes "
             "OP_ENGINE_RESET soft-reset + OP_READ_CONFIG weight readback."),
    "sp701": StdpBoard(
        key="sp701", label="Xilinx SP701 (Spartan-7 xc7s100)",
        part="xc7s100fgga676-2", n_max=352, num_lanes=8, baud=4_000_000,
        bitstream="sp701/snm_infer_sp701_stdp_maxcap352x8.bit",
        spike_mon_base=64, weight_w=16, data_w=24, frac_bits_default=11,
        hardware_validated=True, hardware_validated_date="2026-07-31",
        note="REBUILT 2026-07-31 -- the previous packaged default was "
             "confirmed BROKEN (not just unverified): a stale binary "
             "synthesized before the 2026-07-29 board-top parameter-forwarding "
             "fix, so real hardware ran at N_MAX in 65..96 with scrambled "
             "synapse addressing despite declaring N_MAX=352/NUM_LANES=8. "
             "Rebuilt from today's already-correct RTL + build script "
             "(N_MAX=352, NUM_LANES=8, WEIGHT_W=16, DATA_W=24), WNS +0.162ns, "
             "DRC clean (async-reset RAMB18 warnings only, same class already "
             "accepted elsewhere in this project). Hardware-validated at "
             "frac_bits=11: 40/40 tick outputs and 640/640 learned weights "
             "matched the fixed-point-faithful software reference exactly. A "
             "synthetic on-chip STDP network (3 inputs -> 2 outputs, 6 "
             "STDP-enabled synapses, 20 ticks) trained on-chip and read back "
             "6/6 synapses bit-exact against a fixed-point-faithful "
             "SuperNeuroMAT reference, with correct source-neuron addressing "
             "(0, 1, 2 -- not the previous build's scrambled 62, 63, 0). "
             "Also validated on this board via its own image: microseer "
             "(N=90/K=8), "
             "'bitstreams/sp701/snm_infer_sp701_microseer_N90_K8_cap200_w16.bit', "
             "48/48 test papers exact SW==HW agreement, one-vs-rest 0.6424. "
             "Citation datasets larger than microseer (miniseer/cora/citeseer) "
             "do not fit SP701's LUT budget regardless -- use ZCU104 for those."),
    "zcu104": StdpBoard(
        key="zcu104", label="Xilinx ZCU104 (Zynq UltraScale+ xczu7ev)",
        part="xczu7ev-ffvc1156-2-e", n_max=1024, num_lanes=16, baud=4_000_000,
        bitstream="zcu104/snm_infer_zcu104_stdp_maxcap1024x16.bit",
        spike_mon_base=0, weight_w=8, data_w=16, frac_bits_default=3,
        # Live Windows enumeration maps the responding runtime endpoint to
        # FTDI channel D (SER=...D), which is interface index 3. Interfaces 1
        # and 2 did not answer the read-only geometry probe.
        uart_interface_index=3,
        hardware_validated=True, hardware_validated_date="2026-07-31",
        note="N=1024/K=16 + URAM. LEGACY 8-bit build (unlike the rebuilt "
             "SP701 default, this binary still predates parameter forwarding: "
             "the -generic W16/D24/mon were dropped, "
             "so this runs 8-bit weights / 16-bit membrane, not the wide "
             "datapath the build script requested). Hardware-validated "
             "2026-07-31, unlike SP701's default: probe_spike_words() confirms "
             "the real SPIKE_WORDS=32 -> N_MAX in [993,1024], matching the "
             "declared 1024 (this board's N_MAX/NUM_LANES generics were NOT "
             "silently dropped, only the datapath-width ones were). A tiny "
             "synthetic on-chip STDP network (3 inputs -> 2 outputs, 6 "
             "STDP-enabled synapses, 20 ticks) trained on-chip and read back "
             "bit-exact against a fixed-point-faithful SuperNeuroMAT reference "
             "at this board's own native config (weight_w=8, data_w=16, "
             "frac_bits=3): 6/6 synapses exact. This validates the addressing, "
             "STDP datapath, and capacity of the packaged default itself -- not "
             "the wide recipe. This board is also the target for the "
             "citation-graph datasets: use the per-dataset wide fixed-point + "
             "Option A bitstreams catalogued in datasets.py (miniseer/cora/"
             "citeseer) for those, not this legacy default."),
}


def get_board(key: str = "basys3") -> StdpBoard:
    if key not in BOARDS:
        raise KeyError(f"unknown board {key!r}; available: {sorted(BOARDS)}")
    return BOARDS[key]


def list_boards() -> list[str]:
    return sorted(BOARDS)
