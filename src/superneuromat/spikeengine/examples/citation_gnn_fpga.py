"""Citation-graph SNN (SGNN) classifier on the SpikeEngine FPGA.

Runs a paper->topic citation classifier from the `sgnn-superneuro` benchmark on
real hardware with on-chip STDP, and cross-checks every paper against a
fixed-point-faithful software reference. Works for any of the datasets:

    microseer  (90 neurons)   -> Basys3
    miniseer   (2116 neurons)  -> ZCU104
    cora       (2715 neurons)  -> ZCU104
    citeseer   (3318 neurons)  -> ZCU104

Usage:
    python -m spikeengine.examples.citation_gnn_fpga --dataset microseer --board basys3 --port auto
    python -m spikeengine.examples.citation_gnn_fpga --dataset miniseer  --board zcu104 --port auto
    python -m spikeengine.examples.citation_gnn_fpga --dataset miniseer  --no-hardware   # SW reference

The SGNN dataset sources (the `gnn_citation_networks` module + `configs/`) are
NOT bundled in this package. Point at them with the SGNN_REPO environment
variable (or --sgnn-repo). The per-dataset bitstreams are the packaged
`bitstreams/<board>/...` builds.

Resolved fixed-point recipe (see the project's fixed-point findings): the driving
graph/input weights are rescaled to 2.0 (they only need to push threshold-1
neurons over), frac_bits=13, 16-bit weights / 24-bit membrane, and STDP is GLOBAL
(what the hardware does) -- proven to give the same classification as the sgnn
model's selective STDP for these tasks.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

# resolved hardware config
GRAPH_WEIGHT = 2.0
INPUT_SPIKE = 2.0
FRAC_BITS = 13
WEIGHT_W = 16
DATA_W = 24
STDP_WINDOW = 5

# dataset -> default board (which board the packaged bitstream targets)
DATASET_BOARD = {
    "microseer": "basys3",
    "miniseer": "zcu104",
    "cora": "zcu104",
    "citeseer": "zcu104",
}


# import name -> pip distribution name, where they differ (used by the
# dependency error in main()).
_PIP_NAME = {"yaml": "PyYAML", "sklearn": "scikit-learn"}


def _find_sgnn_repo(explicit: str | None) -> Path:
    for cand in (explicit, os.environ.get("SGNN_REPO")):
        if cand and (Path(cand) / "gnn_citation_networks.py").exists():
            return Path(cand)
    raise SystemExit(
        "SGNN dataset sources not found. Set SGNN_REPO (or --sgnn-repo) to the "
        "sgnn-superneuro checkout containing gnn_citation_networks.py and configs/."
    )


def _q(v, frac_bits, weight_w):
    """Quantize to the signed weight_w fixed-point grid (round + clamp),
    return the on-grid value -- mirrors how the hardware stores int weights."""
    lo, hi = -(1 << (weight_w - 1)), (1 << (weight_w - 1)) - 1
    scale = 1 << frac_bits
    raw = np.clip(np.round(np.asarray(v) * scale), lo, hi)
    return raw / scale


def build_graph(G, yaml, repo: Path, dataset: str, graph_weight: float):
    """Build the SGNN model for `dataset` with the driving weights rescaled."""
    cfg = G.default_config.copy()
    config_path = repo / "configs" / dataset / f"default_{dataset}_config.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        cfg.update(yaml.load(config_file))
    cfg["backend"] = "cpu"
    cfg["graph_weight"] = graph_weight
    graph = G.make_graph(cfg)
    graph.resolution_order = list(graph.topic_neurons)
    return graph, cfg


def _classify(topic_weights, resolution_order):
    topic_weights = sorted(topic_weights, key=lambda x: x[1], reverse=True)
    best_w = topic_weights[0][1]
    ties = [t for t, w in topic_weights if w == best_w]
    if resolution_order:
        ties = [k for k in resolution_order if k in ties]
    return ties


def software_infer(G, graph, paper_id):
    """Fixed-point-faithful software reference with GLOBAL STDP (hardware behavior).
    Fresh model per paper; weights re-quantized after every tick."""
    g = copy.deepcopy(graph)
    snn = g.snn
    snn.enable_stdp = [True] * len(snn.enable_stdp)          # global STDP
    snn.set_weights_from_mat(_q(snn.weight_mat(), FRAC_BITS, WEIGHT_W))
    snn.apos = list(_q(g.config["apos"], FRAC_BITS, WEIGHT_W))
    snn.aneg = list(_q(g.config["aneg"], FRAC_BITS, WEIGHT_W))
    pn = g.paper_neurons[paper_id]
    snn.add_spike(0, pn, float(_q(INPUT_SPIKE, FRAC_BITS, WEIGHT_W)))
    for _ in range(int(g.config["simtime"])):
        snn.simulate(1)
        snn.set_weights_from_mat(_q(snn.weight_mat(), FRAC_BITS, WEIGHT_W))
        snn.shorten_spike_train()
    tw = [(t, snn.get_synapse(pn, tn).weight) for t, tn in g.topic_neurons.items()]
    return g.papers[paper_id].label, _classify(tw, g.resolution_order)


def get_dataset_cap(dataset: str) -> int:
    """Convenience: the built SYN_CAP_PER_LANE for a dataset's bitstream."""
    from superneuromat import spikeengine as se
    return se.get_dataset(dataset).syn_cap_per_lane


def measure_timestep(dev, core_clk_hz: int = 100_000_000) -> dict:
    """Read the hardware cycle counters from the LAST executed tick and report
    the wall-clock time per timestep. Call right after a run_schedule/run_tick.

    Each SuperNeuroMAT tick is a "1 ms" biological timestep; the FPGA computes it
    in microseconds. This returns the compute (neuron+gather) and STDP cycle
    counts, the total wall time, and how it compares to the 1 ms budget.
    """
    tc = dev.read_tick_cycles()
    sc = dev.read_stdp_cycles_full()
    compute_us = tc / core_clk_hz * 1e6
    stdp_us = sc / core_clk_hz * 1e6
    total_us = compute_us + stdp_us
    return {
        "tick_cycles": tc, "stdp_cycles": sc,
        "compute_us": compute_us, "stdp_us": stdp_us, "total_us": total_us,
        "pct_of_1ms": total_us / 1000.0 * 100.0, "within_1ms": total_us < 1000.0,
    }


def hardware_infer(se, dev, graph, paper_id, syn_cap_per_lane=None, max_retries=6):
    """Per-paper on-chip inference: reload initial weights -> inject the paper's
    input current -> run `simtime` ticks with STDP -> read back the paper->topic
    weights -> classify.

    Reloading the full network (tens of thousands of UART commands) for EVERY
    paper occasionally hits a transient stream stall (observed: ~0.07% packet
    loss over a long burst on real hardware, e.g. "sent 48486/48486, recv
    48452/48486") -- not a protocol bug, just serial-link noise over a very long
    session. Retrying is safe because dev.soft_reset() unconditionally
    re-establishes a clean device state before every attempt (no partial state
    carries over from a failed load).

    max_retries raised 2 -> 6 (2026-07-31). A miniseer sweep on ZCU104 (120
    papers x 46,378 frames each) exhausted 2 retries on one paper and failed
    the whole run. The loss is random, not systematic: a standalone load at the
    full in-flight window succeeded in 1.4 s, and every reduced window (448,
    224, 112) also succeeded -- so this is not receive-buffer overflow, and
    shrinking the window would cost up to 5x throughput for no reliability
    gain. More attempts is the proportionate fix; each is ~1.4 s."""
    snn = graph.snn
    n = len(snn.neuron_thresholds)
    paper_idx = int(graph.paper_neurons[paper_id].idx)

    from .. import InferEngineError
    for attempt in range(max_retries + 1):
        try:
            dev.soft_reset()
            # syn_cap_per_lane matches the loaded bitstream's built capacity so the
            # host-side guard reflects the ACTUAL hardware bound (not the device's
            # dense default) -- an over-range synapse write silently corrupts on hw.
            info = se.load_network(dev, snn, frac_bits=FRAC_BITS, weight_w=WEIGHT_W,
                                   data_w=DATA_W, stdp_window=STDP_WINDOW,
                                   syn_cap_per_lane=syn_cap_per_lane)
            break
        except InferEngineError as exc:
            if attempt == max_retries:
                raise
            print(f"    [retry {attempt+1}/{max_retries}] transient load error on "
                  f"paper {paper_id}: {exc}")
    sched = {0: {paper_idx: INPUT_SPIKE}}
    se.run_schedule(dev, sched, total_ticks=int(graph.config["simtime"]),
                    frac_bits=FRAC_BITS, n_neurons=n)
    scale = float(1 << FRAC_BITS)
    topic_weights = []
    for topic, tn in graph.topic_neurons.items():
        topic_idx = int(tn.idx)
        _lane, idx = info["entry_index"][(topic_idx, paper_idx)]
        got_src, raw_w = dev.read_synapse(topic_idx, idx)
        assert got_src == paper_idx, f"readback src mismatch: {got_src} != {paper_idx}"
        topic_weights.append((topic, raw_w / scale))
    return graph.papers[paper_id].label, _classify(topic_weights, graph.resolution_order)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_BOARD))
    ap.add_argument("--board", default=None, help="override the default board for the dataset")
    ap.add_argument("--port", default="auto")
    ap.add_argument("--sgnn-repo", default=None)
    ap.add_argument("--no-hardware", action="store_true", help="software reference only")
    ap.add_argument("--num-test", type=int, default=None, help="limit number of test papers")
    args = ap.parse_args()

    repo = _find_sgnn_repo(args.sgnn_repo)
    sys.path.insert(0, str(repo))
    try:
        import gnn_citation_networks as G
        import xyaml as yaml
    except ModuleNotFoundError as exc:
        # The SGNN checkout brings its own third-party requirements (its xyaml/
        # package needs PyYAML, gnn_citation_networks needs networkx/pandas).
        # None of them are superneuromat dependencies, so a bare install has no
        # reason to carry them and the import fails with an unhelpful
        # "No module named 'yaml'" (seen 2026-07-31 validating from a fresh
        # wheel). Name the package and the fix instead. _find_sgnn_repo above
        # already handles the other half of this -- a missing checkout.
        raise SystemExit(
            f"missing dependency {exc.name!r}, required by the SGNN dataset sources "
            f"at {repo}, not by superneuromat itself.\n"
            f"  install: pip install superneuromat[datasets]\n"
            f"  or:      pip install {_PIP_NAME.get(exc.name, exc.name)}"
        ) from exc

    board = args.board or DATASET_BOARD[args.dataset]
    graph, _cfg = build_graph(G, yaml, repo, args.dataset, GRAPH_WEIGHT)
    papers = graph.selected_papers[: args.num_test] if args.num_test else graph.selected_papers
    print(f"{args.dataset}: {len(graph.snn.neuron_thresholds)} neurons, "
          f"{len(graph.snn.pre_synaptic_neuron_ids)} synapses, {len(papers)} test papers, board={board}")
    print(f"config: graph_weight={GRAPH_WEIGHT} input={INPUT_SPIKE} weight_w={WEIGHT_W} frac_bits={FRAC_BITS}")

    dev = None
    if not args.no_hardware:
        from superneuromat import spikeengine as se
        # the dataset bitstream's geometry (N_MAX/NUM_LANES) differs from the
        # board's packaged default, so size the device from the dataset catalogue.
        ds = se.get_dataset(args.dataset)
        print(f"connecting to {board} on {args.port} (N={ds.neurons}, K={ds.num_lanes}) ...")
        dev = se.connect(port=args.port, board=board, dataset=args.dataset)
        dev.clear_error()
    else:
        se = None

    sw_res, hw_res, agree = [], [], 0
    try:
        for i, pid in enumerate(papers):
            sw = software_infer(G, graph, pid)
            sw_res.append((sw, 0))
            if dev is not None:
                hw = hardware_infer(se, dev, graph, pid, syn_cap_per_lane=ds.syn_cap_per_lane)
                hw_res.append((hw, 0))
                agree += int(sw[1] == hw[1])
                if i < 8:
                    print(f"  paper {pid}: true={sw[0]} sw={sw[1]} hw={hw[1]} "
                          f"{'OK' if sw[1] == hw[1] else 'DIFFER'}")
    finally:
        # Close on ANY exit path (2026-07-31). Previously close() ran only
        # after the loop completed, so an error partway through -- a stream
        # stall on one paper, or Ctrl-C during a long run -- leaked the port.
        # On Windows the port then stays locked until the interpreter exits,
        # so the next attempt fails to open it.
        if dev is not None:
            dev.close()

    sw_acc = G.calculate_accuracy(sw_res, graph.resolution_order, "SW")
    print(f"\nSOFTWARE (fixed-point-faithful): one-vs-rest={sw_acc.accuracy:.4f} "
          f"top1={sw_acc.legacy}/{sw_acc.n}")
    if hw_res:
        hw_acc = G.calculate_accuracy(hw_res, graph.resolution_order, "HW")
        print(f"FPGA (on-chip):                 one-vs-rest={hw_acc.accuracy:.4f} "
              f"top1={hw_acc.legacy}/{hw_acc.n}")
        print(f"SW/HW per-paper agreement:      {agree}/{len(papers)}")


if __name__ == "__main__":
    main()
