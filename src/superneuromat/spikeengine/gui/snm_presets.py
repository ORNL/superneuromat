"""Prebuilt SNN test cases for the GUI (and scripts).

Each preset is plain data describing a small network the GUI can load straight
into its tables: neurons, synapses, optional STDP, per-neuron input spike trains,
step count, and fixed-point format. They mirror the snm_examples.py gallery.

A preset dict has:
    name, desc, frac_bits, steps
    neurons : [ {threshold, leak, reset, refractory, input}, ... ]   (index = id)
    synapses: [ {pre, post, weight, stdp}, ... ]
    stdp    : None | {window, apos:[...], aneg:[...]}
    inputs  : [ {neuron, value, steps:[...]}, ... ]
"""

from __future__ import annotations

import functools
import operator


def _n(threshold=0, leak=0, reset=0, refractory=0, input=False):
    return {"threshold": threshold, "leak": leak, "reset": reset,
                "refractory": refractory, "input": input}


def _s(pre, post, weight=1, stdp=False):
    return {"pre": pre, "post": post, "weight": weight, "stdp": stdp}


def _in(neuron, value, steps):
    return {"neuron": neuron, "value": value, "steps": list(steps)}


def _capacity_mixed_1400():
    """Full-neuron-count structured stress case.

    200 deterministic 7-neuron motifs cover coincidence, inhibition, refractory
    suppression, relay propagation, and local STDP without blowing up the view
    into an unstructured blob.
    """
    neurons, synapses, inputs = [], [], []
    for m in range(200):
        base = 7 * m
        neurons.extend([
            _n(threshold=0, input=True),                 # base + 0
            _n(threshold=0, input=True),                 # base + 1
            _n(threshold=3, leak=1),                     # base + 2
            _n(threshold=0, refractory=1),               # base + 3
            _n(threshold=3, leak=1),                     # base + 4
            _n(threshold=2, refractory=2),               # base + 5
            _n(threshold=4, leak=1, refractory=1),       # base + 6
        ])
        synapses.extend([
            _s(base + 0, base + 2, 2),
            _s(base + 1, base + 2, 2),
            _s(base + 1, base + 3, 1),
            _s(base + 3, base + 2, -2),
            _s(base + 2, base + 4, 2, stdp=True),
            _s(base + 0, base + 4, 1),
            _s(base + 4, base + 5, 2),
            _s(base + 5, base + 6, 2),
            _s(base + 2, base + 6, 1),
        ])
        phase = m % 12
        inputs.append(_in(base + 0, 2 + (m % 2), [phase, phase + 12]))
        inputs.append(_in(base + 1, 2, [((3 * m + 1) % 12), ((3 * m + 7) % 12) + 12]))
    return {
        "desc": ("Full-capacity mixed-behavior showcase: 1400 neurons arranged as 200 local motifs "
              "covering coincidence, inhibition, refractory holdoff, relay propagation, and "
              "sparse STDP on visible subgraphs."),
        "frac_bits": 0,
        "steps": 24,
        "neurons": neurons,
        "synapses": synapses,
        "stdp": {"window": 2, "apos": [2, 1], "aneg": [-1, -1]},
        "inputs": inputs,
    }


def _capacity_synmix_32768():
    """Exact-SYN_DEPTH structured stress case with mixed signs and sparse STDP."""
    n_neurons = 1400
    neurons = [
        _n(
            threshold=(1 if i < 32 else (4 if i % 11 == 0 else 3)),
            leak=(0 if i < 32 else (2 if i % 9 == 0 else 1)),
            refractory=(2 if i % 17 == 0 else (1 if i % 7 == 0 else 0)),
            input=(i < 32),
        )
        for i in range(n_neurons)
    ]
    synapses = []
    for pre in range(n_neurons):
        fanout = 24 if pre < 568 else 23   # 568*24 + 832*23 = 32768
        base_offset = pre // 35
        for hop in range(1, fanout + 1):
            post = (pre + hop * 7 + base_offset) % n_neurons
            if post == pre:
                post = (post + 1) % n_neurons
            weight = (-3, -2, -1, 1, 2, 3)[(pre + hop) % 6]
            synapses.append(_s(pre, post, weight, stdp=(pre % 29 == 0 and hop % 11 == 0)))
    inputs = []
    for group in range(4):
        start = 8 * group
        for idx in range(start, start + 8):
            inputs.append(_in(idx, 3 if idx % 2 == 0 else 2, [2 * group, 2 * group + 8]))
    return {
        "desc": ("Exact-capacity synapse stress case: 1400 neurons with exactly 32,768 directed synapses, "
              "mixed inhibitory/excitatory edges, staggered inputs, refractory variation, and sparse STDP tags."),
        "frac_bits": 0,
        "steps": 16,
        "neurons": neurons,
        "synapses": synapses,
        "stdp": {"window": 2, "apos": [2, 1], "aneg": [-1, -1]},
        "inputs": inputs,
    }


PRESETS = {
    "leak_spike": {
        "desc": "One neuron integrates +8/tick past threshold 20 -> spikes & resets.",
        "frac_bits": 0, "steps": 8,
        "neurons": [_n(threshold=20, leak=0, input=True)],
        "synapses": [], "stdp": None,
        "inputs": [_in(0, 8, range(8))],
    },
    "leak_blocks_spike": {
        "desc": "Two neurons, same input; the leaky one (n1) never reaches threshold.",
        "frac_bits": 0, "steps": 5,
        "neurons": [_n(threshold=15, leak=0, input=True),
                 _n(threshold=15, leak=5, input=True)],
        "synapses": [], "stdp": None,
        "inputs": [_in(0, 6, range(5)), _in(1, 6, range(5))],
    },
    "refractory": {
        "desc": "Neuron spikes, then is silenced for 2 ticks (refractory) despite input.",
        "frac_bits": 0, "steps": 6,
        "neurons": [_n(threshold=0, refractory=2, input=True)],
        "synapses": [], "stdp": None,
        "inputs": [_in(0, 1, range(6))],
    },
    "reset": {
        "desc": "Neuron fires once and its membrane snaps to reset = -3.",
        "frac_bits": 0, "steps": 2,
        "neurons": [_n(threshold=10, reset=-3, input=True)],
        "synapses": [], "stdp": None,
        "inputs": [_in(0, 12, [0])],
    },
    "synapse": {
        "desc": "n0 drives n1 through a weight-10 synapse; n1 fires one tick later.",
        "frac_bits": 0, "steps": 4,
        "neurons": [_n(threshold=5, input=True), _n(threshold=5)],
        "synapses": [_s(0, 1, weight=10)], "stdp": None,
        "inputs": [_in(0, 10, range(4))],
    },
    "and_gate": {
        "desc": "2-input AND: n2 fires only after BOTH n0 and n1 fired the previous tick.",
        "frac_bits": 0, "steps": 7,
        "neurons": [_n(threshold=0, input=True), _n(threshold=0, input=True),
                 _n(threshold=1, leak=10)],
        "synapses": [_s(0, 2, weight=1), _s(1, 2, weight=1)], "stdp": None,
        "inputs": [_in(0, 1, [3, 5]), _in(1, 1, [1, 5])],
    },
    "or_gate": {
        "desc": "2-input OR: n2 fires when either n0 or n1 fired on the previous tick.",
        "frac_bits": 0, "steps": 7,
        "neurons": [_n(threshold=0, input=True), _n(threshold=0, input=True),
                 _n(threshold=0, leak=10)],
        "synapses": [_s(0, 2, weight=1), _s(1, 2, weight=1)], "stdp": None,
        "inputs": [_in(0, 1, [1, 5]), _in(1, 1, [3, 5])],
    },
    "pingpong": {
        "desc": "Recurrent loop n0<->n1: one seed spike bounces back and forth.",
        "frac_bits": 0, "steps": 8,
        "neurons": [_n(threshold=0, input=True), _n(threshold=0)],
        "synapses": [_s(0, 1, weight=1), _s(1, 0, weight=1)], "stdp": None,
        "inputs": [_in(0, 1, [0])],
    },
    "multilayer_pipeline": {
        "desc": "Four-layer feed-forward pipeline: two inputs fan into hidden layers and converge on one output.",
        "frac_bits": 0, "steps": 8,
        "neurons": [
            _n(threshold=0, input=True), _n(threshold=0, input=True),
            _n(threshold=1, leak=10), _n(threshold=1, leak=10), _n(threshold=1, leak=10),
            _n(threshold=1, leak=10), _n(threshold=1, leak=10), _n(threshold=2, leak=10),
        ],
        "synapses": [
            _s(0, 2, 1), _s(0, 3, 1), _s(1, 3, 1), _s(1, 4, 1),
            _s(2, 5, 1), _s(3, 5, 1), _s(3, 6, 1), _s(4, 6, 1),
            _s(5, 7, 1), _s(6, 7, 1),
        ],
        "stdp": None,
        "inputs": [_in(0, 1, [0, 4]), _in(1, 1, [1, 4])],
    },
    "all_to_all_3d": {
        "desc": "Dense 3D visualization case: 24 neurons with all-to-all directed synapses (552 total) for inspecting the general volumetric view.",
        "frac_bits": 0, "steps": 6,
        "neurons": [_n(threshold=2, leak=1, input=(i < 6)) for i in range(24)],
        "synapses": [
            _s(pre, post, weight=((pre + post) % 5) - 2)
            for pre in range(24)
            for post in range(24)
            if pre != post
        ],
        "stdp": None,
        "inputs": [
            _in(0, 3, [0, 3]),
            _in(1, 2, [1, 4]),
            _in(2, 3, [2]),
            _in(3, 2, [0, 5]),
            _in(4, 3, [1]),
            _in(5, 2, [2, 4]),
        ],
    },
    "stdp": {
        "desc": "Pre->post pairing potentiates the synapse (weight 1 -> 3).",
        "frac_bits": 0, "steps": 3,
        "neurons": [_n(threshold=0, input=True), _n(threshold=0, input=True)],
        "synapses": [_s(0, 1, weight=1, stdp=True)],
        "stdp": {"window": 1, "apos": [2], "aneg": [-1]},
        "inputs": [_in(0, 10, [0]), _in(1, 10, [1])],
    },
    # ---- extreme / stress cases (mirror snm_examples.py) ----
    "fanin": {
        "desc": "High fan-in: inputs n0..n14 drive output n15; eight simultaneous inputs make neuron 15 fire.",
        "frac_bits": 0, "steps": 4,
        "neurons": [_n(threshold=0, input=True) for _ in range(15)]
                + [_n(threshold=75, leak=100)],
        "synapses": [_s(i, 15, weight=10) for i in range(15)], "stdp": None,
        "inputs": [_in(i, 1, [0]) for i in range(7)]            # 7 fire @t0 -> no spike
              + [_in(i, 1, [2]) for i in range(8)],          # 8 fire @t2 -> spike @t3
    },
    "inhibition": {
        "desc": "Negative-weight veto: n0->n2 (+10) excites, n1->n2 (-10) cancels it.",
        "frac_bits": 0, "steps": 4,
        "neurons": [_n(threshold=0, input=True), _n(threshold=0, input=True),
                 _n(threshold=5, leak=10)],
        "synapses": [_s(0, 2, weight=10), _s(1, 2, weight=-10)], "stdp": None,
        "inputs": [_in(0, 1, [0, 2]), _in(1, 1, [2])],
    },
    "stdp_saturate": {
        "desc": "STDP into the ceiling: weight 100 +10/pairing -> clamps at +127 (no wrap).",
        "frac_bits": 0, "steps": 6,
        "neurons": [_n(threshold=0, input=True), _n(threshold=0, input=True)],
        "synapses": [_s(0, 1, weight=100, stdp=True)],
        "stdp": {"window": 1, "apos": [10], "aneg": [-1]},
        "inputs": [_in(0, 10, [0, 2, 4]), _in(1, 10, [1, 3, 5])],
    },
    "decimal_leak": {
        "desc": "Fixed-point demo: threshold 2.5, leak 0.5, weight 2.5 (frac_bits=3).",
        "frac_bits": 3, "steps": 6,
        "neurons": [_n(threshold=2.5, leak=0.5, input=True), _n(threshold=1.5)],
        "synapses": [_s(0, 1, weight=2.5)], "stdp": None,
        "inputs": [_in(0, 1.0, range(6))],
    },
    "behavioral_1024": {
        "desc": "1024-neuron behavioral stress test: 256 four-neuron motifs cycling through chain, fan-in, leak, and inhibition.",
        "frac_bits": 0, "steps": 22,
        "neurons": (
            functools.reduce(operator.iadd, [
                [_n(threshold=1, leak=0), _n(threshold=1, leak=0), _n(threshold=1, leak=0), _n(threshold=1, leak=0)]
                if m % 4 == 0 else
                [_n(threshold=1, leak=0), _n(threshold=1, leak=0), _n(threshold=3, leak=0), _n(threshold=1, leak=0)]
                if m % 4 == 1 else
                [_n(threshold=5, leak=1), _n(threshold=1, leak=0), _n(threshold=1, leak=0), _n(threshold=1, leak=0)]
                if m % 4 == 2 else
                [_n(threshold=1, leak=0), _n(threshold=1, leak=0), _n(threshold=1, leak=0), _n(threshold=1, leak=0)]
                for m in range(256)
            ], [])
        ),
        "synapses": (
            functools.reduce(operator.iadd, [
                [_s(4*m + 0, 4*m + 1, 2), _s(4*m + 1, 4*m + 2, 2), _s(4*m + 2, 4*m + 3, 2)]
                if m % 4 == 0 else
                [_s(4*m + 0, 4*m + 2, 2), _s(4*m + 1, 4*m + 2, 2)]
                if m % 4 == 1 else
                []
                if m % 4 == 2 else
                [_s(4*m + 0, 4*m + 2, 2), _s(4*m + 1, 4*m + 2, -2)]
                for m in range(256)
            ], [])
        ),
        "stdp": None,
        "inputs": (
            functools.reduce(operator.iadd, [
                [_in(4*m + 0, 2, [m % 16])]
                if m % 4 == 0 else
                [_in(4*m + 0, 2, [m % 16]), _in(4*m + 1, 2, [m % 16])]
                if m % 4 == 1 else
                [_in(4*m + 0, 3, [m % 16]), _in(4*m + 0, 3, [m % 16 + 1]), _in(4*m + 0, 4, [m % 16 + 3])]
                if m % 4 == 2 else
                [_in(4*m + 0, 2, [m % 16]), _in(4*m + 1, 2, [m % 16])]
                for m in range(256)
            ], [])
        ),
    },
    "capacity_mixed_1400": _capacity_mixed_1400(),
    "max_neurons_1400": {
        "desc": "Packaged Basys3 neuron-capacity example: 1400 neurons with a small driven front end so the full neuron table can be exercised.",
        "frac_bits": 0, "steps": 6,
        "neurons": [_n(threshold=1, leak=0, input=(i < 8)) for i in range(1400)],
        "synapses": [_s(i, i + 8, 2) for i in range(8)],
        "stdp": None,
        "inputs": [_in(i, 2, [0]) for i in range(8)],
    },
    "capacity_synmix_32768": _capacity_synmix_32768(),
    "max_synapse_32768": {
        "desc": "Packaged Basys3 synapse-capacity example: 1024 neurons wired with exactly 32,768 synapses.",
        "frac_bits": 0, "steps": 6,
        "neurons": [_n(threshold=1, leak=0) for _ in range(1024)],
        "synapses": (
            [_s(32*L + src, 32*(L+1) + dst, 1)
             for L in range(31) for src in range(32) for dst in range(32)]
            + [_s(src, 64 + dst, 1) for src in range(32) for dst in range(32)]
        ),
        "stdp": None,
        "inputs": [_in(i, 2, [0]) for i in range(32)],
    },
}


def names():
    return list(PRESETS)


# ---- generated (non-table) examples: board-gated, streamed straight to hardware ----
# These are NOT editable table presets (their networks are too large to render as rows).
# Each entry: {generated: True, boards: [...], desc, runner_module}. The GUI shows a
# generated example only when the selected board is in `boards`, and dispatches BUILD & RUN
# to the named runner module's run_full_capacity(dev, steps, progress) instead of the tables.
GENERATED = {
    "zcu104_full_capacity": {
        "generated": True,
        "boards": ["zcu104"],
        "steps": 10,
        "runner_module": "superneuromat.spikeengine.gui.snm_capacity_example",
        "desc": ("ZCU104 full-capacity hardware stress: configures ALL 4096 neurons and ALL "
              "1,179,648 synapses (dense), then drives input across the array for 10 timesteps "
              "and reads the spike output each step. Generated and streamed straight to the "
              "board -- not shown as editable tables. Requires a live ZCU104 connection."),
    },
    "npu_stdp_digits_classifier": {
        "generated": True,
        "boards": ["npu-stdp:basys3"],
        "entry_point": "run_digits_classifier",
        "runner_module": "superneuromat.spikeengine.gui.snm_digits_example",
        "desc": ("Digits STDP classifier: 64 pixel inputs -> 10 output-class neurons, trained "
              "ON-CHIP via STDP (3-tap depression rule) on the sklearn digits dataset, then "
              "scored with on-chip rate-readout inference -- the on-chip-learning capability "
              "the inference-only NPU array does not have. Hardware-validated bit-exact "
              "training vs SuperNeuroMAT; full-test-set one-vs-rest accuracy 0.909 (recall "
              "0.776), strict top-1 58.4%. Generated and streamed straight to the board -- "
              "not shown as editable tables (640 synapses over ~900 training ticks)."),
    },
}


def is_generated(name) -> bool:
    return name in GENERATED


def names_for_board(board=None):
    """Example names visible for `board`: all editable presets, plus any generated
    example whose `boards` list includes this board. `board` None -> generated examples
    with no board restriction only."""
    out = list(PRESETS)
    for key, meta in GENERATED.items():
        allowed = meta.get("boards")
        if not allowed or (board in allowed):
            out.append(key)
    return out


def get(name):
    if name in GENERATED:
        return GENERATED[name]
    return PRESETS[name]
