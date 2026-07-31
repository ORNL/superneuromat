"""Digits STDP classifier: a generated (non-table) hardware example for the
NPU array (STDP) board, mirroring snm_capacity_example.py's pattern.

64 pixel inputs -> 10 output-class neurons, trained ON-CHIP via STDP (the
capability the inference-only NPU array in snm_npu.py does not have), then
scored with on-chip rate-readout inference against the sklearn digits test
set. Reuses this package's already hardware-validated recipe
(digits_stdp_e2e.py, network.py) rather than re-deriving it -- this module
is a thin GUI-facing wrapper, not a second implementation.

Hardware-validated this session: on-chip training is BIT-EXACT vs
SuperNeuroMAT (640/640 weights); on the full 899-image test set, one-vs-rest
accuracy is 0.909 (recall 0.776), matching software exactly; strict top-1
is 58.4%. See snm_npu_stdp.py's module docstring / _BOARDS["basys3"].note.
"""
from __future__ import annotations

from superneuromat.spikeengine import network as se_network
from superneuromat.spikeengine.examples import digits_stdp_e2e as e2e

DEFAULT_NUM_TEST = 200   # full 899 works but is slow for an interactive GUI run


def run_digits_classifier(dev, steps=None, progress=None,
                          num_train: int | None = None,
                          num_test: int | None = None) -> dict:
    """Train on-chip via STDP, then run on-chip rate-readout inference.

    `steps` is accepted for interface compatibility with the generated-
    example dispatcher (snm_gui.py's _run_generated_example) but unused here
    -- this example's own schedule length is derived from the training set
    size, not a fixed per-run tick count like the capacity stress example.
    """
    Xtr, Xte, ytr, yte, input_max = e2e._load_digits()
    num_train = num_train if num_train is not None else len(Xtr)
    num_test = num_test if num_test is not None else min(len(Xte), DEFAULT_NUM_TEST)

    net, _ins, _outs, sched, total = e2e.build_train_schedule(Xtr, ytr, num_train, input_max)
    if progress:
        progress("train:software-reference", 0, 1)
    sw_W = e2e.software_train(Xtr, ytr, num_train, input_max)

    dev.soft_reset()
    info = se_network.load_network(dev, net, frac_bits=e2e.FRAC_BITS, weight_w=e2e.WEIGHT_W,
                                   data_w=e2e.DATA_W, stdp_window=e2e.STDP_WINDOW)
    if progress:
        progress("train:on-chip", 0, total)
    se_network.run_schedule(dev, sched, total, frac_bits=e2e.FRAC_BITS, n_neurons=74)
    hw_w = se_network.read_weights(dev, info["entry_index"], frac_bits=e2e.FRAC_BITS)
    weight_mismatches = sum(1 for (d, s), v in hw_w.items() if v != sw_W[s, d])

    inf, iin, iout = e2e.build_infer_snn(sw_W)
    dev.soft_reset()
    se_network.load_network(dev, inf, frac_bits=e2e.FRAC_BITS, weight_w=e2e.WEIGHT_W,
                            data_w=e2e.DATA_W, stdp_window=e2e.STDP_WINDOW)
    dev.set_stdp_enable(False)

    correct = 0
    for k in range(num_test):
        if progress:
            progress("infer:on-chip", k, num_test)
        pred = e2e.hardware_infer(dev, iin, iout, Xte[k], input_max)
        correct += int(pred == int(yte[k]))
    acc, recall = e2e.one_vs_rest(
        lambda img: e2e.hardware_infer_answers(dev, iin, iout, img, input_max),
        Xte[:num_test], yte[:num_test])

    summary = (
        f"digits STDP classifier: on-chip trained on {num_train} images "
        f"({weight_mismatches}/640 weight mismatches vs SuperNeuroMAT -> "
        f"{'BIT-EXACT' if weight_mismatches == 0 else 'MISMATCH'}); on-chip "
        f"inference on {num_test} test images: strict top-1 {correct}/{num_test} "
        f"= {correct / num_test * 100:.1f}%, one-vs-rest accuracy {acc:.3f} "
        f"(recall {recall:.3f})."
    )
    return {
        "summary": summary,
        "weight_mismatches": weight_mismatches,
        "top1": correct / num_test,
        "one_vs_rest_acc": acc,
        "one_vs_rest_recall": recall,
        "num_train": num_train,
        "num_test": num_test,
    }
