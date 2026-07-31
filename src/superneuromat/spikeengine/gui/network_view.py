"""2D network view widget (matplotlib inside Qt).

Draws the loaded spiking network as a directed graph: neurons positioned by
``graph_layout``, synapses as arrows annotated with their weight, and neurons
that fired on the current tick highlighted. Switches to a lighter-weight
rendering path (``_draw_network_large``) past a size threshold, so a
few-thousand-neuron network still draws in reasonable time.

This is the always-available view. ``pyvista_network_view`` provides an
optional 3D alternative and falls back here when its dependencies are absent.
"""
from __future__ import annotations

import math

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .graph_layout import network_positions


def _weight_label(s, learned):
    label = f"{s['weight']:g}"
    if learned is not None and s.get("idx") in (learned or {}) and learned[s["idx"]] != s["weight"]:
        label = f"{s['weight']:g}->{learned[s['idx']]:g}"
    return label


def _draw_directed_synapses(ax, pos, synapses, learned=None, *, label_limit=900,
                            arrow_limit=2500, weight_font=6, curved=True):
    valid = [s for s in synapses if s.get("pre") in pos and s.get("post") in pos]
    if not valid:
        return 0

    labelled = 0
    for idx, s in enumerate(valid):
        pre = s["pre"]
        post = s["post"]
        x1, y1 = pos[pre]
        x2, y2 = pos[post]
        if pre == post:
            p1 = (x1 - 0.05, y1 + 0.08)
            p2 = (x1 + 0.05, y1 + 0.08)
            patch = FancyArrowPatch(
                p1,
                p2,
                connectionstyle="arc3,rad=2.0",
                arrowstyle="-|>",
                mutation_scale=8,
                color="#617283",
                lw=1.0,
                alpha=0.75,
                zorder=1,
            )
            ax.add_patch(patch)
            lx, ly = x1, y1 + 0.26
        else:
            if idx < arrow_limit:
                rad = 0.12 if curved and pre < post else -0.12 if curved else 0.0
                patch = FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>",
                    mutation_scale=8,
                    color="#617283",
                    lw=0.9,
                    alpha=0.42,
                    shrinkA=8,
                    shrinkB=8,
                    zorder=1,
                )
                ax.add_patch(patch)
            lx, ly = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if labelled < label_limit:
            ax.text(
                lx,
                ly,
                _weight_label(s, learned),
                fontsize=weight_font,
                color="#1a3e8c",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.72},
                zorder=2,
            )
            labelled += 1
    return labelled


def _draw_network_large(ax, neurons, synapses, fired, learned=None):
    n = len(neurons)
    pos, ncomp, cols = network_positions(neurons, synapses)
    rows = math.ceil(ncomp / cols)
    node_size = max(14.0, min(120.0, 18000.0 / max(n, 1)))

    labelled_edges = _draw_directed_synapses(
        ax, pos, synapses, learned, label_limit=900, arrow_limit=2500, weight_font=5, curved=True
    )

    xs, ys, colors, edgecolors, widths = [], [], [], [], []
    for i in range(n):
        if i not in pos:
            continue
        x, y = pos[i]
        xs.append(x)
        ys.append(y)
        inp = neurons[i].get("input")
        colors.append("#ff6b5e" if i in fired else "#7fb8e6")
        edgecolors.append("#1a8a1a" if inp else "#33465a")
        widths.append(1.0 if (i in fired or inp) else 0.3)
    ax.scatter(xs, ys, s=node_size, c=colors, edgecolors=edgecolors, linewidths=widths, alpha=0.96, zorder=3)

    labelled = 0
    for i in sorted(set(fired) | {j for j, nd in enumerate(neurons) if nd.get("input")}):
        if i not in pos or labelled >= 40:
            continue
        x, y = pos[i]
        ax.text(x, y + 0.02, str(i), ha="center", va="center", fontsize=5, color="#111", zorder=4)
        labelled += 1

    note = ""
    if len(synapses) > labelled_edges:
        note = f", labels shown for {labelled_edges}/{len(synapses)} edges"
    ax.set_xlim(-0.15, cols + 0.15)
    ax.set_ylim(-(rows + 0.15), 0.15)
    ax.set_aspect("equal")
    ax.set_title(
        f"{n} neurons in {ncomp} connected group(s), {len(synapses)} synapses{note} "
        "(circles=neurons, arrows=synapses, text=weight)",
        fontsize=9,
    )


def draw_network(ax, neurons, synapses, fired=None, learned=None, layout="graph"):
    del layout
    fired = set(fired or [])
    ax.clear()
    if not neurons:
        ax.text(0.5, 0.5, "add neurons to see the network", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.set_axis_on()
    _draw_network_large(ax, neurons, synapses, fired, learned)


class MatplotlibNetworkView(QWidget):
    """Legacy network viewer wrapped behind a widget boundary for later replacement."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.net_fig = Figure(figsize=(5, 4))
        self.net_canvas = FigureCanvas(self.net_fig)
        self.net_ax = self.net_fig.add_subplot(111)
        self._net_home = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        zbar = QHBoxLayout()
        graph_lbl = QLabel("Graph: topology layout, neurons = circles, synapses = arrows, edge text = weight")
        graph_lbl.setToolTip("One topology-based graph view for every network size. Use zoom, pan, and Fit to navigate.")
        zbar.addWidget(graph_lbl)

        b_in = QPushButton("Zoom +")
        b_in.clicked.connect(lambda: self.zoom(0.8))
        b_out = QPushButton("Zoom -")
        b_out.clicked.connect(lambda: self.zoom(1.25))
        b_fit = QPushButton("Fit")
        b_fit.clicked.connect(self.fit_view)
        for button in (b_in, b_out, b_fit):
            button.setFixedWidth(70)
            zbar.addWidget(button)
        zbar.addWidget(QLabel("scroll to zoom · pan tool below to drag"))
        zbar.addStretch(1)
        root.addLayout(zbar)

        self.net_toolbar = NavigationToolbar(self.net_canvas, self)
        root.addWidget(self.net_toolbar)
        root.addWidget(self.net_canvas, 1)
        self.net_canvas.mpl_connect("scroll_event", self._on_scroll)

    def set_network(self, neurons, synapses, fired=None, learned=None):
        draw_network(self.net_ax, neurons, synapses, fired=fired, learned=learned, layout="graph")
        self.net_fig.tight_layout()
        self.net_canvas.draw()
        self._net_home = (self.net_ax.get_xlim(), self.net_ax.get_ylim())

    def zoom(self, factor, cx=None, cy=None):
        ax = self.net_ax
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if cx is None:
            cx = (x0 + x1) / 2
        if cy is None:
            cy = (y0 + y1) / 2
        ax.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
        ax.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
        self.net_canvas.draw_idle()

    def fit_view(self):
        if self._net_home:
            (x0, x1), (y0, y1) = self._net_home
            self.net_ax.set_xlim(x0, x1)
            self.net_ax.set_ylim(y0, y1)
            self.net_canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is not self.net_ax:
            return
        self.zoom(0.8 if event.button == "up" else 1.25, event.xdata, event.ydata)
