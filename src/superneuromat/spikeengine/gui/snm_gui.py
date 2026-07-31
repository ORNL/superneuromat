"""SpikeEngine - SNN Console (PyQt5).

The desktop console for SpikeEngine: A Scalable FPGA Platform for Spiking Neural
Networks (built on the SuperNeuroMAT model).

A desktop app to build a spiking network (neurons + synapses + STDP), set per-
neuron input spike trains, run it on the FPGA (or a software preview), and see
the network graph, spike raster, and learned weights. Decimals are supported via
a selectable fixed-point format; inputs are bounds-checked against the chip.

    py snm_gui.py            (needs PyQt5, matplotlib; pyserial for the board)
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import snm_boards, snm_npu, snm_npu_stdp, snm_presets
from . import snm_config as cfg
from . import snm_network as nm
from . import snm_snn_io as snn_io
from .network_view import MatplotlibNetworkView
from .pyvista_network_view import PyVistaNetworkView, pyvista_available
from .snm_network import HW, SNN, fixed_range, format_raster_axes, spike_table

BIO_TIMESTEP_S = 1e-3

UI = {
    "bg": "#F7F9FC",
    "panel": "#FFFFFF",
    "border": "#A9B4C2",
    "border_soft": "#C7D0DB",
    "text": "#15202B",
    "text_muted": "#445468",
    "text_soft": "#5F6C7B",
    "primary": "#0B5FFF",
    "primary_dark": "#0A4BCC",
    "success": "#166534",
    "success_soft": "#EAF7EE",
    "danger": "#B42318",
    "danger_dark": "#8E1C12",
    "danger_soft": "#FDECEC",
    "warning": "#9A3412",
    "warning_soft": "#FFF3E8",
    "teal": "#0F766E",
    "purple": "#5B21B6",
    "blue": "#0F4C81",
    "magenta": "#9D174D",
    "orange": "#C2410C",
    "compare_bg": "#F3F8FF",
    "compare_border": "#87A8D8",
}

SECTION_COLORS = {
    "connection": UI["success"],
    "fixed_point": UI["teal"],
    "examples": UI["purple"],
    "neurons": UI["blue"],
    "synapses": UI["magenta"],
    "inputs": UI["orange"],
    "run": UI["danger"],
}

# --------------------------------------------------------------------------
# Background worker (programming + running is slow over UART)
# --------------------------------------------------------------------------
class RunWorker(QThread):
    done = pyqtSignal(object)          # dict on success, Exception on failure

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:          # noqa: BLE001 - report to UI
            self.done.emit(e)


# --------------------------------------------------------------------------
# Network graph (manual circular layout; no networkx dependency)
# --------------------------------------------------------------------------
def _network_positions(neurons, synapses):
    """Structure-revealing 2D layout (no networkx). Split the graph into connected
    components, lay each out left->right by longest-path depth (feed-forward columns), and
    tile the components in a grid. A chain shows as a row, a layered net as clean columns,
    and 256 little motifs as a grid of separated clusters -- instead of everything collapsed
    onto one ring. Deterministic. Returns (pos, n_components, n_grid_cols)."""
    import math
    from collections import deque
    n = len(neurons)
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    adj = [set() for _ in range(n)]
    for s in synapses:
        a, b = s.get("pre"), s.get("post")
        if isinstance(a, int) and isinstance(b, int) and 0 <= a < n and 0 <= b < n and a != b:
            succ[a].append(b); indeg[b] += 1
            adj[a].add(b); adj[b].add(a)

    # connected components (undirected BFS)
    comp = [-1] * n
    ncomp = 0
    for s0 in range(n):
        if comp[s0] != -1:
            continue
        comp[s0] = ncomp; stack = [s0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = ncomp; stack.append(v)
        ncomp += 1
    members = [[] for _ in range(ncomp)]
    for i in range(n):
        members[comp[i]].append(i)

    # longest-path depth per node (Kahn topo); nodes in cycles keep depth 0
    depth = [0] * n
    rem = indeg[:]
    q = deque(i for i in range(n) if rem[i] == 0)
    while q:
        u = q.popleft()
        for v in succ[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            rem[v] -= 1
            if rem[v] == 0:
                q.append(v)

    # tile components in a grid (biggest first for a tidy arrangement)
    comp_order = sorted(range(ncomp), key=lambda c: (-len(members[c]), c))
    cols = max(1, math.ceil(math.sqrt(ncomp)))
    pos = {}
    CELL = 0.9                      # fraction of a grid cell the component fills
    for slot, c in enumerate(comp_order):
        gx, gy = slot % cols, slot // cols
        mem = members[c]
        dmin = min(depth[i] for i in mem); dmax = max(depth[i] for i in mem)
        dspan = max(1, dmax - dmin)
        by_depth = {}
        for i in mem:
            by_depth.setdefault(depth[i], []).append(i)
        for d, group in by_depth.items():
            group.sort()
            lx = (d - dmin) / dspan
            for j, i in enumerate(group):
                ly = (j + 0.5) / len(group)
                pos[i] = (gx + 0.05 + lx * CELL, -(gy + 0.05 + ly * CELL))
    return pos, ncomp, cols


def _draw_network_large(ax, neurons, synapses, fired):
    import math
    n = len(neurons)
    pos, ncomp, cols = _network_positions(neurons, synapses)
    rows = math.ceil(ncomp / cols)
    node_size = max(5.0, min(80.0, 16000.0 / max(n, 1)))

    if synapses:                    # edges as light directed segments
        qx, qy, qu, qv = [], [], [], []
        for s in synapses:
            a, b = s.get("pre"), s.get("post")
            if a in pos and b in pos and a != b:
                x1, y1 = pos[a]; x2, y2 = pos[b]
                qx.append(x1); qy.append(y1); qu.append(x2 - x1); qv.append(y2 - y1)
        if qx:
            ax.quiver(qx, qy, qu, qv, angles="xy", scale_units="xy", scale=1,
                      width=0.0016, headwidth=3.0, headlength=4.0, headaxislength=3.2,
                      color="#5b6b7a", alpha=0.28, zorder=1)

    xs, ys, colors, edgecolors, widths = [], [], [], [], []
    for i in range(n):
        if i not in pos:
            continue
        x, y = pos[i]; xs.append(x); ys.append(y)
        inp = neurons[i].get("input")
        colors.append("#ff6b5e" if i in fired else "#7fb8e6")
        edgecolors.append("#1a8a1a" if inp else "#33465a")
        widths.append(0.8 if (i in fired or inp) else 0.15)
    ax.scatter(xs, ys, s=node_size, c=colors, edgecolors=edgecolors,
               linewidths=widths, alpha=0.95, zorder=3)

    labelled = 0                    # label only inputs + fired (cap), else it's noise
    for i in sorted(set(fired) | {j for j, nd in enumerate(neurons) if nd.get("input")}):
        if i not in pos or labelled >= 40:
            continue
        x, y = pos[i]
        ax.text(x, y + 0.02, str(i), ha="center", va="center", fontsize=5, color="#111", zorder=4)
        labelled += 1

    ax.set_xlim(-0.15, cols + 0.15); ax.set_ylim(-(rows + 0.15), 0.15); ax.set_aspect("equal")
    ax.set_title(f"{n} neurons in {ncomp} connected group(s), {len(synapses)} synapses "
                 "(green ring=input, red=fired) · scroll or buttons to zoom", fontsize=9)


def _draw_circle(ax, neurons, synapses, fired, learned):
    """Classic circular layout: neurons as labelled dots on a ring, synapses as directed
    arrows carrying the weight (and learned weight, if any). Pan/zoom lets you read a busy one."""
    import math
    n = len(neurons)
    pos = {}
    for i in range(n):
        if n == 1:
            pos[i] = (0.0, 0.0)
        else:
            # neuron 0 on the LEFT, index increasing clockwise, so pre->post reads left->right.
            a = 2 * math.pi * i / n
            pos[i] = (-math.cos(a), math.sin(a))

    detailed = n <= 160 and len(synapses) <= 500     # arrows + weight/id labels stay readable
    if detailed:
        for s in synapses:
            if s["pre"] not in pos or s["post"] not in pos:
                continue
            x1, y1 = pos[s["pre"]]; x2, y2 = pos[s["post"]]
            if s["pre"] == s["post"]:
                p1 = (x1 - 0.06, y1 + 0.11); p2 = (x1 + 0.06, y1 + 0.11)
                ax.add_patch(FancyArrowPatch(
                    p1, p2, connectionstyle="arc3,rad=2.6", arrowstyle="-|>",
                    mutation_scale=12, color="gray", lw=1.5, zorder=2))
                lx, ly = x1, y1 + 0.40
            else:
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops={"arrowstyle": "-|>", "color": "#37474f",
                                            "lw": 1.8, "mutation_scale": 24,
                                            "shrinkA": 17, "shrinkB": 19,
                                            "connectionstyle": "arc3,rad=0.16"})
                lx, ly = 0.45 * x1 + 0.55 * x2, 0.45 * y1 + 0.55 * y2
            lbl = f"{s['weight']:g}"
            if learned is not None and s.get("idx") in (learned or {}) and learned[s["idx"]] != s["weight"]:
                lbl = f"{s['weight']:g}→{learned[s['idx']]:g}"
            ax.text(lx, ly, lbl, fontsize=7, color="#1a3e8c", ha="center",
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.7})
        r = 0.13 if n <= 40 else max(0.03, 5.2 / n)
        for i, (x, y) in pos.items():
            face = "#ff6b5e" if i in fired else "#7fb8e6"
            edge = "#1a8a1a" if neurons[i].get("input") else "#333"
            ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=edge, linewidth=2.0, zorder=3))
            ax.text(x, y, str(i), ha="center", va="center", zorder=4,
                    fontsize=9 if n <= 40 else max(4.0, 260.0 / n), fontweight="bold")
        title = "network (green ring = input, red = fired) · circle layout"
    else:
        qx, qy, qu, qv = [], [], [], []
        for s in synapses:
            if s["pre"] in pos and s["post"] in pos:
                x1, y1 = pos[s["pre"]]; x2, y2 = pos[s["post"]]
                qx.append(x1); qy.append(y1); qu.append(x2 - x1); qv.append(y2 - y1)
        if qx:
            ax.quiver(qx, qy, qu, qv, angles="xy", scale_units="xy", scale=1,
                      width=0.0009, headwidth=3.0, color="#5b6b7a", alpha=0.18, zorder=1)
        xs = [pos[i][0] for i in range(n)]; ys = [pos[i][1] for i in range(n)]
        colors = ["#ff6b5e" if i in fired else "#7fb8e6" for i in range(n)]
        edges = ["#1a8a1a" if neurons[i].get("input") else "#334" for i in range(n)]
        ax.scatter(xs, ys, s=max(5.0, min(40.0, 9000.0 / n)), c=colors, edgecolors=edges,
                   linewidths=0.35, zorder=3)
        for i in sorted(set(fired) | {j for j, nd in enumerate(neurons) if nd.get("input")})[:40]:
            x, y = pos[i]; ax.text(x, y + 0.03, str(i), fontsize=5, ha="center", zorder=4)
        title = ("network (green ring = input, red = fired) · circle layout "
                 "(use zoom, pan, or Fit to inspect details)")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal")
    ax.set_title(title + " · scroll or buttons to zoom", fontsize=9)


# Corrected graph renderer. These definitions intentionally override the older
# fallback above, which collapsed large networks into an unreadable faint ring.
def _weight_label(s, learned):
    label = f"{s['weight']:g}"
    if learned is not None and s.get("idx") in (learned or {}) and learned[s["idx"]] != s["weight"]:
        label = f"{s['weight']:g}->{learned[s['idx']]:g}"
    return label


def _draw_directed_synapses(ax, pos, synapses, learned=None, *, label_limit=900,
                            arrow_limit=2500, weight_font=6, curved=True):
    """Draw synapses as directed arrows with weight labels."""
    valid = [s for s in synapses if s.get("pre") in pos and s.get("post") in pos]
    if not valid:
        return 0

    if len(valid) <= arrow_limit:
        for k, s in enumerate(valid):
            x1, y1 = pos[s["pre"]]
            x2, y2 = pos[s["post"]]
            if s["pre"] == s["post"]:
                start = (x1 - 0.03, y1 + 0.06)
                end = (x1 + 0.03, y1 + 0.06)
                rad = 0.75
                lx, ly = x1, y1 + 0.14
            else:
                start = (x1, y1)
                end = (x2, y2)
                rad = 0.12 * (1 if k % 2 == 0 else -1) if curved else 0.0
                lx, ly = 0.42 * x1 + 0.58 * x2, 0.42 * y1 + 0.58 * y2
            ax.add_patch(FancyArrowPatch(
                start, end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=9,
                color="#344955",
                linewidth=0.75,
                alpha=0.45,
                shrinkA=8,
                shrinkB=10,
                zorder=1,
            ))
            if k < label_limit:
                ax.text(lx, ly, _weight_label(s, learned), fontsize=weight_font,
                        color="#123c8c", ha="center", va="center",
                        bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.72},
                        zorder=4)
    else:
        qx, qy, qu, qv = [], [], [], []
        for s in valid:
            x1, y1 = pos[s["pre"]]
            x2, y2 = pos[s["post"]]
            qx.append(x1); qy.append(y1); qu.append(x2 - x1); qv.append(y2 - y1)
        ax.quiver(qx, qy, qu, qv, angles="xy", scale_units="xy", scale=1,
                  width=0.0012, headwidth=3.0, headlength=4.0,
                  color="#344955", alpha=0.22, zorder=1)
        for s in valid[:label_limit]:
            x1, y1 = pos[s["pre"]]
            x2, y2 = pos[s["post"]]
            ax.text(0.42 * x1 + 0.58 * x2, 0.42 * y1 + 0.58 * y2,
                    _weight_label(s, learned), fontsize=max(4, weight_font - 1),
                    color="#123c8c", ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.05", "fc": "white", "ec": "none", "alpha": 0.60},
                    zorder=4)
    return len(valid)


def _draw_network_large(ax, neurons, synapses, fired, learned=None):
    import math
    n = len(neurons)
    pos, ncomp, cols = _network_positions(neurons, synapses)
    rows = math.ceil(ncomp / cols)
    node_size = max(12.0, min(120.0, 24000.0 / max(n, 1)))

    labelled_edges = _draw_directed_synapses(
        ax, pos, synapses, learned, label_limit=900, arrow_limit=2500,
        weight_font=5.5, curved=False,
    )

    xs, ys, colors, edgecolors, widths = [], [], [], [], []
    for i in range(n):
        if i not in pos:
            continue
        x, y = pos[i]; xs.append(x); ys.append(y)
        inp = neurons[i].get("input")
        colors.append("#ff6b5e" if i in fired else "#7fb8e6")
        edgecolors.append("#1a8a1a" if inp else "#33465a")
        widths.append(1.0 if (i in fired or inp) else 0.35)
    ax.scatter(xs, ys, s=node_size, c=colors, edgecolors=edgecolors,
               linewidths=widths, alpha=0.97, zorder=3)

    labelled = 0
    for i in sorted(set(fired) | {j for j, nd in enumerate(neurons) if nd.get("input")}):
        if i not in pos or labelled >= 80:
            continue
        x, y = pos[i]
        ax.text(x, y + 0.02, str(i), ha="center", va="center", fontsize=5, color="#111", zorder=5)
        labelled += 1

    ax.set_xlim(-0.15, cols + 0.15); ax.set_ylim(-(rows + 0.15), 0.15); ax.set_aspect("equal")
    note = "" if labelled_edges <= 900 else " (first 900 weights labelled)"
    ax.set_title(f"{n} neurons in {ncomp} connected group(s), {len(synapses)} synapses{note} "
                 "(circles=neurons, arrows=synapses, text=weight)", fontsize=9)


def _draw_circle(ax, neurons, synapses, fired, learned):
    """Circular layout with explicit neuron circles, directed arrows, and weight text."""
    import math
    n = len(neurons)
    pos = {}
    for i in range(n):
        if n == 1:
            pos[i] = (0.0, 0.0)
        else:
            a = 2 * math.pi * i / n
            pos[i] = (-math.cos(a), math.sin(a))

    labelled_edges = _draw_directed_synapses(
        ax, pos, synapses, learned, label_limit=900, arrow_limit=2500,
        weight_font=6 if n <= 256 else 4.5, curved=True,
    )
    r = 0.13 if n <= 40 else max(0.008, min(0.035, 8.0 / max(n, 1)))
    for i, (x, y) in pos.items():
        face = "#ff6b5e" if i in fired else "#7fb8e6"
        edge = "#1a8a1a" if neurons[i].get("input") else "#333"
        ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=edge,
                            linewidth=1.3 if (i in fired or neurons[i].get("input")) else 0.45,
                            zorder=3))
        if n <= 220 or i in fired or neurons[i].get("input"):
            ax.text(x, y, str(i), ha="center", va="center", zorder=5,
                    fontsize=8 if n <= 80 else max(3.5, 210.0 / n), fontweight="bold")
    title = "network (circles=neurons, arrows=synapses, text=edge weight)"
    if labelled_edges > 900:
        title += " - first 900 weights labelled"
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal")
    ax.set_title(title + " - scroll or buttons to zoom", fontsize=9)


def draw_network(ax, neurons, synapses, fired=None, learned=None, layout="graph"):
    """Render the network as one consistent graph view.

    Neurons are circles, synapses are directed arrows, and edge text shows weight.
    Zoom/pan/fit handle scale, and the placement is topology-based rather than
    forcing everything onto a ring.
    """
    ax.clear(); ax.axis("off")
    n = len(neurons)
    if n == 0:
        ax.text(0.5, 0.5, "add neurons to see the network", ha="center", va="center")
        return
    fired = set(fired or [])
    _draw_network_large(ax, neurons, synapses, fired, learned)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class SNNConsole(QMainWindow):
    def __init__(self):
        super().__init__()
        self.drv = None
        self.worker = None
        self._redraw_suspended = 0      # >0 while bulk-populating tables (no live redraw)
        # Default to the NPU array (STDP) on Basys3 -- the hardware-validated
        # on-chip-learning board variant this GUI copy exists for -- not
        # snm_boards.current() -- that reflects a static, checked-in config constant
        # (whichever board this repo's dev checkout last ran gen_config.py for), NOT
        # the physically connected hardware, so it was a misleading default.
        self.board = snm_npu_stdp.board_dict("npu-stdp:basys3")
        self.setWindowTitle("SpikeEngine - SNN Console")
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(min(1360, max(1180, avail.width() - 64)),
                        min(900, max(780, avail.height() - 80)))
        else:
            self.resize(1280, 820)
        # NOTE (2026-07-31): a window-level setMaximumWidth/Height(avail.*) was
        # tried here as a catch-all against child widgets forcing the window
        # past the screen edge, but it has a worse side effect: Qt/Windows
        # disables the native maximize button once maximumSize leaves no room
        # to grow into, so "maximize" silently stopped doing anything. Reverted
        # -- the actual offending widgets (the FPGA combo box and the board
        # status label, both below) are capped individually instead, which
        # fixes the real overflow without touching the window's own resize
        # behavior.
        self._apply_accessible_theme()
        self._loading_preset = False
        self._generated_example = None      # armed board-gated generated example, if any
        self._build_ui()
        self._refresh_fixed_point()
        # Run the SAME cascade a manual board switch does (baud, STDP gating, preset
        # list, board info) for the initial selection too -- setCurrentIndex() in
        # _connection_bar() ran before currentIndexChanged was connected, so nothing
        # else initializes these for the startup default.
        self._on_board_change()

    def _apply_accessible_theme(self):
        self.setStyleSheet(
            f"QMainWindow, QWidget{{background:{UI['bg']};color:{UI['text']};}}"
            f"QGroupBox, QTabWidget::pane, QTableWidget, QPlainTextEdit, QTextEdit{{background:{UI['panel']};"
            f"border:1px solid {UI['border_soft']};}}"
            f"QLabel{{color:{UI['text']};}}"
            f"QLineEdit, QSpinBox, QComboBox{{background:{UI['panel']};color:{UI['text']};"
            f"border:1px solid {UI['border']};border-radius:4px;padding:3px 6px;selection-background-color:{UI['primary']};"
            f"selection-color:#FFFFFF;}}"
            f"QPushButton{{background:{UI['panel']};color:{UI['text']};border:1px solid {UI['border']};"
            f"border-radius:5px;padding:4px 10px;font-weight:600;}}"
            f"QPushButton:hover{{background:#EEF4FF;border-color:{UI['primary']};}}"
            f"QPushButton:disabled{{background:#EEF2F6;color:{UI['text_soft']};border-color:{UI['border_soft']};}}"
            f"QCheckBox{{color:{UI['text']};spacing:8px;}}"
            f"QCheckBox::indicator{{width:18px;height:18px;}}"
            f"QTableWidget{{gridline-color:{UI['border_soft']};selection-background-color:#DCEBFF;selection-color:{UI['text']};}}"
            f"QHeaderView::section{{background:#EAF0F7;color:{UI['text']};border:1px solid {UI['border_soft']};padding:4px;font-weight:600;}}"
            f"QTabWidget::pane{{top:-1px;padding-top:2px;}}"
            f"QTabBar::tab{{background:#EEF2F7;color:{UI['text']};border:1px solid {UI['border_soft']};"
            f"padding:7px 14px 6px 14px;margin-right:2px;min-width:88px;}}"
            f"QTabBar::tab:selected{{background:{UI['panel']};border-bottom-color:{UI['panel']};font-weight:700;"
            f"padding:7px 14px 6px 14px;margin-left:0px;margin-right:2px;}}"
        )

    # ---- layout ----
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.addWidget(self._connection_bar())
        outer.addWidget(self._fixedpoint_bar())
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(10)
        split.setStyleSheet(
            f"QSplitter::handle{{background:#D7E0EA;border-left:1px solid {UI['border']};border-right:1px solid {UI['border']};}}"
        )
        self.config_panel = self._config_panel()
        self.output_panel = self._output_panel()
        self.config_panel.setMinimumWidth(520)
        self.output_panel.setMinimumWidth(360)
        split.addWidget(self.config_panel)
        split.addWidget(self.output_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([760, 520])
        self.main_split = split
        outer.addWidget(split, 1)
        self._rebuild_input_grid(preserve=False)   # initial (empty) input grid

    def _connection_bar(self):
        g = self._style_section(QGroupBox("Connection"), SECTION_COLORS["connection"])
        h = QHBoxLayout(g)
        h.addWidget(QLabel("<b>FPGA:</b>"))
        self.board_cb = QComboBox()
        # "Classic" (non-lane, single-core SuperNeuroMAT3) boards hidden from the
        # UI (2026-07-30, user request) -- this project now ships only the NPU
        # lane engine, with or without STDP. snm_boards/boards.yaml and the
        # classic driver path are left in place (not deleted) so this is a
        # one-line revert if classic support is needed again; just uncomment:
        # for name in snm_boards.names():
        #     self.board_cb.addItem(snm_boards.label(name), name)
        # NPU-array (inference-only lane engine) bitstreams appear as their own
        # entries: a different engine + wire protocol, not a mode of the classic boards.
        try:
            for key in snm_npu.keys():
                self.board_cb.addItem(snm_npu.label(key), key)
        except Exception:  # noqa: BLE001, S110 - optional registry must not prevent GUI startup
            pass
        # NPU-array (STDP) bitstreams: same lane-engine wire protocol family as the
        # inference-only NPU array above, but with on-chip STDP learning enabled --
        # a distinct board registry (snm_npu_stdp), so listed as its own entries.
        try:
            for key in snm_npu_stdp.keys():
                self.board_cb.addItem(snm_npu_stdp.label(key), key)
        except Exception:  # noqa: BLE001, S110 - optional registry must not prevent GUI startup
            pass
        ix = self.board_cb.findData(self.board.get("name"))
        if ix >= 0:
            self.board_cb.setCurrentIndex(ix)
        self.board_cb.setToolTip("Target FPGA: sets the timing clock / part shown, the UART "
                                 "baud, and the capacity limits (N_MAX etc.) from the packaged "
                                 "bitstream manifest. At load time the programmed bitstream's "
                                 "real N_MAX is verified against this selection.")
        # Cap the closed box's width (2026-07-31, fixed a real overflow -- the
        # SP701/ZCU104 NPU-STDP labels are full sentences like "validated
        # microseer image (N=90, K=8)" and, combined with board_info's text
        # below, pushed the whole window past the screen edge). Full text is
        # still one click away in the open dropdown, and repeated on hover via
        # the per-item tooltip set below.
        self.board_cb.setMaximumWidth(300)
        for ix in range(self.board_cb.count()):
            self.board_cb.setItemData(ix, self.board_cb.itemText(ix), Qt.ToolTipRole)
        self.board_cb.currentIndexChanged.connect(self._on_board_change)
        h.addWidget(self.board_cb)
        self.board_info = QLabel()
        self.board_info.setMaximumWidth(420)
        h.addWidget(self.board_info)
        h.addSpacing(12)
        h.addWidget(QLabel("Port:"))
        # Resolve at connect time, after the user has selected a board/profile.
        # Prefilling a generic auto-detected port here picked SP701's endpoint
        # even when ZCU104 was selected on a two-board system.
        _default_port = "auto"
        self.port = QLineEdit(_default_port); self.port.setFixedWidth(70)
        self.port.setToolTip("Serial port, e.g. COM5. Use 'auto' to detect the board by USB VID:PID.")
        h.addWidget(self.port)
        # Baud comes from the selected board's packaged-bitstream manifest (the Basys3 image
        # runs at 4 Mbaud), so the GUI opens the port at the same rate as the Python API.
        h.addWidget(QLabel("Baud:"))
        self.baud = QLineEdit(str(snn_io.manifest_baud_for_board(self.board.get("name"))))
        self.baud.setFixedWidth(80); h.addWidget(self.baud)
        self.mock = QCheckBox("Mock (software preview)"); h.addWidget(self.mock)
        self.btn_prog = QPushButton("Program"); self.btn_prog.clicked.connect(self.program_board)
        self.btn_prog.setToolTip(
            "Load the packaged bitstream for the selected board onto the FPGA over "
            "JTAG (volatile -- lost on power-off). Disconnect first if already connected.")
        h.addWidget(self.btn_prog)
        self.btn_conn = QPushButton("Connect"); self.btn_conn.clicked.connect(self.connect); h.addWidget(self.btn_conn)
        self.btn_disc = QPushButton("Disconnect"); self.btn_disc.clicked.connect(self.disconnect); self.btn_disc.setEnabled(False)
        self.btn_disc.setStyleSheet(
            f"QPushButton{{background:{UI['danger']};color:white;font-weight:bold;"
            f"border:1px solid {UI['danger_dark']};border-radius:4px;padding:3px 12px;}}"
            f"QPushButton:hover{{background:{UI['danger_dark']};}}"
            f"QPushButton:disabled{{background:#E7C1BC;color:#FFFDFD;border-color:#D8AAA4;}}")
        h.addWidget(self.btn_disc)
        self.conn_lbl = QLabel("disconnected"); self.conn_lbl.setStyleSheet(f"color:{UI['danger']};font-weight:bold;")
        h.addWidget(self.conn_lbl, 1)
        return g

    def _fixedpoint_bar(self):
        g = self._style_section(QGroupBox("Number format (fixed-point)"), SECTION_COLORS["fixed_point"])
        h = QHBoxLayout(g)
        h.addWidget(QLabel("fractional bits:"))
        self.frac = QSpinBox(); self.frac.setRange(0, 6); self.frac.setValue(3)
        self.frac.valueChanged.connect(self._refresh_fixed_point); h.addWidget(self.frac)
        self.frac_info = QLabel(); h.addWidget(self.frac_info, 1)
        return g

    def _config_panel(self):
        w = QWidget(); v = QVBoxLayout(w)
        # prebuilt examples
        ge = self._style_section(QGroupBox("Prebuilt examples"), SECTION_COLORS["examples"])
        he = QHBoxLayout(ge)
        self.preset_cb = QComboBox()
        self.preset_cb.addItems(snm_presets.names_for_board(self.board.get("name")))
        self.preset_cb.currentTextChanged.connect(
            lambda n: self.preset_desc.setText(snm_presets.get(n)["desc"] if n else ""))
        he.addWidget(self.preset_cb)
        self.btn_load_preset = QPushButton("Load example")
        self.btn_load_preset.clicked.connect(lambda: self.load_preset(self.preset_cb.currentText()))
        he.addWidget(self.btn_load_preset)
        bclr = QPushButton("Clear all")
        bclr.setToolTip("Remove all neurons, synapses, and inputs and reset the view")
        bclr.clicked.connect(self.clear_all); he.addWidget(bclr)
        bimp = QPushButton("Import…")
        bimp.setToolTip("Load an SNN exported from superneuromat (.json)")
        bimp.clicked.connect(self.import_snn); he.addWidget(bimp)
        bexp = QPushButton("Export…")
        bexp.setToolTip("Save this network (+inputs) to ./snn_exports/")
        bexp.clicked.connect(self.export_snn); he.addWidget(bexp)
        self.preset_desc = QLabel(snm_presets.get(self.preset_cb.currentText())["desc"])
        self.preset_desc.setWordWrap(True); he.addWidget(self.preset_desc, 1)
        v.addWidget(ge)
        # neurons
        gn = self._style_section(QGroupBox("Neurons"), SECTION_COLORS["neurons"])
        vn = QVBoxLayout(gn)
        self.ntab = QTableWidget(0, 6)
        self.ntab.setHorizontalHeaderLabels(["id", "threshold", "leak", "reset", "refractory", "active input"])
        self.ntab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ntab.setEditTriggers(QAbstractItemView.AllEditTriggers)
        vn.addWidget(self.ntab)
        hn = QHBoxLayout()
        b = QPushButton("Add neuron"); b.clicked.connect(self.add_neuron); hn.addWidget(b)
        b = QPushButton("Remove selected"); b.clicked.connect(lambda: self._remove_rows(self.ntab, renumber_col0=True)); hn.addWidget(b)
        hn.addStretch(1); vn.addLayout(hn)
        v.addWidget(gn)
        # synapses + STDP
        gs = self._style_section(QGroupBox("Synapses + STDP"), SECTION_COLORS["synapses"])
        vs = QVBoxLayout(gs)
        self.stab = QTableWidget(0, 4)
        self.stab.setHorizontalHeaderLabels(["pre", "post", "weight", "STDP active"])
        self.stab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vs.addWidget(self.stab)
        hs = QHBoxLayout()
        b = QPushButton("Add synapse"); b.clicked.connect(self.add_synapse); hs.addWidget(b)
        b = QPushButton("Remove selected"); b.clicked.connect(lambda: self._remove_rows(self.stab)); hs.addWidget(b)
        hs.addStretch(1); vs.addLayout(hs)
        st = QHBoxLayout()
        self.stdp_on = QCheckBox("STDP"); st.addWidget(self.stdp_on)
        st.addWidget(QLabel("window")); self.stdp_win = QSpinBox(); self.stdp_win.setRange(0, HW.stdp_window_max); self.stdp_win.setValue(1); st.addWidget(self.stdp_win)
        st.addWidget(QLabel("Apos")); self.stdp_apos = QLineEdit("2"); self.stdp_apos.setFixedWidth(90); st.addWidget(self.stdp_apos)
        st.addWidget(QLabel("Aneg")); self.stdp_aneg = QLineEdit("-1"); self.stdp_aneg.setFixedWidth(90); st.addWidget(self.stdp_aneg)
        st.addStretch(1); vs.addLayout(st)
        v.addWidget(gs)
        # inputs -- a step x neuron grid; columns auto-load from the input neurons
        gi = self._style_section(QGroupBox("Inputs (value injected per timestep)"), SECTION_COLORS["inputs"])
        vi = QVBoxLayout(gi)
        vi.addWidget(QLabel(
            "Rows = timesteps, columns = input neurons (auto-loaded from the Neurons "
            "table where active input=1). Type the value to inject at that step; leave blank "
            "for none. Negative values are allowed (inhibitory)."))
        self.itab = QTableWidget(0, 1)
        self.itab.setHorizontalHeaderLabels(["step"])
        self.itab.verticalHeader().setVisible(False)
        self.itab.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._input_cols = []          # neuron id for each value column (col index = i+1)
        vi.addWidget(self.itab)
        hi = QHBoxLayout()
        b = QPushButton("Sync to neurons")
        b.setToolTip("Rebuild the grid from the current input neurons and step count")
        b.clicked.connect(self._rebuild_input_grid); hi.addWidget(b)
        hi.addStretch(1); vi.addLayout(hi)
        v.addWidget(gi)
        # run
        gr = self._style_section(QGroupBox("Run"), SECTION_COLORS["run"])
        hr = QHBoxLayout(gr)
        hr.addWidget(QLabel("steps:")); self.steps = QSpinBox(); self.steps.setRange(1, 100000); self.steps.setValue(10); hr.addWidget(self.steps)
        self.steps.valueChanged.connect(self._sync_inputs)   # grow/shrink the input grid rows
        self.compare_sw = QCheckBox("Cross-check against SuperNeuroMAT")
        self.compare_sw.setChecked(True)
        self.compare_sw.setToolTip(
            "After each run, compare the observed trace against the SuperNeuroMAT "
            "software reference and report any spike/state mismatches."
        )
        self.compare_sw.setStyleSheet(
            f"QCheckBox{{font-weight:bold;color:{UI['primary_dark']};spacing:8px;}}"
            "QCheckBox::indicator{width:22px;height:22px;}"
        )
        hr.addWidget(self.compare_sw)
        hr.addStretch(1)
        self.btn_run = QPushButton("  BUILD && RUN  ")
        self.btn_run.setStyleSheet(
            f"QPushButton{{background:{UI['success']};color:white;font-weight:bold;font-size:14px;padding:8px 18px;"
            f"border-radius:6px;border:1px solid {UI['success']};}}"
            f"QPushButton:hover{{background:#14532D;border-color:#14532D;}}"
        )
        self.btn_run.clicked.connect(self.build_run); hr.addWidget(self.btn_run)
        v.addWidget(gr)
        self.run_lbl = QLabel(""); v.addWidget(self.run_lbl)
        # Live-update the network graph whenever a neuron or synapse cell is EDITED
        # (not just added/removed). Programmatic fills are wrapped in _suspend() so
        # these don't fire mid-populate.
        self.ntab.itemChanged.connect(self._on_table_changed)
        self.stab.itemChanged.connect(self._on_table_changed)
        return w

    def _output_panel(self):
        return self._output_panel_v2()

    def _output_panel_v2(self):
        return self._build_output_tabs_with_viewer()

    def _build_output_tabs_with_viewer(self):
        tabs = QTabWidget()
        self.net_view = PyVistaNetworkView(self) if pyvista_available() else MatplotlibNetworkView(self)
        tabs.addTab(self.net_view, "Network")

        rw = QWidget(); rv = QVBoxLayout(rw)
        self.ras_fig = Figure(figsize=(5, 3)); self.ras_canvas = FigureCanvas(self.ras_fig)
        self.ras_ax = self.ras_fig.add_subplot(111)
        rv.addWidget(self.ras_canvas, 2)
        self.ras_text = QPlainTextEdit(); self.ras_text.setReadOnly(True)
        self.ras_text.setFont(QFont("Consolas", 9)); rv.addWidget(self.ras_text, 1)
        tabs.addTab(rw, "Spike Train")

        self.wtab = QTableWidget(0, 4)
        self.wtab.setHorizontalHeaderLabels(["synapse", "initial w", "learned w", "delta"])
        self.wtab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.wtab, "Weights (STDP)")

        sw = QWidget(); sv = QVBoxLayout(sw)
        note = QLabel(
            "<b>Membrane voltage after each tick.</b>"
            "<ul style='margin:2px 0 2px 14px;-qt-list-indent:1'>"
            "<li>Per tick: leak membrane toward reset &rarr; add synapse + external "
            "inputs &rarr; spike if result is strictly &gt; threshold (then reset)</li>"
            "<li><span style='color:#c0392b'><b>Red 'spike'</b></span> = fired this tick</li>"
            "<li><span style='color:#e67e22'><b>Orange</b></span> = refractory (suppressed)</li>"
            "<li>Non-fired cells show the membrane &mdash; watch a neuron climb toward "
            "(or leak away from) threshold</li></ul>")
        note.setWordWrap(True)
        sv.addWidget(note)
        self.state_tab = QTableWidget(0, 0)
        self.state_tab.setFont(QFont("Consolas", 9))
        sv.addWidget(self.state_tab)
        tabs.addTab(sw, "Neuron state")

        self.timing_text = QTextEdit(); self.timing_text.setReadOnly(True)
        self.timing_text.setHtml(f"<p style='color:{UI['text_soft']}'>Run a network to see the "
                                 "per-timestep on-chip timing analysis.</p>")
        tabs.addTab(self.timing_text, "Timing Analysis")

        cw = QWidget(); cv = QVBoxLayout(cw)
        self.compare_summary = QTextEdit(); self.compare_summary.setReadOnly(True)
        self.compare_summary.setMaximumHeight(180)
        self.compare_summary.setStyleSheet(
            f"QTextEdit{{background:{UI['compare_bg']};border:1px solid {UI['compare_border']};border-radius:4px;padding:4px;color:{UI['text']};}}"
        )
        self.compare_summary.setHtml(
            f"<p style='color:{UI['primary_dark']}'><b>Run a network with cross-check enabled</b> to compare "
            "the observed result against the SuperNeuroMAT software reference.</p>"
        )
        cv.addWidget(self.compare_summary)

        self.compare_trace_tab = QTableWidget(0, 3)
        self.compare_trace_tab.setHorizontalHeaderLabels(["step", "observed", "reference"])
        self.compare_trace_tab.verticalHeader().setVisible(False)
        self.compare_trace_tab.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.compare_trace_tab.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.compare_trace_tab.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.compare_trace_tab.setFont(QFont("Consolas", 9))
        cv.addWidget(self.compare_trace_tab, 1)

        self.compare_details_box = QGroupBox("Mismatch Details")
        dv = QVBoxLayout(self.compare_details_box)
        self.compare_details_note = QLabel(
            "Only mismatches are listed here. When the comparison passes, this section stays hidden."
        )
        self.compare_details_note.setWordWrap(True)
        dv.addWidget(self.compare_details_note)

        self.compare_details_tabs = QTabWidget()

        self.compare_spike_tab = QTableWidget(0, 3)
        self.compare_spike_tab.setHorizontalHeaderLabels(["timestep", "observed spikes", "reference spikes"])
        self.compare_spike_tab.verticalHeader().setVisible(False)
        self.compare_spike_tab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.compare_details_tabs.addTab(self.compare_spike_tab, "Spike mismatches")

        self.compare_state_tab = QTableWidget(0, 5)
        self.compare_state_tab.setHorizontalHeaderLabels(["kind", "timestep", "neuron", "observed", "reference"])
        self.compare_state_tab.verticalHeader().setVisible(False)
        self.compare_state_tab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.compare_details_tabs.addTab(self.compare_state_tab, "State mismatches")

        self.compare_weight_tab = QTableWidget(0, 5)
        self.compare_weight_tab.setHorizontalHeaderLabels(["synapse", "observed", "reference", "delta", "basis"])
        self.compare_weight_tab.verticalHeader().setVisible(False)
        self.compare_weight_tab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.compare_details_tabs.addTab(self.compare_weight_tab, "Weight mismatches")
        dv.addWidget(self.compare_details_tabs)
        cv.addWidget(self.compare_details_box)
        self.compare_details_box.hide()

        tabs.addTab(cw, "Comparison")
        self.tabs = tabs
        self._draw_network()
        return tabs

    # ---- table helpers ----
    @staticmethod
    def _item(text):
        return QTableWidgetItem(str(text))

    @staticmethod
    def _style_section(g, color):
        """Give a QGroupBox a bigger, colored title bar."""
        g.setStyleSheet(
            f"QGroupBox{{font-size:13px;font-weight:bold;border:1px solid {UI['border']};"
            f"border-radius:6px;margin-top:12px;background:{UI['panel']};}}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;"
            "padding:3px 12px;left:8px;color:white;background:%s;border-radius:4px;}" % color)
        return g

    @contextmanager
    def _suspend(self):
        """Suspend live graph redraws while programmatically filling tables, so a
        bulk populate doesn't trigger one redraw per cell (and avoids reentrancy)."""
        self._redraw_suspended += 1
        try:
            yield
        finally:
            self._redraw_suspended -= 1

    def _request_redraw(self):
        if self._redraw_suspended == 0:
            self._draw_network()

    def _on_table_changed(self, item):
        """A neuron/synapse cell was edited -> enforce non-negative leak/refractory
        and refresh the network graph live."""
        # leak (col 2) and refractory (col 4) are magnitudes: clamp negatives to 0.
        if item.tableWidget() is self.ntab and item.column() in (2, 4):
            txt = item.text().strip()
            try:
                bad = float(txt) < 0
            except ValueError:
                bad = False
            if bad:
                field = "leak" if item.column() == 2 else "refractory"
                with self._suspend():
                    item.setText("0")
                self.run_lbl.setText(f"{field} cannot be negative - reset to 0")
        # toggling a neuron's input flag adds/removes an input-grid column
        if item.tableWidget() is self.ntab and item.column() == 5:
            self._sync_inputs()
        self._request_redraw()

    def add_neuron(self):
        self._generated_example = None      # manual editing disarms a generated example
        hw = self._active_hw()
        if self.ntab.rowCount() >= hw.n_max:
            self.run_lbl.setText(f"neuron limit reached for {self.board_cb.currentText()}: N_MAX={hw.n_max}")
            return
        with self._suspend():
            r = self.ntab.rowCount(); self.ntab.insertRow(r)
            vals = [r, "1.0", "0", "0", "0", "1"]
            for c, v in enumerate(vals):
                it = self._item(v)
                if c == 0:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)  # id read-only
                self.ntab.setItem(r, c, it)
        self._request_redraw()
        self._sync_inputs()                  # new neuron may be an input -> new column

    def add_synapse(self):
        hw = self._active_hw()
        if self.stab.rowCount() >= hw.syn_max:
            self.run_lbl.setText(f"synapse limit reached for {self.board_cb.currentText()}: SYN_MAX={hw.syn_max}")
            return
        with self._suspend():
            r = self.stab.rowCount(); self.stab.insertRow(r)
            for c, v in enumerate(["0", "1", "1.0", "0"]):
                self.stab.setItem(r, c, self._item(v))
        self._request_redraw()

    def _rebuild_input_grid(self, preserve=True):
        """Rebuild the input grid: column per input-enabled neuron (auto-loaded from
        the Neurons table), row per timestep. Entered values are preserved across
        rebuilds by (step, neuron)."""
        try:
            neurons = self._gather_neurons()
        except (ValueError, TypeError):
            return
        input_ids = [i for i, nr in enumerate(neurons) if nr["input"]]
        steps = self.steps.value()
        old = {}
        if preserve:
            for t in range(self.itab.rowCount()):
                for ci, nid in enumerate(self._input_cols):
                    it = self.itab.item(t, ci + 1)
                    if it and it.text().strip():
                        old[(t, nid)] = it.text()
        self.itab.blockSignals(True)
        self.itab.clear()
        self._input_cols = input_ids
        self.itab.setColumnCount(1 + len(input_ids))
        self.itab.setRowCount(steps)
        self.itab.setHorizontalHeaderLabels(["step"] + [f"n{i}" for i in input_ids])
        step_bg = QColor(225, 235, 248)
        for t in range(steps):
            sit = self._item(t); sit.setFlags(sit.flags() & ~Qt.ItemIsEditable)
            sit.setBackground(step_bg)
            self.itab.setItem(t, 0, sit)
            for ci, nid in enumerate(input_ids):
                self.itab.setItem(t, ci + 1, self._item(old.get((t, nid), "")))
        self.itab.blockSignals(False)

    def _sync_inputs(self):
        """Resync the grid to the neurons/steps, unless we're mid bulk-populate."""
        if self._redraw_suspended == 0:
            self._rebuild_input_grid(preserve=True)

    def clear_all(self):
        """Remove every neuron, synapse, and input, reset STDP, and clear outputs."""
        with self._suspend():
            self.ntab.setRowCount(0)
            self.stab.setRowCount(0)
            self.stdp_on.setChecked(False)
        self._rebuild_input_grid(preserve=False)   # no input neurons -> empty grid
        self._draw_network()
        self.ras_ax.clear(); self.ras_canvas.draw()
        self.ras_text.setPlainText("")
        self.wtab.setRowCount(0)
        # also clear the Neuron-state trace (was left showing the previous run)
        self.state_tab.clear()
        self.state_tab.setRowCount(0); self.state_tab.setColumnCount(0)
        self.timing_text.setHtml("<p style='color:#777'>Run a network to see the "
                                 "per-timestep on-chip timing analysis.</p>")
        self._clear_comparison_view()
        self.run_lbl.setStyleSheet("")              # drop any green/orange run styling
        self.run_lbl.setText("cleared - add neurons or load an example")

    def _remove_rows(self, tab, renumber_col0=False):
        with self._suspend():
            for r in sorted({i.row() for i in tab.selectedItems()}, reverse=True):
                tab.removeRow(r)
            if renumber_col0:
                for r in range(tab.rowCount()):
                    it = self._item(r); it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    tab.setItem(r, 0, it)
        self._request_redraw()
        if tab is self.ntab:
            self._sync_inputs()              # neuron set changed -> resync columns

    @staticmethod
    def _fmt(v):
        return f"{v:g}"

    def _set_bulk_load_ui(self, busy: bool):
        self._loading_preset = busy
        if hasattr(self, "btn_load_preset"):
            self.btn_load_preset.setEnabled(not busy)
        if hasattr(self, "preset_cb"):
            self.preset_cb.setEnabled(not busy)
        for tab in (getattr(self, "ntab", None), getattr(self, "stab", None), getattr(self, "itab", None)):
            if tab is not None:
                tab.setUpdatesEnabled(not busy)
        QApplication.processEvents()

    def _populate_neuron_table(self, neurons):
        self.ntab.blockSignals(True)
        self.ntab.setRowCount(len(neurons))
        for r, nn in enumerate(neurons):
            vals = [
                r,
                self._fmt(nn.get("threshold", 0)),
                self._fmt(nn.get("leak", 0)),
                self._fmt(nn.get("reset", 0)),
                str(int(nn.get("refractory", 0))),
                "1" if nn.get("input") else "0",
            ]
            for c, v in enumerate(vals):
                it = self._item(v)
                if c == 0:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.ntab.setItem(r, c, it)
        self.ntab.blockSignals(False)

    def _populate_synapse_table(self, synapses):
        self.stab.blockSignals(True)
        self.stab.setRowCount(len(synapses))
        for r, sd in enumerate(synapses):
            vals = [
                str(int(sd.get("pre", 0))),
                str(int(sd.get("post", 0))),
                self._fmt(sd.get("weight", 1)),
                "1" if sd.get("stdp") else "0",
            ]
            for c, v in enumerate(vals):
                self.stab.setItem(r, c, self._item(v))
        self.stab.blockSignals(False)

    def load_preset(self, name):
        """Populate every table from a prebuilt example (snm_presets)."""
        if self._loading_preset:
            return
        if snm_presets.is_generated(name):
            self._load_generated_example(name)
            return
        self._generated_example = None      # a normal preset clears any armed generated run
        p = snm_presets.get(name)
        hw = self._active_hw()
        if len(p["neurons"]) > hw.n_max:
            QMessageBox.warning(
                self, "Preset exceeds board capacity",
                f"'{name}' has {len(p['neurons'])} neurons, but {self.board_cb.currentText()} supports N_MAX={hw.n_max}."
            )
            return
        if len(p["synapses"]) > hw.syn_max:
            QMessageBox.warning(
                self, "Preset exceeds board capacity",
                f"'{name}' has {len(p['synapses'])} synapses, but {self.board_cb.currentText()} supports SYN_MAX={hw.syn_max}."
            )
            return
        self._set_bulk_load_ui(True)
        try:
            with self._suspend():
                self.frac.setValue(p["frac_bits"])
                self.steps.setValue(p["steps"])
                self._populate_neuron_table(p["neurons"])
                self._populate_synapse_table(p["synapses"])
                if p["stdp"]:
                    self.stdp_on.setChecked(True)
                    self.stdp_win.setValue(p["stdp"]["window"])
                    self.stdp_apos.setText(",".join(self._fmt(x) for x in p["stdp"]["apos"]))
                    self.stdp_aneg.setText(",".join(self._fmt(x) for x in p["stdp"]["aneg"]))
                else:
                    self.stdp_on.setChecked(False)
                self._rebuild_input_grid(preserve=False)
                for inp in p["inputs"]:
                    nid = inp["neuron"]
                    if nid in self._input_cols:
                        ci = self._input_cols.index(nid) + 1
                        for t in inp["steps"]:
                            if 0 <= t < self.itab.rowCount():
                                self.itab.item(t, ci).setText(self._fmt(inp["value"]))
            self._draw_network()
            self.run_lbl.setText(f"loaded example '{name}' - press BUILD & RUN")
        finally:
            self._set_bulk_load_ui(False)

    # ---- generated (non-table) hardware examples ----
    def _load_generated_example(self, name):
        """Arm a generated example (e.g. ZCU104 full-capacity). Its network is too large to
        render as editable tables, so we don't populate them -- we show its summary and let
        BUILD & RUN stream it straight to the connected board via its runner module."""
        meta = snm_presets.get(name)
        board = (meta.get("boards") or [None])[0]
        self._generated_example = name
        self.clear_all()                    # empty the tables so no stale network lingers
        self.preset_desc.setText(meta["desc"])
        need = f" - select the {board} board and Connect, then press BUILD & RUN" if board else ""
        self.run_lbl.setStyleSheet("")
        self.run_lbl.setText(f"armed generated example '{name}'{need}")

    def _run_generated_example(self):
        """Stream the armed generated example to the live board on a worker thread."""
        name = self._generated_example
        meta = snm_presets.get(name)
        boards = meta.get("boards") or []
        if boards and self.board.get("name") not in boards:
            QMessageBox.information(
                self, "Wrong board",
                f"'{name}' runs only on {', '.join(boards)}. Select that board first.")
            return
        if self.mock.isChecked() or self.drv is None:
            QMessageBox.information(
                self, "Not connected",
                f"'{name}' streams directly to hardware - Connect to the board first "
                "(Mock is not supported for this full-capacity example).")
            return
        import importlib
        runner = importlib.import_module(meta["runner_module"])
        entry_point = meta.get("entry_point", "run_full_capacity")
        steps = int(meta.get("steps", 10))
        dev = self.drv
        self._gen_progress = (None, 0, 0)

        def progress(phase, done, total):
            self._gen_progress = (phase, done, total)

        def job():
            res = getattr(runner, entry_point)(dev, steps=steps, progress=progress)
            res["_generated"] = name
            return res

        self.btn_run.setEnabled(False)
        self._start_running_anim()
        self.worker = RunWorker(job)
        self.worker.done.connect(self._generated_done)
        self.worker.start()

    def _generated_done(self, result):
        self.btn_run.setEnabled(True)
        self._stop_running_anim(ok=not isinstance(result, Exception))
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Full-capacity run failed", str(result))
            self.run_lbl.setText(f"generated example failed: {result}")
            return
        r = result
        if "summary" in r:
            # generic report shape (e.g. the digits-classifier example): the
            # runner already composed its own message, no capacity-stress
            # fields (n_max/syn_depth/event_cap) to assume are present.
            msg = f"{r['_generated']}: {r['summary']}"
        else:
            spikes = r["spikes_per_step"]
            msg = (f"{r['_generated']}: configured {r['n_max']:,} neurons + {r['syn_depth']:,} "
                   f"synapses in {r['config_s']:.1f}s "
                   f"({r['config_write_rate']:,.0f} writes/s), ran {r['steps']} steps in "
                   f"{r['run_s']:.2f}s. event_cap={r['event_cap']}, "
                   f"direct-driven={r['direct_driven']:,}. spikes/step={spikes}")
        self.run_lbl.setStyleSheet("")
        self.run_lbl.setText(msg)
        QMessageBox.information(self, "Full-capacity run complete", msg)

    # ---- import / export (superneuromat interchange) ----
    def _load_schema(self, data):
        """Populate every table from a snn_io schema dict (import path)."""
        fb = int(data.get("frac_bits", snn_io.DEFAULT_FRAC_BITS))
        inputs = data.get("inputs") or {}
        max_in_t = max((int(t) for t in inputs), default=-1)
        steps = max(int(data.get("steps") or 0), max_in_t + 1, 1)
        with self._suspend():
            self.frac.setValue(fb)
            self.steps.setValue(steps)
            self.ntab.setRowCount(0)
            for nd in data["neurons"]:
                self.add_neuron(); r = self.ntab.rowCount() - 1
                self.ntab.item(r, 1).setText(self._fmt(nd.get("threshold", 0)))
                self.ntab.item(r, 2).setText(self._fmt(nd.get("leak", 0)))
                self.ntab.item(r, 3).setText(self._fmt(nd.get("reset", 0)))
                self.ntab.item(r, 4).setText(str(int(nd.get("refractory", 0))))
                self.ntab.item(r, 5).setText("1" if nd.get("input") else "0")
            self.stab.setRowCount(0)
            for sd in data["synapses"]:
                self.add_synapse(); r = self.stab.rowCount() - 1
                self.stab.item(r, 0).setText(str(int(sd["pre"])))
                self.stab.item(r, 1).setText(str(int(sd["post"])))
                self.stab.item(r, 2).setText(self._fmt(sd.get("weight", 1)))
                self.stab.item(r, 3).setText("1" if sd.get("stdp") else "0")
            st = data.get("stdp")
            if st:
                self.stdp_on.setChecked(True)
                self.stdp_win.setValue(int(st.get("window", 1)))
                self.stdp_apos.setText(",".join(self._fmt(x) for x in (st.get("apos") or [])))
                self.stdp_aneg.setText(",".join(self._fmt(x) for x in (st.get("aneg") or [])))
            else:
                self.stdp_on.setChecked(False)
            self._rebuild_input_grid(preserve=False)
            for t_str, ins in inputs.items():
                t = int(t_str)
                for n_str, v in ins.items():
                    nid = int(n_str)
                    if nid in self._input_cols and 0 <= t < self.itab.rowCount():
                        self.itab.item(t, self._input_cols.index(nid) + 1).setText(self._fmt(v))
        self._draw_network()

    def import_snn(self, path=None):
        """Import an SNN (.json) exported from superneuromat; guard then load."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import SNN", snn_io.export_dir(), "SNN JSON (*.json);;All files (*)")
            if not path:
                return
        try:
            data = snn_io.load_snn(path)
        except Exception as e:  # noqa: BLE001 - file/import errors are reported in the UI
            QMessageBox.critical(self, "Import failed", str(e)); return
        errors, warnings = snn_io.guard_snn(data)
        if errors:
            QMessageBox.critical(self, "Import blocked - does not fit this FPGA",
                                 "\n".join("• " + e for e in errors)); return
        self._load_schema(data)
        msg = (f"imported {len(data['neurons'])} neurons, {len(data['synapses'])} "
               f"synapses from {os.path.basename(path)}")
        if warnings:
            QMessageBox.warning(self, "Imported (with notes)",
                                msg + "\n\nNotes:\n" + "\n".join("• " + w for w in warnings))
        self.run_lbl.setStyleSheet(""); self.run_lbl.setText(msg + " - press BUILD & RUN")

    def export_snn(self):
        """Export the current GUI network (+inputs) to ./snn_exports/ as JSON."""
        if self.ntab.rowCount() == 0:
            QMessageBox.information(self, "Nothing to export", "Add neurons first."); return
        try:
            sched = self._schedule()
        except ValueError as e:
            QMessageBox.warning(self, "Bad input schedule", str(e)); return
        net = self._build_snn(None)
        data = snn_io.snn_to_schema(net, inputs=sched, steps=self.steps.value(),
                                    frac_bits=self.frac.value(),
                                    meta={"source": "fpga-gui"})
        default = os.path.join(snn_io.export_dir(), "snn.json")
        path, _ = QFileDialog.getSaveFileName(self, "Export SNN", default, "SNN JSON (*.json)")
        if not path:
            return
        snn_io.save_snn(data, path=path)
        self.run_lbl.setStyleSheet(""); self.run_lbl.setText(f"exported to {os.path.basename(path)}")

    # ---- read config from tables ----
    def _cell(self, tab, r, c, default=""):
        it = tab.item(r, c)
        return it.text().strip() if it and it.text() else default

    def _gather_neurons(self):
        out = []
        for r in range(self.ntab.rowCount()):
            out.append({
                "threshold": float(self._cell(self.ntab, r, 1, "0")),
                "leak": float(self._cell(self.ntab, r, 2, "0")),
                "reset": float(self._cell(self.ntab, r, 3, "0")),
                "refractory": int(float(self._cell(self.ntab, r, 4, "0"))),
                "input": self._cell(self.ntab, r, 5, "0") in ("1", "yes", "true"),
            })
        return out

    def _gather_synapses(self):
        out = []
        for r in range(self.stab.rowCount()):
            out.append({
                "idx": r,
                "pre": int(float(self._cell(self.stab, r, 0, "0"))),
                "post": int(float(self._cell(self.stab, r, 1, "0"))),
                "weight": float(self._cell(self.stab, r, 2, "1")),
                "stdp": self._cell(self.stab, r, 3, "0") in ("1", "yes", "true"),
            })
        return out

    def _build_snn(self, driver):
        net = SNN(driver=driver, hw=self._active_hw(), frac_bits=self.frac.value())
        for n in self._gather_neurons():
            net.neuron(threshold=n["threshold"], leak=n["leak"], reset=n["reset"],
                       refractory=n["refractory"], inp=n["input"])
        for s in self._gather_synapses():
            net.synapse(s["pre"], s["post"], weight=s["weight"], stdp=s["stdp"])
        if self.stdp_on.isChecked():
            net.stdp(window=self.stdp_win.value(),
                     apos=[float(x) for x in self.stdp_apos.text().split(",") if x.strip()],
                     aneg=[float(x) for x in self.stdp_aneg.text().split(",") if x.strip()])
        return net

    def _schedule(self):
        """Build {timestep: {neuron: value}} from the input grid (row=step,
        column=neuron). Blank cells = no input; raises ValueError (with the
        offending cell) on bad text so the caller can show a clean message."""
        sched = {}
        for t in range(self.itab.rowCount()):
            for ci, nid in enumerate(self._input_cols):
                cell = self._cell(self.itab, t, ci + 1, "")
                if not cell:
                    continue
                try:
                    val = float(cell)
                except ValueError:
                    raise ValueError(f"input grid: step {t}, neuron n{nid}: "
                                     f"'{cell}' is not a number")
                if val != 0:                 # 0 is a no-op injection; skip it
                    sched.setdefault(t, {})[nid] = val
        return sched

    # ---- fixed-point info ----
    def _refresh_fixed_point(self):
        fb = self.frac.value()
        tr = fixed_range("threshold", fb); wr = fixed_range("weight", fb)
        self.frac_info.setText(
            f"resolution = {1/(1<<fb):g}   |   threshold/leak/reset/input range "
            f"[{tr[0]:g}, {tr[1]:g}]   |   weight/STDP range [{wr[0]:g}, {wr[1]:g}]")

    # ---- board selection ----
    def _active_hw(self):
        name = self.board.get("name")
        if snm_npu_stdp.is_npu_stdp(name):
            return snm_npu_stdp.hw_for_board(name)
        if snm_npu.is_npu(name):
            return snm_npu.hw_for_board(name)
        return snn_io.manifest_hw_for_board(name) or HW

    def _is_npu(self) -> bool:
        name = self.board.get("name")
        return snm_npu.is_npu(name) or snm_npu_stdp.is_npu_stdp(name)

    def _is_npu_stdp(self) -> bool:
        return snm_npu_stdp.is_npu_stdp(self.board.get("name"))

    def _on_board_change(self, _ix=None):
        data = self.board_cb.currentData()
        try:
            if snm_npu_stdp.is_npu_stdp(data):
                self.board = snm_npu_stdp.board_dict(data)
            elif snm_npu.is_npu(data):
                self.board = snm_npu.board_dict(data)
            else:
                self.board = snm_boards.get(data)
        except Exception:  # noqa: BLE001 - invalid optional board entries fall back safely
            self.board = snm_boards.current()
        npu = self._is_npu()
        npu_stdp = self._is_npu_stdp()
        self.stdp_win.setRange(0, self._active_hw().stdp_window_max)
        # The inference-only NPU array has no on-chip STDP: grey out every STDP
        # control there (validate() backs this up with a clear error if a loaded
        # file re-enables them). The NPU array (STDP) variant keeps them enabled.
        for w in ("stdp_on", "stdp_win", "stdp_apos", "stdp_aneg"):
            if hasattr(self, w):
                getattr(self, w).setEnabled(not npu or npu_stdp)
        if npu and not npu_stdp and hasattr(self, "stdp_on"):
            self.stdp_on.setChecked(False)
        # Retarget the baud field to the newly selected board's manifest baud.
        if hasattr(self, "baud"):
            if npu_stdp:
                baud = snm_npu_stdp.baud_for(data)
            elif npu:
                baud = snm_npu.baud_for(data)
            else:
                baud = snn_io.manifest_baud_for_board(self.board.get("name"))
            self.baud.setText(str(baud))
        self._refresh_preset_list()
        self._refresh_board_info()

    def _refresh_preset_list(self):
        """Repopulate the example picker for the current board (board-gated generated
        examples appear only for their board), preserving the selection when still valid."""
        if not hasattr(self, "preset_cb"):
            return
        cur = self.preset_cb.currentText()
        names = snm_presets.names_for_board(self.board.get("name"))
        self.preset_cb.blockSignals(True)
        self.preset_cb.clear()
        self.preset_cb.addItems(names)
        if cur in names:
            self.preset_cb.setCurrentText(cur)
        self.preset_cb.blockSignals(False)
        sel = self.preset_cb.currentText()
        if sel:
            self.preset_desc.setText(snm_presets.get(sel)["desc"])

    def _refresh_board_info(self):
        b = self.board
        hw = self._active_hw()
        if self._is_npu():
            npu = b.get("npu", {})
            cap_txt = ("on-chip STDP learning" if self._is_npu_stdp()
                       else "inference-only (no STDP)")
            txt = (f"<span style='color:{UI['text_muted']}'>{b.get('part','?')} &middot; "
                   f"{b.get('core_clk_hz',0)//1_000_000} MHz &middot; "
                   f"NPU array: N={hw.n_max} &middot; K={npu.get('num_lanes','?')} lanes &middot; "
                   f"S={hw.syn_max:,} dense all-to-all &middot; "
                   f"{cap_txt}</span>")
        else:
            txt = (f"<span style='color:{UI['text_muted']}'>{b.get('part','?')} &middot; "
                   f"{b.get('core_clk_hz',0)//1_000_000} MHz &middot; "
                   f"{b.get('bram_kb','?')} Kb BRAM &middot; "
                   f"N_MAX={hw.n_max} &middot; SYN_DEPTH={hw.syn_max}</span>")
        # Short badge inline (keeps the connection bar on one screen width);
        # the full justification goes in the tooltip instead of the label text
        # (2026-07-31, fixed real overflow -- the old inline sentences like
        # "(hardware-validated dataset microseer: bit-exact SW==HW)" and
        # "(bitstream built but NOT yet hardware-validated -- verify results
        # against the software simulator)", stacked next to the equally long
        # NPU-STDP dropdown label, pushed the window past the screen edge).
        tip = None
        if b.get("status") == "stub":
            txt += f" <span style='color:{UI['warning']}'>(stub)</span>"
            tip = "Board wrapper/.xdc not built yet."
        elif b.get("status") == "built_unverified":
            txt += f" <span style='color:{UI['warning']}'>(unverified)</span>"
            tip = ("Bitstream built but NOT yet hardware-validated -- verify "
                   "results against the software simulator.")
        elif b.get("npu", {}).get("hardware_validated"):
            profile = b.get("validation_profile", "selected image")
            txt += f" <span style='color:{UI['success']}'>(validated)</span>"
            tip = f"Hardware-validated {profile}: bit-exact SW==HW."
        self.board_info.setText(txt)
        self.board_info.setToolTip(tip or "")

    # ---- run ----
    def build_run(self):
        if getattr(self, "_generated_example", None):
            self._run_generated_example(); return
        if self.ntab.rowCount() == 0:
            QMessageBox.information(self, "No neurons", "Add at least one neuron."); return
        if not self.mock.isChecked() and self.drv is None:
            QMessageBox.information(self, "Not connected", "Connect to the board (or tick Mock)."); return
        steps = self.steps.value()
        try:
            sched = self._schedule()
        except ValueError as e:
            QMessageBox.warning(self, "Bad input schedule", str(e)); return
        on_fpga = (not self.mock.isChecked()) and (self.drv is not None)
        do_compare = self.compare_sw.isChecked()
        net = self._build_snn(self.drv if on_fpga else None)
        # NPU array outputs spikes only (no vmem/refractory readback): always take
        # the run()+software-reference branch there, whatever the network size.
        trace_readback = ((not on_fpga) or (len(net.neurons) <= 128)) \
            and net.supports_state_readback

        def job():
            net.validate()                       # bounds check (raises with a clear message)
            net.check_schedule(sched)            # reject inputs to non-input neurons (FPGA gates them)
            t0 = time.perf_counter()
            if on_fpga:
                net.build()                                          # program the chip
                t1 = time.perf_counter()
                if trace_readback:
                    train, vmem, refrac = net.run_trace(sched, steps)   # run + read state back
                    state_vmem, state_refrac = vmem, refrac
                    state_source = "fpga"
                    ref_train = ref_vmem = ref_refrac = None
                else:
                    train = net.run(sched, steps)
                    vmem, refrac = None, None
                    ref_train, ref_vmem, ref_refrac = net.software_trace(sched, steps)
                    state_vmem, state_refrac = ref_vmem, ref_refrac
                    state_source = "software_reference"
                t2 = time.perf_counter()
                learned = {s.idx: net.weight(s) for s in net.synapses}
                t3 = time.perf_counter()
                timings = {"config": t1 - t0, "run": t2 - t1, "readback": t3 - t2, "total": t3 - t0,
                               "traced": trace_readback}
            else:
                train, vmem, refrac = net.software_trace(sched, steps)
                state_vmem, state_refrac = vmem, refrac
                state_source = "software_preview"
                ref_train = ref_vmem = ref_refrac = None
                # predicted learned weights via the FPGA's own STDP rule (decimal)
                pw = net.predict_weights(train)
                learned = {s.idx: nm.from_raw(pw[s.idx], net.frac_bits) for s in net.synapses}
                timings = {"total": time.perf_counter() - t0}
            comparison = None
            if do_compare:
                reference_weights = None
                if on_fpga:
                    if ref_train is None:
                        ref_train, ref_vmem, ref_refrac, reference_weights = net.software_reference(sched, steps)
                    else:
                        _, _, _, reference_weights = net.software_reference(sched, steps)
                else:
                    ref_train, ref_vmem, ref_refrac, reference_weights = net.software_reference(sched, steps)
                comparison = {
                    "enabled": True,
                    "observed_label": "FPGA result" if on_fpga else "Software preview",
                    "reference_label": "SuperNeuroMAT reference",
                    "reference_train": ref_train,
                    "reference_vmem": ref_vmem,
                    "reference_refrac": ref_refrac,
                    "trace": nm.compare_trace_outputs(
                        train, ref_train,
                        observed_vmem=vmem, reference_vmem=ref_vmem,
                        observed_refrac=refrac, reference_refrac=ref_refrac,
                    ),
                    "weights": nm.compare_weight_results_reference(net, learned, reference_weights or {}),
                    "note": ("Software preview is generated by the same installed SuperNeuroMAT simulator; "
                          "the reference shown here is a second independently constructed SuperNeuroMAT "
                          "simulation of the same network."
                          if not on_fpga else
                          "Spike, membrane, refractory, and learned weights are checked against a "
                          "separately constructed SuperNeuroMAT software simulation of the same network."),
                }
            return {"net": net, "train": train, "learned": learned, "on_fpga": on_fpga,
                        "vmem": vmem, "refrac": refrac,
                        "state_vmem": state_vmem, "state_refrac": state_refrac,
                        "state_source": state_source,
                        "timings": timings, "comparison": comparison}

        self.btn_run.setEnabled(False)
        self._start_running_anim()
        self.worker = RunWorker(job)
        self.worker.done.connect(self._run_done)
        self.worker.start()

    def _tick_running(self):
        self._run_dots = (self._run_dots + 1) % 4
        self.run_lbl.setText("⏳  running" + " ●" * self._run_dots)

    def _run_done(self, result):
        self.btn_run.setEnabled(True)
        if isinstance(result, Exception):
            self._stop_running_anim(ok=False)
            self.run_lbl.setText("✖ error - " + str(result).splitlines()[0][:80])
            QMessageBox.critical(self, "Run failed", str(result)); return
        self._stop_running_anim(ok=True)
        net = result["net"]; train = result["train"]; learned = result["learned"]
        n = len(net.neurons)
        src = "FPGA" if result["on_fpga"] else "software preview"
        fired = {i for s in train for i in s}
        # network graph
        self._draw_network(fired=fired, learned=learned)
        # raster plot
        self.ras_ax.clear()
        xs = [t for t, s in enumerate(train) for _ in s]; ys = [i for s in train for i in s]
        self.ras_ax.scatter(xs, ys, marker="|", s=200)
        self.ras_ax.set_xlabel("timestep"); self.ras_ax.set_ylabel("neuron")
        self.ras_ax.set_title(f"spike train ({src})")
        format_raster_axes(self.ras_ax, len(train), n)
        self.ras_fig.tight_layout(); self.ras_canvas.draw()
        # raster table
        self.ras_text.setPlainText(spike_table(train, n_neurons=n))
        # weights
        self.wtab.setRowCount(0)
        for s in net.synapses:
            r = self.wtab.rowCount(); self.wtab.insertRow(r)
            init = s.weight; learn = learned.get(s.idx, init)
            for c, val in enumerate([f"{s.pre}->{s.post}", f"{init:g}", f"{learn:g}", f"{learn-init:+g}"]):
                self.wtab.setItem(r, c, self._item(val))
        # per-step neuron state (Vmem + refractory)
        self._fill_state_table(
            net,
            train,
            result.get("state_vmem"),
            result.get("state_refrac"),
            source=result.get("state_source", "fpga"),
        )
        # timing analysis (on-chip tick latency)
        self._fill_timing(net, result["on_fpga"], traced=result.get("timings", {}).get("traced", True))
        self._fill_comparison_view(
            result.get("comparison"),
            observed_train=train,
            observed_vmem=result.get("vmem"),
            observed_refrac=result.get("refrac"),
        )
        total = sum(len(s) for s in train)
        tm = result.get("timings", {})
        if result["on_fpga"] and "config" in tm:
            time_str = (f"   ⏱ config {self._fmt_time(tm['config'])} + run "
                        f"{self._fmt_time(tm['run'])} + read {self._fmt_time(tm['readback'])} "
                        f"= {self._fmt_time(tm['total'])} end-to-end")
        else:
            time_str = f"   ⏱ {self._fmt_time(tm.get('total', 0.0))} end-to-end"
        compare = result.get("comparison")
        compare_str = ""
        if compare and compare.get("enabled"):
            compare_str = f"   |   compare {'PASS' if (compare['trace']['ok'] and compare['weights']['ok']) else 'CHECK'}"
        self.run_lbl.setText(f"{src}: {len(train)} steps, {total} spikes, "
                             f"{n} neurons, {len(net.synapses)} synapses{time_str}{compare_str}")

    def _clear_comparison_view(self):
        self.compare_summary.setHtml(
            f"<p style='color:{UI['primary_dark']}'><b>Run a network with cross-check enabled</b> to compare "
            "the observed result against the SuperNeuroMAT software reference.</p>"
        )
        self.compare_trace_tab.setRowCount(0)
        self.compare_spike_tab.setRowCount(0)
        self.compare_state_tab.setRowCount(0)
        self.compare_weight_tab.setRowCount(0)
        self.compare_details_box.hide()

    def _fill_comparison_view(self, comparison, *, observed_train, observed_vmem, observed_refrac):
        if not comparison or not comparison.get("enabled"):
            self._clear_comparison_view()
            return

        trace = comparison["trace"]
        weights = comparison["weights"]
        self.compare_trace_tab.setHorizontalHeaderLabels(
            ["step", comparison["observed_label"], comparison["reference_label"]]
        )
        self.compare_trace_tab.setRowCount(0)
        steps = max(len(observed_train), len(comparison["reference_train"]))
        for t in range(steps):
            obs = observed_train[t] if t < len(observed_train) else []
            ref = comparison["reference_train"][t] if t < len(comparison["reference_train"]) else []
            r = self.compare_trace_tab.rowCount()
            self.compare_trace_tab.insertRow(r)
            vals = [t, str(list(obs)), str(list(ref))]
            for c, val in enumerate(vals):
                self.compare_trace_tab.setItem(r, c, self._item(val))
        self.compare_trace_tab.resizeColumnToContents(0)

        summary_color = UI["success"] if (trace["ok"] and weights["ok"]) else UI["danger_dark"]
        summary = [
            (f"<h3 style='margin:2px 0;color:{summary_color}'>Comparison "
            f"{'PASS' if (trace['ok'] and weights['ok']) else 'NEEDS REVIEW'}</h3>"),
            (f"<p style='margin:4px 0'><b>Observed:</b> {comparison['observed_label']} "
            f"&nbsp;&middot;&nbsp; <b>Reference:</b> {comparison['reference_label']}</p>"),
            f"<p style='margin:4px 0'>{comparison['note']}</p>",
            ("<p style='margin:4px 0'>The table below is the timestep-by-timestep side-by-side trace. "
            "The detail section only appears when a mismatch exists.</p>"),
            (f"<p style='margin:4px 0'><b>Spike mismatches:</b> {len(trace['spike_mismatches'])} "
            f"&nbsp;&middot;&nbsp; <b>Membrane mismatches:</b> {len(trace['vmem_mismatches'])} "
            f"&nbsp;&middot;&nbsp; <b>Refractory mismatches:</b> {len(trace['refrac_mismatches'])} "
            f"&nbsp;&middot;&nbsp; <b>Weight mismatches:</b> {len(weights['weight_mismatches'])}</p>"),
        ]
        self.compare_summary.setHtml("".join(summary))

        self.compare_spike_tab.setRowCount(0)
        for mm in trace["spike_mismatches"]:
            r = self.compare_spike_tab.rowCount()
            self.compare_spike_tab.insertRow(r)
            vals = [mm["timestep"], str(mm["observed"]), str(mm["reference"])]
            for c, val in enumerate(vals):
                self.compare_spike_tab.setItem(r, c, self._item(val))

        self.compare_state_tab.setRowCount(0)
        for kind, entries in (("vmem", trace["vmem_mismatches"]), ("refrac", trace["refrac_mismatches"])):
            for mm in entries[:500]:
                r = self.compare_state_tab.rowCount()
                self.compare_state_tab.insertRow(r)
                vals = [kind, mm["timestep"], mm["neuron"], mm["observed"], mm["reference"]]
                for c, val in enumerate(vals):
                    self.compare_state_tab.setItem(r, c, self._item("" if val is None else val))

        self.compare_weight_tab.setRowCount(0)
        for mm in weights["weight_mismatches"]:
            r = self.compare_weight_tab.rowCount()
            self.compare_weight_tab.insertRow(r)
            vals = [
                f"{mm['pre']}->{mm['post']}",
                f"{mm['observed']:g}",
                f"{mm['reference']:g}",
                f"{mm['delta']:+g}",
                "SuperNeuroMAT STDP reference",
            ]
            for c, val in enumerate(vals):
                self.compare_weight_tab.setItem(r, c, self._item(val))

        has_details = bool(trace["spike_mismatches"] or trace["vmem_mismatches"] or
                           trace["refrac_mismatches"] or weights["weight_mismatches"])
        self.compare_details_box.setVisible(has_details)

    def _fill_timing(self, net, on_fpga, traced=True):
        """Professional per-timestep timing report: real on-chip cycle counts when
        run on the FPGA, an analytical estimate (from the cycle-accurate sim) in
        Mock mode."""
        clk = self.board.get("core_clk_hz", cfg.CORE_CLK_HZ)
        n = len(net.neurons); s = len(net.synapses)
        stdp = "on" if self.stdp_on.isChecked() else "off"
        us = lambda c: c / clk * 1e6
        accel = lambda c: BIO_TIMESTEP_S / (c / clk) if c else 0.0
        accel_txt = lambda c: f"{accel(c):.3g}x"
        h = ["<h2 style='margin:2px 0'>Timing Analysis</h2>",
             (f"<p style='margin:2px 0'><b>Network:</b> {n} neurons, {s} synapses, "
             f"STDP {stdp} &nbsp;&middot;&nbsp; <b>core clock:</b> {clk/1e6:g} MHz "
             f"({1e9/clk:g} ns/cycle) &nbsp;&middot;&nbsp; "
             f"<b>bio reference:</b> {BIO_TIMESTEP_S * 1e3:g} ms/timestep</p>")]
        cyc = getattr(net, "tick_cycles", None)
        if on_fpga and cyc:
            tot = [c["total"] for c in cyc]
            avg = sum(tot) / len(tot)
            asyn = sum(c["synapse"] for c in cyc) / len(cyc)
            aneu = sum(c["neuron"] for c in cyc) / len(cyc)
            astd = sum(c["stdp"] for c in cyc) / len(cyc)
            aout = sum(c["output"] for c in cyc) / len(cyc)
            h.append("<p style='margin:6px 0 2px'><b>Measured on-chip</b> "
                     "(tick_start &rarr; tick_done, one timestep for all neurons):</p>")
            h.append("<table border=1 cellpadding=5 cellspacing=0 style='border-collapse:collapse'>"
                     "<tr style='background:#eef'><th>per timestep</th><th>clock cycles</th><th>time</th>"
                     "<th>bio acceleration</th></tr>"
                     f"<tr><td>average</td><td align=right><b>{avg:.0f}</b></td>"
                     f"<td align=right>{us(avg):.2f} &micro;s</td>"
                     f"<td align=right><b>{accel_txt(avg)}</b></td></tr>"
                     f"<tr><td>min / max</td><td align=right>{min(tot)} / {max(tot)}</td>"
                     f"<td align=right>{us(min(tot)):.2f} / {us(max(tot)):.2f} &micro;s</td>"
                     f"<td align=right>{accel_txt(min(tot))} / {accel_txt(max(tot))}</td></tr></table>")
            h.append("<p style='margin:6px 0 2px'><b>Phase breakdown (avg cycles):</b></p>")
            h.append("<table border=1 cellpadding=5 cellspacing=0 style='border-collapse:collapse'>"
                     "<tr style='background:#eef'><th>phase</th><th>cycles</th><th>share</th></tr>")
            for name, v in [("synapse accumulate", asyn), ("neuron update", aneu),
                            ("STDP", astd), ("output / overhead", aout)]:
                h.append(f"<tr><td>{name}</td><td align=right>{v:.0f}</td>"
                         f"<td align=right>{(100*v/avg if avg else 0):.1f}%</td></tr>")
            h.append("</table>")
            h.append(f"<p style='margin:6px 0 2px'><b>Throughput:</b> {clk/avg:,.0f} timesteps/s "
                     f"&nbsp;&middot;&nbsp; <b>run total:</b> {len(tot)} steps = {sum(tot):,} cycles "
                     f"= {us(sum(tot))/1e3:.3f} ms of on-chip compute</p>")
        else:
            floor, worst = net.estimate_tick_cycles()
            if on_fpga:
                why = ("state trace was skipped for speed on this large FPGA run"
                       if not traced else
                       "the loaded bitstream does not expose the on-chip cycle counter "
                       "&mdash; rebuild &amp; reprogram the FPGA to measure it")
            else:
                why = "no board connected (Mock mode)"
            h.append(f"<p style='margin:6px 0 2px'><b>Estimated</b> ({why}); analytical "
                     f"model from the cycle-accurate simulation of this build:</p>")
            h.append("<table border=1 cellpadding=5 cellspacing=0 style='border-collapse:collapse'>"
                     "<tr style='background:#eef'><th>case</th><th>cycles/timestep</th><th>time</th>"
                     "<th>throughput</th><th>bio acceleration</th></tr>"
                     f"<tr><td>floor (no synaptic activity)</td><td align=right>{floor}</td>"
                     f"<td align=right>{us(floor):.2f} &micro;s</td><td align=right>{clk/floor:,.0f}/s</td>"
                     f"<td align=right><b>{accel_txt(floor)}</b></td></tr>"
                     f"<tr><td>worst ({s} synapses every tick)</td><td align=right>{worst}</td>"
                     f"<td align=right>{us(worst):.2f} &micro;s</td><td align=right>{clk/worst:,.0f}/s</td>"
                     f"<td align=right><b>{accel_txt(worst)}</b></td></tr></table>")
            h.append(f"<p style='color:#555;margin:6px 0 2px'>model: cycles &asymp; 132 + 15&middot;N + "
                     f"24&middot;S = 132 + 15&middot;{n} + 24&middot;S. Connect a board and run to measure "
                     f"the real on-chip latency.</p>")
        h.append("<p style='color:#777;font-size:11px;margin-top:8px'>This is the <b>silicon compute "
                 "time</b> per timestep (synapse + neuron + STDP + output), independent of the UART "
                 "link. The end-to-end time in the status bar additionally includes host&harr;chip "
                 "communication.</p>")
        self.timing_text.setHtml("".join(h))

    @staticmethod
    def _fmt_time(s):
        """Human-readable duration: µs / ms / s as appropriate."""
        if s < 1e-3:
            return f"{s * 1e6:.0f} µs"
        if s < 1.0:
            return f"{s * 1e3:.1f} ms"
        return f"{s:.2f} s"

    def _fill_state_table(self, net, train, vmem, refrac, source="fpga"):
        """Show each neuron's membrane voltage after every tick (rows=time,
        cols=neuron). Fired cells are red, refractory cells orange; the tooltip
        gives the threshold so you can see why a neuron does or doesn't fire."""
        t = self.state_tab
        t.clear()
        if not vmem:
            t.setRowCount(1); t.setColumnCount(1)
            t.setHorizontalHeaderLabels(["state"])
            t.setVerticalHeaderLabels(["info"])
            msg = QTableWidgetItem("Neuron state trace is not available for this run.")
            msg.setToolTip("State trace is unavailable for this result.")
            t.setItem(0, 0, msg)
            t.resizeColumnsToContents()
            return
        steps = len(vmem); n = len(net.neurons)
        t.setRowCount(steps); t.setColumnCount(n)
        t.setHorizontalHeaderLabels([f"n{i}" for i in range(n)])
        t.setVerticalHeaderLabels([f"t{ti}" for ti in range(steps)])
        source_tip = {
            "fpga": "Observed FPGA state trace.",
            "software_reference": "Software reference state trace shown because FPGA state readback was skipped for speed on this large run.",
            "software_preview": "Software preview state trace.",
        }.get(source, "State trace.")
        fired_bg = QColor(255, 150, 140); refrac_bg = QColor(255, 220, 150)
        for ti in range(steps):
            fset = set(train[ti])
            for i in range(n):
                v = vmem[ti][i]; thr = net.neurons[i].threshold
                r = refrac[ti][i] if refrac and ti < len(refrac) else 0
                if i in fset:
                    # fired: membrane crossed threshold then reset, so the post-tick
                    # value is the reset (not the peak) -- show 'spike', not a stale 0.
                    it = QTableWidgetItem("spike")
                    it.setBackground(fired_bg)
                    fnt = it.font(); fnt.setBold(True); it.setFont(fnt)
                    it.setToolTip(f"neuron {i} FIRED at t{ti}: this tick's membrane rose "
                                  f"above threshold {thr:g}, so it spiked and reset to "
                                  f"{net.neurons[i].reset:g}\n{source_tip}")
                else:
                    it = QTableWidgetItem(f"{v:g}")
                    if r:
                        it.setBackground(refrac_bg)
                        it.setToolTip(f"neuron {i} refractory ({r} ticks left) at t{ti}; Vmem={v:g}\n{source_tip}")
                    else:
                        it.setToolTip(f"neuron {i}: Vmem={v:g} vs threshold {thr:g} at t{ti}\n{source_tip}")
                t.setItem(ti, i, it)
        t.resizeColumnsToContents()

    def connect(self):
        try:
            if self.mock.isChecked():
                self.drv = None
                self.conn_lbl.setText("Ready (mock)")
                self.conn_lbl.setStyleSheet(f"color:{UI['success']};font-weight:bold;")
            elif self._is_npu_stdp():
                # NPU array (STDP): same lane-engine family, its own connector
                # (health-checked inside; error message explains the classic-image case).
                self.drv = snm_npu_stdp.connect(self.port.text(), self.board["name"],
                                                int(self.baud.text()))
                self.conn_lbl.setText("Ready (NPU-STDP)")
                self.conn_lbl.setStyleSheet(f"color:{UI['success']};font-weight:bold;")
            elif self._is_npu():
                # NPU array: different engine + protocol -> its own connector
                # (health-checked inside; error message explains the classic-image case).
                self.drv = snm_npu.connect(self.port.text(), self.board["name"],
                                           int(self.baud.text()))
                self.conn_lbl.setText("Ready (NPU)")
                self.conn_lbl.setStyleSheet(f"color:{UI['success']};font-weight:bold;")
            else:
                self.drv = nm.snm.connect(self.port.text(), int(self.baud.text()),
                                          n_max=self._active_hw().n_max,
                                          board=self.board.get("name"))
                self.drv.clear_error()
                self.drv.read_status()
                self.conn_lbl.setText("Ready")
                self.conn_lbl.setStyleSheet(f"color:{UI['success']};font-weight:bold;")
            self.btn_conn.setEnabled(False)
            self.btn_disc.setEnabled(True)
            # Lock the board selector while connected: switching boards (especially
            # classic <-> NPU, which use entirely different driver classes with no
            # common interface) would otherwise leave self.drv pointing at a live but
            # protocol-mismatched device -- the next Build & Run would hit an unclear
            # AttributeError instead of a clean "reconnect" prompt.
            self.board_cb.setEnabled(False)
            self.btn_prog.setEnabled(False)   # can't reprogram while the UART link is open
        except Exception as e:  # noqa: BLE001 - connection errors are reported in the UI
            QMessageBox.critical(self, "Connect failed", str(e))

    def disconnect(self):
        if self.drv is not None and QMessageBox.question(
                self, "Disconnect",
                "Disconnect from the FPGA?\nThe board keeps running its current "
                "configuration; you'll need to reconnect to run again.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            if self.drv is not None:
                self.drv.close()
        finally:
            self.drv = None
            self.conn_lbl.setText("disconnected")
            self.conn_lbl.setStyleSheet(f"color:{UI['danger']};font-weight:bold;")
            self.btn_conn.setEnabled(True)
            self.btn_disc.setEnabled(False)
            self.board_cb.setEnabled(True)
            self.btn_prog.setEnabled(True)

    def program_board(self):
        """Load the selected board's packaged bitstream onto the FPGA over JTAG
        (Vivado backend -- the stock Digilent driver already installed on this
        machine works with it; openFPGALoader would need a Zadig/WinUSB rebind
        that breaks Vivado's own JTAG access afterward). Volatile load only
        (lost on power-off), matching this project's policy of validating on
        volatile before ever touching a board's flash. Runs on a worker thread
        since a real JTAG program call takes 10-20+ seconds."""
        if self.drv is not None:
            QMessageBox.information(self, "Disconnect first",
                                    "Disconnect from the board before reprogramming it.")
            return
        name = self.board.get("name")
        npu = self._is_npu()
        npu_stdp = self._is_npu_stdp()
        label = self.board.get("label", name)

        # Confirm the board is actually plugged in (USB VID:PID enumeration -- works
        # regardless of which JTAG driver is bound, since that's a lower-level concern
        # than basic USB enumeration) before ever invoking Vivado. Fails OPEN: only a
        # confirmed empty scan blocks the button: a hardware feature this wasn't
        # able to introspect never blocks it, matching program_fpga()'s own
        # verify_board_identity() philosophy. The NPU-STDP registry lives outside the
        # shipping manifest, so its spec isn't introspectable
        # this way -- skip detection there and rely on Vivado's own failure message.
        # Only NPU-STDP boards are offered by this copy (classic and non-STDP
        # NPU are stubbed out -- see snm_npu.py / snm_driver.py), so detection
        # always takes the fail-open path here. The board-specific USB probe
        # that used to run for the other kinds relied on the pre-STDP
        # spikeengine package's fpga/fpga_runtime modules, which this package
        # does not ship (they are archived under legacy/); rather than import
        # something that no longer exists, we let Vivado report a missing board
        # itself, which it does clearly.
        detected = True
        if not detected:
            QMessageBox.warning(
                self, "No board detected",
                f"No USB device matching {label}'s expected VID:PID was found on this "
                "machine. Plug in the board (check the cable and that it's powered on), "
                "then try Program again.")
            return

        kind = "NPU-array (STDP)" if npu_stdp else ("NPU-array" if npu else "classic")
        if QMessageBox.question(
                self, "Program FPGA",
                f"Load the {kind} bitstream for {label} "
                "onto the connected board over JTAG?\n\nThis overwrites whatever design "
                "is currently running (volatile load -- lost on power-off; the board's "
                "flash is not touched).",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return

        def job():
            if npu_stdp:
                from superneuromat.spikeengine import program as spikeengine_program
                return spikeengine_program.program(
                    snm_npu_stdp.base_board(name),
                    bitstream=snm_npu_stdp.bitstream_path(name))
            # classic and non-STDP NPU are unreachable here: the board dropdown
            # never offers them in this copy (see snm_npu.py's module
            # docstring). This branch exists only as a clear failure if that
            # invariant is ever broken, rather than importing a module
            # (superneuromat.spikeengine.fpga) this copy does not have.
            raise RuntimeError(
                "Classic and non-STDP NPU boards are not supported by this "
                "GUI copy -- only NPU-STDP. See snm_npu.py/snm_driver.py.")

        for w in (self.btn_prog, self.btn_conn, self.board_cb):
            w.setEnabled(False)
        self.run_lbl.setStyleSheet(
            f"font-size:13px;font-weight:bold;color:white;background:{UI['warning']};"
            "padding:5px 14px;border-radius:8px;")
        self.run_lbl.setText("programming...")
        self._prog_worker = RunWorker(job)
        self._prog_worker.done.connect(self._program_done)
        self._prog_worker.start()

    def _program_done(self, result):
        self.btn_prog.setEnabled(True)
        self.btn_conn.setEnabled(True)
        self.board_cb.setEnabled(True)
        if isinstance(result, Exception):
            self.run_lbl.setStyleSheet(
                f"font-size:13px;font-weight:bold;color:white;background:{UI['danger']};"
                "padding:5px 14px;border-radius:8px;")
            self.run_lbl.setText("program failed")
            QMessageBox.critical(self, "Programming failed", str(result))
            return
        self.run_lbl.setStyleSheet(
            f"font-size:13px;font-weight:bold;color:white;background:{UI['success']};"
            "padding:5px 14px;border-radius:8px;")
        self.run_lbl.setText("programmed - press Connect")
        QMessageBox.information(self, "Programmed",
                                "Bitstream loaded. Press Connect to start using it.")

    def _start_running_anim(self):
        self._run_dots = 0
        self.run_lbl.setStyleSheet(
            f"font-size:16px;font-weight:bold;color:white;background:{UI['warning']};"
            "padding:6px 16px;border-radius:8px;")
        self.run_lbl.setText("running")
        if getattr(self, "_run_timer", None) is None:
            self._run_timer = QTimer(self)
            self._run_timer.timeout.connect(self._tick_running)
        self._run_timer.start(280)

    def _stop_running_anim(self, ok=True):
        if getattr(self, "_run_timer", None):
            self._run_timer.stop()
        bg = UI["success"] if ok else UI["danger"]
        self.run_lbl.setStyleSheet(
            f"font-size:13px;font-weight:bold;color:white;background:{bg};"
            f"padding:5px 14px;border-radius:8px;")

    def _draw_network(self, fired=None, learned=None):
        # During live editing a cell may be momentarily half-typed (e.g. "" or "-");
        # ignore the transient parse error -- the next keystroke redraws cleanly.
        try:
            neurons = self._gather_neurons(); synapses = self._gather_synapses()
        except (ValueError, TypeError):
            return
        self.net_view.set_network(neurons, synapses, fired=fired, learned=learned)

    # ---- Network view zoom / pan ----
    def _zoom_network(self, factor, cx=None, cy=None):
        """Scale the Network view by ``factor`` (<1 zooms in) about (cx, cy) or the center."""
        self.net_view.zoom(factor, cx, cy)

    def _fit_network(self):
        self.net_view.fit_view()

    def _on_net_scroll(self, event):
        self.net_view._on_scroll(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SNNConsole(); win.show()
    # --import <file>: preload an exported SNN (used by spikeengine.open_in_fpga_gui)
    args = sys.argv[1:]
    if "--import" in args:
        i = args.index("--import")
        if i + 1 < len(args):
            win.import_snn(args[i + 1])
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
