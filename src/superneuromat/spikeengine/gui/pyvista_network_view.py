"""Optional 3D network view (PyVista/VTK inside Qt).

Renders the network in 3D with neurons placed by ``graph_layout``'s spherical
layout, letting large graphs be inspected by rotating rather than fighting
2D edge overlap. ``RenderState`` tracks what is currently drawn so ticks
update incrementally instead of rebuilding the whole scene.

OPTIONAL: pyvista/pyvistaqt/vtk are not declared dependencies of this
package. ``pyvista_available()`` reports whether they imported, and
``pyvista_import_error_message()`` explains what is missing; the GUI falls
back to the 2D ``network_view`` when they are absent, so nothing breaks.
Note that VTK needs a real OpenGL context -- this view cannot be exercised
in a headless/offscreen environment.
"""
# VTK/PyVista APIs vary across backends and versions. Rendering and actor
# cleanup are deliberately best-effort so an optional 3D view cannot crash the
# main GUI; broad catches in this module are UI isolation boundaries.
# ruff: noqa: BLE001, S110
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .graph_layout import network_positions_3d

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception as exc:
    pv = None
    QtInteractor = None
    _PYVISTA_IMPORT_ERROR = exc
else:
    _PYVISTA_IMPORT_ERROR = None


VIEW = {
    "bg": "#F7F9FC",
    "text": "#15202B",
    "text_soft": "#445468",
    "danger": "#B42318",
    "danger_soft": "#FDECEC",
    "label_bg": "#FFFFFF",
    "label_text": "#15202B",
    "label_weight_bg": "#FFF7ED",
    "label_weight_text": "#7C2D12",
    "edge_bg": "#7A8CA2",
    "edge_out": "#9A3412",
    "edge_in": "#166534",
    "node_base": "#8EB9E8",
    "node_fired": "#D64545",
    "node_focus": "#D6E9FF",
    "node_focus_fired": "#F4A29B",
    "node_selected": "#D4A72C",
    "node_selected_ring": "#F2D27A",
}


@dataclass
class RenderState:
    neurons: list
    synapses: list
    fired: set
    learned: dict | None
    selected_neuron: int | None
    mode: str


def pyvista_available():
    return pv is not None and QtInteractor is not None


def pyvista_import_error_message():
    if _PYVISTA_IMPORT_ERROR is None:
        return ""
    return str(_PYVISTA_IMPORT_ERROR)


def _weight_label(s, learned):
    label = f"{s['weight']:g}"
    if learned is not None and s.get("idx") in (learned or {}) and learned[s["idx"]] != s["weight"]:
        label = f"{s['weight']:g}->{learned[s['idx']]:g}"
    return label


class PyVistaNetworkView(QWidget):
    """Interactive 3D network viewer shell for the SpikeEngine GUI.

    This starts with safe dependency handling and scene plumbing so we can swap
    it into the GUI incrementally without crashing when PyVista/VTK is absent.
    """

    def __init__(self, parent=None, *, allow_detach=True):
        super().__init__(parent)
        self._allow_detach = allow_detach
        self._state = RenderState(neurons=[], synapses=[], fired=set(), learned=None, selected_neuron=None, mode="outgoing")
        self._actors: list = []
        self._background_actors: list = []
        self._overlay_actors: list = []
        self._pick_actors: dict[object, int] = {}
        self._node_centers: dict[int, np.ndarray] = {}
        self._point_pick_nodes: dict[int, np.ndarray] = {}
        self._node_radius = 0.05
        self._overview_label_budget = 80
        self._layout_cache_key = None
        self._layout_cache_positions: dict[int, np.ndarray] = {}
        self._fullscreen_window = None
        self._fullscreen_view = None
        self._camera_initialized = False
        self._camera_needs_reset = True
        self._background_dirty = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        top = QHBoxLayout()
        self.graph_label = QLabel("View: 3D network")
        self.graph_label.setToolTip("Background topology remains faint. Selecting a neuron highlights its visible neighborhood.")
        top.addWidget(self.graph_label)

        top.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["outgoing", "incoming", "both"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top.addWidget(self.mode_combo)

        top.addWidget(QLabel("Neuron:"))
        self.select_spin = QSpinBox()
        self.select_spin.setMinimum(0)
        self.select_spin.setMaximum(0)
        self.select_spin.setToolTip("Select a neuron index to inspect its visible connections.")
        self.select_spin.valueChanged.connect(self.select_neuron)
        top.addWidget(self.select_spin)

        self.select_btn = QPushButton("Select")
        self.select_btn.clicked.connect(lambda: self.select_neuron(self.select_spin.value()))
        top.addWidget(self.select_btn)

        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.clicked.connect(self.clear_selection)
        top.addWidget(self.clear_btn)

        if self._allow_detach:
            self.fullscreen_btn = QPushButton("Open In Window")
            self.fullscreen_btn.clicked.connect(self._open_fullscreen_viewer)
            top.addWidget(self.fullscreen_btn)
        else:
            close_btn = QPushButton("Close Window")
            close_btn.clicked.connect(lambda: self.window().close())
            top.addWidget(close_btn)

        self.selection_label = QLabel("Selected: none")
        self.selection_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.selection_label, 1)
        root.addLayout(top)

        if not pyvista_available():
            self.status_label = QLabel(
                "3D view unavailable. Install pyvista, pyvistaqt, and vtk to enable the network viewer."
            )
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet(f"color:{VIEW['danger']}; background:{VIEW['danger_soft']}; padding:8px;")
            root.addWidget(self.status_label, 1)
            return

        self.plotter = QtInteractor(self)
        root.addWidget(self.plotter.interactor, 1)
        self._init_scene()

    def _init_scene(self):
        self.plotter.set_background(VIEW["bg"])
        self.plotter.enable_anti_aliasing("fxaa")
        self.plotter.camera_position = [
            (3.6, 2.8, 2.3),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        self.plotter.view_isometric(render=False)
        self.plotter.reset_camera(render=False)
        self.plotter.enable_point_picking(
            callback=self._on_point_picked,
            show_message=False,
            left_clicking=True,
            show_point=False,
            use_picker=False,
            clear_on_no_selection=False,
            pickable_window=True,
        )

    def set_network(self, neurons, synapses, fired=None, learned=None):
        prev_neurons = self._state.neurons
        prev_synapses = self._state.synapses
        prev_fired = self._state.fired
        self._state = RenderState(
            neurons=list(neurons),
            synapses=list(synapses),
            fired=set(fired or []),
            learned=learned,
            selected_neuron=self._state.selected_neuron,
            mode=self.mode_combo.currentText(),
        )
        if self._state.neurons != prev_neurons or self._state.synapses != prev_synapses:
            self._layout_cache_key = None
            self._layout_cache_positions = {}
            self._camera_needs_reset = True
            self._background_dirty = True
        elif self._state.fired != prev_fired:
            self._background_dirty = True
        self._apply_quality_profile()
        self.select_spin.setMaximum(max(0, len(self._state.neurons) - 1))
        if self._state.selected_neuron is not None and self._state.selected_neuron >= len(self._state.neurons):
            self._state.selected_neuron = None
        default_selection = min(self.select_spin.value(), len(self._state.neurons) - 1) if self._state.neurons else 0
        self.select_spin.blockSignals(True)
        self.select_spin.setValue(default_selection)
        self.select_spin.blockSignals(False)
        self._render(full=self._background_dirty, preserve_camera=not self._camera_needs_reset)
        self._sync_fullscreen_view()

    def select_neuron(self, neuron_idx: int | None):
        if neuron_idx is None:
            self.clear_selection()
            return
        if neuron_idx < 0 or neuron_idx >= len(self._state.neurons):
            return
        self._state.selected_neuron = neuron_idx
        self.select_spin.setValue(neuron_idx)
        self._render(preserve_camera=True)
        self._sync_fullscreen_view()

    def clear_selection(self):
        self._state.selected_neuron = None
        self.selection_label.setText("Selected: none")
        self._render(preserve_camera=True)
        self._sync_fullscreen_view()

    def _on_mode_changed(self, mode):
        self._state.mode = mode
        self._render(preserve_camera=True)
        self._sync_fullscreen_view()

    def _remove_actor_group(self, actors):
        if not pyvista_available():
            return
        for actor in list(actors):
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        actors.clear()

    def _clear_scene_actors(self):
        if not pyvista_available():
            return
        try:
            self.plotter.clear_actors()
        except Exception:
            for actor in self._actors:
                try:
                    self.plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
        self._actors.clear()
        self._background_actors.clear()
        self._overlay_actors.clear()
        self._pick_actors.clear()
        self._node_centers.clear()
        self._point_pick_nodes.clear()

    def _open_fullscreen_viewer(self):
        if not pyvista_available() or not self._allow_detach:
            return
        try:
            if self._fullscreen_window is not None and self._fullscreen_window.isVisible():
                self._fullscreen_window.activateWindow()
                self._fullscreen_window.raise_()
                return
        except RuntimeError:
            self._drop_fullscreen_refs()

        self._fullscreen_window = QWidget(None, Qt.Window)
        self._fullscreen_window.setWindowTitle("SpikeEngine Network View")
        self._fullscreen_window.setAttribute(Qt.WA_DeleteOnClose, True)
        self._fullscreen_window.destroyed.connect(self._on_fullscreen_window_closed)
        QShortcut("Escape", self._fullscreen_window, activated=self._fullscreen_window.close)
        layout = QVBoxLayout(self._fullscreen_window)
        layout.setContentsMargins(0, 0, 0, 0)
        self._fullscreen_view = PyVistaNetworkView(self._fullscreen_window, allow_detach=False)
        layout.addWidget(self._fullscreen_view)
        self._sync_fullscreen_view()
        self._fullscreen_window.showMaximized()

    def _on_fullscreen_window_closed(self, *_args):
        self._drop_fullscreen_refs()

    def _drop_fullscreen_refs(self):
        self._fullscreen_window = None
        self._fullscreen_view = None

    def _sync_fullscreen_view(self):
        if self._fullscreen_view is None:
            return
        try:
            self._fullscreen_view.mode_combo.blockSignals(True)
            self._fullscreen_view.mode_combo.setCurrentText(self._state.mode)
            self._fullscreen_view.mode_combo.blockSignals(False)
            self._fullscreen_view.set_network(
                self._state.neurons,
                self._state.synapses,
                fired=self._state.fired,
                learned=self._state.learned,
            )
            if self._state.selected_neuron is not None:
                self._fullscreen_view.select_neuron(self._state.selected_neuron)
            else:
                self._fullscreen_view.clear_selection()
        except RuntimeError:
            self._drop_fullscreen_refs()

    def _render(self, full=None, preserve_camera=False):
        if not pyvista_available():
            return
        if full is None:
            full = self._background_dirty or not self._background_actors
        prior_camera = None
        if preserve_camera:
            try:
                prior_camera = self.plotter.camera_position
            except Exception:
                prior_camera = None
        try:
            if not self._state.neurons:
                self._clear_scene_actors()
                self.selection_label.setText("Selected: none")
                self.plotter.render()
                return
            if full:
                self._clear_scene_actors()
            self._node_centers = {idx: xyz.copy() for idx, xyz in self._get_layout_positions().items()}
            if full:
                self._render_background_scene()
                self._background_dirty = False
            else:
                prior_overlay = list(self._overlay_actors)
                self._remove_actor_group(self._overlay_actors)
                self._actors = [actor for actor in self._actors if actor not in prior_overlay]
                self._pick_actors.clear()
            self._render_overlay_scene()
            self._update_camera(prior_camera if preserve_camera else None)
            self.plotter.render()
        except Exception:
            self._clear_scene_actors()
            self._node_centers = {idx: xyz.copy() for idx, xyz in self._get_layout_positions().items()}
            self._render_fallback_scene()
            self._update_camera(prior_camera if preserve_camera else None)
            self.plotter.render()

    def _render_background_scene(self):
        self._add_background_edges()
        self._add_background_neurons()

    def _render_overlay_scene(self):
        self._add_selected_edges()
        self._add_focus_neurons_and_labels()
        self._add_selection_highlight()

    def _render_fallback_scene(self):
        if not self._node_centers:
            self.selection_label.setText("Selected: none")
            return
        pts = np.asarray(list(self._node_centers.values()), dtype=float)
        actor = self.plotter.add_points(
            pts,
            color=VIEW["node_base"],
            opacity=0.95,
            point_size=self._background_point_size(),
            render_points_as_spheres=True,
            pickable=False,
            render=False,
        )
        self._track_actor(actor, layer="background")
        if self._state.selected_neuron is not None and self._state.selected_neuron in self._node_centers:
            center = self._node_centers[self._state.selected_neuron]
            labels = pv.PolyData(np.asarray([center], dtype=float))
            labels["label"] = [str(self._state.selected_neuron)]
            actor = self.plotter.add_point_labels(
                labels,
                "label",
                point_size=0,
                font_size=13,
                text_color=VIEW["label_text"],
                shape="rounded_rect",
                shape_color=VIEW["label_bg"],
                shape_opacity=0.92,
                margin=2,
                always_visible=True,
                show_points=False,
                pickable=False,
                render=False,
            )
            self._track_actor(actor, layer="overlay")
            self.selection_label.setText(f"Selected: neuron {self._state.selected_neuron} (fallback)")
        else:
            self.selection_label.setText("Selected: none")

    def _apply_quality_profile(self):
        if not pyvista_available():
            return
        try:
            if len(self._state.neurons) >= 900:
                self.plotter.disable_anti_aliasing()
            else:
                self.plotter.enable_anti_aliasing("fxaa")
        except Exception:
            pass

    def _layout_signature(self):
        neuron_sig = tuple(
            (
                n.get("id"),
                n.get("threshold"),
                n.get("leak"),
                n.get("reset"),
                n.get("refractory"),
                n.get("input"),
            )
            for n in self._state.neurons
        )
        syn_sig = tuple(
            (
                s.get("pre"),
                s.get("post"),
                s.get("weight"),
                s.get("stdp"),
            )
            for s in self._state.synapses
        )
        return (neuron_sig, syn_sig)

    def _get_layout_positions(self):
        signature = self._layout_signature()
        if signature != self._layout_cache_key:
            pos3d = network_positions_3d(self._state.neurons, self._state.synapses)
            self._layout_cache_positions = {idx: np.array(xyz, dtype=float) for idx, xyz in pos3d.items()}
            self._layout_cache_key = signature
        return self._layout_cache_positions

    def _add_background_edges(self):
        if not self._state.synapses:
            return
        selected = self._state.selected_neuron
        points = []
        lines = []
        point_idx = 0
        for s in self._state.synapses:
            pre = s.get("pre")
            post = s.get("post")
            if pre not in self._node_centers or post not in self._node_centers or pre == post:
                continue
            if selected is not None and (pre == selected or post == selected):
                continue
            curve = self._curve_points(
                self._node_centers[pre],
                self._node_centers[post],
                bend_scale=self._background_bend_scale(pre, post),
                lift_scale=self._background_lift_scale(pre, post),
                samples=self._background_curve_samples(),
            )
            curve = self._trim_curve_to_neuron_surfaces(curve)
            points.extend(curve)
            lines.extend([len(curve), *range(point_idx, point_idx + len(curve))])
            point_idx += len(curve)
        if points:
            mesh = pv.PolyData(np.asarray(points, dtype=float))
            mesh.lines = np.asarray(lines, dtype=np.int64)
            actor = self.plotter.add_mesh(
                mesh,
                color=VIEW["edge_bg"],
                opacity=self._background_edge_opacity(),
                line_width=self._background_edge_width(),
                render_lines_as_tubes=False,
                pickable=False,
                render=False,
            )
            self._track_actor(actor, layer="background")

    def _add_selected_edges(self):
        selected = self._state.selected_neuron
        if selected is None or selected not in self._node_centers:
            return

        weight_points = []
        weight_labels = []
        edge_pairs = {(s.get("pre"), s.get("post")) for s in self._state.synapses}
        for s in self._state.synapses:
            pre = s.get("pre")
            post = s.get("post")
            if pre not in self._node_centers or post not in self._node_centers:
                continue

            outgoing = pre == selected
            incoming = post == selected
            if self._state.mode == "outgoing" and not outgoing:
                continue
            if self._state.mode == "incoming" and not incoming:
                continue
            if self._state.mode == "both" and not (incoming or outgoing):
                continue

            if pre == post == selected:
                midpoint = self._add_self_loop(self._node_centers[selected], s)
                if midpoint is not None:
                    weight_points.append(midpoint)
                    weight_labels.append(_weight_label(s, self._state.learned))
                continue
            if pre == post:
                continue

            start = self._node_centers[pre]
            end = self._node_centers[post]
            reciprocal = (post, pre) in edge_pairs and pre != post
            if outgoing and incoming:
                bend = 0.0
                lift = 0.0
            elif outgoing:
                bend = 0.11 if reciprocal else 0.075
                lift = 0.09 if reciprocal else 0.03
            else:
                bend = -0.11 if reciprocal else -0.075
                lift = -0.09 if reciprocal else -0.03
            curve = self._curve_points(start, end, bend_scale=bend, lift_scale=lift, samples=24)
            curve = self._trim_curve_to_neuron_surfaces(curve)
            color = VIEW["edge_out"] if outgoing else VIEW["edge_in"]
            spline = pv.Spline(curve, len(curve) * 3)
            tube = spline.tube(radius=0.0072 if self._state.mode != "both" else 0.006)
            actor = self.plotter.add_mesh(
                tube,
                color=color,
                opacity=0.96,
                smooth_shading=True,
                pickable=False,
                render=False,
            )
            self._track_actor(actor, layer="overlay")
            self._add_arrow(curve, color=color)
            weight_points.append(curve[len(curve) // 2] + np.array([0.0, 0.0, 0.04]))
            weight_labels.append(_weight_label(s, self._state.learned))

        if weight_points:
            labels = pv.PolyData(np.asarray(weight_points))
            labels["label"] = weight_labels
            actor = self.plotter.add_point_labels(
                labels,
                "label",
                point_size=0,
                font_size=14,
                text_color=VIEW["label_weight_text"],
                shape="rounded_rect",
                shape_color=VIEW["label_weight_bg"],
                shape_opacity=0.98,
                margin=2,
                always_visible=True,
                show_points=False,
                pickable=False,
                render=False,
            )
            self._track_actor(actor, layer="overlay")

    def _add_background_neurons(self):
        self._point_pick_nodes = {idx: center for idx, center in self._node_centers.items()}
        highlight_nodes = self._selected_neighbor_nodes()
        selected_or_highlight = set(highlight_nodes)
        if self._state.selected_neuron is not None:
            selected_or_highlight.add(self._state.selected_neuron)

        base_points = []
        fired_points = []
        for idx, center in self._node_centers.items():
            if idx in selected_or_highlight:
                continue
            if idx in self._state.fired:
                fired_points.append(center)
            else:
                base_points.append(center)

        if base_points:
            actor = self.plotter.add_points(
                np.asarray(base_points, dtype=float),
                color=VIEW["node_base"],
                opacity=0.90,
                point_size=self._background_point_size(),
                render_points_as_spheres=True,
                pickable=False,
                render=False,
            )
            self._track_actor(actor, layer="background")
        if fired_points:
            actor = self.plotter.add_points(
                np.asarray(fired_points, dtype=float),
                color=VIEW["node_fired"],
                opacity=0.94,
                point_size=self._background_point_size() + 1,
                render_points_as_spheres=True,
                pickable=False,
                render=False,
            )
            self._track_actor(actor, layer="background")

        label_points, label_text = self._overview_labels(highlight_nodes)
        if label_points:
            labels = pv.PolyData(np.asarray(label_points))
            labels["label"] = label_text
            actor = self.plotter.add_point_labels(
                labels,
                "label",
                point_size=0,
                font_size=11 if len(label_points) > 40 else 13,
                text_color=VIEW["label_text"],
                shape="rounded_rect",
                shape_color=VIEW["label_bg"],
                shape_opacity=0.84 if len(label_points) > 40 else 0.92,
                margin=2,
                always_visible=True,
                show_points=False,
                pickable=False,
                render=False,
            )
            self._track_actor(actor, layer="background")

    def _add_focus_neurons_and_labels(self):
        highlight_nodes = self._selected_neighbor_nodes()
        focus_nodes = []
        if self._state.selected_neuron is not None and self._state.selected_neuron in self._node_centers:
            focus_nodes.append(self._state.selected_neuron)
        focus_nodes.extend(sorted(idx for idx in highlight_nodes if idx in self._node_centers))
        if not focus_nodes:
            return

        label_points = []
        label_text = []
        for idx in focus_nodes:
            center = self._node_centers[idx]
            if idx == self._state.selected_neuron:
                face = VIEW["node_selected"]
                radius = self._node_radius * 1.02
            elif idx in self._state.fired:
                face = VIEW["node_focus_fired"]
                radius = self._node_radius * 0.94
            else:
                face = VIEW["node_focus"]
                radius = self._node_radius * 0.94
            edge_sphere = pv.Sphere(
                radius=radius,
                center=center,
                theta_resolution=self._focus_sphere_resolution(),
                phi_resolution=self._focus_sphere_resolution(),
            )
            actor = self.plotter.add_mesh(
                edge_sphere,
                color=face,
                smooth_shading=True,
                specular=0.08,
                specular_power=8,
                ambient=0.34,
                render=False,
            )
            self._track_actor(actor, layer="overlay")
            self._pick_actors[actor] = idx
            label_points.append(center)
            label_text.append(str(idx))

        labels = pv.PolyData(np.asarray(label_points, dtype=float))
        labels["label"] = label_text
        actor = self.plotter.add_point_labels(
            labels,
            "label",
            point_size=0,
            font_size=13,
            text_color=VIEW["label_text"],
            shape="rounded_rect",
            shape_color=VIEW["label_bg"],
            shape_opacity=0.94,
            margin=2,
            always_visible=True,
            show_points=False,
            pickable=False,
            render=False,
        )
        self._track_actor(actor, layer="overlay")

    def _overview_label_stride(self) -> int:
        n = len(self._node_centers)
        if n <= self._overview_label_budget:
            return 1
        return max(1, math.ceil(n / self._overview_label_budget))

    def _add_selection_highlight(self):
        selected = self._state.selected_neuron
        if selected is None or selected not in self._node_centers:
            self.selection_label.setText("Selected: none")
            return
        self.selection_label.setText(f"Selected: neuron {selected} ({self._state.mode})")
        center = self._node_centers[selected]
        ring = pv.Sphere(radius=self._node_radius * 1.22, center=center, theta_resolution=24, phi_resolution=24)
        actor = self.plotter.add_mesh(
            ring,
            color=VIEW["node_selected_ring"],
            opacity=0.28,
            smooth_shading=True,
            ambient=0.34,
            pickable=False,
            render=False,
        )
        self._track_actor(actor, layer="overlay")
        focus = pv.Sphere(radius=self._node_radius * 0.72, center=center, theta_resolution=24, phi_resolution=24)
        actor = self.plotter.add_mesh(
            focus,
            color=VIEW["node_selected"],
            opacity=0.88,
            smooth_shading=True,
            specular=0.1,
            specular_power=8,
            ambient=0.34,
            pickable=False,
            render=False,
        )
        self._track_actor(actor, layer="overlay")

    def _selected_neighbor_nodes(self):
        selected = self._state.selected_neuron
        if selected is None:
            return set()
        nodes = set()
        for s in self._state.synapses:
            pre = s.get("pre")
            post = s.get("post")
            if pre == selected:
                nodes.add(post)
            if post == selected:
                nodes.add(pre)
        nodes.discard(selected)
        return nodes

    def _add_arrow(self, curve: np.ndarray, color: str):
        if len(curve) < 8:
            return
        tip_idx = len(curve) - 1
        base_idx = max(0, len(curve) - 5)
        tip = curve[tip_idx]
        base = curve[base_idx]
        direction = tip - base
        if np.linalg.norm(direction) < 1e-9:
            return
        unit = direction / np.linalg.norm(direction)
        height = 0.078
        radius = 0.019
        tip = tip - unit * (self._node_radius * 0.10)
        center = tip - unit * (height * 0.5)
        arrow = pv.Cone(
            center=center,
            direction=unit,
            height=height,
            radius=radius,
            resolution=20,
            capping=True,
        )
        actor = self.plotter.add_mesh(arrow, color=color, smooth_shading=True, opacity=0.92, render=False)
        self._track_actor(actor, layer="overlay")

    def _add_self_loop(self, center: np.ndarray, synapse):
        theta = np.linspace(-0.15 * math.pi, 1.20 * math.pi, 70)
        loop_r = self._node_radius * 1.9
        points = np.column_stack(
            [
                center[0] + loop_r * np.cos(theta),
                center[1] + self._node_radius * 1.0 + loop_r * np.sin(theta),
                center[2] + self._node_radius * 0.65 * np.sin(theta * 0.55),
            ]
        )
        spline = pv.Spline(points, len(points) * 3)
        tube = spline.tube(radius=0.0068)
        actor = self.plotter.add_mesh(tube, color=VIEW["edge_out"], opacity=0.96, smooth_shading=True, render=False)
        self._track_actor(actor, layer="overlay")
        self._add_arrow(points, color=VIEW["edge_out"])
        return points[len(points) // 2] + np.array([0.0, 0.0, 0.04])

    def _on_point_picked(self, point):
        if point is None:
            return
        point = np.asarray(point, dtype=float)
        best_idx = None
        best_dist2 = None
        search_nodes = self._point_pick_nodes or self._node_centers
        for idx, center in search_nodes.items():
            dist2 = float(np.sum((center - point) ** 2))
            if best_dist2 is None or dist2 < best_dist2:
                best_idx = idx
                best_dist2 = dist2
        if best_idx is not None:
            self.select_neuron(best_idx)

    def _fit_camera_to_scene(self):
        if not self._node_centers:
            return
        pts = np.asarray(list(self._node_centers.values()))
        center = pts.mean(axis=0)
        extent = np.ptp(pts, axis=0)
        radius = max(0.9, float(np.max(extent)) * 0.9 + 0.6)
        self.plotter.camera_position = [
            (center[0] + radius * 1.5, center[1] + radius * 1.2, center[2] + radius * 1.0),
            tuple(center),
            (0.0, 0.0, 1.0),
        ]
        self._camera_initialized = True
        self._camera_needs_reset = False

    def _update_camera(self, preserved_camera):
        if preserved_camera is not None and self._camera_initialized and not self._camera_needs_reset:
            try:
                self.plotter.camera_position = preserved_camera
                return
            except Exception:
                pass
        self._fit_camera_to_scene()

    def _trim_curve_to_neuron_surfaces(self, curve: np.ndarray) -> np.ndarray:
        if len(curve) < 6:
            return curve
        trimmed = curve.copy()
        start_center = trimmed[0].copy()
        end_center = trimmed[-1].copy()
        start_dir = trimmed[2] - start_center
        end_dir = trimmed[-3] - end_center
        start_norm = np.linalg.norm(start_dir)
        end_norm = np.linalg.norm(end_dir)
        if start_norm > 1e-9:
            trimmed[0] = start_center + (start_dir / start_norm) * (self._node_radius * 1.04)
        if end_norm > 1e-9:
            trimmed[-1] = end_center + (end_dir / end_norm) * (self._node_radius * 1.04)
        return trimmed

    def _track_actor(self, actor, *, layer: str):
        if actor is None:
            return
        self._actors.append(actor)
        if layer == "background":
            self._background_actors.append(actor)
        else:
            self._overlay_actors.append(actor)

    def _background_point_size(self) -> float:
        n = max(1, len(self._state.neurons))
        if n <= 32:
            return 14
        if n <= 128:
            return 11
        if n <= 512:
            return 8
        return 6

    def _focus_sphere_resolution(self) -> int:
        n = max(1, len(self._state.neurons))
        if n <= 32:
            return 24
        if n <= 128:
            return 18
        return 14

    def _overview_label_limit(self) -> int:
        n = max(1, len(self._state.neurons))
        if n <= 24:
            return n
        if n <= 128:
            return 42
        if n <= 512:
            return 56
        return 72

    def _background_curve_samples(self) -> int:
        n = max(1, len(self._state.synapses))
        if n <= 64:
            return 14
        if n <= 512:
            return 10
        return 7

    def _background_edge_width(self) -> float:
        if len(self._state.synapses) <= 128:
            return 1.4
        if len(self._state.synapses) <= 512:
            return 1.1
        return 1.0

    def _background_edge_opacity(self) -> float:
        if len(self._state.synapses) <= 128:
            return 0.22
        if len(self._state.synapses) <= 512:
            return 0.18
        return 0.14

    def _background_bend_scale(self, pre: int, post: int) -> float:
        start = self._node_centers[pre]
        end = self._node_centers[post]
        length = float(np.linalg.norm(end - start))
        if length < 1e-9:
            return 0.0
        return min(0.09, max(0.015, length * 0.08))

    def _background_lift_scale(self, pre: int, post: int) -> float:
        return 0.018 if ((pre + post) % 2 == 0) else -0.018

    def _overview_labels(self, highlight_nodes: set[int]):
        focus_nodes = set(highlight_nodes)
        if self._state.selected_neuron is not None:
            focus_nodes.add(self._state.selected_neuron)
        limit = self._overview_label_limit()
        if len(self._node_centers) <= limit:
            sampled = sorted(self._node_centers)
        else:
            stride = max(1, math.ceil(len(self._node_centers) / limit))
            sampled = sorted(idx for idx in self._node_centers if idx % stride == 0)
        combined = []
        seen = set()
        for idx in sampled + sorted(focus_nodes):
            if idx in self._node_centers and idx not in seen:
                seen.add(idx)
                combined.append(idx)
        return [self._node_centers[idx] for idx in combined], [str(idx) for idx in combined]

    @staticmethod
    def _curve_points(start: np.ndarray, end: np.ndarray, bend_scale: float, lift_scale: float = 0.0, samples: int = 18) -> np.ndarray:
        delta = end - start
        norm = np.linalg.norm(delta)
        if norm < 1e-9:
            return np.repeat(start[None, :], samples, axis=0)
        unit = delta / norm
        probe = np.array([0.0, 0.0, 1.0])
        perp = np.cross(unit, probe)
        if np.linalg.norm(perp) < 1e-6:
            probe = np.array([0.0, 1.0, 0.0])
            perp = np.cross(unit, probe)
        perp /= np.linalg.norm(perp)
        midpoint = 0.5 * (start + end)
        control = midpoint + bend_scale * perp + np.array([0.0, 0.0, lift_scale], dtype=float)
        t = np.linspace(0.0, 1.0, samples)
        omt = 1.0 - t
        return (
            (omt[:, None] ** 2) * start[None, :]
            + (2.0 * omt[:, None] * t[:, None]) * control[None, :]
            + (t[:, None] ** 2) * end[None, :]
        )
