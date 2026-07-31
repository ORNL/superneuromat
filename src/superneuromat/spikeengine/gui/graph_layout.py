"""Node-placement algorithms for the GUI's network views.

Computes 2D and 3D coordinates for a spiking network's neurons so the graph
is readable rather than overlapping: connected components are laid out
separately, then relaxed apart (``_relax_component_positions`` /
``_relax_component_positions_3d``) so components do not collide, with
``_fibonacci_sphere_point`` giving an even spread for the 3D case.

Pure geometry -- no Qt, no rendering, no hardware. Consumed by
``network_view`` (matplotlib 2D) and ``pyvista_network_view`` (optional 3D).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


def _relax_component_positions(pos, members, min_sep=0.16, iterations=20):
    original = {i: pos[i] for i in members}
    for _ in range(iterations):
        delta = {i: [0.0, 0.0] for i in members}
        for idx, a in enumerate(members):
            ax, ay = pos[a]
            for b in members[idx + 1:]:
                bx, by = pos[b]
                dx = bx - ax
                dy = by - ay
                dist2 = dx * dx + dy * dy
                if dist2 < 1e-12:
                    dx, dy = 0.001, 0.0
                    dist2 = dx * dx
                dist = math.sqrt(dist2)
                if dist >= min_sep:
                    continue
                push = 0.5 * (min_sep - dist)
                ux = dx / dist
                uy = dy / dist
                delta[a][0] -= ux * push
                delta[a][1] -= uy * push
                delta[b][0] += ux * push
                delta[b][1] += uy * push
        for i in members:
            ox, oy = original[i]
            dx, dy = delta[i]
            x, y = pos[i]
            x += dx + 0.10 * (ox - x)
            y += dy + 0.10 * (oy - y)
            pos[i] = (x, y)


def _relax_component_positions_3d(pos, members, min_sep=0.22, iterations=28):
    original = {i: pos[i] for i in members}
    for _ in range(iterations):
        delta = {i: [0.0, 0.0, 0.0] for i in members}
        for idx, a in enumerate(members):
            ax, ay, az = pos[a]
            for b in members[idx + 1:]:
                bx, by, bz = pos[b]
                dx = bx - ax
                dy = by - ay
                dz = bz - az
                dist2 = dx * dx + dy * dy + dz * dz
                if dist2 < 1e-12:
                    dx, dy, dz = 0.001, 0.0, 0.001
                    dist2 = dx * dx + dz * dz
                dist = math.sqrt(dist2)
                if dist >= min_sep:
                    continue
                push = 0.5 * (min_sep - dist)
                ux = dx / dist
                uy = dy / dist
                uz = dz / dist
                delta[a][0] -= ux * push
                delta[a][1] -= uy * push
                delta[a][2] -= uz * push
                delta[b][0] += ux * push
                delta[b][1] += uy * push
                delta[b][2] += uz * push
        for i in members:
            ox, oy, oz = original[i]
            dx, dy, dz = delta[i]
            x, y, z = pos[i]
            x += dx + 0.08 * (ox - x)
            y += dy + 0.08 * (oy - y)
            z += dz + 0.08 * (oz - z)
            pos[i] = (x, y, z)


def _fibonacci_sphere_point(index, count):
    if count <= 1:
        return (0.0, 0.0, 1.0)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - (2.0 * index + 1.0) / count
    radial = math.sqrt(max(0.0, 1.0 - z * z))
    theta = golden_angle * index
    return (math.cos(theta) * radial, math.sin(theta) * radial, z)


def _fibonacci_volume_point(index, count):
    ux, uy, uz = _fibonacci_sphere_point(index, count)
    radius = ((index + 0.5) / max(1, count)) ** (1.0 / 3.0)
    return (ux * radius, uy * radius, uz * radius)


def _component_basis(slot, count):
    w = np.array(_fibonacci_sphere_point(slot, max(1, count)), dtype=float)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-9:
        w = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        w /= w_norm
    probe = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(w, probe))) > 0.9:
        probe = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(w, probe)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    spin = (2.0 * math.pi * ((slot * 0.61803398875) % 1.0))
    cs = math.cos(spin)
    sn = math.sin(spin)
    return (cs * u + sn * v, -sn * u + cs * v, w)


def network_positions(neurons, synapses):
    """Deterministic topology-driven 2D positions for network viewers.

    Returns `(pos, n_components, n_grid_cols)`, where `pos` maps neuron index to
    a 2D point. Connected components are laid out in a stable tiled arrangement,
    and each component is arranged by longest-path depth where possible.
    """
    n = len(neurons)
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    adj = [set() for _ in range(n)]
    for s in synapses:
        a, b = s.get("pre"), s.get("post")
        if isinstance(a, int) and isinstance(b, int) and 0 <= a < n and 0 <= b < n and a != b:
            succ[a].append(b)
            indeg[b] += 1
            adj[a].add(b)
            adj[b].add(a)

    comp = [-1] * n
    ncomp = 0
    for start in range(n):
        if comp[start] != -1:
            continue
        comp[start] = ncomp
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = ncomp
                    stack.append(v)
        ncomp += 1

    members = [[] for _ in range(ncomp)]
    for i in range(n):
        members[comp[i]].append(i)

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

    comp_order = sorted(range(ncomp), key=lambda c: (-len(members[c]), c))
    cols = max(1, math.ceil(math.sqrt(ncomp)))
    pos = {}
    cell = 0.9
    for slot, c in enumerate(comp_order):
        gx, gy = slot % cols, slot // cols
        mem = members[c]
        dmin = min(depth[i] for i in mem)
        dmax = max(depth[i] for i in mem)
        dspan = max(1, dmax - dmin)
        by_depth = {}
        for i in mem:
            by_depth.setdefault(depth[i], []).append(i)
        for d, group in by_depth.items():
            group.sort()
            lx = (d - dmin) / dspan
            for j, i in enumerate(group):
                ly = (j + 0.5) / len(group)
                pos[i] = (gx + 0.05 + lx * cell, -(gy + 0.05 + ly * cell))
        _relax_component_positions(pos, mem)
    return pos, ncomp, cols


def network_positions_3d(neurons, synapses):
    """Deterministic 3D positions with true volumetric component spread."""
    pos2d, ncomp, cols = network_positions(neurons, synapses)
    del ncomp, cols
    n = len(neurons)
    succ = [[] for _ in range(n)]
    adj = [set() for _ in range(n)]
    for s in synapses:
        a, b = s.get("pre"), s.get("post")
        if isinstance(a, int) and isinstance(b, int) and 0 <= a < n and 0 <= b < n and a != b:
            succ[a].append(b)
            adj[a].add(b)
            adj[b].add(a)

    comp = [-1] * n
    ncomp = 0
    for start in range(n):
        if comp[start] != -1:
            continue
        comp[start] = ncomp
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = ncomp
                    stack.append(v)
        ncomp += 1

    members = [[] for _ in range(ncomp)]
    for i in range(n):
        members[comp[i]].append(i)

    comp_order = sorted(range(ncomp), key=lambda c: (-len(members[c]), c))
    comp_slot = {c: slot for slot, c in enumerate(comp_order)}
    cloud_radius = max(0.55, 1.15 * (max(1, ncomp) ** (1.0 / 3.0)))
    pos3d = {}
    for c, mem in enumerate(members):
        cx, cy, cz = _fibonacci_volume_point(comp_slot[c], max(1, ncomp))
        cx *= cloud_radius
        cy *= cloud_radius
        cz *= cloud_radius
        basis_u, basis_v, basis_w = _component_basis(comp_slot[c], max(1, ncomp))
        ordered = sorted(mem, key=lambda i: (round(pos2d[i][0], 6), round(pos2d[i][1], 6), i))
        xs = [pos2d[i][0] for i in ordered]
        ys = [pos2d[i][1] for i in ordered]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = max(1e-9, x_max - x_min)
        y_span = max(1e-9, y_max - y_min)
        local_scale = 0.16 + 0.08 * min(len(mem) ** (1.0 / 3.0), 4.0)
        use_volume = len(mem) > 10
        for rank, i in enumerate(ordered):
            if use_volume:
                ux, uy, uz = _fibonacci_volume_point(rank, len(ordered))
            else:
                ux, uy, uz = _fibonacci_sphere_point(rank, len(ordered))
            px, py = pos2d[i]
            depth_bias = ((px - x_min) / x_span) - 0.5
            lateral_bias = ((py - y_min) / y_span) - 0.5
            local = (
                basis_u * (0.95 * ux + 0.55 * depth_bias)
                + basis_v * (0.95 * uy + 0.25 * lateral_bias)
                + basis_w * (1.15 * uz - 0.30 * lateral_bias)
            )
            x3 = cx + local_scale * float(local[0])
            y3 = cy + local_scale * float(local[1])
            z3 = cz + local_scale * float(local[2])
            pos3d[i] = (x3, y3, z3)
        _relax_component_positions_3d(pos3d, mem, min_sep=max(0.14, local_scale * 0.65))
    return pos3d
