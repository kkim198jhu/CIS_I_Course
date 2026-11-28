"""
Plots for assignment 4 (provided). Everything anatomical is drawn in tube
coordinates (x_tube = F_tube^-1 p_R), where the tube runs from z = 0 down to
z = -L_tube.

    plot_scene(scene, log)                 3-D view of the anatomy and the tip path
    plot_slices(scene, log, z_values)      free space at several depths, with the tip trace
    plot_distances(log, scene)             D(S_k) vs time with d_safe / d_range
    plot_speed(log, scene)                 tip speed vs the active v_safe limit
    plot_joint_usage(log, limits, dt)      |q_dot| / q_dot_max and |q_ddot| / q_ddot_max
    plot_approach(log, F_init, F_goal)     coordination of the approach move
    plot_speed_limits(speed_limit, scene)  your v_S(d) curves, one per structure
"""

import numpy as np
import matplotlib.pyplot as plt

from .frames import frame_Rp, rot_log

_PHASE_COLORS = {"approach": "tab:blue", "cooperative": "tab:red"}
_TYPE_COLORS = {"A": "0.6", "B": "tab:orange", "C": "tab:purple", "D": "tab:green"}


def _tube_coords(scene, P):
    return np.array([scene.to_tube(p) for p in P]) if len(P) else np.zeros((0, 3))


def plot_scene(scene, log=None, ax=None, show_approach: bool = False, elev: float = 20, azim: float = -60):
    if ax is None:
        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot(projection="3d")
    th = np.linspace(0, 2 * np.pi, 48)
    for s in scene.structures:
        col = _TYPE_COLORS.get(s.type, "k")
        if s.kind == "Tube":
            z = np.linspace(-s.length, 0, 12)
            T, Z = np.meshgrid(th, z)
            ax.plot_wireframe(s.r_tube * np.cos(T), s.r_tube * np.sin(T), Z, color=col, lw=0.4, alpha=0.5)
        elif s.kind == "Donut":
            u, v = np.meshgrid(th, np.linspace(0, 2 * np.pi, 16))
            X = (s.r_donut + s.rho_donut * np.cos(v)) * np.cos(u) + s.c[0]
            Y = (s.r_donut + s.rho_donut * np.cos(v)) * np.sin(u) + s.c[1]
            Z = s.rho_donut * np.sin(v) + s.c[2]
            ax.plot_surface(X, Y, Z, color=col, alpha=0.35, linewidth=0)
        elif s.kind == "Sphere":
            u, v = np.meshgrid(th, np.linspace(0, np.pi, 24))
            ax.plot_surface(s.rho * np.cos(u) * np.sin(v) + s.c[0], s.rho * np.sin(u) * np.sin(v) + s.c[1],
                            s.rho * np.cos(v) + s.c[2], color=col, alpha=0.25, linewidth=0)
        elif s.kind == "HalfPlane":
            r, t = np.meshgrid(np.linspace(0, scene.r_tube, 2), th)
            ax.plot_surface(r * np.cos(t) + s.c[0], r * np.sin(t) + s.c[1], 0 * r + s.c[2], color=col, alpha=0.5)
    if log is not None and len(log):
        for phase in sorted(set(log.phase_labels)):
            if phase == "approach" and not show_approach:
                continue
            X = _tube_coords(scene, log.phase(phase).p)
            ax.plot(X[:, 0], X[:, 1], X[:, 2], color=_PHASE_COLORS.get(phase, "k"), lw=1.5, label=phase)
        ax.legend(loc="upper left")
    lim = 25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-scene.L_tube - 5, 15)
    ax.set_box_aspect((1, 1, (scene.L_tube + 20) / (2 * lim)))
    ax.set_xlabel("x_tube (mm)"); ax.set_ylabel("y_tube (mm)"); ax.set_zlabel("z_tube (mm)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("Anatomy (tube coordinates)")
    return ax


def plot_slices(scene, log=None, z_values=(-10, -15, -40, -70, -80, -85), half_width: float = 12.0, n: int = 121):
    """
    Each panel is a horizontal slice of the tube at depth z: shaded where the tip
    may NOT go (some D_k < 0), with D = 0 contours per structure and the part of
    the tip trace within +-2 mm of that depth.
    """
    z_values = list(z_values)
    cols = min(3, len(z_values))
    rows = int(np.ceil(len(z_values) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    g = np.linspace(-half_width, half_width, n)
    GX, GY = np.meshgrid(g, g)
    X = _tube_coords(scene, log.phase("cooperative").p) if (log is not None and len(log)) else None
    for ax, z in zip(axes.ravel(), z_values):
        D = np.empty((len(scene.structures), n, n))
        for k, s in enumerate(scene.structures):
            for i in range(n):
                for j in range(n):
                    D[k, i, j] = s.distance_at(scene.from_tube([GX[i, j], GY[i, j], z]))
        blocked = (D.min(axis=0) < 0).astype(float)
        ax.contourf(GX, GY, blocked, levels=[0.5, 1.5], colors=["0.85"])
        for k, s in enumerate(scene.structures):
            if s.type == "D" or D[k].min() > s.d_range:
                continue
            ax.contour(GX, GY, D[k], levels=[0.0], colors=[_TYPE_COLORS.get(s.type, "k")], linewidths=1.2)
            ax.contour(GX, GY, D[k], levels=[s.d_safe] if s.d_safe > 0 else [], colors=[_TYPE_COLORS.get(s.type, "k")],
                       linewidths=0.6, linestyles="dashed")
        if X is not None:
            sel = np.abs(X[:, 2] - z) <= 2.0
            ax.plot(X[sel, 0], X[sel, 1], ".", ms=2, color="tab:red")
        ax.set_aspect("equal")
        ax.set_title(f"z_tube = {z:g} mm")
        ax.set_xlabel("x_tube"); ax.set_ylabel("y_tube")
    for ax in axes.ravel()[len(z_values):]:
        ax.axis("off")
    fig.suptitle("Free space (white) by depth; dashed = d_safe; red = tip within +-2 mm")
    fig.tight_layout()
    return fig


def plot_distances(log, scene, phase: str = "cooperative"):
    L = log.phase(phase)
    t, D = L.t, L.d
    N = len(scene.structures)
    cols = 4
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), sharex=True, squeeze=False)
    for k, (ax, s) in enumerate(zip(axes.ravel(), scene.structures)):
        ax.plot(t, D[:, k], color=_TYPE_COLORS.get(s.type, "k"))
        ax.axhline(0, color="k", lw=0.8)
        if s.d_safe > 0:
            ax.axhline(s.d_safe, color="tab:red", ls="--", lw=0.8, label="d_safe")
        ax.axhline(s.d_range, color="0.5", ls=":", lw=0.8, label="d_range")
        ax.set_ylim(min(-0.5, float(np.min(D[:, k])) - 0.5), max(s.d_range * 1.5, 1))
        ax.set_title(s.name, fontsize=9)
    for ax in axes.ravel()[N:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Distance from the tool tip to each structure (mm)")
    fig.tight_layout()
    return fig


def plot_speed(log, scene, phase: str = "cooperative"):
    L = log.phase(phase)
    t, P, D = L.t, L.p, L.d
    speed = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.diff(t)
    d_int = np.minimum(D[:-1], D[1:])
    limit = np.full(len(speed), np.inf)
    for k, s in enumerate(scene.structures):
        if np.isfinite(s.v_safe) and s.d_safe > 0:
            limit = np.where(d_int[:, k] <= s.d_safe, np.minimum(limit, s.v_safe), limit)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t[1:], speed, lw=1, label="tip speed")
    lim_plot = np.where(np.isfinite(limit), limit, np.nan)
    ax.plot(t[1:], lim_plot, color="tab:red", lw=1.5, label="active v_safe")
    viol = speed > np.where(np.isfinite(limit), limit, np.inf) * 1.02
    if np.any(viol):
        ax.plot(t[1:][viol], speed[viol], "rx", label="violation")
    ax.set_xlabel("t (s)"); ax.set_ylabel("mm/s"); ax.legend()
    ax.set_title("Tool-tip speed")
    fig.tight_layout()
    return fig


def plot_joint_usage(log, limits, dt, phase=None):
    L = log.phase(phase) if phase else log
    t, qd = L.t, L.qdot
    qdd = np.diff(qd, axis=0) / dt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for k in range(6):
        a1.plot(t, np.abs(qd[:, k]) / limits.qd_max[k], label=f"q{k + 1}")
        a2.plot(t[1:], np.abs(qdd[:, k]) / limits.qdd_max[k])
    for a in (a1, a2):
        a.axhline(1.0, color="k", ls="--", lw=0.8)
    a1.set_ylabel("|qdot| / max"); a2.set_ylabel("|qddot| / max"); a2.set_xlabel("t (s)")
    a1.legend(ncol=6, fontsize=8)
    fig.suptitle("Joint velocity / acceleration usage")
    fig.tight_layout()
    return fig


def plot_approach(log, F_init, F_goal):
    L = log.phase("approach")
    R0, p0 = frame_Rp(F_init)
    R1, p1 = frame_Rp(F_goal)
    span = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / span
    rel = L.p - p0
    s_lin = rel @ u / span
    dev = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
    th_tot = np.linalg.norm(rot_log(R1 @ R0.T))
    s_rot = np.array([np.linalg.norm(rot_log(R @ R0.T)) for R in L.R]) / th_tot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    a1.plot(L.t, s_lin, label="linear fraction")
    a1.plot(L.t, s_rot, "--", label="rotation fraction")
    a1.set_xlabel("t (s)"); a1.set_ylabel("fraction complete"); a1.legend()
    a2.plot(L.t, dev * 1e3)
    a2.set_xlabel("t (s)"); a2.set_ylabel("distance from straight line (um)")
    fig.suptitle("Approach move")
    fig.tight_layout()
    return fig


def plot_speed_limits(speed_limit, scene, d_min: float = -1.0, d_max: float = 15.0):
    """Plots speed_limit(structure, d) (your gain design, as a speed) for every structure."""
    d = np.linspace(d_min, d_max, 400)
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in scene.structures:
        ax.plot(d, [speed_limit(s, di) for di in d], label=s.name)
    ax.set_xlabel("distance D_S (mm)"); ax.set_ylabel("max tip speed (mm/s)")
    ax.set_title("Speed limit (= kappa_S * phi_max) vs distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
