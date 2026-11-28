"""
Checks a TrajectoryLog against the assignment's requirements (provided). The
graders run the same checks, so use these to see where you stand.

    res_a = evaluate_approach(log, F_init, F_start, robot.limits, robot.dt)
    res_c = evaluate_cooperative(log, scene, robot.limits, robot.dt)
    print_report(res_a); print_report(res_c)
"""

import numpy as np

from data_types.nominal_types import Frame
from .frames import frame_Rp, rot_log, tilt_angle

SHAFT_LENGTH = 150.0      # tip -> wrist, mm (|p_WT|)

# pass/fail thresholds
APPROACH_PATH_TOL = 0.5         # mm, max distance of the tip from the straight line
APPROACH_AXIS_TOL = np.deg2rad(1.0)
APPROACH_SYNC_TOL = 0.01        # max |s_linear - s_rotation|
APPROACH_POS_TOL = 0.1          # mm, final position error
APPROACH_ROT_TOL = 1e-3         # rad, final rotation error
SPEED_TOL = 0.02                # relative slack on v_safe (finite-difference speeds)


def _limit_ratios(log, limits, dt):
    qdot = log.qdot
    vel = float(np.max(np.abs(qdot) / limits.qd_max)) if len(qdot) else 0.0
    if len(qdot) > 1:
        qddot = np.diff(qdot, axis=0) / dt
        acc = float(np.max(np.abs(qddot) / limits.qdd_max))
    else:
        acc = 0.0
    return vel, acc


def evaluate_approach(log, F_init: Frame, F_goal: Frame, limits, dt) -> dict:
    L = log.phase("approach") if "approach" in log.phase_labels else log
    if len(L) == 0:
        return dict(name="approach", passed=False, error="no approach samples in the log")
    R0, p0 = frame_Rp(F_init)
    R1, p1 = frame_Rp(F_goal)
    P, Rs = L.p, L.R
    span = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / span
    rel = P - p0
    along = rel @ u
    path_dev = float(np.max(np.linalg.norm(rel - np.outer(along, u), axis=1)))
    s_lin = along / span

    a_tot = rot_log(R1 @ R0.T)
    th_tot = np.linalg.norm(a_tot)
    a_k = np.array([rot_log(R @ R0.T) for R in Rs])
    th_k = np.linalg.norm(a_k, axis=1)
    s_rot = th_k / th_tot if th_tot > 1e-9 else s_lin
    mask = th_k > 1e-3
    if th_tot > 1e-9 and np.any(mask):
        cosang = np.clip((a_k[mask] @ a_tot) / (th_k[mask] * th_tot), -1, 1)
        axis_dev = float(np.max(np.arccos(cosang)))
    else:
        axis_dev = 0.0
    sync = float(np.max(np.abs(s_lin - s_rot)))

    pos_err = float(np.linalg.norm(P[-1] - p1))
    rot_err = float(np.linalg.norm(rot_log(Rs[-1].T @ R1)))
    vel, acc = _limit_ratios(L, limits, dt)
    stopped = bool(np.all(np.abs(L.qdot[-1]) < 1e-9))
    checks = {
        "tip path straight (max dev <= %.2f mm)" % APPROACH_PATH_TOL: path_dev <= APPROACH_PATH_TOL,
        "fixed rotation axis (max dev <= %.1f deg)" % np.rad2deg(APPROACH_AXIS_TOL): axis_dev <= APPROACH_AXIS_TOL,
        "coordinated (|s_lin - s_rot| <= %.2f)" % APPROACH_SYNC_TOL: sync <= APPROACH_SYNC_TOL,
        "reached F_start (pos <= %.2f mm, rot <= %.0e rad)" % (APPROACH_POS_TOL, APPROACH_ROT_TOL):
            pos_err <= APPROACH_POS_TOL and rot_err <= APPROACH_ROT_TOL,
        "joint velocity limits respected": vel <= 1 + 1e-6,
        "joint acceleration limits respected": acc <= 1 + 1e-6,
        "stopped at the end": stopped,
    }
    return dict(name="approach", duration=float(L.t[-1] - L.t[0]), path_dev_mm=path_dev,
                axis_dev_deg=float(np.rad2deg(axis_dev)), sync_err=sync, final_pos_err_mm=pos_err,
                final_rot_err_rad=rot_err, max_vel_ratio=vel, max_acc_ratio=acc,
                rc_nonzero=int(np.count_nonzero(L.rc)), checks=checks, passed=all(checks.values()))


def evaluate_cooperative(log, scene, limits, dt, stop_distance: float = 0.2) -> dict:
    L = log.phase("cooperative") if "cooperative" in log.phase_labels else log
    if len(L) < 2:
        return dict(name="cooperative", passed=False, error="fewer than 2 cooperative samples in the log")
    t, P, Rs, D = L.t, L.p, L.R, L.d
    if np.isnan(D).any():
        D = np.array([scene.distances(p) for p in P])
    steps = np.diff(t)
    speed = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.where(steps > 0, steps, np.inf)
    d_int = np.minimum(D[:-1], D[1:])                       # distance over each interval

    per = []
    for k, s in enumerate(scene.structures):
        row = dict(name=s.name, min_d=float(np.min(D[:, k])), penetrated=bool(np.min(D[:, k]) < 0.0),
                   v_safe=s.v_safe, d_safe=s.d_safe, max_speed_in_dsafe=None, speed_ok=True)
        if np.isfinite(s.v_safe) and s.d_safe > 0:
            inside = d_int[:, k] <= s.d_safe
            if np.any(inside):
                vmax = float(np.max(speed[inside]))
                row["max_speed_in_dsafe"] = vmax
                row["speed_ok"] = vmax <= s.v_safe * (1 + SPEED_TOL)
        per.append(row)

    k_stop = scene.stopper_index
    d_final = float(D[-1, k_stop])
    X = np.array([scene.to_tube(p) for p in P])
    lateral = np.hypot(X[:, 0], X[:, 1])
    tilts = np.array([tilt_angle(R) for R in Rs])

    # shaft containment: points from the tip up toward the wrist that lie inside the tube's length
    shaft_r = 0.0
    for p, R in zip(P, Rs):
        pts = p + np.outer(np.linspace(0, SHAFT_LENGTH, 31), R[:, 2])
        xs = np.array([scene.to_tube(x) for x in pts])
        sel = (xs[:, 2] <= 0) & (xs[:, 2] >= -scene.L_tube)
        if np.any(sel):
            shaft_r = max(shaft_r, float(np.max(np.hypot(xs[sel, 0], xs[sel, 1]))))

    vel, acc = _limit_ratios(L, limits, dt)
    stopped = bool(np.all(np.abs(L.qdot[-1]) < 1e-9))
    checks = {
        "no structure penetrated (all D >= 0)": not any(r["penetrated"] for r in per),
        "speed <= v_safe within d_safe of every structure": all(r["speed_ok"] for r in per),
        "halted within %.1f mm of the stopper" % stop_distance: 0.0 <= d_final <= stop_distance,
        "robot stopped at the end": stopped,
        "tool shaft stays inside the tube": shaft_r <= scene.r_tube,
        "joint velocity limits respected": vel <= 1 + 1e-6,
        "joint acceleration limits respected": acc <= 1 + 1e-6,
    }
    return dict(name="cooperative", duration=float(t[-1] - t[0]), final_stopper_distance=d_final,
                structures=per, max_speed=float(np.max(speed)), mean_centerline_dev=float(np.mean(lateral)),
                max_centerline_dev=float(np.max(lateral)), max_tilt_deg=float(np.rad2deg(np.max(tilts))),
                max_shaft_radius=shaft_r, max_vel_ratio=vel, max_acc_ratio=acc,
                rc_nonzero=int(np.count_nonzero(L.rc)), checks=checks, passed=all(checks.values()))


def print_report(res: dict):
    print(f"=== {res.get('name', '?')} : {'PASS' if res.get('passed') else 'FAIL'}")
    if "error" in res:
        print("  error:", res["error"])
        return
    for k, v in res.items():
        if k in ("name", "checks", "structures", "passed"):
            continue
        print(f"  {k:24s} {v:.4g}" if isinstance(v, float) else f"  {k:24s} {v}")
    if "structures" in res:
        print(f"  {'structure':18s} {'min D':>8s} {'d_safe':>7s} {'v_safe':>7s} {'max v in d_safe':>16s}")
        for r in res["structures"]:
            vs = "-" if r["max_speed_in_dsafe"] is None else f"{r['max_speed_in_dsafe']:.3f}"
            vsafe = "-" if not np.isfinite(r["v_safe"]) else f"{r['v_safe']:.1f}"
            flag = "" if (r["speed_ok"] and not r["penetrated"]) else "  <-- VIOLATION"
            print(f"  {r['name']:18s} {r['min_d']:8.3f} {r['d_safe']:7.1f} {vsafe:>7s} {vs:>16s}{flag}")
    for k, ok in res["checks"].items():
        print(f"  [{'x' if ok else ' '}] {k}")
