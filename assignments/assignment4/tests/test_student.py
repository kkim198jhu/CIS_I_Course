"""
Tests for YOUR code in src/student. Run from the assignment4 folder:

    pytest tests/test_student.py -m "not slow"     # quick unit tests
    pytest tests/test_student.py                   # + the full simulated runs (~20 s)

They check correctness and the hard requirements, not the quality of your
design -- passing them is necessary, not sufficient.
"""

import numpy as np
import pytest

from src import simulation as sim, Frame, Rot, vct3
from src.simulation import constants as C
from src.tools import (TrajectoryLog, pose_error_body, frame_Rp, make_frame, rot_exp, rot_log,
                       evaluate_approach, evaluate_cooperative, print_report)


@pytest.fixture
def robot():
    return sim.create_robot()


@pytest.fixture
def scene():
    return sim.build_scene()


def _random_q(rng, robot):
    lo, hi = robot.limits.q_min, robot.limits.q_max
    return rng.uniform(lo * 0.8 + hi * 0.2, lo * 0.2 + hi * 0.8)


# ------------------------------------------------------------------ jacobian.py

def _numerical_jacobian(pose_fn, q, h=1e-6):
    J = np.zeros((6, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = h
        J[:, k] = pose_error_body(pose_fn(q - e), pose_fn(q + e)) / (2 * h)
    return J


def test_jacobian_RW_matches_finite_differences(student, robot):
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = _random_q(rng, robot)
        J = student.jacobian.jacobian_RW(q)
        assert J.shape == (6, 6)
        assert np.allclose(J, _numerical_jacobian(robot.wrist_pose, q), atol=1e-5 * (1 + np.abs(J).max()))


def test_jacobian_RT_matches_finite_differences(student, robot):
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = _random_q(rng, robot)
        J = student.jacobian.jacobian_RT(q, robot.F_WT)
        assert np.allclose(J, _numerical_jacobian(robot.tool_pose, q), atol=1e-5 * (1 + np.abs(J).max()))


def test_jacobian_RT_with_rotated_tool(student, robot):
    F_WT = make_frame(rot_exp([0.3, -0.2, 0.5]), [10.0, -20.0, -150.0])
    q = _random_q(np.random.default_rng(2), robot)
    pose = lambda qq: robot.wrist_pose(qq) * F_WT
    assert np.allclose(student.jacobian.jacobian_RT(q, F_WT), _numerical_jacobian(pose, q), atol=1e-3)


# ------------------------------------------------------------------ wrench.py

def test_pure_force_at_wrist_resolved_at_tip(student):
    phi_W = np.array([1.0, -2.0, 3.0])
    zeta_T = student.wrench.resolve_wrench_to_tool(np.concatenate([np.zeros(3), phi_W]), C.F_WT)
    p = frame_Rp(C.F_WT)[1]
    assert np.allclose(zeta_T[3:], phi_W)                           # same force
    assert np.allclose(zeta_T[:3], -np.cross(p, phi_W))             # moment about the tip


def test_wrench_resolution_with_rotated_tool(student):
    R = rot_exp([0.4, 0.1, -0.7])
    p = np.array([5.0, 0.0, -100.0])
    F_WT = make_frame(R, p)
    zeta_W = np.array([10.0, -5.0, 2.0, 0.5, 1.5, -4.0])
    zeta_T = student.wrench.resolve_wrench_to_tool(zeta_W, F_WT)
    # equivalent wrench: same force; and the moment of (force at T) about W reproduces tau_W
    assert np.allclose(R @ zeta_T[3:], zeta_W[3:])
    assert np.allclose(R @ zeta_T[:3] + np.cross(p, zeta_W[3:]), zeta_W[:3])


def test_wrench_in_robot_frame(student):
    R = rot_exp([0.2, 0.3, -0.1])
    z = np.arange(6.0)
    out = student.wrench.wrench_in_robot_frame(z, R)
    assert np.allclose(out, np.concatenate([R @ z[:3], R @ z[3:]]))


# ------------------------------------------------------------------ gains.py

def test_free_space_gain(student):
    k = student.gains.kappa_free_space(C.PHI_MAX)
    assert k * C.PHI_MAX == pytest.approx(C.V_FREE, rel=0.1)


def test_speed_limits_respect_v_safe_and_never_stall(student, scene):
    g = student.gains
    for s in scene.structures:
        for d in np.linspace(-1.0, 3 * s.d_range, 200):
            v = g.speed_limit(s, d)
            assert v > 0, f"{s.name}: zero speed at d={d:.2f} would stall the tool"
            assert v <= C.V_FREE + 1e-9
            if np.isfinite(s.v_safe) and d <= s.d_safe:
                assert v <= s.v_safe + 1e-9, f"{s.name}: {v:.3f} mm/s > v_safe at d={d:.2f}"
            assert g.kappa_structure(s, d, C.PHI_MAX) == pytest.approx(v / C.PHI_MAX)


def test_combined_gain_is_most_restrictive(student, scene):
    g = student.gains
    rng = np.random.default_rng(3)
    for _ in range(20):
        d = rng.uniform(-0.5, 15, size=len(scene))
        k = g.kappa_combined(scene.structures, d, C.PHI_MAX)
        assert k <= min(g.kappa_structure(s, di, C.PHI_MAX) for s, di in zip(scene.structures, d)) + 1e-12


# ------------------------------------------------------------------ virtual_fixtures.py

def test_joint_motion_bounds_keep_the_robot_happy(student, robot):
    """Any dq inside the bounds, sent as dq / dt, must not trip SendCommand's limit flags."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        lb, ub = student.virtual_fixtures.joint_motion_bounds(robot.read_state().q, robot.read_state().qdot,
                                                              robot.limits, robot.dt)
        assert np.all(lb <= ub + 1e-12)
        dq = rng.uniform(lb, ub)
        rc = robot.send_command(dq / robot.dt)
        assert rc == 0, f"rc={rc} for a command inside your bounds"


def test_joint_motion_bounds_at_a_travel_limit(student, robot):
    q = robot.tool_inv_kins(C.F_RT_START)                         # q2 on its upper limit
    lb, ub = student.virtual_fixtures.joint_motion_bounds(q, np.zeros(6), robot.limits, robot.dt)
    assert ub[1] <= 1e-9 and lb[1] < 0


def test_structure_constraints_block_motion_into_a_sphere(student, scene, robot):
    s6 = scene.structures[5]                                      # sphere at (-10, 0, -40) tube coords
    p = scene.from_tube([0.0, 0.0, -40.0 + np.sqrt((15.0 + 1.0) ** 2 - 100.0)])   # 1 mm above it
    J_p = np.hstack([np.eye(3), np.zeros((3, 3))])                # pretend the first 3 joints move the tip
    G, h = student.virtual_fixtures.structure_constraints(scene, p, J_p)
    assert G.shape[1] == 6 and G.shape[0] == h.shape[0] >= 1
    assert np.all(h >= -1e-12), "standing still must be feasible when nothing is penetrated"
    toward = -s6.gradient_at(p) * 2.0                             # 2 mm straight into the sphere
    assert np.any(G @ np.concatenate([toward, np.zeros(3)]) > h), "moving into S6 must violate a constraint"
    away = s6.gradient_at(p) * 0.5
    assert np.all(G @ np.concatenate([away, np.zeros(3)]) <= h + 1e-9)


def test_structure_constraints_ignore_far_structures(student, scene):
    p = scene.from_tube([0.0, 0.0, 30.0])                         # 30 mm above the tube entrance
    G, h = student.virtual_fixtures.structure_constraints(scene, p, np.hstack([np.eye(3), np.zeros((3, 3))]))
    assert G.shape == (0, 6) and h.shape == (0,)


def test_centering_bias(student, scene):
    vf = student.virtual_fixtures
    assert np.allclose(vf.centering_bias(scene, scene.from_tube([0, 0, -50])), 0.0, atol=1e-12)
    f = vf.centering_bias(scene, scene.from_tube([3.0, 4.0, -50]))
    assert f @ np.array([3.0, 4.0, 0.0]) < 0 and abs(f[2]) < 1e-12       # points at the axis
    assert np.linalg.norm(f) < C.PHI_MAX


# ------------------------------------------------------------------ approach.py

def test_interpolate_pose_endpoints_and_midpoint(student):
    A, B = C.F_RT_INIT, C.F_RT_START
    ap = student.approach
    for s, F in ((0.0, A), (1.0, B)):
        assert np.allclose(pose_error_body(ap.interpolate_pose(A, B, s), F), 0.0, atol=1e-9)
    M = ap.interpolate_pose(A, B, 0.5)
    (R0, p0), (R1, p1), (Rm, pm) = frame_Rp(A), frame_Rp(B), frame_Rp(M)
    assert np.allclose(pm, (p0 + p1) / 2)
    a_tot, a_mid = rot_log(R1 @ R0.T), rot_log(Rm @ R0.T)
    assert np.allclose(a_mid, a_tot / 2, atol=1e-9)                 # half the angle, same axis


def test_motion_profile(student, robot):
    ap = student.approach
    plan = ap.plan_approach(robot, C.F_RT_INIT, C.F_RT_START, robot.read_state().q)
    assert plan.T > 0
    t = np.linspace(-1, plan.T + 1, 500)
    s = np.array([ap.motion_fraction(ti, plan) for ti in t])
    assert s[0] == 0.0 and s[-1] == 1.0 and np.all(np.diff(s) >= -1e-12)


def test_approach_step_tracks_a_small_target(student, robot):
    st = robot.read_state()
    F = robot.tool_pose(st.q)
    R, p = frame_Rp(F)
    target = make_frame(R, p + np.array([0.0, 0.0, 0.004]))          # 4 um: inside one period's acceleration
    dq = student.approach.approach_step(st, F, student.jacobian.jacobian_RT(st.q, robot.F_WT), target,
                                        robot.limits, robot.dt)
    assert np.allclose(pose_error_body(robot.tool_pose(st.q + dq), target), 0.0, atol=1e-6)


# ------------------------------------------------------------------ full runs

@pytest.fixture(scope="module")
def full_run(student):
    robot = sim.create_robot()
    scene = sim.build_scene()
    log = TrajectoryLog(scene.names)
    student.approach.move_tool_to(robot, C.F_RT_START, log)
    student.cooperative.cooperative_control_mode(robot, scene, log)
    return robot, scene, log


@pytest.mark.slow
def test_full_approach(full_run):
    robot, scene, log = full_run
    res = evaluate_approach(log, C.F_RT_INIT, C.F_RT_START, robot.limits, robot.dt)
    print_report(res)
    assert res["passed"], [k for k, v in res["checks"].items() if not v]


@pytest.mark.slow
def test_full_cooperative(full_run):
    robot, scene, log = full_run
    res = evaluate_cooperative(log, scene, robot.limits, robot.dt)
    print_report(res)
    assert res["passed"], [k for k, v in res["checks"].items() if not v]
