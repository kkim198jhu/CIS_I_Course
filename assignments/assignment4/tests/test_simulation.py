"""
Tests of the provided simulation and tools. These pass as shipped; read them as
examples of how the API behaves.
"""

import numpy as np
import pytest

from src import simulation as sim, Frame, Rot, vct3
from src.simulation import constants as C
from src.tools import (TrajectoryLog, solve_lsqlin, LsqProblem, rot_exp, rot_log, pose_error_body,
                       frame_Rp, make_frame)


# ------------------------------------------------------------------ robot

def test_create_robot_holds_tool_at_F_init():
    robot = sim.create_robot()
    st = robot.read_state()
    R, p = frame_Rp(robot.tool_pose(st.q))
    R0, p0 = frame_Rp(C.F_RT_INIT)
    assert np.allclose(p, p0, atol=1e-9) and np.allclose(R, R0, atol=1e-9)
    assert np.all(st.qdot == 0) and st.t == 0.0


def test_start_pose_is_reachable_and_q2_on_its_limit():
    robot = sim.create_robot()
    q = robot.tool_inv_kins(C.F_RT_START)
    assert np.allclose(q[:3], [150.0, 300.0, 450.0]) and np.allclose(q[3:], 0.0)
    assert q[1] == pytest.approx(C.Q_MAX[1])


def test_send_command_within_limits_sets_no_flags():
    robot = sim.create_robot()
    v = np.array([0.1, -0.1, 0.2, 0.001, 0.0, -0.001])      # well below one period's acceleration
    rc = robot.send_command(v)
    st = robot.read_state()
    assert rc == 0 and st.t == pytest.approx(C.DT)
    assert np.allclose(st.qdot, v)


def test_send_command_flags_velocity_and_acceleration_limits():
    robot = sim.create_robot()
    rc = robot.send_command(np.array([1000.0, 0, 0, 0, 0, 0]))
    assert rc & sim.RC_VELOCITY and rc & sim.RC_ACCEL
    # from rest, one period can only reach qdd_max * dt
    assert robot.read_state().qdot[0] == pytest.approx(C.QDD_MAX[0] * C.DT)


def test_send_command_travel_limit_is_a_hard_stop():
    robot = sim.create_robot()
    q_target = robot.tool_inv_kins(C.F_RT_START)
    robot = sim.RobotSim(q_target)                           # q2 sits on its +300 mm limit
    rc = robot.send_command(np.array([0, 0.4, 0, 0, 0, 0]))
    assert rc & sim.RC_TRAVEL
    assert robot.read_state().q[1] == pytest.approx(C.Q_MAX[1])


def test_send_command_rejects_nan():
    robot = sim.create_robot()
    with pytest.raises(ValueError):
        robot.send_command(np.array([np.nan, 0, 0, 0, 0, 0]))


def test_halt_brings_robot_to_rest():
    robot = sim.create_robot()
    for _ in range(50):
        robot.send_command(np.array([20.0, 0, 0, 0, 0, 0]))
    assert not robot.is_stopped()
    robot.halt()
    assert robot.is_stopped()


# ------------------------------------------------------------------ force sensor

def test_hand_force_is_bounded_and_mostly_downward():
    hand = sim.HandForceModel()
    for t in np.linspace(0, 60, 121):
        phi = hand.force_R(t)
        assert np.linalg.norm(phi) <= C.PHI_MAX + 1e-9
        assert phi[2] <= -C.PHI_MAX * np.cos(C.WOBBLE_TILT_MAX) + 1e-9


def test_wrist_wrench_is_hand_force_in_wrist_coordinates():
    robot = sim.create_robot()
    st = robot.read_state()
    R_RW = robot.wrist_pose(st.q).R.matrix
    assert np.allclose(st.zeta_W[:3], 0.0)                              # torques first, zero at the wrist
    assert np.allclose(R_RW @ st.zeta_W[3:], robot.force_model.force_R(st.t))


# ------------------------------------------------------------------ anatomy

def test_tube_distances():
    tube = sim.Tube("A", np.inf, 0.0, 10.0, r_tube=10.0, length=100.0)
    assert tube.distance([0, 0, -50]) == pytest.approx(10.0)
    assert tube.distance([4, 0, -50]) == pytest.approx(6.0)
    assert tube.distance([0, 12, -50]) == pytest.approx(-2.0)           # through the wall
    assert tube.distance([0, 6, 3]) == pytest.approx(5.0)               # above the entrance: distance to the rim
    assert np.allclose(tube.gradient([4, 0, -50]), [-1, 0, 0])          # toward the centerline


def test_donut_is_a_torus():
    d = sim.Donut("B", 2.0, 3.0, 10.0, c=[0, 0, 0], r_donut=15.0, rho_donut=2.0)
    assert d.distance([0, 0, 0]) == pytest.approx(13.0)
    assert d.distance([15, 0, 0]) == pytest.approx(-2.0)
    assert d.distance([15, 0, 5]) == pytest.approx(3.0)                 # above the core circle
    assert d.distance([10, 0, 30]) == pytest.approx(np.hypot(5, 30) - 2)


def test_sphere_and_halfplane():
    s = sim.Sphere("C", 1.0, 3.0, 10.0, c=[0, 0, 0], rho=15.0)
    assert s.distance([20, 0, 0]) == pytest.approx(5.0)
    h = sim.HalfPlane("D", 1.0, 3.0, 10.0, c=[0, 0, -100], n=[0, 0, 1])
    assert h.distance([0, 0, 2.5]) == pytest.approx(2.5)                # local coords are relative to c


@pytest.mark.parametrize("k", range(8))
def test_gradients_match_finite_differences(k):
    scene = sim.build_scene()
    s = scene.structures[k]
    rng = np.random.default_rng(k)
    for _ in range(20):
        x = rng.uniform([-9, -9, -99], [9, 9, -1]) - s.c          # inside the tube, structure coordinates
        if s.kind in ("Tube", "Donut") and np.hypot(x[0], x[1]) < 0.5:
            continue
        g = s.gradient(x)
        assert np.linalg.norm(g) == pytest.approx(1.0)
        assert np.allclose(g, s.numerical_gradient(x), atol=1e-5)


def test_scene_layout_and_frames():
    scene = sim.build_scene()
    assert len(scene) == 8 and scene.stopper_index == 7
    assert [s.type for s in scene] == list("ABBBBCCD")
    p = np.array([751.0, 302.0, 250.0])
    assert np.allclose(scene.from_tube(scene.to_tube(p)), p)
    d = scene.distances(frame_Rp(C.F_RT_START)[1])
    assert d[0] == pytest.approx(10.0) and d[7] == pytest.approx(100.0)
    assert np.all(d > 0)


# ------------------------------------------------------------------ tools

def test_rot_log_exp_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.normal(size=3)
        a *= rng.uniform(0, 3.1) / np.linalg.norm(a)
        assert np.allclose(rot_log(rot_exp(a)), a, atol=1e-8)


def test_pose_error_body_is_exact():
    F = make_frame(rot_exp([0.2, -0.1, 0.4]), [1, 2, 3])
    G = make_frame(rot_exp([-0.3, 0.5, 0.1]), [4, -2, 7])
    eta = pose_error_body(F, G)
    R, p = frame_Rp(F)
    assert np.allclose(R @ rot_exp(eta[:3]), frame_Rp(G)[0]) and np.allclose(p + R @ eta[3:], frame_Rp(G)[1])


def test_solver_unconstrained_and_bounded():
    rng = np.random.default_rng(1)
    A, b = rng.normal(size=(9, 6)), rng.normal(size=9)
    assert np.allclose(solve_lsqlin(A, b).x, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-6)
    res = solve_lsqlin(np.eye(3), [5, -5, 0.5], lb=-np.ones(3), ub=np.ones(3))
    assert res.success and np.allclose(res.x, [1, -1, 0.5], atol=1e-8)


def test_solver_inequality_is_a_projection():
    # closest point to b with x_0 >= 0.5
    res = solve_lsqlin(np.eye(3), [-1, 2, 3], G=[[-1, 0, 0]], h=[-0.5])
    assert res.success and np.allclose(res.x, [0.5, 2, 3], atol=1e-8)


def test_solver_reports_infeasible():
    assert not solve_lsqlin(np.eye(2), [0, 0], G=[[1, 0], [-1, 0]], h=[-1, -1]).success


def test_lsq_problem_weights():
    prob = LsqProblem(1)
    prob.add_objective([[1.0]], [0.0], weight=1.0)
    prob.add_objective([[1.0]], [1.0], weight=3.0)
    assert prob.solve().x[0] == pytest.approx(0.75, abs=1e-8)


def test_log_records_and_writes_csv(tmp_path):
    log = TrajectoryLog(["S1", "S2"])
    log.record(0.0, np.zeros(6), np.zeros(6), C.F_RT_START, distances=[1.0, 2.0], phase="cooperative")
    log.record(0.02, np.ones(6), np.zeros(6), C.F_RT_START, distances=[0.5, 1.5], phase="cooperative")
    assert log.p.shape == (2, 3) and log.d.shape == (2, 2) and len(log.phase("cooperative")) == 2
    log.to_csv(tmp_path / "log.csv")
    assert (tmp_path / "log.csv").read_text().count("\n") == 3
