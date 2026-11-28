"""
YOUR CODE -- the autonomous approach move F_init -> F_start (handout "Approach move").

Requirements:
* the tool tip p_RT(t) moves in a straight line from p_init to p_start,
* R_RT(t) rotates about one fixed axis from R_init to R_start,
* both are coordinated: the same fraction s(t) of each is complete at any time,
* the robot starts and ends at rest and never exceeds the joint velocity /
  acceleration limits (if it does, SendCommand clips your command and the
  tip leaves the straight line).

Structure: plan_approach() picks a motion profile s(t), interpolate_pose()
turns s into an intermediate goal, approach_step() solves one period's
optimization problem. move_tool_to() at the bottom (provided) is the
handout's MoveToolTo loop that calls them.
"""

from dataclasses import dataclass

import numpy as np

from data_types.nominal_types import Frame
from src.tools.frames import frame_Rp, make_frame, rot_exp, rot_log, pose_error_body
from src.tools.solver import LsqProblem

from .jacobian import jacobian_RT
from .virtual_fixtures import joint_motion_bounds


@dataclass
class ApproachPlan:
    """Whatever your profile needs. T (total duration, s) is required by the loop below."""
    F_start: Frame
    F_goal: Frame
    T: float
    # TODO: add the parameters of your motion profile (peak speed, acceleration, ...)


def interpolate_pose(F_start: Frame, F_goal: Frame, s: float) -> Frame:
    """
    input:  F_start, F_goal -- tool poses; s in [0, 1] -- fraction of the move completed
    output: the intermediate pose: position a fraction s along the straight line,
            orientation a fraction s of the way around the single fixed rotation axis
            taking R_start to R_goal.  (tools.rot_log / rot_exp help.)
    """
    raise NotImplementedError("TODO: interpolate_pose")


def plan_approach(robot, F_start: Frame, F_goal: Frame, q_start) -> ApproachPlan:
    """
    input:  robot   -- RobotSim (robot.limits, robot.tool_inv_kins(F, seed) are useful)
            F_start -- current tool pose; F_goal -- target tool pose; q_start -- current joints
    output: an ApproachPlan whose profile s(t) respects the joint limits.

    Hint: along the path the joints follow q(s), so
        qdot = q'(s) sdot,     qddot = q'(s) sddot + q''(s) sdot^2.
    Sampling q(s) with inverse kinematics gives you q' and q''.
    """
    raise NotImplementedError("TODO: plan_approach")


def motion_fraction(t: float, plan: ApproachPlan) -> float:
    """
    input:  t -- time since the start of the move (s); plan -- from plan_approach
    output: s(t), from 0 (t <= 0) rising smoothly to exactly 1 (t >= plan.T)
    """
    raise NotImplementedError("TODO: motion_fraction")


def approach_step(state, F_RT: Frame, J_RT, F_target: Frame, limits, dt) -> np.ndarray:
    """
    One period of Step 3 for the approach.

    input:  state    -- RobotState (q, qdot, ...)
            F_RT     -- current tool pose;  J_RT -- its body Jacobian (6x6)
            F_target -- where the tool should be at the end of this period
            limits, dt
    output: dq (6,), the joint increment to command (the loop sends dq / dt)

    Typical form:  min || eta_desired - J_RT dq ||^2  s.t. joint motion bounds,
    with eta_desired from tools.pose_error_body(F_RT, F_target). Think about how
    to weight rotation (radians) against translation (mm).
    """
    raise NotImplementedError("TODO: approach_step")


# =====================================================================
# PROVIDED -- the MoveToolTo loop from the handout. You should not need to
# change it; it calls the functions above.
# =====================================================================

def close_enough(F_a: Frame, F_b: Frame, pos_tol: float, rot_tol: float) -> bool:
    eta = pose_error_body(F_a, F_b)
    return bool(np.linalg.norm(eta[3:]) <= pos_tol and np.linalg.norm(eta[:3]) <= rot_tol)


def move_tool_to(robot, F_goal: Frame, log=None, pos_tol: float = 0.05, rot_tol: float = 1e-4,
                 max_time: float = 120.0) -> ApproachPlan:
    """MoveToolTo(F_goal): uniform, coordinated motion of F_RT from where it is now to F_goal."""
    dt, limits = robot.dt, robot.limits
    state = robot.read_state()
    t0 = state.t
    F_start = robot.tool_pose(state.q)
    plan = plan_approach(robot, F_start, F_goal, state.q)

    def record(state, F_RT):
        if log is not None:
            log.record(state.t, state.q, state.qdot, F_RT, phase="approach", rc=robot.last_rc)

    while True:
        state = robot.read_state()
        F_RT = robot.tool_pose(state.q)
        J_RT = jacobian_RT(state.q, robot.F_WT)
        record(state, F_RT)
        if motion_fraction(state.t - t0, plan) >= 1.0 and close_enough(F_RT, F_goal, pos_tol, rot_tol):
            robot.halt(on_step=lambda: record(robot.read_state(), robot.tool_pose(robot.read_state().q)))
            return plan
        if state.t - t0 > max_time:
            raise RuntimeError(f"move_tool_to: goal not reached after {max_time} s")
        F_intermediate = interpolate_pose(F_start, F_goal, motion_fraction(state.t - t0 + dt, plan))
        dq = approach_step(state, F_RT, J_RT, F_intermediate, limits, dt)
        robot.send_command(dq / dt)
