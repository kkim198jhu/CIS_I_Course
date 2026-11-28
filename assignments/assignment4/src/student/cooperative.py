"""
YOUR CODE -- admittance control through the tube with SDF
virtual fixtures (assignment's "Cooperative control").

Goal: the surgeon's hand pushes the tool (the simulated force is mostly
downward with a slowly wobbling sideways part). The tip should move through
the tube quickly, stay as close as possible to the centerline, keep the tool
axis as close to vertical as possible, never enter a structure, obey v_safe
near each structure, and halt within 0.2 mm of the type-D stopper.

Per the assignment's "Cooperative control", the compliance law ignores tau and only uses the force, so
you formulate the problem in terms of the tip displacement dp_RT. The tool
may tilt slightly if the optimizer finds that helps.

Structure: desired_tip_motion() is the admittance law (gains.py + bias),
cooperative_step() solves one period's optimization problem with the
virtual fixtures. cooperative_control_mode() at the bottom (provided) is the
handout's CooperativeControlMode loop that calls them.
"""

import numpy as np

from data_types.nominal_types import Frame
from src.simulation.constants import PHI_MAX, STOP_DISTANCE
from src.tools.frames import frame_Rp, rot_log
from src.tools.solver import LsqProblem

from .gains import kappa_combined
from .jacobian import jacobian_RT
from .wrench import resolve_wrench_to_tool, wrench_in_robot_frame
from .virtual_fixtures import joint_motion_bounds, structure_constraints, centering_bias


def desired_tip_motion(phi_R, p_R, scene, dt, phi_max: float = PHI_MAX) -> np.ndarray:
    """
    The admittance law, robot coordinates.

    input:  phi_R -- hand force resolved at the tip, robot coordinates (N)
            p_R   -- tip position, robot coordinates;  scene -- the anatomy
            dt    -- controller period
    output: dp_desired (3,) -- the tip displacement you would like this period,
            before the virtual fixtures are applied: kappa(x) times (force + your
            centering bias), times dt. Make sure the bias can't push the speed
            over the limit your gains impose.
    """
    raise NotImplementedError("TODO: desired_tip_motion")


def cooperative_step(state, F_RT: Frame, J_RT, scene, dp_desired, limits, dt) -> np.ndarray:
    """
    One period of Step 3 for cooperative control.

    input:  state -- RobotState;  F_RT, J_RT -- tool pose and body Jacobian
            scene -- anatomy;  dp_desired -- from desired_tip_motion (robot coordinates)
            limits, dt
    output: dq (6,)

    Things to put in the problem:
    * tip motion should match dp_desired -- note J_RT gives epsilon in TOOL
      coordinates; rotate by R_RT to get robot coordinates,
    * the tool axis should stay vertical (the angle between R_RT z and z small),
    * structure_constraints() and joint_motion_bounds(),
    * a fallback if the solver reports failure (what is a safe command then?).
    """
    raise NotImplementedError("TODO: cooperative_step")


# =====================================================================
# PROVIDED -- the CooperativeControlMode loop from the handout. You should
# not need to change it; it calls the functions above.
# =====================================================================

def cooperative_control_mode(robot, scene, log=None, stop_distance: float = STOP_DISTANCE,
                             max_time: float = 300.0) -> float:
    """
    CooperativeControlMode(F_WT): compliant motion of the tool tip under the
    hand force, subject to the SDF virtual fixtures, until the tip is within
    stop_distance of the type-D stopper. Returns the final stopper distance.
    """
    dt, limits = robot.dt, robot.limits
    k_stop = scene.stopper_index
    t0 = robot.read_state().t

    def record(state):
        F_RT = robot.tool_pose(state.q)
        d = scene.distances(F_RT.p.vec.ravel())
        if log is not None:
            log.record(state.t, state.q, state.qdot, F_RT, distances=d, phase="cooperative", rc=robot.last_rc)
        return F_RT, d

    while True:
        state = robot.read_state()
        F_RT, d = record(state)
        R_RT, p_RT = frame_Rp(F_RT)
        if d[k_stop] <= stop_distance:
            robot.halt(on_step=lambda: record(robot.read_state()))
            return float(scene.distances(robot.tool_pose(robot.read_state().q).p.vec.ravel())[k_stop])
        if state.t - t0 > max_time:
            raise RuntimeError(f"cooperative_control_mode: stopper not reached after {max_time} s "
                               f"(distance {d[k_stop]:.3f} mm)")
        J_RT = jacobian_RT(state.q, robot.F_WT)
        zeta_T = resolve_wrench_to_tool(state.zeta_W, robot.F_WT)
        zeta_RT = wrench_in_robot_frame(zeta_T, R_RT)
        dp_desired = desired_tip_motion(zeta_RT[3:], p_RT, scene, dt)
        dq = cooperative_step(state, F_RT, J_RT, scene, dp_desired, limits, dt)
        robot.send_command(dq / dt)
