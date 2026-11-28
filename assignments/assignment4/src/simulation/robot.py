"""
The simulated robot: ReadState() and SendCommand() from the handout.

The kinematics are assignment 2's nominal robot (assignments/assignment2/src/robot.py,
same F_RW(q) and inverse kinematics); this module adds the velocity-commanded
dynamics, the tool offset F_WT and the wrist force sensor. There are no hidden
kinematic errors in assignment 4: fwd_kins() is the truth.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from data_types.nominal_types import Frame
from assignments.assignment2.src.robot import Robot

from . import constants as C
from .force_sensor import HandForceModel

RC_TRAVEL = 1       # rc & 1: at least one joint at a travel limit
RC_VELOCITY = 2     # rc & 2: at least one joint at a velocity limit
RC_ACCEL = 4        # rc & 4: at least one joint at an acceleration limit


@dataclass(frozen=True)
class JointLimits:
    q_min: np.ndarray
    q_max: np.ndarray
    qd_max: np.ndarray      # |q_dot| <= qd_max
    qdd_max: np.ndarray     # |q_ddot| <= qdd_max


DEFAULT_LIMITS = JointLimits(C.Q_MIN.copy(), C.Q_MAX.copy(), C.QD_MAX.copy(), C.QDD_MAX.copy())


@dataclass(frozen=True)
class RobotState:
    """What ReadState() returns."""
    t: float                # simulated time, s
    q: np.ndarray           # joint positions (mm, mm, mm, rad, rad, rad)
    qdot: np.ndarray        # joint velocities
    zeta_W: np.ndarray      # wrist wrench [tau_W (N mm), phi_W (N)], wrist coordinates


class RobotSim(Robot):
    """
    Velocity-controlled robot. Each send_command(v_cmd) advances simulated time
    by dt after enforcing, in this order:
      1. |v| <= qd_max                       (rc |= 2)
      2. |v - q_dot| <= qdd_max dt            (rc |= 4)
      3. q_min <= q + v dt <= q_max           (rc |= 1; a hard stop, applied last)
    The handout's TA pseudocode applies travel limits first; doing them last
    means a joint can never be driven through a travel limit by the
    acceleration limit, which is what a physical hard stop does.
    """

    def __init__(self, q0, F_WT: Frame = C.F_WT, dt: float = C.DT, limits: JointLimits = DEFAULT_LIMITS,
                 force_model: Optional[HandForceModel] = None):
        super().__init__("A4 robot", Frame.eye(), limits.q_min, limits.q_max, 0.0, limits.qd_max, limits.qdd_max)
        self.F_WT = F_WT
        self.dt = float(dt)
        self.limits = limits
        self.force_model = force_model if force_model is not None else HandForceModel()
        self._q = np.asarray(q0, dtype=np.float64).reshape(6).copy()
        self.check_limits(self._q)
        self._qdot = np.zeros(6)
        self._t = 0.0
        self.theta_cmd = self._q.copy()     # keeps inherited inv_kins() seeding on the current pose
        self.last_rc = 0
        self.n_commands = 0
        self.rc_counts = {RC_TRAVEL: 0, RC_VELOCITY: 0, RC_ACCEL: 0}

    def __repr__(self):
        return f"RobotSim(t={self._t:.3f}s, q={np.round(self._q, 4).tolist()})"

    # ------------------------------------------------------------ handout API
    def read_state(self) -> RobotState:
        """[q, q_dot, zeta_W] <- ReadState()"""
        R_RW = self.fwd_kins(self._q).R.matrix
        return RobotState(t=self._t, q=self._q.copy(), qdot=self._qdot.copy(),
                          zeta_W=self.force_model.wrench_W(self._t, R_RW))

    def send_command(self, v_cmd) -> int:
        """rc <- SendCommand(v_cmd): v_cmd are joint velocities (mm/s, rad/s)."""
        v = np.asarray(v_cmd, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(v)):
            raise ValueError(f"send_command: non-finite velocity command {v.tolist()}")
        lim, dt, tol = self.limits, self.dt, 1e-9
        rc = 0

        v1 = np.clip(v, -lim.qd_max, lim.qd_max)
        if np.any(np.abs(v1 - v) > tol):
            rc |= RC_VELOCITY

        dv = lim.qdd_max * dt
        v2 = np.clip(v1, self._qdot - dv, self._qdot + dv)
        if np.any(np.abs(v2 - v1) > tol):
            rc |= RC_ACCEL

        q_new = self._q + v2 * dt
        q_clipped = np.clip(q_new, lim.q_min, lim.q_max)
        if np.any(np.abs(q_clipped - q_new) > tol):
            rc |= RC_TRAVEL
            v2 = (q_clipped - self._q) / dt

        self._q = q_clipped
        self._qdot = v2
        self._t += dt
        self.theta_cmd = self._q.copy()
        self.last_rc = rc
        self.n_commands += 1
        for bit in self.rc_counts:
            if rc & bit:
                self.rc_counts[bit] += 1
        return rc

    # ------------------------------------------------------------ helpers
    @property
    def t(self) -> float:
        return self._t

    def wrist_pose(self, q) -> Frame:
        """F_RW(q)"""
        return self.fwd_kins(q)

    def tool_pose(self, q) -> Frame:
        """F_RT(q) = F_RW(q) F_WT"""
        return self.fwd_kins(q) * self.F_WT

    def tool_inv_kins(self, F_RT: Frame, seed=None) -> np.ndarray:
        """q such that tool_pose(q) == F_RT (nearest wrist branch to `seed`)."""
        return self.inv_kins(F_RT * self.F_WT.inv(), seed=seed)

    def is_stopped(self, tol: float = 1e-9) -> bool:
        return bool(np.all(np.abs(self._qdot) <= tol))

    def halt(self, on_step: Optional[Callable[[], None]] = None, max_steps: int = 1000) -> int:
        """
        SendCommand(0) until the robot has stopped (the acceleration limit can
        make that take several periods). Calls on_step() after every command,
        e.g. to keep logging. Returns the number of commands sent.
        """
        n = 0
        while not self.is_stopped() and n < max_steps:
            self.send_command(np.zeros(6))
            n += 1
            if on_step is not None:
                on_step()
        return n


def create_robot(F_RT_init: Frame = C.F_RT_INIT, F_WT: Frame = C.F_WT, dt: float = C.DT,
                 limits: JointLimits = DEFAULT_LIMITS, force_model: Optional[HandForceModel] = None) -> RobotSim:
    """A stopped robot holding the tool at F_RT_init (wrist branch nearest q4 = q5 = q6 = 0)."""
    proto = RobotSim(np.clip(np.zeros(6), limits.q_min, limits.q_max), F_WT, dt, limits, force_model)
    q0 = proto.tool_inv_kins(F_RT_init, seed=np.zeros(6))
    return RobotSim(q0, F_WT, dt, limits, force_model)
