"""
The simulated surgeon's hand and the wrist force/torque sensor.

The handout's wrench ordering is used everywhere: zeta = [tau (3), phi (3)],
torques first, forces second. The hand pushes on the tool near the wrist, so
the sensed torque at the wrist is zero; the force is generated in robot
coordinates and reported in wrist coordinates (zeta_W).
"""

from typing import Callable, Optional

import numpy as np

from . import constants as C


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class HandForceModel:
    """
    phi_R(t) = -phi_max Rz(omega_z t) Rx(tilt(t)) z,  tilt(t) = tilt_max sin(omega_x t)

    Pass `force_fn(t) -> phi_R` to replace the default profile (e.g. to test a
    controller against a constant sideways push).
    """

    def __init__(self, phi_max: float = C.PHI_MAX, omega_x: float = C.OMEGA_X, omega_z: float = C.OMEGA_Z,
                 tilt_max: float = C.WOBBLE_TILT_MAX, force_fn: Optional[Callable[[float], np.ndarray]] = None):
        self.phi_max = float(phi_max)
        self.omega_x = float(omega_x)
        self.omega_z = float(omega_z)
        self.tilt_max = float(tilt_max)
        self.force_fn = force_fn

    def force_R(self, t: float) -> np.ndarray:
        """Hand force at time t, robot coordinates (N)."""
        if self.force_fn is not None:
            return np.asarray(self.force_fn(t), dtype=np.float64).reshape(3)
        tilt = self.tilt_max * np.sin(self.omega_x * t)
        return -self.phi_max * (_rz(self.omega_z * t) @ _rx(tilt) @ np.array([0.0, 0.0, 1.0]))

    def wrench_W(self, t: float, R_RW: np.ndarray) -> np.ndarray:
        """zeta_W = [tau_W, phi_W] as the wrist sensor reports it, magnitude-limited."""
        phi_W = np.asarray(R_RW).T @ self.force_R(t)
        tau_W = np.zeros(3)
        return np.concatenate([_clip_norm(tau_W, C.TAU_MAX), _clip_norm(phi_W, C.PHI_MAX)])


def _clip_norm(v: np.ndarray, vmax: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * (vmax / n) if n > vmax else v
