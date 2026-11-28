"""
YOUR CODE -- resolving the measured wrist wrench at the tool tip.

Wrench ordering (handout): zeta = [tau (3, N mm); phi (3, N)] -- torques first.

The force sensor reports zeta_W at the origin of the wrist frame W, in W
coordinates. The handout's controller first resolves it at the tool tip:
the force is the same vector (just re-expressed in T), but the moment about
the tip differs from the moment about the wrist by (lever arm) x force.

NOTE: the 6x6 matrix printed in the handout for this step has a typo; derive
the transform yourself from the definition of a moment.
"""

import numpy as np

from src.tools.frames import skew, frame_Rp


def resolve_wrench_to_tool(zeta_W, F_WT) -> np.ndarray:
    """
    input:  zeta_W -- [tau_W, phi_W] measured at and expressed in W
            F_WT   -- pose of the tool frame T relative to W
    output: zeta_T -- [tau_T, phi_T], the equivalent wrench at the origin of T, in T coordinates
    """
    raise NotImplementedError("TODO: resolve_wrench_to_tool")


def wrench_in_robot_frame(zeta_T, R_RT) -> np.ndarray:
    """
    input:  zeta_T -- a wrench expressed in tool coordinates; R_RT -- tool orientation (3x3)
    output: zeta_RT -- the same wrench expressed in robot coordinates
    (The handout writes R_RW here; with F_WT = [I, p] that is the same matrix.)
    """
    raise NotImplementedError("TODO: wrench_in_robot_frame")
