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
    Resolve a wrist wrench at the tool tip
    Note: torque_about_wrist = torque_about_tip + offset x force
    
    input:  zeta_W -- [tau_W, phi_W] measured at and expressed in W
            F_WT   -- pose of the tool frame T relative to W
    output: zeta_T -- [tau_T, phi_T], the equivalent wrench at the origin of T, in T coordinates
    """
    
    zeta_W = np.asarray(zeta_W, dtype=float).reshape(6)
    
    #Get torque and force
    tau_W = zeta_W[:3]
    phi_W = zeta_W[3:]
    
    #Get the rotation and translation
    R_WT, p_WT = frame_Rp(F_WT)
    
    #Force in tool coordinates
    phi_T = R_WT.T @ phi_W
    
    #Set it to the tool tip instead of the wrist
    torque_from_offset = np.cross(p_WT, phi_W)
    tau_T = R_WT.T @ (tau_W - torque_from_offset)
    
    return np.concatenate((tau_T, phi_T))


def wrench_in_robot_frame(zeta_T, R_RT) -> np.ndarray:
    """
    input:  zeta_T -- a wrench expressed in tool coordinates; R_RT -- tool orientation (3x3)
    output: zeta_RT -- the same wrench expressed in robot coordinates
    (The handout writes R_RW here; with F_WT = [I, p] that is the same matrix.)
    """
    raise NotImplementedError("TODO: wrench_in_robot_frame")
