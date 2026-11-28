"""
YOUR CODE -- robot Jacobians.

Handout convention:
    F(q + dq) = F(q) dF,   dF ~= [I + sk(alpha), epsilon],   eta = [alpha; epsilon] = J(q) dq

so J is the *body* Jacobian: alpha and epsilon are expressed in the coordinates
of the moving frame itself (F_RW for J_RW, F_RT for J_RT). Rows 0-2 are alpha,
rows 3-5 are epsilon; column k is joint k+1.

Robot (handout "Robot Kinematics"):
    F_RW(q) = F_c(q1, q2, q3) F_cw F_w(q4, q5, q6) F_we
    F_c = translation q1 x + q2 y + q3 z,   F_cw = [I, (500, 0, 0)]
    F_w = Rx(q4) Ry(q5) Rz(q6),             F_we = [I, (100, 0, 0)]

Hints:
* For a prismatic joint the tip moves along d_k (robot coordinates) and does not rotate.
* For a revolute joint with axis omega_k (robot coordinates) through the wrist
  center, the origin of F_RW moves by omega_k x (lever arm). Remember the
  intermediate rotations when expressing omega_5 and omega_6 in robot coordinates.
* Convert the robot-coordinate motions to body coordinates with R_RW^T.
* Check your answer numerically: tools.pose_error_body(F(q), F(q + h e_k)) / h
  should match column k for small h (tests/test_student.py does exactly this).

You may ask an AI system to help write these, provided you cite the prompt you
used and check the result (handout).
"""

import numpy as np

from src.tools.frames import skew, frame_Rp


def jacobian_RW(q) -> np.ndarray:
    """
    input:  q -- 6 joint values (mm, mm, mm, rad, rad, rad)
    output: 6x6 body Jacobian of F_RW(q), rows [alpha; epsilon]
    """
    raise NotImplementedError("TODO: jacobian_RW")


def jacobian_RT(q, F_WT) -> np.ndarray:
    """
    input:  q -- joint values; F_WT -- tool tip relative to the wrist (Frame)
    output: 6x6 body Jacobian of F_RT(q) = F_RW(q) F_WT, rows [alpha_T; epsilon_T]

    Hint: dF_T = F_WT^-1 dF_W F_WT. Work out what that does to alpha and epsilon.
    """
    raise NotImplementedError("TODO: jacobian_RT")
