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
from data_types.nominal_types import Rot



def jacobian_RW(q) -> np.ndarray:
    """
    Compute the wrist body Jacobian for the current joint configuration.
    
    input:  q -- 6 joint values (mm, mm, mm, rad, rad, rad)
    output: 6x6 body Jacobian of F_RW(q), rows [alpha; epsilon]
            Rows 0 - 2 describe rotation in radians
            Rows 3 - 5 describe translation in millimeters
            Each column corresponds to one joint
    """
    #Gets it with 6 values
    q = np.asarray(q, dtype=float).reshape(6)
    
    #Gets the rotation matrices
    Rx = Rot('x', q[3]).matrix
    Ry = Rot('y', q[4]).matrix
    Rz = Rot('z', q[5]).matrix
    
    #Gets the rotation matrix
    R = Rx @ Ry @ Rz
    
    #Note that the the 5th's axis is affect by the 4th and the 6th is affect by the 5th and the 4th
    axes = [np.array([1.0, 0.0, 0.0]), Rx @ np.array([0.0, 1.0, 0.0]), Rx @ Ry @ np.array([0.0, 0.0, 1.0])]
    
    #From assignment 100 meters away from the rotated x direction = lever arm (look above in the file docstring)
    lever_arm = R @ np.array([100.0, 0.0, 0.0])
    
    #Intialize it
    J = np.zeros((6, 6))
    
    #Put the translation value on the 1st, 2nd, and 3rd joint
    J[3:, :3] = R.T
    
    #Put the rotation value on the 4th, 5th, and 6th value
    for j in range(3):
        axis = axes[j]
        #Gets current joint’s rotation axis, which it expresses it in robot coords
        J[:3, j + 3] = R.T @ axis
        
        #How does the rot. changes the position of the wrist origin
        linear_motion = np.cross(axis, lever_arm)
        J[3:, j + 3] = R.T @ linear_motion
    
    return J


def jacobian_RT(q, F_WT) -> np.ndarray:
    """
    Compute the tool-tip body Jacobian.
    
    input:  q -- joint values; F_WT -- tool tip relative to the wrist (Frame)
    output: 6x6 body Jacobian of F_RT(q) = F_RW(q) F_WT, rows [alpha_T; epsilon_T]
    
    Hint: dF_T = F_WT^-1 dF_W F_WT. Work out what that does to alpha and epsilon.
    """
    #Reuses J_RW from the function above
    J_RW = jacobian_RW(q)
    
    #Gets the tool's orientation in relation to the wrist
    R_WT, p_WT = frame_Rp(F_WT)
    
    #Intialize dat things
    J_RT = np.zeros((6, 6))
    
    #Process all 6 joints
    for k in range(6):
        angular_W = J_RW[:3, k]
        linear_W = J_RW[3:, k]
        
        #Calculates the change in position caused by by wrist rotation at the tool's tip
        offset_motion = np.cross(angular_W, p_WT)
        
        #Gets it in tool coordinates
        J_RT[:3, k] = R_WT.T @ angular_W
        J_RT[3:, k] = R_WT.T @ (linear_W + offset_motion)
    
    return J_RT
