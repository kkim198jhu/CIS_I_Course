"""
Small-motion frame math in plain numpy (provided).

Convention (handout): an incremental motion eta = [alpha, epsilon] acts on the
right of a frame,
    F(q + dq) = F(q) dF,   dF ~= [I + sk(alpha), epsilon],
so alpha and epsilon are expressed in the coordinates of F itself ("body"
coordinates). To get the same motion in the coordinates F is expressed in
(e.g. the robot base), rotate both parts by F.R: alpha_R = R alpha, eps_R = R eps.
"""

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame


def skew(v) -> np.ndarray:
    """sk(v), so that sk(v) @ w == cross(v, w)."""
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def unskew(M) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def rot_exp(alpha) -> np.ndarray:
    """Rotation matrix for the rotation vector alpha (axis * angle), Rodrigues' formula."""
    a = np.asarray(alpha, dtype=np.float64).reshape(3)
    th = np.linalg.norm(a)
    K = skew(a)
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * (K @ K)


def rot_log(R) -> np.ndarray:
    """Rotation vector alpha (axis * angle, angle in [0, pi]) with rot_exp(alpha) == R."""
    R = np.asarray(R, dtype=np.float64)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-7:
        return unskew(R)   # first order: R ~= I + sk(alpha)
    if np.pi - th < 1e-5:
        # near pi: axis from the symmetric part, sign from the (small) skew part
        B = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / np.sqrt(max(B[k, k], 1e-15))
        axis /= np.linalg.norm(axis)
        s = unskew(R)
        if axis @ s < 0:
            axis = -axis
        return th * axis
    return th / np.sin(th) * unskew(R)


def frame_Rp(F: Frame):
    """(R 3x3, p (3,)) numpy views of a Frame."""
    return F.R.matrix, F.p.vec.ravel()


def make_frame(R, p) -> Frame:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return Frame(Rot(matrix=np.asarray(R, dtype=np.float64)), vct3(*p))


def pose_error_body(F_current: Frame, F_goal: Frame) -> np.ndarray:
    """
    eta = [alpha, epsilon] (body coordinates of F_current) with
    F_goal = F_current [rot_exp(alpha), epsilon]. Exact, not just first order.
    """
    R, p = frame_Rp(F_current)
    Rg, pg = frame_Rp(F_goal)
    return np.concatenate([rot_log(R.T @ Rg), R.T @ (pg - p)])


def rotation_angle(R) -> float:
    return float(np.linalg.norm(rot_log(R)))


def tilt_angle(R) -> float:
    """Angle between R z and z (rad): how far the tool axis is tilted."""
    zz = np.clip(np.asarray(R)[2, 2], -1.0, 1.0)
    return float(np.arccos(zz))
