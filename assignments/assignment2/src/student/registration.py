"""
Rigid registration helpers: point-set registration, frame averaging and
marker-body refinement.
"""

from typing import List

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody


def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    TODO (DONE): implement (your own -- do not call the simulator's solver).

    Rigid point-set registration: find the least-squares Frame F such that
    F * a[i] ~= b[i] for every corresponding pair. Think about what happens
    with fewer than 3 points, with collinear points, and with a reflection.

    input:
        a - points in the source frame (e.g. a marker body's local layout)
        b - the same points observed in the target frame, same order
    output: Frame mapping a -> b
    """
    #Error Handling
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Nonvalid inputs for either a or b")
    
    # b_i = R * a_i + p
    # Gets all the points for x, y, and z
    A = np.array([point.vec.ravel() for point in a])
    B = np.array([point.vec.ravel() for point in b])
    
    # Gets averages for all, so b_avg = R * a_avg + p 
    a_centroid = np.mean(A, axis=0)
    b_centroid = np.mean(B, axis=0)
    
    #Sets them to the origin, so don't worry about translations
    #b_i - b_avg = (R * a_i + p) - (R * a_avg + p )
    #b_i - b_avg = R (a_i - a_avg)
    a_i_origin = A - a_centroid
    b_i_origin = B - b_centroid
    
    #Error check to make sure colinear
    rank_a = np.linalg.matrix_rank(a_i_origin)
    rank_b = np.linalg.matrix_rank(b_i_origin)
    if (rank_a == 0 or rank_b == 0):
        raise ValueError("All points are the same point")
    
    if (rank_a == 1 or rank_b == 1):
        raise ValueError("All points lie on a line")
    
    #Gets a 3x3 matrix for singular value decomp.
    H = a_i_origin.T @ b_i_origin
    
    # H = U * sum(V)^t
    U, singular_values, Vt = np.linalg.svd(H)
    
    #From that we get the rotation prediction, as R_pred * U = V, where U is source direction and V is matching target directions
    R_prediction = Vt.T @ U.T
    
    # +1 is good proper rotation, while -1 is a reflection, so we need to flip it back
    R_prediction_det = np.linalg.det(R_prediction)
    if (R_prediction_det < 0):
        Vt[-1, :] *= -1
        R_prediction = Vt.T @ U.T
    
    # Gets the translation
    p = b_centroid - R_prediction @ a_centroid
    
    F = Frame (Rot(matrix=R_prediction), vct3(p))
    
    return F


def average_frames(frames: List[Frame]) -> Frame:
    """
    TODO: implement.

    Average of several noisy estimates of the same Frame. Rotations cannot
    simply be averaged element-wise -- the result must be a proper
    rotation matrix.

    input:
        frames - List[Frame], all estimates of the same transform
    output: the averaged Frame
    """
    n = len(frames)
    if (n == 0):
        raise ValueError("List is empty.")
    
    rot = []
    translation = []
    for frame in frames:
        rot.append(frame.R.matrix)
        translation.append(frame.p.vec.ravel())
    
    translation_avg = np.mean(translation, axis=0)
    rot_avg = np.mean(rot, axis=0)
    
    # This gets SVD to make it into a proper rotation
    U, singular_values, Vt = np.linalg.svd(rot_avg)
    
    # Now averaging for averaged rotation matrix itself, so R = U sum (Vt)
    R_prediction = U @ Vt
    # +1 is good proper rotation, while -1 is a reflection, so we need to flip it back
    R_prediction_det = np.linalg.det(R_prediction)
    if (R_prediction_det < 0):
        Vt[-1, :] *= -1
        R_prediction = U @ Vt
    
    return Frame(Rot(matrix=R_prediction), vct3(translation_avg))


def refine_marker_body(trk: Tracker, placed: PlacedBody, n_readings: int = 30) -> List[vct3]:
    """
    TODO: implement.

    "It is possible to improve your estimate of a_m,k through a calibration
    process involving multiple observations of the marker body using the
    tracking camera and then averaging them."

    Take n_readings SampleMarkerBodies readings, use
    most_recent_observed_markers(...) plus your register_points to bring
    each reading's b_m,k back into the body's own frame, average, and push
    the result into the body with update_marker_body(...). Note that each
    reading costs delta_t_samp of simulated time, and that the tracker
    jiggles between readings.

    input:
        trk - Tracker the body is tracked by
        placed - PlacedBody whose body's nominal marker positions should be refined
        n_readings - number of readings to average
    output: the new List[vct3] of marker positions (also stored on the body)
    """
    raise NotImplementedError("TODO: implement refine_marker_body")
