"""
Pose-error metrics used to report calibration accuracy.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import math

from data_types.nominal_types import Frame


def frame_error(F_a: Frame, F_b: Frame) -> Tuple[float, float]:
    """
    #TODO (DONE)
    The mismatch between two frames is defined as
    dF = F_a^-1 F_b ~= Frame(I + sk(alpha), epsilon); report its size.

    input:
        F_a - reference Frame (e.g. ground truth)
        F_b - Frame to compare (e.g. your prediction)
    output:
        (angle_rad, dist_mm) - ||alpha|| (the rotation angle of dF.R) and
            ||epsilon|| (the norm of dF.p)
    """
    #Gets dF
    dF = F_a.inv() * F_b
    
    #Get the dist_mm sqrt(x^2 + y^2 + z^2)
    dist_mm = dF.p.norm()
    
    #Get the angle_rad
    _, angle_rad = dF.R.axis_angle()
    
    return angle_rad, dist_mm


def summarize_errors(errors: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """
    TODO (DONE): implement.
    Calculate the number of observations, RMS errors, and maximum errors for orientation and position.
    
    input:
        errors - list of (angle_rad, dist_mm) pairs from frame_error
    output:
        dict with at least the keys
            n, rms_angle, max_angle, rms_pos, max_pos
    """
    
    #Gets length and initialized variables
    n = len (errors)
    max_pos = -math.inf
    max_angle = -math.inf
    total_pos = 0
    total_angle = 0
    
    #Error Handling just in case
    if n == 0:
        raise ValueError("errors must contain at least one error pair")
    
    for i in range(n):
        #Gets the max of each
        if (max_angle < errors[i][0]):
            max_angle = errors[i][0]
        
        if (max_pos < errors[i][1]):
            max_pos = errors[i][1]
        
        # Gets the squaring of errors
        total_pos += errors[i][1]**2
        total_angle += errors[i][0]**2
    
    # Gets the rms values
    rms_angle = math.sqrt(total_angle/n)
    rms_pos = math.sqrt(total_pos/n)
    
    return { "n" : n,
             "rms_angle": rms_angle,
             "max_angle": max_angle,
             "rms_pos": rms_pos,
             "max_pos": max_pos}
