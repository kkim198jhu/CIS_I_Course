"""
Provided helpers for assignment 4 (don't edit):

    frames         skew / rot_exp / rot_log / pose_error_body ... (small-motion frame math)
    solver         solve_lsqlin, LsqProblem (constrained least squares for Step 3)
    logger         TrajectoryLog (the required time trace)
    evaluation     evaluate_approach / evaluate_cooperative / print_report (what grading checks)
    visualization  plots (import it explicitly: `from src.tools import visualization as viz`)
"""

from . import frames, solver, logger, evaluation
from .frames import skew, unskew, rot_exp, rot_log, frame_Rp, make_frame, pose_error_body, tilt_angle
from .solver import solve_lsqlin, LsqProblem, LsqResult
from .logger import TrajectoryLog
from .evaluation import evaluate_approach, evaluate_cooperative, print_report

__all__ = [
    "frames", "solver", "logger", "evaluation",
    "skew", "unskew", "rot_exp", "rot_log", "frame_Rp", "make_frame", "pose_error_body", "tilt_angle",
    "solve_lsqlin", "LsqProblem", "LsqResult", "TrajectoryLog",
    "evaluate_approach", "evaluate_cooperative", "print_report",
]
