"""
CIS I -- Assignment 4: Robot Virtual Fixtures.

    src/simulation/   the robot, force sensor and anatomy (provided, don't edit)
    src/tools/        provided helpers: frame math, the constrained least-squares
                      solver, logging, evaluation and plotting
    src/student/      YOUR code
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_types.nominal_types import vct3, Rot, Frame  # noqa: E402

from . import simulation, tools  # noqa: E402

__all__ = ["vct3", "Rot", "Frame", "simulation", "tools"]
