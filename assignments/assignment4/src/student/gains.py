"""
YOUR CODE -- admittance gain functions kappa_S (handout "Cooperative control").

Free-space admittance law:  dp_RT = kappa_A(x) phi_T  (per unit time), with
kappa_A(free space) * phi_max ~= 25 mm/s.

You design one gain curve per structure S as a function of the distance
d = D_S(x_S). It is usually easiest to think in speeds: since ||phi|| <= phi_max,
    kappa_S(d) = v_S(d) / phi_max
bounds the tip speed by v_S(d). Write v_S(d) in speed_limit(); the other
functions follow from it.

Your curves must
* enforce speed <= v_safe whenever d <= d_safe (the graders measure the actual
  tip speed from your log, so leave yourself some margin),
* be followable by the robot: slowing down from speed v to v_safe takes
  distance (v^2 - v_safe^2) / (2 a), and a is limited by the 25 mm/s^2 joint
  acceleration limit -- this may force a top speed inside the tube below 25 mm/s,
* not make the tool stall (if the gain reaches 0 the tool can never move again).

Figure 2 in the handout shows piecewise-linear examples; you may use curves.
Explain and justify your choices in the notebook (this is graded).

Structure fields you may use: s.type ('A'/'B'/'C'/'D'), s.v_safe (inf for the
tube), s.d_safe, s.d_range.
"""

import numpy as np

from src.simulation.constants import PHI_MAX, V_FREE

# TODO: your design parameters go here (and get explained in the notebook)


def kappa_free_space(phi_max: float = PHI_MAX) -> float:
    """kappa_A in free space, mm / (s N), so that kappa_A phi_max ~= V_FREE."""
    raise NotImplementedError("TODO: kappa_free_space")


def speed_limit(structure, d: float) -> float:
    """
    input:  structure -- a src.simulation.anatomy.Structure; d -- distance D_S of the tip from it (mm)
    output: v_S(d), the maximum tool-tip speed (mm/s) that structure allows at that distance
    """
    raise NotImplementedError("TODO: speed_limit")


def kappa_structure(structure, d: float, phi_max: float = PHI_MAX) -> float:
    """kappa_S(d) = v_S(d) / phi_max"""
    raise NotImplementedError("TODO: kappa_structure")


def kappa_combined(structures, distances, phi_max: float = PHI_MAX) -> float:
    """
    input:  structures -- list of Structure; distances -- matching list of D_S values
    output: the single gain the admittance law uses at this tip position
            (hint: which structure should win?)
    """
    raise NotImplementedError("TODO: kappa_combined")
