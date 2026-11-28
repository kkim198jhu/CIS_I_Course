"""
Every number from the assignment 4 handout lives here, with a note wherever
the handout was ambiguous and a choice had to be made. Units are mm, radians,
seconds and newtons throughout.

Students: read this file, but don't edit it -- the grader uses its own copy.
"""

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

# ----------------------------------------------------------------- timing

DT = 0.020  # controller period Delta t, s

# ----------------------------------------------------------------- robot (handout "Robot Kinematics and joint constraints")
# Same cartesian-base + spherical-wrist robot as assignment 2:
#   F_RW(q) = F_c(q1, q2, q3) F_cw F_w(q4, q5, q6) F_we,  F_w = Rx(q4) Ry(q5) Rz(q6)

Q_MIN = np.array([0.0, -300.0, 0.0, -0.75 * np.pi, -0.75 * np.pi, -np.pi])
Q_MAX = np.array([600.0, 300.0, 600.0, 0.75 * np.pi, 0.75 * np.pi, np.pi])
QD_MAX = np.array([100.0, 100.0, 100.0, np.pi / 6, np.pi / 6, np.pi / 6])      # mm/s, rad/s
QDD_MAX = np.array([25.0, 25.0, 25.0, np.pi / 24, np.pi / 24, np.pi / 24])     # mm/s^2, rad/s^2
# NOTE: at F_start the robot needs q2 = 300 mm, i.e. joint 2 sits exactly on its
# upper travel limit. The tool tip therefore cannot move toward +y_robot while it
# is inside the tube; controllers must respect this as a joint-limit constraint.

# ----------------------------------------------------------------- tool and poses

F_WT = Frame(Rot(), vct3(0.0, 0.0, -150.0))   # tool tip relative to the wrist

def rot_zyx(rx: float, ry: float, rz: float) -> Rot:
    """The handout's Rot(rx, ry, rz) in the ZYX convention: R = Rz(rz) Ry(ry) Rx(rx)."""
    return Rot('z', rz) * Rot('y', ry) * Rot('x', rx)

F_RT_INIT = Frame(rot_zyx(-np.pi / 4, np.pi / 4, np.pi / 6), vct3(800.0, 200.0, 400.0))
F_TUBE = Frame(Rot(), vct3(750.0, 300.0, 300.0))
F_RT_START = F_TUBE

# ----------------------------------------------------------------- force sensor and "surgeon's hand"

PHI_MAX = 5.0               # ||phi_RW|| limit, N
TAU_MAX = PHI_MAX / 600.0   # ||tau_RW|| limit, N mm (value as printed in the handout; the
                            # simulated hand applies no torque at the wrist, so it never binds)
V_FREE = 25.0               # kappa_A(free space) * phi_max ~= 25 mm/s

# The hand pushes "down" (-z of the robot) with a slowly wobbling lateral component:
#   phi_R(t) = -phi_max Rz(omega_z t) Rx(tilt(t)) z,   tilt(t) = WOBBLE_TILT_MAX sin(omega_x t)
# The handout writes Rx(omega_x t); taken literally the force would point straight up
# at t = 1/0.05 = 20 s, which contradicts "constant downward force with a small lateral
# component", so the tilt is bounded by WOBBLE_TILT_MAX instead.
OMEGA_X = 0.05 * np.pi
OMEGA_Z = 0.10 * np.pi
WOBBLE_TILT_MAX = 0.35      # rad (~20 deg)

# ----------------------------------------------------------------- anatomy (handout "For the TA" object list)

R_TUBE = 10.0
L_TUBE = 100.0
D_RANGE = 10.0              # same d_range for every structure
STOP_DISTANCE = 0.2         # halt within this distance of the type-D stopper

# (factory, type, v_safe [mm/s], d_safe [mm], d_range [mm], geometry...) -- positions are relative to F_tube
STRUCTURE_SPECS = [
    ("Tube",      dict(type="A", v_safe=np.inf, d_safe=0.0, d_range=D_RANGE, r_tube=R_TUBE, length=L_TUBE)),
    ("Donut",     dict(type="B", v_safe=2.0, d_safe=3.0, d_range=D_RANGE, c=[3.0, 4.0, -10.0], r_donut=15.0, rho_donut=2.0)),
    ("Donut",     dict(type="B", v_safe=2.0, d_safe=3.0, d_range=D_RANGE, c=[-3.0, -4.0, -15.0], r_donut=15.0, rho_donut=2.0)),
    ("Donut",     dict(type="B", v_safe=3.0, d_safe=5.0, d_range=D_RANGE, c=[3.0, 4.0, -L_TUBE + 20.0], r_donut=15.0, rho_donut=2.0)),
    ("Donut",     dict(type="B", v_safe=2.0, d_safe=3.0, d_range=D_RANGE, c=[0.0, 0.0, -L_TUBE + 15.0], r_donut=7.0, rho_donut=2.0)),
    ("Sphere",    dict(type="C", v_safe=1.0, d_safe=3.0, d_range=D_RANGE, c=[-10.0, 0.0, -40.0], rho=15.0)),
    ("Sphere",    dict(type="C", v_safe=1.0, d_safe=3.0, d_range=D_RANGE, c=[6.0, 8.0, -70.0], rho=15.0)),
    ("HalfPlane", dict(type="D", v_safe=1.0, d_safe=3.0, d_range=D_RANGE, c=[0.0, 0.0, -L_TUBE], n=[0.0, 0.0, 1.0])),
]
