"""
YOUR CODE -- constraints and bias terms for the Step 3 optimization problem.

Every period the controller solves for the joint increment dq (6 unknowns):

    dq = argmin  sum_i w_i || A_i dq - b_i ||^2
         s.t.    G dq <= h          (e.g. virtual fixtures)
                 lb <= dq <= ub     (joint limits)

This module builds the G/h and lb/ub pieces and the centering bias. The
objective terms are put together in approach.py and cooperative.py.
"""

import numpy as np

# TODO: your design parameters (minimum clearances, gains, ...) go here


def joint_motion_bounds(q, qdot, limits, dt):
    """
    Box constraints on the joint increment for one period.

    input:  q, qdot -- current joint positions / velocities (from ReadState)
            limits  -- JointLimits(q_min, q_max, qd_max, qdd_max)
            dt      -- controller period (s)
    output: (lb, ub), each shape (6,), so that lb <= dq <= ub keeps
              q_min <= q + dq <= q_max,
              |dq / dt| <= qd_max,
              |dq / dt - qdot| <= qdd_max dt
    Think about: what if these windows don't overlap? (lb must be <= ub.)
    A little margin below the exact limits avoids round-off tripping the robot's rc flags.
    """
    raise NotImplementedError("TODO: joint_motion_bounds")


def structure_constraints(scene, p_R, J_p):
    """
    Linearized SDF virtual fixtures.

    input:  scene -- Scene (iterate scene.structures; each has distance_at(p_R),
                     gradient_at(p_R) in robot coordinates, d_range, type, ...)
            p_R   -- tool tip position, robot coordinates (3,)
            J_p   -- 3x6 matrix, tip displacement in robot coordinates per dq
    output: (G, h) with G of shape (m, 6) so that G dq <= h keeps the tip out of
            every structure within d_range. To first order
                D_S(p + dp) ~= D_S(p) + grad D_S . dp
            Decide how much clearance to keep from each structure type, and how
            fast the tip may use up the clearance it has.
            Return np.zeros((0, 6)), np.zeros(0) if no structure is in range.
    """
    raise NotImplementedError("TODO: structure_constraints")


def centering_bias(scene, p_R) -> np.ndarray:
    """
    input:  scene, p_R -- as above
    output: a bias "force" (N, robot coordinates, shape (3,)) that pulls the tip
            toward the tube's centerline (scene.to_tube(p_R) gives tube coordinates;
            the tube axis is the tube z axis). It is added to the hand force in
            the admittance law, so keep it well below phi_max or it will fight the user.
    """
    raise NotImplementedError("TODO: centering_bias")
