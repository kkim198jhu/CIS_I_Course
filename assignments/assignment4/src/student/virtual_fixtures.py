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
LIMIT_MARGIN = 0.999999
MIN_CLEARANCE = 0.1
CENTERING_GAIN = 0.1
MAX_CENTERING_FORCE = 0.5
CLEARANCE_FRACTION = 0.2


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
    
    #Gets current joint positions
    q = np.asarray(q, dtype=float).reshape(6)
    #Gets current joint velocities
    qdot = np.asarray(qdot, dtype=float).reshape(6)
    
    #The time must be positive and finite at least...
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time (dt) must be finite and positive")
    
    #Gets/intializes the outputs
    lb = np.zeros(6)
    ub = np.zeros(6)
    
    #Goes through each joints
    for k in range(6):
        # Small margin for rounding errors.
        v_max = limits.qd_max[k] * LIMIT_MARGIN
        a_max = limits.qdd_max[k] * LIMIT_MARGIN
        
        #Gets how much velocity changes during one step from the acceleration and time
        dv = a_max * dt
        
        #Then calculates how far the joints can travel with 0 being the min as that makes sense
        distance_lower = max(q[k] - limits.q_min[k], 0.0)
        distance_upper = max(limits.q_max[k] - q[k], 0.0)
        
        #Get the breaking magnitudes, which tells us how fast the joint may move toward either limit (while still preserving the breaking rooms)
        #Equation used: v * dt + v² / (2*a) <= d    ->    v <= sqrt((a*dt)² + 2*a*d) - a*dt
        brake_lower = np.sqrt(dv**2 + 2 * a_max * distance_lower) - dv
        brake_upper = np.sqrt(dv**2 + 2 * a_max * distance_upper) - dv
        
        #Choose the strictest lower bound between 3 options:
        # 1. limits.q_min[k] - q[k]: Finish the step above the lower position limit
        # 2. -min(v_max, brake_lower) * dt: Respect the speed and braking limits when moving toward the lower limit
        # 3. (qdot[k] - dv) * dt: Avoid decreasing velocity faster than the acceleration limit allows
        lb[k] = max(limits.q_min[k] - q[k], -min(v_max, brake_lower) * dt, (qdot[k] - dv) * dt)
        
        #Same as above, but with the upper bound
        ub[k] = min(limits.q_max[k] - q[k], min(v_max, brake_upper) * dt, (qdot[k] + dv) * dt)
        
        # Make sure that the lower bound isn't above the upper bound
        if lb[k] > ub[k]:
            raise ValueError("Lower bound is higher than upper bound")
    
    return lb, ub


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
