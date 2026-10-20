"""
Data-collection workflow: which robot poses to visit, how to read them, and
how to register the robot base to the tracker.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from data_types.nominal_types import vct3, Frame

from ..robot import Robot
from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody
from .registration import average_frames

from ..simulation.clock import clock
from ..simulation.tracker import sample_marker_bodies
from ..simulation import constants

@dataclass
class CalibrationData:
    """One calibration data set.

    thetas[j] is the commanded (encoder) joint vector and frames[j] the
    calibration body's observed pose relative to the tracker, F_ee,j (this
    handout's tracker has no jiggle, so the tracker frame is a fixed
    reference). t_collect is the simulated time the collection took.
    """
    thetas: List[np.ndarray] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    t_collect: float = 0.0

    def __len__(self):
        return len(self.thetas)


def measure_home_offsets(rob: Robot, n_homings: int = 5) -> np.ndarray:
    """
    #TODO (DONE): implement.

    HomeRobot() drives the robot to theta* = 0 and returns the encoder
    values there, i.e. -delta_theta_sys(home) - delta_theta_rnd. Repeat it
    and average to estimate the systematic joint offsets at home (after the
    first homing the move is tiny, so each repeat costs about t_settle).

    input:
        rob - the robot (rob.home_robot())
        n_homings - how many homings to average
    output: estimated delta_theta_sys at home, np.ndarray of 6
    """
    
    if (n_homings < 1):
        raise ValueError("Not enough homings")
    
    readings = []
    for i in range(n_homings):
        readings.append(rob.home_robot())
    
    # Encoder readings = negative of the systematic offsets plus noise
    return -np.mean(readings, axis = 0)


# this should contain a desired accuracy which will affect the number of calibration poses
def plan_calibration_poses(rob: Robot, trk: Tracker, mbee_local_positions: List[vct3], desired_accuracy: float, **kwargs) -> List[np.ndarray]:
    """
    #TODO: implement -- this is a design task.

    Choose the joint vectors the robot will visit during calibration to reach the desired accuracy. Every
    parameter of your kinematic model must be excited by the data (axis
    tilts need travel along each axis; the wrist axes and F_we need wrist
    rotations; the joint distortion needs several stops along each joint),
    every pose must keep all calibration-body markers inside the tracker's
    viewing volume (Tracker.is_visible, using the nominal rob.F_rob and
    rob.fwd_kins -- they are only tens of mm off), and the order of the
    poses determines the motion time. The wrist moves at 0.1 rad/s: a pi/2
    turn costs ~16 s.

    The handout's method 1: decide how closely spaced the samples along an
    axis must be from the error bounds (delta_sys^trk, omega_sys^trk,
    delta_sys^rob, omega_sys^rob), then sweep each axis in turn from the
    home position. Joints 1-3 are decoupled from 4-6.

    input:
        rob - the Robot (nominal kinematics + limits)
        trk - the Tracker (for visibility checks)
        mbee_local_positions - the calibration body's marker layout a*_k
        desired_accuracy - target RMS orientation error in rad (the limiting
            metric); use it to size the plan -- how many distinct poses and
            how often each is read (a pose listed twice in a row is
            re-commanded and re-read, costing only t_settle + delta_t_samp)
        kwargs - whatever your design needs (grid sizes, margins, ...)
    output: ordered List of joint vectors (np.ndarray of 6)
    """
    
    if (desired_accuracy <= 0):
        raise ValueError("Desired_accuracy must be positive")
    
    poses = []
    home = np.zeros(6)
    ratio = constants.TARGET_ANG_ERR_RAD / desired_accuracy
    n_samples = 30

    
    
    for k in range(6):
        #Gets a list of random values between the min and the max
        sample_values = np.linspace(rob.theta_min[k], rob.theta_max[k], n_samples)
        accepted = 0
        
        for value in sample_values:
            #Creates a candidate joint configuration
            theta = home.copy()
            theta[k] = value
            
            #Gets the frame shift to the world frame
            F_world = rob.F_rob * rob.fwd_kins(theta)
            
            #Gets the world positions from the marker body attached to the robot’s end effector
            world_positions = []
            for marker in mbee_local_positions:
                world_positions.append(F_world * marker)
                
            if rob.within_limits(theta) and trk.is_visible(world_positions):
                poses.append(theta.copy())
                accepted += 1
                
        print(f"Joint {k + 1}: {accepted}/{n_samples} poses accepted")
    #If there are no visible calibrations
    if not poses:
        raise ValueError("No visible calibration poses found")

    return poses


def collect_calibration_data(rob: Robot, trk: Tracker, mbee_placed: PlacedBody,
                             thetas: List[np.ndarray], repeats: int = 1) -> CalibrationData:
    """
    # TODO (DONE): implement.

    Visit each joint vector with rob.move_joints, read the calibration body
    with sample_marker_bodies and store its frame relative to the tracker.
    Decide what to do when the body is not visible (None), and whether/how
    to use repeats (re-sampling costs 0.03 s; re-commanding the same pose
    costs t_settle and re-rolls the joint repeatability noise).

    input:
        rob, trk - robot and tracker
        mbee_placed - the robot calibration body (attached)
        thetas - joint vectors to visit, in order
        repeats - readings per pose
    output: CalibrationData
    """
    
    #Error handling
    if (repeats < 1):
        raise ValueError("Repeats is 0 or below")
    
    # Starts the clock and initialize the data
    data = CalibrationData()
    t_start = clock.elapsed
    
    #Goes through all the theatas
    for i in range (len(thetas)):
        
        # Moves the joints
        rob.move_joints(thetas[i])
        
        #Track the observations through each repeat
        for j in range(repeats):
            # Gets the observations
            observations = sample_marker_bodies(trk)
            observed = observations[mbee_placed.body.name]
            
            # Makes sure that it's visible
            if observed is None:
                raise ValueError("Calibration body isn’t visible")
            
            # Add it to the data
            data.thetas.append(thetas[i].copy())
            data.frames.append(observed.frame)
    
    #Gets the clock elapsed overall
    data.t_collect = clock.elapsed - t_start
    return data


def estimate_base_frame(data: CalibrationData, kins: Robot) -> Frame:
    """
    #TODO (DONE): implement.

    With the calibration body's frame being exactly F_ee, every sample
    satisfies  F_ee,j = F_X * kins.fwd_kins(theta_j)  for one constant
    unknown F_X (the robot base expressed in the tracker's frame). Estimate
    F_X from all samples (your average_frames may help).

    input:
        data - CalibrationData
        kins - the kinematics to use (the nominal Robot at first; later your
            ImprovedKinematics)
    output: F_X
    """
    
    #Error handling
    if len(data.frames) != len(data.thetas) or len(data.frames) == 0:
        raise ValueError("Thetas and frames aren't equal lengths")
    
    base_frames = []
    for j in range(len(data.thetas)):
        #Finds where the robot -> tracker: Fx_j
        # data.frames (F_ee,j) : end effector -> tracker
        # kins.fwd_kins(data.thetas[j])^-1 : base -> end effector
        # End effector is attachment at end of robot that tracks pose relative to robot base
        Fx_j = data.frames[j] * kins.fwd_kins(data.thetas[j]).inv()
        base_frames.append(Fx_j)
    
    return average_frames(base_frames)
