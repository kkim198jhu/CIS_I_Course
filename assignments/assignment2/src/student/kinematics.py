"""
The calibrated kinematic model: ImprovedFwdKins / ImprovedInvKins.
"""

from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from data_types.nominal_types import vct3, Rot, Frame
from uncertainty_networks.se3 import log_so3, exp_se3, exp_so3

from ..robot import Robot
from .workflow import CalibrationData


class ImprovedKinematics(Robot):
    """
    TODO: implement -- the core deliverable.

    A Robot whose fwd_kins / inv_kins account for the calibrated error
    model. You choose the parameterization; the handout's error model is
        theta*_k = theta_k + delta_theta_k,sys(theta_k)   (all joints; ||.|| <= delta_sys, slope <= omega_sys)
        d*_k = R(phi_k) d_k                                (k = 2..6, ||phi_k|| <= phi_max; d*_1 = d_1)
        F*_we = F_we dF_ee                                 (||alpha_ee|| <= 0.01, ||epsilon_ee|| <= 0.5)
    plus the constant base frame F_X relating the robot base to the
    tracker (see workflow.estimate_base_frame). HomeRobot() gives you
    delta_theta_sys at home directly (workflow.measure_home_offsets); pass
    it as joint_offsets. Every parameter you include must be identifiable
    from your data -- look at the singular values of the Jacobian.

    A parameter vector that covers this model, for example:
        params[0:6]     xi_X          twist of the base frame about F_X_init, F_X = F_X_init * Exp(xi_X)
        params[6:16]    (t1, t2) x 5  tilt of d_2, d_3, n_4, n_5, n_6, two components each in the
                                      plane perpendicular to the nominal axis
        params[16:22]   xi_we         twist of the tool frame, F_we = F_WE * Exp(xi_we)
        params[22:]     c_k,j         per-joint distortion polynomial coefficients
                                      delta_theta_k,sys = offset_k + sum_j c_k,j (theta_k / scale_k)^j
    with every entry zero meaning "the nominal robot at F_X_init".

    The nominal Robot's attributes (AXES, F_CW, F_WE, limits, timing,
    wrist_solutions) are inherited and may be reused.
    """

    def __init__(self, robot: Robot, F_X_init: Frame, joint_offsets=None, n_poly: int = 2):
        super().__init__(robot.name + "_improved", robot.F_rob, robot.theta_min, robot.theta_max,
                         robot.t_settle, robot.theta_dot_max, robot.theta_ddot_max)
        self.F_X_init = F_X_init  # starting estimate of the base frame (from the nominal fit)
        self.joint_offsets = np.zeros(6) if joint_offsets is None else np.asarray(joint_offsets, dtype=np.float64).reshape(6)
        self.n_poly = n_poly      # e.g. the degree of a per-joint distortion polynomial
        self.params = self.initial_params()

    # ------------------------------------------------------------------ parameterization

    def initial_params(self) -> np.ndarray:
        """
        TODO (DONE): implement.
        Creates the initial guesses for the corrections
        
        output: the parameter vector that makes this model identical to the
                nominal kinematics with base frame F_X_init -- with the
                layout in the class docstring that is simply
                np.zeros(6 + 10 + 6 + 6 * self.n_poly)
        """
        
        return np.zeros(6 + 10 + 6 + 6 * self.n_poly)

    def base_frame(self, params: Optional[np.ndarray] = None) -> Frame:
        """
        TODO (DONE): implement.
        Applies the first six correction parameters to the starting base-frame
        
        output: F_X (robot base relative to the tracker) for params (default:
                self.params) -- e.g. F_X_init * Exp(params[0:6]), see
                uFrame.perturb / uncertainty_networks.se3.exp_se3
        """
        
        if params is None:
            params = self.params
        
        f_X = self.F_X_init * Frame(exp_se3(params[0:6]))
        return f_X

    # ------------------------------------------------------------------ kinematics

    def fwd_kins(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """
        ImprovedFwdKins(theta).
        TODO (DONE): implement.
        Predicts the end-effector pose with the formula F_ee = F_c * F_cw * F_w * F_we^(corrected)
            F_c : Moving carriage -> robot base
            F_cw : Unrotated wrist frame -> carriage
            F_w : Rotated wrist frame -> unrotated wrist frame
            F_we^(corrected) : End effector → rotated wrist frame
            
        input: theta - 6 encoder joint values; params - defaults to self.params
        output: F_ee relative to the robot base frame, with the calibrated
                error model (joint offsets + distortion, tilted axes, tool
                frame offset) applied
        """
        
        # Corrects how far each joint moves or rotates
        #Makes sure we have params
        if params is None:
            params = self.params
        
        #Gets the joint offsets
        theta = np.asarray(theta, dtype=float).reshape(6)
        theta_corrected = theta + self.joint_offsets
        
        coefficients = params[22:].reshape(6, self.n_poly)
        
        #calculates theta*_k = theta_k + delta_theta_k,sys(theta_k) (Look at ImprovedKinematics Docstring above)
        #theta_k^corrected = theta_k + offsets_k + c_k,1 (theta_k / s_k) + c_k,2(theta_k / s_k) ^2
        scales = np.maximum(np.abs(self.theta_min), np.abs(self.theta_max))
        theta_normalized = theta / scales
        
        for k in range(6):
            for j in range(self.n_poly):
                theta_corrected[k] += (coefficients[k, j] * theta_normalized[k] ** (j + 1))
        
        
        
        # Accounts for robot's tilted axes that are not perfectly aligned with x, y, and z.
        #Each row contains 2 tilt values for 1 axis
        tilts = params[6:16].reshape(5, 2)
        
        #Starts the corrected axis list, so axis 1 is unchanged
        axes = [self.AXES[0]]  
        
        #Gets the nominal axies from 2 - 6
        for k in range(1, 6):
            t1, t2 = tilts[k - 1]
            
            #Nominal x-axis
            if k % 3 == 0:      
                phi = np.array([0.0, t1, t2])
                
            # Nominal y-axis
            elif k % 3 == 1:    
                phi = np.array([t1, 0.0, t2])
            
            # Nominal z-axis
            else:             
                phi = np.array([t1, t2, 0.0])
            
            #Gets the nominal direction * rotation matrix from the rotation vector (exp_so3)
            #Implements d*_k = R(phi_k) d_k from docstring above (in ImprovedKinematics doc string)
            corrected_axis = vct3(exp_so3(phi) @ self.AXES[k].vec.ravel())
            axes.append(corrected_axis)
        
        
        
        # Calculate translation produced by the first three joint
        #Not all 6 joints as the first 3 joints only do translation and the next 3 do rotation
        #Find this out at _actual_fwd_kins in Robot the class we inherited from
        p_c = (
            axes[0] * theta_corrected[0]
            + axes[1] * theta_corrected[1]
            + axes[2] * theta_corrected[2]
        )
        
        #Gain the translation in a frame (F_c)
        F_c = Frame(Rot.eye(), p_c)
        
        # Calculate rotations about the three corrected wrist axes
        #Next 3 do rotation to get R_w = R_4 * R_5 * R_6.
        R_w = Rot.eye()
        for k in range(3, 6):
            rotation = Rot(matrix=exp_so3(axes[k].vec.ravel() * theta_corrected[k]))
            R_w = R_w * rotation
        
        #Wrist rotation has no translation of its own
        F_w = Frame(R_w, vct3())
        
        
        # Apply the formula from above where 
        #F_ee = F_c * F_cw * F_w * F_we^(corrected)
        # Frame(exp_se3(params[16:22])) = docstring above: xi_we -> twist of the tool frame, F_we = F_WE * Exp(xi_we)
        F_we_corrected = self.F_WE * Frame(exp_se3(params[16:22]))

        # End-effector pose relative to the robot base
        return F_c * self.F_CW * F_w * F_we_corrected 

    def predict_in_tracker(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """The end-effector pose in the tracker frame: base_frame * fwd_kins."""
        return self.base_frame(params) * self.fwd_kins(theta, params)

    def inv_kins(self, F: Frame, seed=None, tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
        """
        ImprovedInvKins(F) -- provided.

        There is no closed form once the axes are tilted, so this iterates
        on your fwd_kins: both nominal wrist branches (Robot.inv_kins'
        closed form) are refined with damped Gauss-Newton / Levenberg-
        Marquardt steps on the pose error [alpha; epsilon] of
        fwd_kins(theta)^-1 F, and the in-limit solution nearest `seed`
        (default: the current command) is returned. Works as soon as your
        fwd_kins does.

        input: F - desired F_ee relative to the robot base frame
        output: theta with fwd_kins(theta) ~= F; raises ValueError if no
                branch satisfies the joint limits
        """
        seed = self.theta_cmd if seed is None else np.asarray(seed, dtype=np.float64).reshape(6)
        p_w = F.p - F.R * self.F_WE.p - self.F_CW.p   # nominal wrist center -> seeds for the linear joints
        step_max = np.array([50.0] * 3 + [0.3] * 3)
        h = np.array([1e-3] * 3 + [1e-6] * 3)

        def pose_delta(theta):
            dF = self.fwd_kins(theta).inv() * F
            return np.concatenate([log_so3(dF.R.matrix), dF.p.vec.ravel()])

        candidates = []
        for wrist in self.wrist_solutions(F.R):
            theta = np.concatenate([[p_w.x, p_w.y, p_w.z], wrist])
            lam, delta = 1e-4, pose_delta(theta)
            for _ in range(max_iter):
                cost = float(delta @ delta)
                if np.sqrt(cost) < tol:
                    break
                J = np.zeros((6, 6))
                for k in range(6):
                    th = theta.copy()
                    th[k] += h[k]
                    J[:, k] = (pose_delta(th) - delta) / h[k]
                while True:  # Levenberg-Marquardt: shrink the step until the cost decreases
                    JtJ = J.T @ J
                    step = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), J.T @ delta)
                    trial = theta + np.clip(step, -step_max, step_max)
                    delta_trial = pose_delta(trial)
                    if float(delta_trial @ delta_trial) < cost or lam > 1e6:
                        theta, delta, lam = trial, delta_trial, max(lam / 10.0, 1e-12)
                        break
                    lam *= 10.0
            theta[3] = (theta[3] + np.pi) % (2 * np.pi) - np.pi
            theta[5] = (theta[5] + np.pi) % (2 * np.pi) - np.pi
            if self.within_limits(theta):
                candidates.append(theta)
        if not candidates:
            raise ValueError(f"inv_kins: no joint solution within limits for {F!r}")
        return min(candidates, key=lambda th: float(np.linalg.norm(th - seed)))

    # ------------------------------------------------------------------ fitting

    def residuals(self, params: np.ndarray, data: CalibrationData) -> np.ndarray:
        """
        TODO (DONE): implement.
        Tells the fitting algorithm how your current model is wrong
        
        input: params - candidate parameter vector; data - CalibrationData
        output: flat residual vector comparing predict_in_tracker(theta_j,
                params) with data.frames[j] for every sample (choose how to
                weigh rotation vs translation)
        """
        
        #Error handling
        if len(data.frames) != len(data.thetas) or len(data.frames) == 0:
            raise ValueError("Need nonempty, paired joint values and frames")
        
        errors = []
        for j in range(len(data.thetas)):
            #Gets the predicted values
            predicted = self.predict_in_tracker(data.thetas[j], params)
            
            #Gets the observed values
            observed = data.frames[j]
            
            #Does delta F = F^-1_predicted * F^observed (look at frame_error for formula)
            dF = predicted.inv() * observed
            
            #Gets the errors and weights to them based on their target accuracy (in assignment 2)
            rot_error = log_so3(dF.R.matrix)
            trans_error = dF.p.vec.ravel()
            
            rot_error = rot_error/0.001
            trans_error = trans_error/0.2
            
            error = np.concatenate((rot_error, trans_error))
            errors.append(error)
        
        return np.concatenate(errors)
    def fit(self, data: CalibrationData, **kwargs):
        """
        TODO(DONE): implement.
        Minimize the sum of squared residuals
        
        Non-linear least squares over the parameters (scipy.optimize is
        allowed). Must set self.params. Return whatever diagnostics you want
        to report (final cost, iterations, ...).
        """
        
        result = least_squares(self.residuals, self.params, args=(data,), **kwargs)
        self.params = result.x
        return result

    def jacobian_rank_report(self, data: CalibrationData, params: Optional[np.ndarray] = None) -> dict:
        """
        TODO (DONE): implement.

        Numerically differentiate residuals w.r.t. params and report the
        singular values (and which parameter directions are poorly
        determined). Use it to justify your parameterization and your
        choice of calibration poses in the report.
        """
        
        #Gets some params as default is self.params
        if (params is None):
            params = self.params
        
        #Gets the residue vector and an empty matrix
        res_vector = self.residuals(params, data)
        mat = np.zeros((len(res_vector), len(params)))
        
        #Finds how each residual changes when one parameter changes
        h = 10**-6
        for i in range(len(params)):
            c_param = params.copy()
            c_param[i] += h
            
            mod_res_vector = self.residuals(c_param, data)
            mat[:, i] = (mod_res_vector - res_vector) / h
        
        # SVD 
        U, singular_values, Vt = np.linalg.svd(mat, full_matrices=False)
        rank = np.linalg.matrix_rank(mat)
        
        return {
            "rank": rank,
            "n_parameters": len(params),
            "singular_values": singular_values,
            "parameter_directions": Vt
        }

    # ------------------------------------------------------------------ reporting

    def parameters_report(self) -> Dict[str, np.ndarray]:
        """
        TODO: implement.

        The handout (method 1 (e)) asks your program to report the updated
        kinematic parameters in physical terms, and your debugging runs to
        compare them with ground truth (rob.ground_truth_parameters()).
        Report at least: the axis tilt vectors / unit axes, the angles
        between the axes, the per-joint offsets and scale factors, and the
        tool-frame offset. Note which of these are only defined up to the
        base-frame gauge (and so are better compared as angles between axes).
        """
        
        
        # Corrects how far each joint moves or rotates
        #Makes sure we have params
        params = self.params
        
        # Accounts for robot's tilted axes that are not perfectly aligned with x, y, and z.
        #Each row contains 2 tilt values for 1 axis
        tilts = params[6:16].reshape(5, 2)
        
        #Starts the corrected tilt_vectors, with axis 1 unchanged for both
        tilt_vectors = [np.zeros(3)]
        axes = [self.AXES[0]]
        
        #Gets the nominal axies from 2 - 6
        for k in range(1, 6):
            t1, t2 = tilts[k - 1]
            
            #Nominal x-axis
            if k % 3 == 0:      
                phi = np.array([0.0, t1, t2])
                
            # Nominal y-axis
            elif k % 3 == 1:    
                phi = np.array([t1, 0.0, t2])
            
            # Nominal z-axis
            else:             
                phi = np.array([t1, t2, 0.0])
            
            #Gets the nominal direction * rotation matrix from the rotation vector (exp_so3)
            #Implements d*_k = R(phi_k) d_k from docstring above (in ImprovedKinematics doc string)
            corrected_axis = vct3(exp_so3(phi) @ self.AXES[k].vec.ravel())
            tilt_vectors.append(phi)
            axes.append(corrected_axis)
        
        #Tilt_vectors finds how each nominal axis was rotated.
        tilt_vectors = np.array(tilt_vectors)
        #Unit_axes where each corrected joint axis points
        unit_axes = np.array([axis.vec.ravel() for axis in axes])
        
        # Find the angle between axises using u dot v = cos(alpha), therefore alpha = arccos(u dot v)
        axis_angles = np.zeros((6,6))
        for i in range (6):
            for j in range (6):
                dot_product = np.dot(unit_axes[i], unit_axes[j])
                dot_product_mod = np.clip(dot_product, -1, 1)
                axis_angles[i, j] = np.arccos(dot_product_mod)
        
        coefficients = params[22:].reshape(6, self.n_poly)
        # This finds whether a joint moves slightly more or less than commanded (1 is the original movement)
        #calculates theta*_k = theta_k + delta_theta_k,sys(theta_k) (Look at ImprovedKinematics Docstring above)
        #theta_k^corrected = theta_k + offsets_k + c_k,1 (theta_k / s_k) + c_k,2(theta_k / s_k) ^2
        scales = np.maximum(np.abs(self.theta_min), np.abs(self.theta_max))
        first_coeff = coefficients[:, 0]
        coeff_normalized = first_coeff / scales + 1
        
        # Apply the formula from above where 
        #F_ee = F_c * F_cw * F_w * F_we^(corrected)
        # Frame(exp_se3(params[16:22])) = docstring above: xi_we -> twist of the tool frame, F_we = F_WE * Exp(xi_we)
        tool_frame = Frame(exp_se3(params[16:22]))
        
        #Gets the translation and rotation
        tool_trans = tool_frame.p.vec.ravel()
        tool_rot = log_so3(tool_frame.R.matrix)
        
        #Returns the keys from docstring
        return {
            "axis_tilt_vectors": tilt_vectors,
            "unit_axes": unit_axes,
            "axis_angles_rad": axis_angles,
            "joint_offsets": self.joint_offsets.copy(),
            "joint_scale_factors": coeff_normalized,
            "tool_translation_offset_mm": tool_trans,
            "tool_rotation_offset_rad": tool_rot,
        }
