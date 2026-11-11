"""Statistical estimate of the registration error and the predicted TRE."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import Frame
from uncertainty_networks.se3 import log_se3
from ..mesh import Mesh
from .icp import IcpResult


def registration_covariance(result: IcpResult, F0_reg: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: implement.
    Left/CT convention: F*_reg = Exp(delta) . F_reg, eta_reg = [alpha; eps] relative to F0_reg.
    input: IcpResult, the initial guess F0_reg
    output: mu (6,) estimate of eta_reg, C_reg (6,6) covariance of delta
    """
    #Intialize everything
    mask = result.inlier_mask
    closest = result.closest[mask]
    norms = result.normals[mask]
    residuals = result.residuals[mask]
    count = len(residuals)
    
    #Error handling: Have enough to match the noise
    if count <= 6:
        raise ValueError("Need more than six inlier matches to estimate covariance")
    
    #Matches constrain the frame (small = can have a large change without changing the frame)
    A = np.column_stack((np.cross(closest, norms), norms))
    _, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    
    #Error handling
    for value in singular_values:
        if value <= 0.001:
            raise ValueError("Matches need to be nonzero (or not close to zero)")
    
    #To get Covariance = sigma² × inverse(A.T @ A)
    #Estimate residual noise levels by first getting the sigma_sq like in assignment 1
    sigma_sq = float(residuals @ residuals) / (count - 6)
    
    #Convert the noise into frame uncertainty
    #Note that: A = U @ S @ Vt from SVD
    #And then A.T x A = Vt.T @ S @ U.T @ U @ S @ Vt = Vt.T @ S² @ Vt
    #inverse(A.T @ A) = Vt.T @ S⁻² @ Vt (which is the equation below)
    #we want this because to know uncertain those six frame components
    C_reg = sigma_sq * (Vt.T @ np.diag(1.0 / singular_values**2) @ Vt)
    
    #Averages the transpose to restore symmetry 
    C_reg = (C_reg + C_reg.T)/2
    
    #Describes the correction from the initial frame
    #Gets the total adjustment from your initial registration guess to the final ICP estimate
    correction = result.F_reg * F0_reg.inv()
    T = np.eye(4)
    T[:3, :3] = correction.R.matrix
    T[:3, 3] = correction.p.vec.ravel()
    mu = log_se3(T)
    
    return mu, C_reg


def predicted_tre(mesh: Mesh, F_reg: Frame, C_reg: np.ndarray) -> dict:
    """
    TODO: implement.
    input: mesh, estimated F_reg, C_reg (6,6)
    output: {"rms", "mean", "max", "p90"} of the predicted TRE over the mesh vertices
    """
    raise NotImplementedError("TODO: implement predicted_tre")
