"""Statistical estimate of the registration error and the predicted TRE."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import Frame
from uncertainty_networks.se3 import log_se3
from ..mesh import Mesh
from .icp import IcpResult


def registration_covariance(result: IcpResult, F0_reg: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO(DONE): implement.
    Estimate the correction from F0_reg and local frame uncertainty.
    
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
    TODO(DONE): implement.
    Summarize per-vertex RMS position uncertainties in millimeters.
    
    input: mesh, estimated F_reg, C_reg (6,6)
    output: {"rms", "mean", "max", "p90"} of the predicted TRE over the mesh vertices
    """
    
    #Gets the covariance matrix
    C_reg = np.asarray(C_reg, dtype=float)
    
    #Error handling: Matrix is valid
    if C_reg.shape != (6, 6) or not np.isfinite(C_reg).all():
        raise ValueError("C_reg must be a finite 6 by 6 covariance matrix")
    
    #Error handling: there are vertices
    if mesh.n_vertices == 0:
        raise ValueError("The mesh must contain vertices (nonzero)")
    
    #Prep and store identity matrix
    vertex_rms = np.empty(mesh.n_vertices)
    identity = np.eye(3)
    
    #Get one vertex at a time
    for i, point in enumerate(mesh.vertices):
        x, y, z = point
        
        #Skew matrix from class
        skew_point = np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0]])
        
        #Stack the matrix to get a 3x6 with the identity on the bottom
        J = np.column_stack((-skew_point, identity))
        
        #Note that point error ≈ J @ frame error, so
        #point cov = J @ C_reg @ J.T
        point_cov = J @ C_reg @ J.T
        
        variance_sum = float(point_cov[0, 0] + point_cov[1, 1] + point_cov[2, 2])
        vertex_rms[i] = np.sqrt(max(variance_sum, 0.0))
    
    return {
        "rms": float(np.sqrt(np.mean(vertex_rms**2))),
        "mean": float(np.mean(vertex_rms)),
        "max": float(np.max(vertex_rms)),
        "p90": float(np.percentile(vertex_rms, 90))}