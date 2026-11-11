"""Rigid registration: point-set registration, coarse initialisation and ICP."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame
from ..mesh import Mesh, PatchBox

from .closest_point import closest_points_on_mesh

@dataclass
class IcpResult:
    """Output of icp()."""
    F_reg: Frame                  # reference body -> CT
    residuals: np.ndarray         # (K,) ||F_reg d_k - closest_k|| for every sample
    inlier_mask: np.ndarray       # (K,) bool, False for rejected samples
    closest: np.ndarray           # (K,3) closest mesh points of the last iteration
    normals: np.ndarray           # (K,3) mesh normals at those points
    n_iter: int
    converged: bool
    history: List[float] = field(default_factory=list)   # rms inlier residual per iteration


def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    TODO: implement (your own; do not call the simulator's solver).
    Rigid point-set registration: find the least-squares Frame F such that
    F * a[i] ~= b[i] for every corresponding pair. Think about what happens
    with fewer than 3 points, with collinear points, and with a reflection. (same as a2)
    
    input: paired points a_i, b_i (>= 3)
    output: least-squares Frame F with F * a_i ~= b_i
    """
    #Error Handling
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("Nonvalid inputs for either a or b")
    
    # b_i = R * a_i + p
    # Gets all the points for x, y, and z
    A = np.array([point.vec.ravel() for point in a])
    B = np.array([point.vec.ravel() for point in b])
    
    # Gets averages for all, so b_avg = R * a_avg + p 
    a_centroid = np.mean(A, axis=0)
    b_centroid = np.mean(B, axis=0)
    
    #Sets them to the origin, so don't worry about translations
    #b_i - b_avg = (R * a_i + p) - (R * a_avg + p )
    #b_i - b_avg = R (a_i - a_avg)
    a_i_origin = A - a_centroid
    b_i_origin = B - b_centroid
    
    #Error check to make sure colinear
    rank_a = np.linalg.matrix_rank(a_i_origin)
    rank_b = np.linalg.matrix_rank(b_i_origin)
    if (rank_a == 0 or rank_b == 0):
        raise ValueError("All points are the same point")
    
    if (rank_a == 1 or rank_b == 1):
        raise ValueError("All points lie on a line")
    
    #Gets a 3x3 matrix for singular value decomp.
    H = a_i_origin.T @ b_i_origin
    
    # H = U * sum(V)^t
    U, singular_values, Vt = np.linalg.svd(H)
    
    #From that we get the rotation prediction, as R_pred * U = V, where U is source direction and V is matching target directions
    R_prediction = Vt.T @ U.T
    
    # +1 is good proper rotation, while -1 is a reflection, so we need to flip it back
    R_prediction_det = np.linalg.det(R_prediction)
    if (R_prediction_det < 0):
        Vt[-1, :] *= -1
        R_prediction = Vt.T @ U.T
    
    # Gets the translation
    p = b_centroid - R_prediction @ a_centroid
    
    F = Frame (Rot(matrix=R_prediction), vct3(p))
    
    return F


def coarse_registration(d_by_patch: Dict[int, np.ndarray], boxes: List[PatchBox]) -> Frame:
    """
    TODO: implement.
    Gives ICP a rough starting alignment between your measured tip points and the CT model
    
    input: {patch id: (n_i,3) tip points in reference coordinates}, the patch boxes
    output: a rough F_reg (reference -> CT) from patch centroids <-> box centres
    """
    #Makes the CT dictionary with id as keys and the box as the items
    boxes_by_id = {}
    for box in boxes:
        boxes_by_id[box.id] = box
    
    #Creates two lists
    ref_centers = []
    ct_centers = []
    
    #Go through the sample patches
    for patch_id, tip_points in d_by_patch.items():
        #If it isn't valid, then skip
        if len(tip_points) == 0:
            continue
        
        #Build one matched pair for that patch
        center_ref = np.mean(tip_points, axis=0)
        center_ct = boxes_by_id[patch_id].p
        
        #Add it to the loop
        ref_centers.append(vct3(center_ref))
        ct_centers.append(vct3(center_ct))
    
    return register_points(ref_centers, ct_centers)


def icp(d: np.ndarray, mesh: Mesh, F_init: Frame, max_iter: int = 100, tol: float = 1e-4,
        reject: Optional[str] = "mad") -> IcpResult:
    """
    TODO: implement.
    Refine a reference-to-CT frame using tip points d
    
    input: d (K,3) tip points in reference coordinates, mesh, initial F_reg,
           iteration cap, convergence tolerance, outlier-rejection method
    output: IcpResult
    """
    
    d = np.asarray(d, dtype=float)
    #Reject errors/outliers
    if d.ndim != 2 or d.shape[1] != 3 or len(d) < 3:
        raise ValueError("ICP needs 3 3D tip points at least")
    
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter and tol are not currently positive")
    
    if reject not in (None, "mad"):
        raise ValueError("reject must be None or 'mad'")
    
    #Transform the tips into CT and find their mesh matches.
    def find_matches(F: Frame):
        #Measures it in CT coordinates
        x = d @ F.R.matrix.T + F.p.vec.ravel()
        
        #Find the closest points to the mesh and normalize it
        closest, triangle_indices, residuals = closest_points_on_mesh(x, mesh)
        normals = mesh.face_normals[triangle_indices]
        
        # Keep every match originally
        imask = np.ones(len(d), dtype=bool)
        
        #Check if it is median absolute deviation
        if reject == "mad":
            median = np.median(residuals)
            mad = np.median(np.abs(residuals - median))
            cutoff = median + 3.5 * max((1/0.6745) * mad, tol)
            #Removes all them that is outside the mean
            imask = residuals <= cutoff
        
        #If there isn't enough inliers then we reject
        if np.count_nonzero(imask) < 3:
            raise ValueError("ICP found fewer than three inlier matches")
        
        return x, closest, residuals, normals, imask
    
    #Start with the initial frame
    F_reg = F_init
    x, closest, residuals, normals, imask = find_matches(F_reg)
    history = []
    converged = False
    
    #Iterate for max_iter
    for i in range(max_iter):
        #Get the inliers
        x_in = x[imask]
        q_in = closest[imask]
        n_in = normals[imask]
        
        #Solve for rotation and translation that improve those matches
        A = np.column_stack((np.cross(x_in, n_in), n_in))
        b = np.sum(n_in * (q_in - x_in), axis=1)
        delta, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        
        #Put the rot and translation into a frame
        if rank < 6:
            source = [vct3(point) for point in x_in]
            target = [vct3(point) for point in q_in]
            correction = register_points(source, target)
        else:
            angle = float(np.linalg.norm(delta[:3]))
            R_delta = Rot()
            if angle > 0:
                axis = vct3(delta[:3] / angle)
                R_delta = Rot(axis, angle=angle)
            correction = Frame(R_delta, vct3(delta[3:]))
        
        #Apply the correction and then find the matches again
        F_reg = correction * F_reg
        x_new, closest, residuals, normals, new_mask = find_matches(F_reg)
        
        #Record the data
        step_mm = np.linalg.norm(x_new[imask] - x_in, axis=1).max()
        x = x_new
        imask = new_mask
        history.append(float(np.sqrt(np.mean(residuals[imask]**2))))
        
        #Can decide when to stop
        if step_mm <= tol:
            converged = True
            break
    
    #Return the results
    return IcpResult(
        F_reg=F_reg,
        residuals=residuals,
        inlier_mask=imask,
        closest=closest,
        normals=normals,
        n_iter=len(history),
        converged=converged,
        history=history)
