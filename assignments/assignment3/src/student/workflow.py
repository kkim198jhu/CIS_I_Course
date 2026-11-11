"""The end-to-end registration workflow: design, patch plan, data collection, registration."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import time

from .icp import coarse_registration, icp
from .uncertainty import registration_covariance, predicted_tre
from .report import RegistrationReport, residual_statistics
from ..mesh import Mesh, PatchBox
from ..session import Session, SceneDesign, PointerPlacement
from .report import RegistrationReport
from data_types.nominal_types import vct3, Rot, Frame


@dataclass
class SampleSet:
    """Tip positions collected through the tracker."""
    handles: List[PointerPlacement] = field(default_factory=list)
    patch_ids: List[int] = field(default_factory=list)
    d_ref: List[np.ndarray] = field(default_factory=list)   # (3,) tip in reference-body coordinates
    t_collect: float = 0.0

    def __len__(self) -> int:
        return len(self.d_ref)

    def as_array(self) -> np.ndarray:
        return np.array(self.d_ref).reshape(-1, 3)


def design_scene() -> SceneDesign:
    """
    TODO(DONE): implement -- return the same values the notebook's Step 1 cell uses
    (grading calls this instead of the notebook).
    """
    F_tracker_approx = Frame(Rot(), vct3(0, 0, -1500))
    p_tip = vct3(0, 0, -150)
    ref_marker_positions = [vct3(-40, -30, 0), vct3(40, -30, 0), vct3(0, 40, 0), vct3(10, 0, 25)]

    ptr_marker_positions = [vct3(-30, -20, 0), vct3(30, -20, 0), vct3(0, 30, 0), vct3(10, 5, 20)]
    
    return SceneDesign(F_tracker_approx, ref_marker_positions, ptr_marker_positions, p_tip)


def plan_patches(boxes: List[PatchBox], n_patches: int, samples_per_patch: int) -> List[int]:
    """
    TODO(DONE): implement.
    Return one patch ID (number that labels a region) for each planned pointer touch (one simulated placement of the pointer tip)
    
    input: the patch boxes, how many patches to use, samples per patch
    output: ordered list of patch ids to call place_pointer with (repeats allowed)
    """
    #Error handling
    if not 1 <= n_patches <= len(boxes):
        raise ValueError("n_patches must be between 1 and the number of boxes")
    if samples_per_patch < 1:
        raise ValueError("samples_per_patch must be positive")
    
    plan = []
    for box in boxes[:n_patches]:
        for i in range(samples_per_patch):
            plan.append(box.id)
    
    return plan


def collect_samples(session: Session, plan: List[int]) -> SampleSet:
    """
    TODO(DONE): implement.
    Collect tip positions in reference-body coordinates for a patch plan
    
    For every patch id: session.place_pointer, session.sample_marker_bodies,
    d_k = F_ref^-1 F_ptr p_tip; skip (and log) samples where a body was invisible.
    input: session, patch plan
    output: SampleSet
    """
    
    #Initalize the variables
    samples = SampleSet()
    t_start = session.elapsed()
    p_tip = session.design.p_tip
    
    
    for patch_id in plan:
        #Place pointer and read the tracker
        handle = session.place_pointer(patch_id)
        observations = session.sample_marker_bodies()
        
        ref_obs = observations["ref"]
        ptr_obs = observations["ptr"]
        
        #Check that it can see the bodies
        if ref_obs is None or ptr_obs is None:
            print(f"Skipped sample {handle.sample_index} on patch {patch_id}: a body was not visible")
            continue
        
        #Express tip's position in reference fra e
        F_ref = ref_obs.frame
        F_ptr = ptr_obs.frame
        
        tip_tracker = F_ptr * p_tip
        tip_ref = F_ref.inv() * tip_tracker
        
        #Add it to samples
        samples.handles.append(handle)
        samples.patch_ids.append(patch_id)
        samples.d_ref.append(tip_ref.vec.ravel())
    
    #Record the time
    samples.t_collect = session.elapsed() - t_start
    return samples


def run_registration(session: Session, mesh: Mesh, boxes: List[PatchBox], **kwargs) -> RegistrationReport:
    """
    TODO: implement -- the headless entry point used for grading:
    Collect samples, register reference to CT, and return the report.
    
    plan -> collect -> coarse_registration -> icp -> uncertainty -> RegistrationReport.
    input: session, mesh, boxes, your own keyword options
    output: RegistrationReport
    """
    #Starts all the clocks
    reality_start = time.perf_counter()
    sim_start = session.elapsed()
    
    #Gets the session's initial guess
    F0_reg = session.initial_registration_guess()
    
    #Get the correct parameters from kwargs
    n_patches = kwargs.get("n_patches", len(boxes))
    samples_per_patch = kwargs.get("samples_per_patch", 4)
    
    #Plan patches and collect samples
    plan = plan_patches(boxes, n_patches, samples_per_patch)
    samples = collect_samples(session, plan)
    
    #Needs at least 3 samples for a plane
    if len(samples) < 3:
        raise ValueError("Too few visible samples for registration")
    
    #Group measured points by patch id
    d = samples.as_array()
    by_patch = {}
    
    #Goes through each sample
    for patch_id, point in zip(samples.patch_ids, d):
        #add it if it doesn't exist yet, or add to current
        if patch_id not in by_patch:
            by_patch[patch_id] = []
        by_patch[patch_id].append(point)
    
    #Turns this into a 2D array with [x, y, z] readings and allows for coarse_registration
    for patch_id in by_patch:
        by_patch[patch_id] = np.array(by_patch[patch_id])
    
    #Find the alignment and estimate its uncertainty
    F_coarse = coarse_registration(by_patch, boxes)
    result = icp(
        d, mesh, F_coarse,
        max_iter=kwargs.get("max_iter", 100),
        tol=kwargs.get("tol", 1e-4),
        reject=kwargs.get("reject", "mad"))
    mu, C_reg = registration_covariance(result, F0_reg)
    prediction = predicted_tre(mesh, result.F_reg, C_reg)
    
    #Creates one counter record for each planned patch.
    patch_counts = {}
    for patch_id in plan:
        if patch_id not in patch_counts:
            patch_counts[patch_id] = {
                "patch_id": int(patch_id),
                "n_samples": 0,
                "n_used": 0,
                "n_rejected": 0,
            }
    
    #Record what happened to each collected sample
    sample_records = []
    for i, handle in enumerate(samples.handles):
        patch_id = samples.patch_ids[i]
        used = bool(result.inlier_mask[i])
        
        #Check if it is used or rejected
        patch_counts[patch_id]["n_samples"] += 1
        if used:
            patch_counts[patch_id]["n_used"] += 1
        else:
            patch_counts[patch_id]["n_rejected"] += 1
        
        #Add it to the sample records
        sample_records.append({
            "sample_index": int(handle.sample_index),
            "patch_id": int(patch_id),
            "t_sim": float(handle.t_sim),
            "d_ref": samples.d_ref[i].tolist(),
            "residual_mm": float(result.residuals[i]),
            "used": used,
            "reason": "inlier" if used else "ICP outlier"})
    
    #Calculate the residual statistics and elapsed time
    stats = residual_statistics(result.residuals[result.inlier_mask])
    simulated_time = session.elapsed() - sim_start
    
    #Return a bunch for the report
    return RegistrationReport(
        F_reg=result.F_reg,
        patches=list(patch_counts.values()),
        samples=sample_records,
        residual_stats=stats,
        t_simulated=simulated_time,
        t_wall=time.perf_counter() - reality_start,
        n_icp_iterations=result.n_iter,
        converged=result.converged,
        eta_reg={"mu": mu, "C_reg": C_reg, "convention": "left_ct"},
        predicted_tre=prediction
    )
