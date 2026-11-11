"""The end-to-end registration workflow: design, patch plan, data collection, registration."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

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
    plan -> collect -> coarse_registration -> icp -> uncertainty -> RegistrationReport.
    input: session, mesh, boxes, your own keyword options
    output: RegistrationReport
    """
    raise NotImplementedError("TODO: implement run_registration")
