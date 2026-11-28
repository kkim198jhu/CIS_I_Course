"""
The anatomy of Figure 2: a tube at F_tube with the S1..S8 structures embedded in
it. build_scene() returns a Scene whose structures are already placed relative
to the robot (S.F_R = F_tube S.F_S), so the per-structure loop in the handout's
CooperativeControlMode reduces to scene.distances(p_RT) / scene.gradients(p_RT).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from data_types.nominal_types import Frame

from . import constants as C
from .anatomy import Structure, FACTORIES, _as3


@dataclass
class Scene:
    F_tube: Frame
    structures: List[Structure] = field(default_factory=list)
    r_tube: float = C.R_TUBE
    L_tube: float = C.L_TUBE

    def __post_init__(self):
        for k, s in enumerate(self.structures, start=1):
            s.place(self.F_tube)
            s.name = f"S{k}:{s.kind}({s.type})"
        self._R = self.F_tube.R.matrix.copy()
        self._p = self.F_tube.p.vec.ravel().copy()

    def __len__(self):
        return len(self.structures)

    def __iter__(self):
        return iter(self.structures)

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.structures]

    @property
    def stopper_index(self) -> int:
        """Index of the (single) type-D structure."""
        return next(k for k, s in enumerate(self.structures) if s.type == "D")

    @property
    def tube(self) -> Structure:
        return next(s for s in self.structures if s.type == "A")

    def to_tube(self, p_R) -> np.ndarray:
        """x = F_tube^-1 p_R, a robot-frame point in tube coordinates."""
        return self._R.T @ (_as3(p_R) - self._p)

    def from_tube(self, x_tube) -> np.ndarray:
        return self._R @ _as3(x_tube) + self._p

    def distances(self, p_R) -> np.ndarray:
        """[D(S_1), ..., D(S_N)] for the tool tip at p_R (robot coordinates)."""
        return np.array([s.distance_at(p_R) for s in self.structures])

    def gradients(self, p_R) -> np.ndarray:
        """N x 3 array; row k is grad D(S_k) with respect to p_R, in robot coordinates."""
        return np.array([s.gradient_at(p_R) for s in self.structures])


def build_scene(F_tube: Frame = C.F_TUBE, specs=C.STRUCTURE_SPECS) -> Scene:
    structures = [FACTORIES[kind](**kw) for kind, kw in specs]
    return Scene(F_tube=F_tube, structures=structures)
