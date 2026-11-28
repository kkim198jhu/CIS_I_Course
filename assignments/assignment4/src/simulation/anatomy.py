"""
Anatomic structures with signed distance functions (handout "For the TA").

Every structure S carries
    type            'A' (tube), 'B' (donut), 'C' (sphere) or 'D' (stopper)
    v_safe, d_safe  max tool-tip speed when within d_safe of S
    d_range         beyond this distance the virtual fixture for S is irrelevant
    F_S             pose of S relative to F_tube (a pure translation by c here)
    distance(x_S)   D_S, x_S given in S's own coordinates
    gradient(x_S)   grad D_S (unit vector, S coordinates)

Sign convention, for every type: D_S > 0 where the tool tip is allowed to be and
grad D_S points the way D_S increases, i.e. away from the forbidden region.
For the tube that means "away from the wall, toward the centerline".

Once a structure has been placed in a scene (Scene sets `F_R = F_tube F_S`), the
`*_at(p_R)` helpers take tool-tip positions in robot coordinates and return
gradients in robot coordinates as well.
"""

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

_EPS = 1e-9


def _as3(x) -> np.ndarray:
    if isinstance(x, vct3):
        return x.vec.ravel().astype(np.float64)
    return np.asarray(x, dtype=np.float64).reshape(3)


class Structure:
    """Base class. Subclasses implement distance() and gradient() in local coordinates."""

    kind = "Structure"

    def __init__(self, type: str, v_safe: float, d_safe: float, d_range: float, c=(0.0, 0.0, 0.0)):
        self.type = str(type)
        self.v_safe = float(v_safe)
        self.d_safe = float(d_safe)
        self.d_range = float(d_range)
        self.c = _as3(c)
        self.F_S = Frame(Rot(), vct3(*self.c))     # relative to F_tube
        self.name = self.kind
        self.place(Frame.eye())                    # until a Scene places it: F_tube = I

    # ------------------------------------------------------------ placement
    def place(self, F_tube: Frame):
        """Set F_R = F_tube F_S (pose of S relative to the robot)."""
        self.F_R = F_tube * self.F_S
        self._R = self.F_R.R.matrix.copy()
        self._p = self.F_R.p.vec.ravel().copy()

    def to_local(self, p_R) -> np.ndarray:
        """x_S = F_R^-1 p_R"""
        return self._R.T @ (_as3(p_R) - self._p)

    def distance_at(self, p_R) -> float:
        return self.distance(self.to_local(p_R))

    def gradient_at(self, p_R) -> np.ndarray:
        """Gradient of D_S with respect to the tip position in robot coordinates."""
        return self._R @ self.gradient(self.to_local(p_R))

    # ------------------------------------------------------------ to override
    def distance(self, x_S) -> float:
        raise NotImplementedError

    def gradient(self, x_S) -> np.ndarray:
        raise NotImplementedError

    def numerical_gradient(self, x_S, h: float = 1e-6) -> np.ndarray:
        """Central differences of distance(); handy for checking gradient()."""
        x = _as3(x_S)
        g = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            g[i] = (self.distance(x + e) - self.distance(x - e)) / (2 * h)
        return g

    def __repr__(self):
        return (f"{self.name}(type={self.type}, v_safe={self.v_safe:g}, d_safe={self.d_safe:g}, "
                f"d_range={self.d_range:g}, c={self.c.tolist()})")


class Tube(Structure):
    """
    Hollow tube of inner radius r_tube running from z = 0 down the -z axis of F_tube.
    D = distance to the closest point on the wall (or the entry rim when z > 0),
    negative if the tip has gone through the wall.
    """

    kind = "Tube"

    def __init__(self, type, v_safe, d_safe, d_range, r_tube, length):
        super().__init__(type, v_safe, d_safe, d_range)
        self.r_tube = float(r_tube)
        self.length = float(length)

    def _closest(self, x):
        r_xy = np.array([x[0], x[1], 0.0])
        r = np.linalg.norm(r_xy)
        u = r_xy / r if r > _EPS else np.zeros(3)   # on the axis every wall point is equally close
        a = self.r_tube * u
        b = np.array([a[0], a[1], min(x[2], 0.0)])  # closest point on the wall
        return r, b

    def distance(self, x_S) -> float:
        x = _as3(x_S)
        r, b = self._closest(x)
        if r <= _EPS:
            return float(np.hypot(self.r_tube, max(x[2], 0.0)))
        dist = float(np.linalg.norm(x - b))
        outside = x[2] < 0.0 and r > self.r_tube
        return -dist if outside else dist

    def gradient(self, x_S) -> np.ndarray:
        x = _as3(x_S)
        r, b = self._closest(x)
        if r <= _EPS:
            # on the centerline D is a maximum in every lateral direction
            z = max(x[2], 0.0)
            return np.array([0.0, 0.0, z / np.hypot(self.r_tube, z)])
        d = x - b
        n = np.linalg.norm(d)
        if n <= _EPS:
            return -np.array([x[0], x[1], 0.0]) / r   # on the wall: point inward
        outside = x[2] < 0.0 and r > self.r_tube
        return -d / n if outside else d / n


class Donut(Structure):
    """Torus: core circle of radius r_donut in the local xy plane, tube radius rho_donut."""

    kind = "Donut"

    def __init__(self, type, v_safe, d_safe, d_range, c, r_donut, rho_donut):
        super().__init__(type, v_safe, d_safe, d_range, c)
        self.r_donut = float(r_donut)
        self.rho_donut = float(rho_donut)

    def _displacement(self, x):
        r_xy = np.array([x[0], x[1], 0.0])
        r = np.linalg.norm(r_xy)
        # on the axis every core point is equally close; pick +x deterministically
        u = r_xy / r if r > _EPS else np.array([1.0, 0.0, 0.0])
        # closest point on the core circle, which lies in the local z = 0 plane. (The handout
        # writes b = [a_x, a_y, x_z]; that is the distance to an infinite cylinder, not a torus.)
        b = self.r_donut * u
        return x - b

    def distance(self, x_S) -> float:
        return float(np.linalg.norm(self._displacement(_as3(x_S))) - self.rho_donut)

    def gradient(self, x_S) -> np.ndarray:
        d = self._displacement(_as3(x_S))
        n = np.linalg.norm(d)
        return d / n if n > _EPS else np.array([0.0, 0.0, 1.0])


class Sphere(Structure):
    kind = "Sphere"

    def __init__(self, type, v_safe, d_safe, d_range, c, rho):
        super().__init__(type, v_safe, d_safe, d_range, c)
        self.rho = float(rho)

    def distance(self, x_S) -> float:
        return float(np.linalg.norm(_as3(x_S)) - self.rho)

    def gradient(self, x_S) -> np.ndarray:
        x = _as3(x_S)
        n = np.linalg.norm(x)
        return x / n if n > _EPS else np.array([0.0, 0.0, 1.0])


class HalfPlane(Structure):
    """Plane through c (relative to F_tube) with outward normal n; D = (x - c) . n."""

    kind = "HalfPlane"

    def __init__(self, type, v_safe, d_safe, d_range, c, n):
        super().__init__(type, v_safe, d_safe, d_range, c)
        n = _as3(n)
        self.n = n / np.linalg.norm(n)

    def distance(self, x_S) -> float:
        return float(_as3(x_S) @ self.n)   # x_S is already relative to c

    def gradient(self, x_S) -> np.ndarray:
        return self.n.copy()


FACTORIES = {"Tube": Tube, "Donut": Donut, "Sphere": Sphere, "HalfPlane": HalfPlane}
