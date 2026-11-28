"""
Constrained linear least squares (provided), the workhorse of Step 3 of the
midlevel controller:

    x* = argmin_x  sum_i w_i ||A_i x - b_i||^2
         subject to  G x <= h,   C x == d,   lb <= x <= ub

Two ways to use it:

    res = solve_lsqlin(A, b, G=G, h=h, lb=lb, ub=ub)

or, building the problem one term at a time (what the controllers in
src/student do):

    prob = LsqProblem(6)
    prob.add_objective(J_p, dp_des, weight=1.0, name="admittance")
    prob.add_objective(np.eye(6), np.zeros(6), weight=1e-4, name="regularize")
    prob.add_inequality(G, h, name="virtual fixtures")
    prob.set_bounds(lb, ub)
    res = prob.solve()
    if res.success: dq = res.x

The method is Lawson & Hanson's exact LSI -> LDP -> NNLS reduction (the same
algorithm behind MATLAB-style lsqlin with inequality constraints), with the
variables rescaled so millimetres and radians are treated evenly. For 6
unknowns it takes well under a millisecond.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls


@dataclass
class LsqResult:
    x: np.ndarray
    success: bool
    message: str
    cost: float
    max_violation: float = 0.0


def _rows(M, n):
    if M is None:
        return np.zeros((0, n))
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    if M.shape[1] != n:
        raise ValueError(f"constraint matrix has {M.shape[1]} columns, expected {n}")
    return M


def solve_lsqlin(A, b, G=None, h=None, C=None, d=None, lb=None, ub=None, x0=None,
                 tol: float = 1e-9) -> LsqResult:
    """
    Lawson & Hanson's LSI method (Solving Least Squares Problems, ch. 20-23):
    equality constraints are eliminated with a null-space basis, the remaining
    problem is reduced to a least-distance program  min ||z|| s.t. E z >= f,
    and that is solved exactly with non-negative least squares. `x0` is
    accepted for compatibility and ignored (the method needs no starting point).
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = A.shape[1]
    G = _rows(G, n)
    h = np.zeros(0) if h is None else np.asarray(h, dtype=np.float64).reshape(-1)
    C = _rows(C, n)
    d = np.zeros(0) if d is None else np.asarray(d, dtype=np.float64).reshape(-1)
    lb = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=np.float64).reshape(n)
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=np.float64).reshape(n)
    if G.shape[0] != h.size or C.shape[0] != d.size or A.shape[0] != b.size:
        raise ValueError("A/b, G/h or C/d sizes do not match")
    if np.any(lb > ub + tol):
        k = int(np.argmax(lb - ub))
        return LsqResult(np.clip(np.zeros(n), np.minimum(lb, ub), np.maximum(lb, ub)), False,
                         f"infeasible bounds: lb[{k}]={lb[k]:.4g} > ub[{k}]={ub[k]:.4g}", np.inf, float(lb[k] - ub[k]))

    # bounds become inequality rows
    eye = np.eye(n)
    fin_u, fin_l = np.isfinite(ub), np.isfinite(lb)
    Gall = np.vstack([G, eye[fin_u], -eye[fin_l]])
    hall = np.concatenate([h, ub[fin_u], -lb[fin_l]])

    # scale variables (x = s * y) so mm and radians are treated evenly
    col = np.linalg.norm(np.vstack([A, Gall, C]), axis=0)
    s = np.where(col > 1e-12, 1.0 / np.maximum(col, 1e-12), 1.0)
    As, Gs, Cs = A * s, Gall * s, C * s

    # equality constraints: y = y_p + N w
    if Cs.shape[0]:
        U, sv, Vt = np.linalg.svd(Cs)
        rank = int(np.sum(sv > 1e-10 * max(sv.max(), 1e-300)))
        y_p = np.linalg.lstsq(Cs, d, rcond=None)[0]
        N = Vt[rank:].T
        if np.linalg.norm(Cs @ y_p - d) > 1e-8 * (1 + np.linalg.norm(d)):
            return LsqResult(s * y_p, False, "inconsistent equality constraints", np.inf,
                             float(np.linalg.norm(Cs @ y_p - d)))
    else:
        y_p, N = np.zeros(n), np.eye(n)
    k = N.shape[1]
    Ar, br = As @ N, b - As @ y_p
    Gr, hr = Gs @ N, hall - Gs @ y_p

    # regularize a rank-deficient objective very slightly so R below is invertible
    Q, R = np.linalg.qr(np.vstack([Ar, 1e-10 * max(1.0, np.abs(Ar).max()) * np.eye(k)]))
    br_ext = np.concatenate([br, np.zeros(k)])
    c = Q.T @ br_ext                                  # unconstrained optimum: w = R^-1 c

    if Gr.shape[0] == 0:
        w = np.linalg.solve(R, c)
        status_ok, msg = True, "unconstrained optimum"
    else:
        # LDP: with z = R w - c, minimize ||z|| s.t. E z >= f, where E = -Gr R^-1, f = Gr R^-1 c - hr
        Rinv = np.linalg.inv(R)
        E = -Gr @ Rinv
        f = Gr @ (Rinv @ c) - hr
        # scale rows so each constraint has unit norm (the LDP solution does not change)
        rn = np.linalg.norm(E, axis=1)
        keep = rn > 1e-14
        if np.any(~keep) and np.any(f[~keep] > tol):
            return LsqResult(s * y_p, False, "infeasible: a constraint row is zero but requires > 0", np.inf,
                             float(np.max(f[~keep])))
        E, f = E[keep] / rn[keep, None], f[keep] / rn[keep]
        M = np.vstack([E.T, f[None, :]])
        e = np.zeros(k + 1)
        e[-1] = 1.0
        u, _ = nnls(M, e, maxiter=50 * M.shape[1] + 100)
        r = M @ u - e
        if abs(r[-1]) < 1e-12:
            return LsqResult(s * y_p, False, "infeasible constraints (LDP has no solution)", np.inf, np.inf)
        z = -r[:k] / r[-1]
        w = np.linalg.solve(R, z + c)
        status_ok, msg = True, "optimal"

    x = s * (y_p + N @ w)
    viol = max([0.0]
               + ([float(np.max(G @ x - h))] if G.shape[0] else [])
               + ([float(np.max(np.abs(C @ x - d)))] if C.shape[0] else [])
               + [float(np.max(lb - x)), float(np.max(x - ub))])
    scale = 1.0 + float(np.max(np.abs(np.concatenate([hall, d, [0.0]]))))
    ok = status_ok and viol <= 1e-7 * scale
    if not ok and status_ok:
        msg = f"constraint violation {viol:.3g} after solve"
    res = A @ x - b
    return LsqResult(x, ok, msg, float(res @ res), viol)


@dataclass
class LsqProblem:
    """Accumulates weighted objective terms and constraints for solve_lsqlin()."""
    n: int
    _A: List[np.ndarray] = field(default_factory=list)
    _b: List[np.ndarray] = field(default_factory=list)
    _G: List[np.ndarray] = field(default_factory=list)
    _h: List[np.ndarray] = field(default_factory=list)
    _C: List[np.ndarray] = field(default_factory=list)
    _d: List[np.ndarray] = field(default_factory=list)
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None
    names: List[Tuple[str, str]] = field(default_factory=list)

    def add_objective(self, A, b, weight: float = 1.0, name: str = ""):
        """Adds weight * ||A x - b||^2 (weight is applied as sqrt(weight) on both sides)."""
        A = np.atleast_2d(np.asarray(A, dtype=np.float64))
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if A.shape != (b.size, self.n):
            raise ValueError(f"objective '{name}': A is {A.shape}, b has {b.size} entries, n={self.n}")
        w = np.sqrt(float(weight))
        self._A.append(w * A)
        self._b.append(w * b)
        self.names.append(("objective", name))
        return self

    def add_inequality(self, G, h, name: str = ""):
        """Adds G x <= h."""
        G = np.atleast_2d(np.asarray(G, dtype=np.float64)).reshape(-1, self.n)
        self._G.append(G)
        self._h.append(np.asarray(h, dtype=np.float64).reshape(-1))
        self.names.append(("inequality", name))
        return self

    def add_equality(self, C, d, name: str = ""):
        """Adds C x == d."""
        C = np.atleast_2d(np.asarray(C, dtype=np.float64)).reshape(-1, self.n)
        self._C.append(C)
        self._d.append(np.asarray(d, dtype=np.float64).reshape(-1))
        self.names.append(("equality", name))
        return self

    def set_bounds(self, lb, ub):
        """lb <= x <= ub (intersected with any bounds already set)."""
        lb = np.asarray(lb, dtype=np.float64).reshape(self.n)
        ub = np.asarray(ub, dtype=np.float64).reshape(self.n)
        self.lb = lb if self.lb is None else np.maximum(self.lb, lb)
        self.ub = ub if self.ub is None else np.minimum(self.ub, ub)
        return self

    def solve(self, x0=None) -> LsqResult:
        if not self._A:
            raise ValueError("LsqProblem.solve(): no objective terms")
        cat = lambda L, shape: np.concatenate(L) if L else np.zeros(shape)
        return solve_lsqlin(np.vstack(self._A), cat(self._b, 0),
                            G=np.vstack(self._G) if self._G else None, h=cat(self._h, 0) if self._G else None,
                            C=np.vstack(self._C) if self._C else None, d=cat(self._d, 0) if self._C else None,
                            lb=self.lb, ub=self.ub, x0=x0)
