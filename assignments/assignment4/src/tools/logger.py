"""
TrajectoryLog (provided): the "Output [t, q, q_dot, F_RT, D(S_1..S_N)] to log"
lines of the handout's MoveToolTo / CooperativeControlMode. This log is the
required program output: a time trace of the tool pose and the distances to
the anatomy.
"""

import csv
from typing import List, Optional, Sequence

import numpy as np

from data_types.nominal_types import Frame


class TrajectoryLog:
    def __init__(self, structure_names: Optional[Sequence[str]] = None):
        self.structure_names = list(structure_names) if structure_names is not None else []
        self._rows: List[dict] = []

    def __len__(self):
        return len(self._rows)

    def __repr__(self):
        phases = sorted(set(self.phase_labels))
        return f"TrajectoryLog({len(self)} samples, phases={phases})"

    def record(self, t: float, q, qdot, F_RT: Frame, distances=None, phase: str = "", rc: int = 0):
        R = np.array(F_RT.R.matrix, dtype=np.float64)
        p = np.array(F_RT.p.vec.ravel(), dtype=np.float64)
        m = len(self.structure_names)
        d = np.full(m, np.nan) if distances is None else np.asarray(distances, dtype=np.float64).reshape(-1)
        if m == 0 and distances is not None:
            self.structure_names = [f"S{k + 1}" for k in range(d.size)]
            m = d.size
            for row in self._rows:          # back-fill earlier rows recorded without distances
                row["d"] = np.full(m, np.nan)
        if d.size != m:
            raise ValueError(f"expected {m} distances, got {d.size}")
        self._rows.append(dict(t=float(t), q=np.array(q, dtype=np.float64).reshape(6),
                               qdot=np.array(qdot, dtype=np.float64).reshape(6), R=R, p=p, d=d,
                               phase=str(phase), rc=int(rc)))

    # ------------------------------------------------------------ array views
    def _stack(self, key):
        if not self._rows:
            return np.zeros((0,))
        return np.array([r[key] for r in self._rows])

    @property
    def t(self): return self._stack("t")
    @property
    def q(self): return self._stack("q")
    @property
    def qdot(self): return self._stack("qdot")
    @property
    def p(self): return self._stack("p")
    @property
    def R(self): return self._stack("R")
    @property
    def d(self): return self._stack("d")
    @property
    def rc(self): return self._stack("rc")
    @property
    def phase_labels(self): return [r["phase"] for r in self._rows]

    def phase(self, name: str) -> "TrajectoryLog":
        """The samples recorded with phase == name, as a new log."""
        out = TrajectoryLog(self.structure_names)
        out._rows = [r for r in self._rows if r["phase"] == name]
        return out

    def to_csv(self, path):
        names = self.structure_names
        header = (["t", "phase", "rc"] + [f"q{k}" for k in range(1, 7)] + [f"qdot{k}" for k in range(1, 7)]
                  + ["px", "py", "pz"] + [f"R{i}{j}" for i in range(1, 4) for j in range(1, 4)]
                  + [f"D[{n}]" for n in names])
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for r in self._rows:
                w.writerow([f"{r['t']:.4f}", r["phase"], r["rc"]] + [f"{v:.6g}" for v in r["q"]]
                           + [f"{v:.6g}" for v in r["qdot"]] + [f"{v:.6f}" for v in r["p"]]
                           + [f"{v:.8f}" for v in r["R"].ravel()] + [f"{v:.6f}" for v in r["d"]])
