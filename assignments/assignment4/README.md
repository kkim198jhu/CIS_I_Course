# CIS I — Assignment 4: Robot Virtual Fixtures

You will write the midlevel controller for a surgical robot. It has two phases:

1. An autonomous, coordinated approach move to the entrance of a tube.
2. Cooperative (admittance) control down the tube. Signed-distance-function (SDF) virtual fixtures
   keep the tool tip clear of critical anatomy and stop it just short of the stopper at the bottom.

The notebook drives everything. Your code lives in Python modules under `src/student/`, which
are unit-tested with `pytest`.

## What's in this folder

```
assignment4/
├── requirements.txt
├── src/
│   ├── simulation/        -- the robot (ReadState / SendCommand), simulated hand + force
│   │                         sensor, anatomy (Tube / Donut / Sphere / HalfPlane), constants (don't edit)
│   ├── tools/             -- provided helpers (don't edit):
│   │                           frames.py         small-motion frame math (skew, rot_log, pose_error_body, ...)
│   │                           solver.py         constrained least squares: LsqProblem / solve_lsqlin
│   │                           logger.py         TrajectoryLog (the required time trace)
│   │                           evaluation.py     the checks the graders run
│   │                           visualization.py  plots
│   └── student/           <- YOUR code (stubs to fill in, see below)
├── tests/
│   ├── test_simulation.py -- provided, passes as shipped; read it as an example
│   └── test_student.py    -- tests for your code
└── notebooks/
    └── assignment4_template.ipynb   <- start here
```

The robot is the same Cartesian-base, spherical-wrist robot as assignment 2. Its kinematics are
imported from `../assignment2/src/robot.py`, and the frame types come from the repository's
`data_types` package, so keep the whole course repository checked out, not just this folder.
The handout's function names map to snake_case: `ReadState` → `robot.read_state()`,
`SendCommand` → `robot.send_command()`, `MoveToolTo` → `approach.move_tool_to()`,
`CooperativeControlMode` → `cooperative.cooperative_control_mode()`.

## What you write (`src/student/`)

Each file's docstring explains the inputs, outputs and conventions. Suggested order:

| file | functions |
|---|---|
| `jacobian.py` | `jacobian_RW(q)`, `jacobian_RT(q, F_WT)` |
| `wrench.py` | `resolve_wrench_to_tool`, `wrench_in_robot_frame` |
| `virtual_fixtures.py` | `joint_motion_bounds`, `structure_constraints`, `centering_bias` |
| `approach.py` | `interpolate_pose`, `plan_approach`, `motion_fraction`, `approach_step` |
| `gains.py` | `kappa_free_space`, `speed_limit`, `kappa_structure`, `kappa_combined` |
| `cooperative.py` | `desired_tip_motion`, `cooperative_step` |

The control loops at the bottom of `approach.py` and `cooperative.py` are the handout's
pseudocode. They are provided and call your functions. In the notebook you also write up your
formulation, your κ design and its rationale, and any AI prompts you used.

## Setup

You need **Python 3.9 or newer**. From the `assignment4/` folder:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
jupyter notebook notebooks/assignment4_template.ipynb
pytest tests/test_simulation.py                 # should pass as shipped
pytest tests/test_student.py -m "not slow"      # your unit tests
pytest tests/test_student.py                    # + full simulated runs (~20 s)
```

## Conventions and clarifications to the handout

These are also documented in `src/simulation/constants.py`:

- **Units:** mm, radians, seconds, newtons.
- **Wrench order:** `zeta = [tau; phi]`, torques first.
- **Jacobians** follow the handout: `F(q + dq) = F(q) [I + sk(alpha), epsilon]`, so they are body Jacobians.
- **`Rot(rx, ry, rz)` in `F_init`** uses the ZYX convention: `R = Rz(rz) Ry(ry) Rx(rx)`.
- **`d_range`** is 10 mm for every structure.
- **Donut distance:** the closest point on the core circle is `[a_x, a_y, 0]`. The handout's
  `[a_x, a_y, x_z]` describes an infinite cylinder.
- **Hand force:** `phi_R(t) = -phi_max Rz(w_z t) Rx(tilt(t)) z` with `tilt(t) = 0.35 sin(w_x t)`.
  Taken literally, the handout's `Rx(w_x t)` would point the force straight up after 20 s.
- **Wrench transform:** the 6x6 matrix printed in the handout has a typo. Derive the transform yourself.
- **Joint 2 limit:** at `F_start`, joint 2 sits exactly on its +300 mm travel limit, so the tip
  cannot move toward +y of the robot while inside the tube.
- **`SendCommand` limit order:** it applies velocity, then acceleration, then travel limits. The
  travel limit acts as a hard stop, so running into it at speed shows up as an acceleration violation.
