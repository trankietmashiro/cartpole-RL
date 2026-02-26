"""
cartpole_run.py
===============
Entry point for running VI or PI on the CartPole system.

Usage
-----
    python cartpole_run.py --algo vi
    python cartpole_run.py --algo pi
    python cartpole_run.py --algo vi --retrain
    python cartpole_run.py --algo vi --model-path models/my_vi.pth

Adding a new system
-------------------
1.  Write a dynamics function: f(state, action) -> state_dot
2.  Create a Discretizer for that system's state space
3.  Build a solver with those components — everything else is automatic:

        solver = ValueIterationSolver(
            disc                   = my_disc,
            actions                = my_actions,
            dynamics_fn            = my_dynamics,
            discrete_transition_fn = discrete_transition,
            target_state           = my_target,
            Q=my_Q, R=my_R, dt=my_dt, ...
        )
        V_net = solver.solve(...)
"""

import argparse
import numpy as np
from cartpole import dynamics, Discretizer, discrete_transition

from dp_solver  import DPSolver
from fitted_vi_solver  import ValueIterationSolver
from approx_pi_solver  import PolicyIterationSolver


# ──────────────────────────────────────────────────────────────────────────────
# Solver factory
# ──────────────────────────────────────────────────────────────────────────────

SOLVERS = {
    "vi": ValueIterationSolver,
    "pi": PolicyIterationSolver,
}


def build_cartpole_solver(algorithm: str) -> DPSolver:
    """Instantiate the requested solver wired up to the CartPole system."""
    disc = Discretizer(
        lows= [-1.0, -2.0, -np.pi, -2.0],
        highs=[ 1.0,  2.0,  np.pi,  2.0],
        n_bins=[21, 21, 21, 21],
    )
    actions = np.linspace(-2.0, 2.0, 11).tolist()
    xd      = np.zeros(4)          # target: upright, centred, stationary
    Q       = np.diag([5.0, 0.5, 5.0, 0.5])
    R       = 0.01

    return SOLVERS[algorithm](
        disc                   = disc,
        actions                = actions,
        dynamics_fn            = dynamics,
        discrete_transition_fn = discrete_transition,
        target_state           = xd,
        Q                      = Q,
        R                      = R,
        dt                     = 0.1,
        method                 = "rk4",
        gamma                  = 0.99,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole DP solver (VI or PI).")
    parser.add_argument(
        "--algo", choices=["vi", "pi"], default="vi",
        help="Algorithm: 'vi' = value iteration, 'pi' = policy iteration.",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Ignore any saved checkpoint and retrain from scratch.",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Override the default checkpoint path.",
    )
    args = parser.parse_args()

    model_path = args.model_path or f"models/nn_{args.algo}_weights.pth"

    # ── Build solver ──────────────────────────────────────────────────────────
    solver = build_cartpole_solver(args.algo)

    # ── Load or train ─────────────────────────────────────────────────────────
    loaded = not args.retrain and solver.load_checkpoint(model_path)

    if not loaded:
        reason = "--retrain flag" if args.retrain else "no checkpoint found"
        print(f"[main] Training from scratch ({reason}).")
        solver.solve(
            hidden     = 64,
            lr         = 1e-3,
            fit_epochs = 150,
            fit_batch  = 256,
            iterations = 50,
            theta      = 1e-4,
        )
        solver.save_checkpoint(model_path)

    # ── Analysis & plot ───────────────────────────────────────────────────────
    solver.tie_analysis()

    solver.plot_policy(
        dim_x     = 0,              # cart position  on y-axis
        dim_theta = 2,              # pole angle     on x-axis
        xlabel    = "Pole angle θ",
        ylabel    = "Cart position x",
        title     = f"CartPole {args.algo.upper()} — greedy policy",
        save_path = f"figures/{args.algo}_policy.png",
    )
