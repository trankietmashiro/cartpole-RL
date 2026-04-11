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

States are sampled uniformly from the continuous state space rather than
enumerated from a discrete grid.  This is the natural fit for fitted VI / PI
where V_net is a neural network that generalises across the continuous space.

Adding a new system
-------------------
1.  Write a dynamics function: f(state, action) -> state_dot
2.  Write a continuous_transition function: (dyn, x, a, xd, Q, R, dt,
        lows, highs, method) -> (x_next, r)
3.  Build a solver with lows, highs, n_samples — no Discretizer needed:

        solver = ValueIterationSolver(
            lows                     = my_lows,
            highs                    = my_highs,
            n_samples                = 20_000,
            actions                  = my_actions,
            dynamics_fn              = my_dynamics,
            continuous_transition_fn = my_continuous_transition,
            target_state             = my_target,
            Q=my_Q, R=my_R, dt=my_dt, ...
        )
        V_net = solver.solve(...)
"""

import argparse
import numpy as np
from cartpole import dynamics, continuous_transition

from dp_solver         import DPSolver
from fitted_vi_solver  import ValueIterationSolver
from approx_pi_solver  import PolicyIterationSolver


# ──────────────────────────────────────────────────────────────────────────────
# Solver factory
# ──────────────────────────────────────────────────────────────────────────────

SOLVERS = {
    "vi": ValueIterationSolver,
    "pi": PolicyIterationSolver,
}


def build_cartpole_solver(algorithm: str, n_samples: int = 20_000) -> DPSolver:
    """Instantiate the requested solver wired up to the CartPole system."""
    lows    = [-5.0, -10.0, -np.pi, -20.0]
    highs   = [ 5.0,  10.0,  np.pi,  20.0]
    actions = np.linspace(-20.0, 20.0, 41).tolist()
    xd      = np.zeros(4)                       # target: upright, centred, stationary
    Q       = np.diag([20.0, 0, 20.0, 0])
    R       = 0

    return SOLVERS[algorithm](
        lows                     = lows,
        highs                    = highs,
        n_samples                = n_samples,
        actions                  = actions,
        dynamics_fn              = dynamics,
        continuous_transition_fn = continuous_transition,
        target_state             = xd,
        Q                        = Q,
        R                        = R,
        dt                       = 0.1,
        method                   = "rk4",
        gamma                    = 0.99,
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
    parser.add_argument(
        "--n-samples", type=int, default=100_000,
        help="Number of states to sample for training (default: 20000).",
    )
    args = parser.parse_args()

    model_path = args.model_path or f"models/nn_{args.algo}_weights.pth"

    # ── Build solver ──────────────────────────────────────────────────────────
    solver = build_cartpole_solver(args.algo, n_samples=args.n_samples)

    # ── Load or train ─────────────────────────────────────────────────────────
    loaded = not args.retrain and solver.load_checkpoint(model_path)

    if not loaded:
        reason = "--retrain flag" if args.retrain else "no checkpoint found"
        print(f"[main] Training from scratch ({reason}).")
        solver.solve(
            hidden     = 256,
            lr         = 1e-3,
            fit_epochs = 150,
            fit_batch  = 256,
            iterations = 50,
            theta      = 1e-4,
            cache_path = f"cache/{args.algo}_transitions.npz",
        )
        solver.save_checkpoint(model_path)

    # ── Analysis & plot ───────────────────────────────────────────────────────
    solver.tie_analysis()

    solver.plot_policy(
        dim_x     = 0,              # cart position  on y-axis
        dim_theta = 2,              # pole angle     on x-axis
        n_grid    = 30,
        xlabel    = "Pole angle θ",
        ylabel    = "Cart position x",
        title     = f"CartPole {args.algo.upper()} — greedy policy",
        save_path = f"figures/{args.algo}_policy.png",
    )
