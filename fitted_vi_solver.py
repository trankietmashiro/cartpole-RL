"""
vi_solver.py
============
Fitted value iteration.

Each outer iteration:
  targets(s) = min_a [ r(s, a)  +  γ · V_old(s') ]

Convergence:
  max_s | targets(s) − V_old(s) | < theta
"""

import torch
from dp_solver import DPSolver


class ValueIterationSolver(DPSolver):
    """
    Subclass of DPSolver that implements the value-iteration target:

        y(s) = min_a [ r(s, a) + γ · V(s') ]

    Convergence is measured by the Bellman residual (the max absolute
    difference between the new targets and the current value estimates)
    before fitting begins each iteration.
    """

    # VI needs no extra state — _pre_solve_hook is inherited as a no-op.

    def _compute_targets(
        self, transitions: dict
    ) -> tuple[torch.Tensor, bool]:
        """
        Stack Q(s, a) for every action into an (A, N) matrix, then take
        the row-wise max to get the Bellman optimality targets.

        Also computes the Bellman residual for convergence checking.
        """
        x_s        = transitions["x_s"]        # (N, D)
        s2_tensors = transitions["s2_tensors"]  # list of (N, D), one per action
        r_tensors  = transitions["r_tensors"]   # list of (N,),   one per action

        self.V_net.eval()
        with torch.no_grad():
            # Build Q-value matrix: entry [a, n] = r(s_n, a) + γ·V(s'_n)
            q_values = torch.stack(
                [r_tensors[ai] + self.gamma * self.V_net(s2_tensors[ai])
                 for ai in range(len(self.actions))],
                dim=0,
            )                                   # (A, N)

            # Bellman optimality targets: best Q over actions for each state
            # Reward is a cost (always >= 0), so we MINIMISE not maximise
            targets, _ = q_values.min(dim=0)   # (N,)

            # Bellman residual: max absolute change in value estimates
            v_old  = self.V_net(x_s)           # (N,)
            delta = (targets - v_old).abs().mean().item()

        print(f"  Bellman residual = {delta:.6f}  (threshold = {self.theta:.6f})")
        return targets, delta < self.theta

    def _post_fit_hook(self, transitions: dict) -> bool:
        # VI convergence is checked in _compute_targets before fitting.
        # Nothing to do after the fit.
        return False
