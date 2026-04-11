"""
approx_pi_solver.py
===================
Neural-network policy iteration.

Each outer iteration:
  Evaluation  – targets(s) = r(s, π(s)) + γ · V(s')
              – fit V_net to those targets
  Improvement – π(s) ← argmax_a [ r(s, a) + γ · V(s') ]

Convergence:
  Policy did not change after an improvement step.
"""

from collections import Counter

import numpy as np
import torch
from dp_solver import DPSolver


class PolicyIterationSolver(DPSolver):
    """
    Subclass of DPSolver that implements policy iteration.

    The key difference from VI:
      - _compute_targets  evaluates only the *current* policy (one action per state)
      - _post_fit_hook    improves the policy greedily and checks for stability

    Policy is stored as a dict keyed by integer state index (0 … N-1) since
    states are now continuous samples rather than hashable grid tuples.
    """

    def _pre_solve_hook(self) -> None:
        """Initialise a uniform policy: every state maps to the middle action."""
        mid_action  = self.actions[len(self.actions) // 2]
        self.policy = {i: mid_action for i in range(len(self.states))}
        print(f"[PI] Policy initialised with action = {mid_action:.2f} for all states.")

    # ── Policy evaluation targets ─────────────────────────────────────────────

    def _compute_targets(
        self, transitions: dict
    ) -> tuple[torch.Tensor, bool]:
        """
        For each state index i, look up the policy action π(i) and compute:

            y(i) = r(s_i, π(i)) + γ · V(s'_i)

        Uses advanced tensor indexing to select the right action for each
        state in a single vectorised operation (no Python loop over states).
        """
        s2_tensors = transitions["s2_tensors"]   # list of (N, D), one per action
        r_tensors  = transitions["r_tensors"]    # list of (N,),   one per action

        # Map each state index to the index of its policy action
        action_to_idx = {a: i for i, a in enumerate(self.actions)}
        N      = len(self.states)
        pi_idx = torch.tensor(
            [action_to_idx[self.policy[i]] for i in range(N)],
            dtype=torch.long, device=self.device,
        )  # (N,)

        state_idx = torch.arange(N, device=self.device)   # (N,)

        self.V_net.eval()
        with torch.no_grad():
            # Stack value and reward tensors into (A, N) matrices
            v_next_all = torch.stack(
                [self.V_net(s2_tensors[ai]) for ai in range(len(self.actions))],
                dim=0,
            )   # (A, N)
            r_all = torch.stack(r_tensors, dim=0)   # (A, N)

            # Advanced indexing: for state n pick row pi_idx[n]
            v_next_pi = v_next_all[pi_idx, state_idx]   # (N,)
            r_pi      = r_all     [pi_idx, state_idx]   # (N,)

            targets = r_pi + self.gamma * v_next_pi     # (N,)

        # Convergence for PI is checked after improvement, not here
        return targets, False

    # ── Policy improvement ────────────────────────────────────────────────────

    def _post_fit_hook(self, transitions: dict) -> bool:
        """
        Greedy policy improvement:
            π(i) ← argmax_a [ r(s_i, a) + γ · V(s'_i) ]

        Returns True if the policy did not change (stable → converged).
        """
        s2_tensors = transitions["s2_tensors"]
        r_tensors  = transitions["r_tensors"]

        self.V_net.eval()
        with torch.no_grad():
            # Q-value matrix (A, N)
            q_values = torch.stack(
                [r_tensors[ai] + self.gamma * self.V_net(s2_tensors[ai])
                 for ai in range(len(self.actions))],
                dim=0,
            )
            best_idx = q_values.argmin(dim=0).cpu().numpy()   # (N,) action indices

        # Update policy and check stability
        stable = True
        N      = len(self.states)
        for i in range(N):
            new_a = self.actions[best_idx[i]]
            if new_a != self.policy[i]:
                stable = False
            self.policy[i] = new_a

        # Print action distribution for monitoring
        counts = Counter(self.policy.values())
        print("  Action distribution after improvement:")
        for a in self.actions:
            c = counts.get(a, 0)
            print(f"    {a:>5.1f}: {c:>6} states  ({100.0 * c / N:5.2f}%)")

        if stable:
            print("  Policy is stable.")
        return stable
