"""
pi_solver.py
============
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
    """

    def _pre_solve_hook(self) -> None:
        """Initialise a uniform policy: every state maps to the middle action."""
        mid_action   = self.actions[len(self.actions) // 2]
        self.policy  = {s: mid_action for s in self.states}
        print(f"[PI] Policy initialised with action = {mid_action:.2f} for all states.")

    # ── Policy evaluation targets ─────────────────────────────────────────────

    def _compute_targets(
        self, transitions: dict
    ) -> tuple[torch.Tensor, bool]:
        """
        For each state s, look up the policy action π(s) and compute:

            y(s) = r(s, π(s)) + γ · V(s')

        Uses advanced tensor indexing to select the right action for each
        state in a single vectorised operation (no Python loop over states).
        """
        s2_tensors = transitions["s2_tensors"]   # list of (N, D), one per action
        r_tensors  = transitions["r_tensors"]    # list of (N,),   one per action

        # Map each state to the index of its policy action
        action_to_idx = {a: i for i, a in enumerate(self.actions)}
        pi_idx = torch.tensor(
            [action_to_idx[self.policy[s]] for s in self.states],
            dtype=torch.long, device=self.device,
        )  # (N,)

        N          = len(self.states)
        state_idx  = torch.arange(N, device=self.device)   # (N,)

        self.V_net.eval()
        with torch.no_grad():
            # Stack value and reward tensors into (A, N) matrices
            v_next_all = torch.stack(
                [self.V_net(s2_tensors[ai]) for ai in range(len(self.actions))],
                dim=0,
            )   # (A, N)
            r_all = torch.stack(r_tensors, dim=0)   # (A, N)

            # Advanced indexing: for state n pick row pi_idx[n]
            # v_next_all[pi_idx, state_idx] gives (N,) with each element
            # being V(s') under the current policy action for that state.
            v_next_pi = v_next_all[pi_idx, state_idx]   # (N,)
            r_pi      = r_all     [pi_idx, state_idx]   # (N,)

            targets = r_pi + self.gamma * v_next_pi     # (N,)

        # Convergence for PI is checked after improvement, not here
        return targets, False

    # ── Policy improvement ────────────────────────────────────────────────────

    def _post_fit_hook(self, transitions: dict) -> bool:
        """
        Greedy policy improvement:
            π(s) ← argmax_a [ r(s, a) + γ · V(s') ]

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
            best_idx = q_values.argmax(dim=0).cpu().numpy()   # (N,) action indices

        # Update policy and check stability
        stable = True
        for i, s in enumerate(self.states):
            new_a = self.actions[best_idx[i]]
            if new_a != self.policy[s]:
                stable = False
            self.policy[s] = new_a

        # Print action distribution for monitoring
        counts = Counter(self.policy.values())
        N      = len(self.policy)
        print("  Action distribution after improvement:")
        for a in self.actions:
            c = counts.get(a, 0)
            print(f"    {a:>5.1f}: {c:>6} states  ({100.0 * c / N:5.2f}%)")

        if stable:
            print("  Policy is stable.")
        return stable
