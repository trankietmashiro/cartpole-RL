"""
fitted_q_solver.py
==================
Fitted Q-Iteration (FQI) with cost minimization.

Each outer iteration:
  targets(s, a) = c(s, a)  +  γ · min_a' Q_old(s', a')

Convergence:
  max_{s,a} | targets(s, a) − Q_old(s, a) | < theta

Key difference from VI
----------------------
VI learns V(s) → scalar with a state-only network.
FQI learns Q(s, a) → scalar with a (state ∥ action) network (input_dim = D + 1).
Training data has N × A rows (one per state–action pair) rather than N rows.

The base DPSolver.solve() hardcodes both input_dim = D and the (N, D) training
tensor, so this class overrides solve() to fix those two things.  Everything
else — _precompute_transitions, _fit_net, checkpointing, greedy_action,
plot_policy, tie_analysis — is reused from the base class unchanged, except
greedy_action which must use argmin Q(s, a) instead of argmax.
"""

import numpy as np
import torch
import torch.optim as optim

from dp_solver import DPSolver, NeuralNet


class FittedQSolver(DPSolver):
    """
    Subclass of DPSolver that implements Fitted Q-Iteration.

    The Q-network takes a concatenated (state, action) vector as input:

        Q_net : R^{D+1} → R

    Training targets (Bellman optimality for costs):

        y(s, a) = c(s, a) + γ · min_a' Q(s', a')

    The solve() loop is overridden to:
      1. Build Q_net with input_dim = D + 1 instead of D.
      2. Pre-build state–action tensors sa_tensors (list of (N, D+1), one per action).
      3. Pass sa_tensors into the augmented transitions dict so that
         _compute_targets and _fit_net receive the right inputs.
    """

    # ── Override solve() to wire up the Q-network ─────────────────────────────

    def solve(
        self,
        hidden      : int   = 64,
        lr          : float = 1e-3,
        fit_epochs  : int   = 100,
        fit_batch   : int   = 256,
        iterations  : int   = 50,
        theta       : float = 1e-4,
        cache_path  : str   = None,
    ) -> NeuralNet:
        """
        Identical structure to DPSolver.solve() except:
          - V_net (reused as Q_net) has input_dim = D + 1
          - transitions dict is augmented with sa_tensors (list of (N, D+1))
          - _fit_net is called with sa_flat (N*A, D+1) rather than x_s (N, D)
        """
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta     = theta
        print(f"[{self.__class__.__name__}] Using device: {self.device}")

        D = len(self.lows)
        # Q_net input = [state (D) ∥ action (1)]
        self.V_net     = NeuralNet(input_dim=D + 1, hidden=hidden).to(self.device)
        self.optimizer = optim.Adam(self.V_net.parameters(), lr=lr)

        self._pre_solve_hook()
        transitions = self._precompute_transitions(cache_path=cache_path)

        # ── Pre-build (s, a) tensors for every action ─────────────────────────
        # sa_tensors[ai]  : (N, D+1) — concat(state, scalar action ai)
        # sa2_tensors[ai] : list of (N, D+1) per next-action aj
        #                   used inside _compute_targets to evaluate min_a' Q(s',a')
        x_s = transitions["x_s"]   # (N, D)
        N   = x_s.shape[0]
        A   = len(self.actions)

        sa_tensors = []
        for a in self.actions:
            a_col = torch.full((N, 1), float(a), dtype=torch.float32, device=self.device)
            sa_tensors.append(torch.cat([x_s, a_col], dim=1))   # (N, D+1)

        # For each current action ai, build all next-state–action tensors:
        # sa2_tensors[ai][aj] = concat(s2_tensors[ai], a_col_j)  (N, D+1)
        sa2_tensors = []
        for ai in range(A):
            s2 = transitions["s2_tensors"][ai]   # (N, D)
            sa2_per_aj = []
            for a in self.actions:
                a_col = torch.full((N, 1), float(a), dtype=torch.float32, device=self.device)
                sa2_per_aj.append(torch.cat([s2, a_col], dim=1))   # (N, D+1)
            sa2_tensors.append(sa2_per_aj)

        transitions["sa_tensors"]  = sa_tensors   # list[A] of (N, D+1)
        transitions["sa2_tensors"] = sa2_tensors  # list[A][A] of (N, D+1)

        # Flat (N*A, D+1) training inputs — same ordering as targets in
        # _compute_targets (actions as outer loop, states as inner)
        sa_flat = torch.cat(sa_tensors, dim=0)   # (N*A, D+1)

        # ── Main iteration loop ───────────────────────────────────────────────
        for i in range(iterations):
            print(f"\n=== {self.__class__.__name__} iter {i} ===")

            targets, pre_converged = self._compute_targets(transitions)
            # targets shape: (N*A,) — fit Q_net on all state–action pairs
            self._fit_net(sa_flat, targets, fit_epochs, fit_batch)
            post_converged = self._post_fit_hook(transitions)

            if pre_converged or post_converged:
                print("Converged — stopping early.")
                break

        return self.V_net

    # ── Bellman targets ───────────────────────────────────────────────────────

    def _compute_targets(
        self, transitions: dict
    ) -> tuple[torch.Tensor, bool]:
        """
        For every (state, action) pair compute:

            y(s, a) = c(s, a) + γ · min_a' Q(s', a')

        where s' = f(s, a) and the min is over the discrete action set.

        Returns
        -------
        targets    : (N*A,) tensor — flattened in action-major order
        converged  : bool          – True when Bellman residual < theta
        """
        sa_tensors  = transitions["sa_tensors"]   # list[A] of (N, D+1)
        sa2_tensors = transitions["sa2_tensors"]  # list[A][A] of (N, D+1)
        r_tensors   = transitions["r_tensors"]    # list[A] of (N,)

        A = len(self.actions)

        self.V_net.eval()
        with torch.no_grad():
            targets_per_action = []
            for ai in range(A):
                # min_a' Q(s'_{s,ai}, a') — shape (N,)
                # Stack Q over all next actions: (A, N), then take row-wise min
                q_next = torch.stack(
                    [self.V_net(sa2_tensors[ai][aj]) for aj in range(A)],
                    dim=0,
                ).min(dim=0).values   # (N,)

                target_ai = r_tensors[ai] + self.gamma * q_next   # (N,)
                targets_per_action.append(target_ai)

            targets = torch.cat(targets_per_action, dim=0)   # (N*A,)

            # Bellman residual: max absolute difference across all (s, a) pairs
            sa_flat = torch.cat(sa_tensors, dim=0)   # (N*A, D+1)
            q_old   = self.V_net(sa_flat)             # (N*A,)
            delta   = (targets - q_old).abs().max().item()

        print(f"  Bellman residual = {delta:.6f}  (threshold = {self.theta:.6f})")
        return targets, delta < self.theta

    def _post_fit_hook(self, transitions: dict) -> bool:
        # FQI convergence is fully handled in _compute_targets (pre-fit check).
        return False

    # ── Greedy action: argmin_a Q(s, a) ──────────────────────────────────────

    def greedy_action(self, x: np.ndarray) -> float:
        """
        Return argmin_a Q(x, a) for a continuous state.

        Overrides the base-class argmax (reward maximisation) with argmin
        (cost minimisation) and uses Q_net's (s ∥ a) input format.
        """
        self.V_net.eval()
        best_a, best_q = None, np.inf
        with torch.no_grad():
            for a in self.actions:
                sa = torch.tensor(
                    np.append(x, float(a)).astype(np.float32),
                    device=self.device,
                ).unsqueeze(0)   # (1, D+1)
                q = self.V_net(sa).item()
                if q < best_q:
                    best_q, best_a = q, a
        return best_a
