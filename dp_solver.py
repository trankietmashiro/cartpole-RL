"""
dp_solver.py
============
Abstract base class for neural-network dynamic-programming solvers.

Shared infrastructure (used by both VI and PI):
  - NeuralNet          – the value-function approximator
  - DPSolver           – ABC with the solve() template and all helpers
      · _enumerate_states / _state_to_continuous / _states_to_tensor
      · _precompute_transitions   (done once, reused every iteration)
      · _fit_net                  (regress V_net onto targets)
      · save/load checkpoint
      · greedy_action             (query V_net for best action)
      · plot_policy               (generic 2-D heatmap over any two dims)
      · tie_analysis

Subclasses must implement three small methods:
  _pre_solve_hook()              – initialise any extra state (e.g. PI policy dict)
  _compute_targets(transitions)  – return (targets tensor, converged bool)
  _post_fit_hook(transitions)    – called after fitting; return converged bool
                                   (PI uses this for policy improvement)
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.optim as optim


# ──────────────────────────────────────────────────────────────────────────────
# Value-function network  V_θ(s) → scalar
# ──────────────────────────────────────────────────────────────────────────────

class NeuralNet(nn.Module):
    """Small MLP: continuous state vector → scalar value estimate."""

    def __init__(self, input_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (batch,)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base solver
# ──────────────────────────────────────────────────────────────────────────────

class DPSolver(ABC):
    """
    Base class for neural-network DP solvers.

    Parameters
    ----------
    disc                   : Discretizer  – state-space grid
    actions                : list[float]  – discrete action set
    dynamics_fn            : callable     – continuous system dynamics  f(s, a)
    discrete_transition_fn : callable     – (dynamics, disc, s, a, ...) → (s', r)
    target_state           : np.ndarray   – desired equilibrium
    Q, R                   : cost matrices / scalar
    dt, method, gamma      : integration step, integrator name, discount factor
    """

    def __init__(
        self,
        disc,
        actions                : list,
        dynamics_fn,
        discrete_transition_fn,
        target_state           : np.ndarray,
        Q                      : np.ndarray,
        R,
        dt     : float = 0.02,
        method : str   = "rk4",
        gamma  : float = 0.99,
    ):
        self.disc                   = disc
        self.actions                = actions
        self.dynamics_fn            = dynamics_fn
        self.discrete_transition_fn = discrete_transition_fn
        self.target_state           = target_state
        self.Q                      = Q
        self.R                      = R
        self.dt                     = dt
        self.method                 = method
        self.gamma                  = gamma
        self.states                 = self._enumerate_states()

        # Set during solve()
        self.V_net    : NeuralNet | None       = None
        self.optimizer: optim.Optimizer | None = None
        self.device   : torch.device | None    = None
        self.theta    : float                  = 1e-4

    # ── State-space helpers ───────────────────────────────────────────────────

    def _enumerate_states(self) -> list:
        """All state index tuples from the discretizer grid."""
        grids = [range(n) for n in self.disc.n_bins.tolist()]
        return [
            tuple(idx)
            for idx in np.array(np.meshgrid(*grids, indexing="ij"))
                         .reshape(len(grids), -1).T
        ]

    def _state_to_continuous(self, s: tuple) -> np.ndarray:
        """Bin-centre coordinates for a state index tuple."""
        centres = []
        for dim, idx in enumerate(s):
            lo   = self.disc.lows[dim]
            hi   = self.disc.highs[dim]
            n    = self.disc.n_bins[dim]
            step = (hi - lo) / n
            centres.append(lo + (idx + 0.5) * step)
        return np.array(centres, dtype=np.float32)

    def _states_to_tensor(self, states: list) -> torch.Tensor:
        """Batch-convert a list of state tuples to a (N, D) float tensor."""
        arr = np.stack([self._state_to_continuous(s) for s in states], axis=0)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    # ── Transition pre-computation ────────────────────────────────────────────

    def _precompute_transitions(self) -> dict:
        """
        Compute every (state, action) transition once and cache as tensors.

        Returns a dict:
            "x_s"        : (N, D)  – current-state features
            "s2_tensors" : list of (N, D), one per action  – next-state features
            "r_tensors"  : list of (N,),   one per action  – immediate rewards
        """
        print("Pre-computing transitions …")
        s2_tensors, r_tensors = [], []

        for a in self.actions:
            s2_list, r_list = [], []
            for s in self.states:
                s2, r = self.discrete_transition_fn(
                    self.dynamics_fn, self.disc, s, a,
                    self.target_state, self.Q, self.R, self.dt,
                    method=self.method,
                )
                s2_list.append(s2)
                r_list.append(float(r))
            s2_tensors.append(self._states_to_tensor(s2_list))
            r_tensors.append(
                torch.tensor(r_list, dtype=torch.float32, device=self.device)
            )

        print(f"  {len(self.states)} states × {len(self.actions)} actions cached.")
        return {
            "x_s"        : self._states_to_tensor(self.states),
            "s2_tensors" : s2_tensors,
            "r_tensors"  : r_tensors,
        }

    # ── Shared regression loop ────────────────────────────────────────────────

    def _fit_net(
        self,
        x_s       : torch.Tensor,
        targets   : torch.Tensor,
        fit_epochs: int,
        fit_batch : int,
        loss_theta: float = 1e-6,
    ) -> None:
        """
        Fit V_net to regression targets by minimising MSE for up to
        `fit_epochs` epochs with mini-batch gradient descent.
        Stops early if the per-epoch loss change drops below `loss_theta`.
        """
        loss_fn   = nn.MSELoss()
        N         = x_s.shape[0]
        indices   = torch.arange(N, device=self.device)
        prev_loss = float("inf")

        self.V_net.train()
        for epoch in range(fit_epochs):
            perm          = indices[torch.randperm(N, device=self.device)]
            epoch_loss    = 0.0
            n_batches     = 0

            for start in range(0, N, fit_batch):
                idx    = perm[start : start + fit_batch]
                v_pred = self.V_net(x_s[idx])
                loss   = loss_fn(v_pred, targets[idx])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if abs(prev_loss - avg_loss) < loss_theta:
                print(f"    fit converged at epoch {epoch + 1}  loss={avg_loss:.6f}")
                return
            prev_loss = avg_loss

    # ── Abstract interface (subclasses implement these three methods) ──────────

    def _pre_solve_hook(self) -> None:
        """Initialise any iteration-specific state before the solve loop."""
        pass

    @abstractmethod
    def _compute_targets(
        self, transitions: dict
    ) -> tuple[torch.Tensor, bool]:
        """
        Compute regression targets for the current iteration.
        Returns (targets: Tensor shape (N,), converged: bool).
        Convergence can be checked here (VI) or left to _post_fit_hook (PI).
        """

    @abstractmethod
    def _post_fit_hook(self, transitions: dict) -> bool:
        """
        Called after _fit_net each iteration.
        PI uses this for policy improvement and stable-check.
        VI returns False (its convergence is handled in _compute_targets).
        Returns True if converged.
        """

    # ── Main solve loop (template method) ────────────────────────────────────

    def solve(
        self,
        hidden    : int   = 64,
        lr        : float = 1e-3,
        fit_epochs: int   = 100,
        fit_batch : int   = 256,
        iterations: int   = 50,
        theta     : float = 1e-4,
    ) -> "NeuralNet":
        """
        Run the DP algorithm and return the trained V_net.

        Each iteration:
            1. _compute_targets  – Bellman targets + optional pre-convergence check
            2. _fit_net          – regress V_net onto those targets
            3. _post_fit_hook    – post-fit logic + optional post-convergence check
        """
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theta     = theta
        print(f"[{self.__class__.__name__}] Using device: {self.device}")

        input_dim      = len(self.disc.n_bins)
        self.V_net     = NeuralNet(input_dim=input_dim, hidden=hidden).to(self.device)
        self.optimizer = optim.Adam(self.V_net.parameters(), lr=lr)

        self._pre_solve_hook()
        transitions = self._precompute_transitions()

        for i in range(iterations):
            print(f"\n=== {self.__class__.__name__} iter {i} ===")

            targets, pre_converged = self._compute_targets(transitions)
            self._fit_net(transitions["x_s"], targets, fit_epochs, fit_batch)
            post_converged = self._post_fit_hook(transitions)

            if pre_converged or post_converged:
                print("Converged — stopping early.")
                break

        return self.V_net

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, model_path: str) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.V_net.state_dict(), model_path)
        print(f"[checkpoint] Saved → {model_path}")

    def load_checkpoint(self, model_path: str, hidden: int = 64) -> bool:
        """Load weights into V_net. Returns True on success, False if file missing."""
        if not Path(model_path).exists():
            return False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim   = len(self.disc.n_bins)
        self.V_net  = NeuralNet(input_dim=input_dim, hidden=hidden).to(self.device)
        self.V_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.V_net.eval()
        print(f"[checkpoint] Loaded ← {model_path}")
        return True

    # ── Greedy-policy query ───────────────────────────────────────────────────

    def greedy_action(self, s: tuple) -> float:
        """Return argmax_a Q(s,a) for the given state using current V_net."""
        self.V_net.eval()
        best_a, best_q = None, -np.inf
        with torch.no_grad():
            for a in self.actions:
                s2, r = self.discrete_transition_fn(
                    self.dynamics_fn, self.disc, s, a,
                    self.target_state, self.Q, self.R, self.dt,
                    method=self.method,
                )
                x2 = torch.tensor(
                    self._state_to_continuous(s2),
                    dtype=torch.float32, device=self.device,
                ).unsqueeze(0)
                q = float(r) + self.gamma * self.V_net(x2).item()
                if q > best_q:
                    best_q, best_a = q, a
        return best_a

    # ── Generic 2-D policy heatmap ────────────────────────────────────────────

    def plot_policy(
        self,
        dim_x     : int,
        dim_theta : int,
        fixed_bins: dict = None,
        xlabel    : str  = "Dimension θ",
        ylabel    : str  = "Dimension x",
        title     : str  = "Greedy policy",
        save_path : str  = None,
    ):
        """
        2-D heatmap of the greedy policy over two chosen state dimensions,
        with all other dimensions fixed to their middle bins (or as specified).

        Parameters
        ----------
        dim_x, dim_theta : int   – state dimensions for the y- and x-axes
        fixed_bins       : dict  – {dim: bin_index} for the remaining dimensions
        """
        n_dims = len(self.disc.n_bins)
        if fixed_bins is None:
            fixed_bins = {
                d: int(self.disc.n_bins[d] // 2)
                for d in range(n_dims) if d not in (dim_x, dim_theta)
            }

        nx = self.disc.n_bins[dim_x]
        nt = self.disc.n_bins[dim_theta]

        def bin_centre(dim, idx):
            lo, hi, n = self.disc.lows[dim], self.disc.highs[dim], self.disc.n_bins[dim]
            return lo + (idx + 0.5) * (hi - lo) / n

        x_labels = [f"{bin_centre(dim_x,    i):.2f}" for i in range(nx)]
        t_labels = [f"{bin_centre(dim_theta, i):.2f}" for i in range(nt)]

        # Build greedy action grid
        action_grid = np.empty((nx, nt))
        self.V_net.eval()
        with torch.no_grad():
            for xi in range(nx):
                for ti in range(nt):
                    s_list            = [None] * n_dims
                    s_list[dim_x]     = xi
                    s_list[dim_theta] = ti
                    for d, b in fixed_bins.items():
                        s_list[d] = b
                    action_grid[xi, ti] = self.greedy_action(tuple(s_list))

        unique_actions = sorted(set(self.actions))
        n_act    = len(unique_actions)
        act_idx  = {a: i for i, a in enumerate(unique_actions)}
        idx_grid = np.vectorize(act_idx.__getitem__)(action_grid)

        cmap   = plt.cm.get_cmap("RdYlGn", n_act)
        norm   = mcolors.BoundaryNorm(np.arange(-0.5, n_act, 1), cmap.N)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(idx_grid, origin="lower", aspect="auto",
                       cmap=cmap, norm=norm, interpolation="nearest")

        ax.set_xticks(range(nt))
        ax.set_xticklabels(t_labels, rotation=45, ha="right")
        ax.set_yticks(range(nx))
        ax.set_yticklabels(x_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fixed_str = ", ".join(f"dim{d}=bin{b}" for d, b in fixed_bins.items())
        ax.set_title(f"{title}\n({fixed_str} fixed at middle bins)")

        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_act))
        cbar.ax.set_yticklabels([f"{a:.1f}" for a in unique_actions])
        cbar.set_label("Action (force)")

        for xi in range(nx):
            for ti in range(nt):
                ax.text(ti, xi, f"{action_grid[xi, ti]:.1f}",
                        ha="center", va="center", fontsize=7, color="black")

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot → {save_path}")
        plt.show()
        return fig, ax

    # ── Tie analysis ──────────────────────────────────────────────────────────

    def tie_analysis(self, tie_eps: float = 1e-4) -> None:
        """Count states where all actions produce nearly identical Q-values."""
        self.V_net.eval()
        tied = 0
        with torch.no_grad():
            for s in self.states:
                qs = []
                for a in self.actions:
                    s2, r = self.discrete_transition_fn(
                        self.dynamics_fn, self.disc, s, a,
                        self.target_state, self.Q, self.R, self.dt,
                        method=self.method,
                    )
                    x2 = torch.tensor(
                        self._state_to_continuous(s2),
                        dtype=torch.float32, device=self.device,
                    ).unsqueeze(0)
                    qs.append(float(r) + self.gamma * self.V_net(x2).item())
                if (max(qs) - min(qs)) < tie_eps:
                    tied += 1
        n = len(self.states)
        print(f"\nTied states (within {tie_eps:g}): {tied}/{n}  "
              f"({100.0 * tied / n:.2f}%)")
