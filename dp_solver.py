"""
dp_solver.py
============
Abstract base class for neural-network dynamic-programming solvers.

States are now sampled uniformly from the continuous state space rather than
enumerated from a discrete grid.  The value function is always represented by
V_net (a neural network), so no Discretizer is needed.

Shared infrastructure (used by both VI and PI):
  - NeuralNet          – the value-function approximator
  - DPSolver           – ABC with the solve() template and all helpers
      · _sample_states            (done once at init)
      · _precompute_transitions   (done once before the solve loop)
      · _fit_net                  (regress V_net onto targets)
      · save/load checkpoint
      · greedy_action             (query V_net for best action)
      · plot_policy               (generic 2-D heatmap over any two dims)
      · tie_analysis

Subclasses must implement three small methods:
  _pre_solve_hook()              – initialise any extra state (e.g. PI policy)
  _compute_targets(transitions)  – return (targets tensor, converged bool)
  _post_fit_hook(transitions)    – called after fitting; return converged bool
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
    lows, highs              : array-like  – continuous state-space bounds
    n_samples                : int         – number of states to sample
    actions                  : list[float] – discrete action set
    dynamics_fn              : callable    – continuous dynamics  f(x, a) → x_dot
    continuous_transition_fn : callable    – (dynamics_fn, x, a, xd, Q, R, dt,
                                              lows, highs, method) → (x_next, r)
    target_state             : np.ndarray  – desired equilibrium
    Q, R                     : cost matrices / scalar
    dt, method, gamma        : integration step, integrator name, discount factor
    """

    def __init__(
        self,
        lows,
        highs,
        n_samples              : int,
        actions                : list,
        dynamics_fn,
        continuous_transition_fn,
        target_state           : np.ndarray,
        Q                      : np.ndarray,
        R,
        dt     : float = 0.02,
        method : str   = "rk4",
        gamma  : float = 0.99,
    ):
        self.lows                    = np.asarray(lows,  dtype=np.float32)
        self.highs                   = np.asarray(highs, dtype=np.float32)
        self.n_samples               = n_samples
        self.actions                 = actions
        self.dynamics_fn             = dynamics_fn
        self.continuous_transition_fn = continuous_transition_fn
        self.target_state            = target_state
        self.Q                       = Q
        self.R                       = R
        self.dt                      = dt
        self.method                  = method
        self.gamma                   = gamma

        # states: list of np.ndarray, shape (D,), sampled once at construction
        self.states: list[np.ndarray] = self._sample_states()

        # Set during solve()
        self.V_net    : NeuralNet | None       = None
        self.optimizer: optim.Optimizer | None = None
        self.device   : torch.device | None    = None
        self.theta    : float                  = 1e-4

    # ── State-space helpers ───────────────────────────────────────────────────

    def _sample_states(self) -> list:
        """Sample n_samples continuous states uniformly from [lows, highs]."""
        return [
            np.random.uniform(self.lows, self.highs).astype(np.float32)
            for _ in range(self.n_samples)
        ]

    def _states_to_tensor(self, states: list) -> torch.Tensor:
        """Stack a list of continuous state arrays into a (N, D) float tensor."""
        arr = np.stack(states, axis=0)   # (N, D)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    # ── Transition pre-computation ────────────────────────────────────────────

    def _precompute_transitions(self, cache_path: str = None) -> dict:
        """
        Roll out every (state, action) pair once and cache as tensors.

        If cache_path is given and the file exists, loads from disk instead
        of recomputing.  If the file does not exist, computes and saves it.

        The cache stores the full (x, a, x', r) dataset as a .npz file:
            x_s   : (N, D)      – sampled states
            x2_<i>: (N, D)      – next states for action index i
            r_<i> : (N,)        – rewards for action index i

        Returns a dict:
            "x_s"        : (N, D) tensor
            "s2_tensors" : list of (N, D) tensors, one per action
            "r_tensors"  : list of (N,) tensors,   one per action
        """
        # ── Try loading from cache ────────────────────────────────────────────
        if cache_path and Path(cache_path).exists():
            print(f"Loading transition cache from {cache_path} …")
            data = np.load(cache_path)

            # Validate cached action count matches current solver config
            cached_n_actions = sum(1 for k in data.files if k.startswith("x2_"))
            if cached_n_actions != len(self.actions):
                print(
                    f"  WARNING: cache has {cached_n_actions} actions but solver "
                    f"has {len(self.actions)} — discarding stale cache and recomputing."
                )
                data.close()
            else:
                # Restore states so self.states matches the cached x_s exactly
                x_s_np      = data["x_s"]                          # (N, D)
                self.states = [x_s_np[i] for i in range(len(x_s_np))]

                s2_tensors = [
                    torch.tensor(data[f"x2_{i}"], dtype=torch.float32, device=self.device)
                    for i in range(len(self.actions))
                ]
                r_tensors = [
                    torch.tensor(data[f"r_{i}"], dtype=torch.float32, device=self.device)
                    for i in range(len(self.actions))
                ]
                print(f"  Loaded {len(self.states)} states × {len(self.actions)} actions.")
                return {
                    "x_s"        : torch.tensor(x_s_np, dtype=torch.float32, device=self.device),
                    "s2_tensors" : s2_tensors,
                    "r_tensors"  : r_tensors,
                }

        # ── Compute from scratch ──────────────────────────────────────────────
        print("Pre-computing transitions …")
        x2_arrays, r_arrays = [], []

        for a in self.actions:
            x2_list, r_list = [], []
            for x in self.states:
                x_next, r = self.continuous_transition_fn(
                    self.dynamics_fn, x, a,
                    self.target_state, self.Q, self.R, self.dt,
                    lows=self.lows, highs=self.highs,
                    method=self.method,
                )
                x2_list.append(np.asarray(x_next, dtype=np.float32))
                r_list.append(float(r))

            x2_arrays.append(np.stack(x2_list, axis=0))   # (N, D)
            r_arrays.append(np.array(r_list, dtype=np.float32))   # (N,)

        x_s_np = np.stack(self.states, axis=0)   # (N, D)

        # ── Save to disk ──────────────────────────────────────────────────────
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            save_dict = {"x_s": x_s_np}
            for i, (x2, r) in enumerate(zip(x2_arrays, r_arrays)):
                save_dict[f"x2_{i}"] = x2
                save_dict[f"r_{i}"]  = r
            np.savez_compressed(cache_path, **save_dict)
            print(f"  Transition cache saved → {cache_path}")

        print(f"  {len(self.states)} states × {len(self.actions)} actions cached.")
        s2_tensors = [
            torch.tensor(x2, dtype=torch.float32, device=self.device)
            for x2 in x2_arrays
        ]
        r_tensors = [
            torch.tensor(r, dtype=torch.float32, device=self.device)
            for r in r_arrays
        ]
        return {
            "x_s"        : torch.tensor(x_s_np, dtype=torch.float32, device=self.device),
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
            perm       = indices[torch.randperm(N, device=self.device)]
            epoch_loss = 0.0
            n_batches  = 0

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
        hidden      : int   = 64,
        lr          : float = 1e-3,
        fit_epochs  : int   = 100,
        fit_batch   : int   = 256,
        iterations  : int   = 50,
        theta       : float = 1e-4,
        cache_path  : str   = None,
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

        input_dim      = len(self.lows)
        self.V_net     = NeuralNet(input_dim=input_dim, hidden=hidden).to(self.device)
        self.optimizer = optim.Adam(self.V_net.parameters(), lr=lr)

        self._pre_solve_hook()
        transitions = self._precompute_transitions(cache_path=cache_path)

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

    def load_checkpoint(self, model_path: str, hidden: int = 256) -> bool:
        """Load weights into V_net. Returns True on success, False if file missing."""
        if not Path(model_path).exists():
            return False
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim     = len(self.lows)
        self.V_net    = NeuralNet(input_dim=input_dim, hidden=hidden).to(self.device)
        self.V_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.V_net.eval()
        print(f"[checkpoint] Loaded ← {model_path}")
        return True

    # ── Greedy-policy query ───────────────────────────────────────────────────

    def greedy_action(self, x: np.ndarray) -> float:
        """Return argmax_a Q(x, a) for a continuous state using current V_net."""
        self.V_net.eval()
        # Reward is actually a cost, so we want the action with LOWEST Q-value
        best_a, best_q = None, np.inf
        with torch.no_grad():
            for a in self.actions:
                x_next, r = self.continuous_transition_fn(
                    self.dynamics_fn, x, a,
                    self.target_state, self.Q, self.R, self.dt,
                    lows=self.lows, highs=self.highs,
                    method=self.method,
                )
                x2 = torch.tensor(
                    np.asarray(x_next, dtype=np.float32),
                    device=self.device,
                ).unsqueeze(0)
                q = float(r) + self.gamma * self.V_net(x2).item()
                if q < best_q:
                    best_q, best_a = q, a
        return best_a

    # ── Generic 2-D policy heatmap ────────────────────────────────────────────

    def plot_policy(
        self,
        dim_x     : int,
        dim_theta : int,
        n_grid    : int  = 30,
        fixed_vals: dict = None,
        xlabel    : str  = "Dimension θ",
        ylabel    : str  = "Dimension x",
        title     : str  = "Greedy policy",
        save_path : str  = None,
    ):
        """
        2-D heatmap of the greedy policy over two chosen state dimensions,
        with all other dimensions fixed to their midpoints (or as specified).

        Parameters
        ----------
        dim_x, dim_theta : int   – state dimensions for the y- and x-axes
        n_grid           : int   – resolution of the evaluation grid
        fixed_vals       : dict  – {dim: continuous_value} for remaining dims
        """
        n_dims = len(self.lows)
        if fixed_vals is None:
            fixed_vals = {
                d: 0.5 * (float(self.lows[d]) + float(self.highs[d]))
                for d in range(n_dims) if d not in (dim_x, dim_theta)
            }

        x_vals = np.linspace(self.lows[dim_x],    self.highs[dim_x],    n_grid)
        t_vals = np.linspace(self.lows[dim_theta], self.highs[dim_theta], n_grid)

        action_grid = np.empty((n_grid, n_grid))
        self.V_net.eval()
        with torch.no_grad():
            for xi, xv in enumerate(x_vals):
                for ti, tv in enumerate(t_vals):
                    s = np.array(
                        [fixed_vals.get(d, 0.0) for d in range(n_dims)],
                        dtype=np.float32,
                    )
                    s[dim_x]     = xv
                    s[dim_theta] = tv
                    action_grid[xi, ti] = self.greedy_action(s)

        unique_actions = sorted(set(self.actions))
        n_act   = len(unique_actions)
        act_idx = {a: i for i, a in enumerate(unique_actions)}
        idx_grid = np.vectorize(act_idx.__getitem__)(action_grid)

        cmap = plt.cm.get_cmap("RdYlGn", n_act)
        norm = mcolors.BoundaryNorm(np.arange(-0.5, n_act, 1), cmap.N)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(idx_grid, origin="lower", aspect="auto",
                       cmap=cmap, norm=norm, interpolation="nearest",
                       extent=[t_vals[0], t_vals[-1], x_vals[0], x_vals[-1]])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fixed_str = ", ".join(f"dim{d}={v:.2f}" for d, v in fixed_vals.items())
        ax.set_title(f"{title}\n({fixed_str} fixed)")

        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_act))
        cbar.ax.set_yticklabels([f"{a:.1f}" for a in unique_actions])
        cbar.set_label("Action (force)")

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
            for x in self.states:
                qs = []
                for a in self.actions:
                    x_next, r = self.continuous_transition_fn(
                        self.dynamics_fn, x, a,
                        self.target_state, self.Q, self.R, self.dt,
                        lows=self.lows, highs=self.highs,
                        method=self.method,
                    )
                    x2 = torch.tensor(
                        np.asarray(x_next, dtype=np.float32),
                        device=self.device,
                    ).unsqueeze(0)
                    qs.append(float(r) + self.gamma * self.V_net(x2).item())
                if (max(qs) - min(qs)) < tie_eps:
                    tied += 1

        n = len(self.states)
        print(f"\nTied states (within {tie_eps:g}): {tied}/{n}  "
              f"({100.0 * tied / n:.2f}%)")
