import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.optim as optim
from cartpole import dynamics, Discretizer, discrete_transition


# ── Default checkpoint path ───────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/nn_vi_weights.pth"


# ──────────────────────────────────────────────────────────────────────────────
# Neural network value function  V_θ(s) → scalar
# ──────────────────────────────────────────────────────────────────────────────

class NeuralNet(nn.Module):
    """Small MLP that maps a continuous state vector to a scalar value."""

    def __init__(self, input_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (batch,)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: NeuralNet,
    model_path: str = DEFAULT_MODEL_PATH,
) -> None:
    """Save neural-network weights to disk."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[checkpoint] Model → {model_path}")


def load_checkpoint(
    model: NeuralNet,
    model_path: str = DEFAULT_MODEL_PATH,
) -> "NeuralNet | None":
    """
    Load weights into *model* and return it.
    Returns None if the file is missing.
    """
    if not Path(model_path).exists():
        return None
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"[checkpoint] Loaded model ← {model_path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Helper: discrete state index tuple → continuous bin-centre vector
# ──────────────────────────────────────────────────────────────────────────────

def state_to_continuous(s: tuple, disc: Discretizer) -> np.ndarray:
    """Return the bin-centre coordinates for state index tuple s."""
    centres = []
    for dim, idx in enumerate(s):
        lo   = disc.lows[dim]
        hi   = disc.highs[dim]
        n    = disc.n_bins[dim]
        step = (hi - lo) / n
        centres.append(lo + (idx + 0.5) * step)
    return np.array(centres, dtype=np.float32)


def states_to_tensor(
    states: list, disc: Discretizer, device: torch.device
) -> torch.Tensor:
    """Batch-convert a list of state tuples to a (N, D) float tensor."""
    arr = np.stack([state_to_continuous(s, disc) for s in states], axis=0)
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# Main value-iteration loop
# ──────────────────────────────────────────────────────────────────────────────

def value_iteration(
    disc: Discretizer,
    states: list,
    actions: list,
    target: np.ndarray,
    Q: np.ndarray,
    R,
    dt: float        = 0.02,
    method: str      = "rk4",
    gamma: float     = 0.99,
    lr: float        = 1e-3,
    hidden: int      = 64,
    fit_epochs: int  = 100,
    fit_batch: int   = 256,
    iterations: int  = 50,
    theta: float     = 1e-4,
) -> NeuralNet:
    """
    Neural-network value iteration.

    Each outer iteration:
      1. Compute Bellman optimality targets using the *current* (frozen) V_net:
             y(s) = max_a  [ r(s, a)  +  γ · V_θ_old(s') ]
      2. Fit a *fresh* copy of V_net to those targets by minimising
             ½ · (V_θ(s) − y(s))²   via Adam for `fit_epochs` epochs.
      3. Check convergence:  max_s |y(s) − V_θ_old(s)| < theta → stop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    V_net     = NeuralNet(input_dim=4, hidden=hidden).to(device)
    optimizer = optim.Adam(V_net.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    # Pre-compute transitions for every (state, action) pair — these are fixed.
    # s2_tensors[a_idx] : (N, D) tensor of next-state features
    # r_tensors[a_idx]  : (N,)   tensor of rewards
    print("Pre-computing transitions …")
    N = len(states)
    x_s = states_to_tensor(states, disc, device)           # (N, D)

    s2_tensors = []
    r_tensors  = []
    for a in actions:
        s2_list, r_list = [], []
        for s in states:
            s2, r = discrete_transition(dynamics, disc, s, a, target,
                                        Q, R, dt, method=method)
            s2_list.append(s2)
            r_list.append(float(r))
        s2_tensors.append(states_to_tensor(s2_list, disc, device))
        r_tensors.append(torch.tensor(r_list, dtype=torch.float32, device=device))
    print("Transitions ready.")

    indices = torch.arange(N, device=device)

    for vi_iter in range(iterations):
        print(f"\n=== VI iter {vi_iter} ===")

        # ── Step 1: compute Bellman optimality targets (frozen net) ──────
        V_net.eval()
        with torch.no_grad():
            # Stack Q-values: (A, N)
            q_values = torch.stack([
                r_tensors[ai] + gamma * V_net(s2_tensors[ai])
                for ai in range(len(actions))
            ], dim=0)                                       # (A, N)

            targets, _ = q_values.max(dim=0)               # (N,)

            # Convergence check against old value estimates
            v_old  = V_net(x_s)                            # (N,)
            delta  = (targets - v_old).abs().max().item()
        print(f"  Bellman residual (max |y - V_old|) = {delta:.6f}")

        if delta < theta:
            print("Value function converged — done.")
            break

        # ── Step 2: fit V_net to the new targets ─────────────────────────
        V_net.train()
        prev_loss = float("inf")
        for epoch in range(fit_epochs):
            perm = indices[torch.randperm(N, device=device)]

            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, N, fit_batch):
                idx    = perm[start : start + fit_batch]
                v_pred = V_net(x_s[idx])
                loss   = loss_fn(v_pred, targets[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if abs(prev_loss - avg_loss) < 1e-6:
                print(f"    fit converged at epoch {epoch+1}  loss={avg_loss:.6f}")
                break
            prev_loss = avg_loss

    return V_net


def enumerate_states(disc: Discretizer) -> list:
    grids = [range(n) for n in disc.n_bins.tolist()]
    return [
        tuple(idx)
        for idx in np.array(np.meshgrid(*grids, indexing="ij"))
                     .reshape(len(grids), -1).T
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_policy_x_theta(
    V_net: NeuralNet,
    disc: Discretizer,
    actions: list,
    target_state: np.ndarray,
    Q: np.ndarray,
    R,
    dt: float,
    method: str,
    gamma: float,
    xdot_bin: int = None,
    thetadot_bin: int = None,
    title: str = "Policy: action vs x and θ",
    save_path: str = None,
):
    """
    Plot a 2D heatmap of the greedy policy (derived directly from V_net) as a
    function of cart position (x) and pole angle (theta), with x_dot and
    theta_dot fixed to their middle bins (nearest to zero).
    """
    device = next(V_net.parameters()).device

    n_x     = disc.n_bins[0]
    n_xdot  = disc.n_bins[1]
    n_th    = disc.n_bins[2]
    n_thdot = disc.n_bins[3]

    if xdot_bin     is None: xdot_bin     = n_xdot  // 2
    if thetadot_bin is None: thetadot_bin = n_thdot // 2

    def bin_centres(dim):
        lo, hi, n = disc.lows[dim], disc.highs[dim], disc.n_bins[dim]
        step = (hi - lo) / n
        return [lo + (i + 0.5) * step for i in range(n)]

    x_labels  = [f"{v:.2f}" for v in bin_centres(0)]
    th_labels = [f"{np.degrees(v):.1f}°" for v in bin_centres(2)]

    # Derive greedy action for each (xi, ti) by querying V_net
    action_grid = np.empty((n_x, n_th))
    V_net.eval()
    with torch.no_grad():
        for xi in range(n_x):
            for ti in range(n_th):
                s = (xi, xdot_bin, ti, thetadot_bin)
                best_a, best_q = None, -np.inf
                for a in actions:
                    s2, r = discrete_transition(
                        dynamics, disc, s, a, target_state, Q, R, dt, method=method
                    )
                    x2 = torch.tensor(
                        state_to_continuous(s2, disc), dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    q = float(r) + gamma * V_net(x2).item()
                    if q > best_q:
                        best_q = q
                        best_a = a
                action_grid[xi, ti] = best_a

    unique_actions = sorted(set(actions))
    n_act   = len(unique_actions)
    act_idx = {a: i for i, a in enumerate(unique_actions)}
    idx_grid = np.vectorize(act_idx.__getitem__)(action_grid)

    cmap   = plt.cm.get_cmap("RdYlGn", n_act)
    bounds = np.arange(-0.5, n_act, 1)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        idx_grid, origin="lower", aspect="auto",
        cmap=cmap, norm=norm, interpolation="nearest",
    )

    ax.set_xticks(range(n_th))
    ax.set_xticklabels(th_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_x))
    ax.set_yticklabels(x_labels)
    ax.set_xlabel("Pole angle θ")
    ax.set_ylabel("Cart position x")
    ax.set_title(
        f"{title}\n"
        f"(ẋ bin={xdot_bin}, θ̇ bin={thetadot_bin} — velocities fixed at middle bins)"
    )

    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_act))
    cbar.ax.set_yticklabels([str(a) for a in unique_actions])
    cbar.set_label("Action (force)")

    for xi in range(n_x):
        for ti in range(n_th):
            ax.text(
                ti, xi, str(action_grid[xi, ti]),
                ha="center", va="center", fontsize=7, color="black",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole value iteration (neural net).")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retraining even if a saved checkpoint already exists.",
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH,
        help=f"Path for the .pth model weights  (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    # ── Shared hyper-parameters ───────────────────────────────────────────
    disc = Discretizer(
        lows= [-1.0, -1.0, -np.pi / 12, -1.0],
        highs=[ 1.0,  1.0,  np.pi / 12,  1.0],
        n_bins=[11, 11, 11, 11],
    )
    actions = np.round(np.linspace(-2.0, 2.0, 9), 2).tolist()
    xd      = np.zeros(4)
    Q       = np.diag([5.0, 0.5, 5.0, 0.5])
    R       = 0.01
    states  = enumerate_states(disc)

    # ── Load or train ─────────────────────────────────────────────────────
    V_net = NeuralNet(input_dim=4, hidden=64)

    if not args.retrain and Path(args.model_path).exists():
        print("[main] Checkpoint found — loading saved model.")
        V_net = load_checkpoint(V_net, args.model_path)
    else:
        if args.retrain and Path(args.model_path).exists():
            print("[main] --retrain flag set — ignoring existing checkpoint.")
        else:
            print("[main] No checkpoint found — training from scratch.")

        V_net = value_iteration(
            disc, states, actions, xd, Q, R,
            dt=0.1, method="rk4",
            gamma=0.99, lr=1e-3,
            hidden=64, fit_epochs=150,
            fit_batch=256, iterations=50,
            theta=1e-4,
        )

        save_checkpoint(V_net, args.model_path)

    # ── Tie analysis ──────────────────────────────────────────────────────
    device  = next(V_net.parameters()).device
    tie_eps = 1e-4
    tied    = 0
    for s in states:
        qs = []
        for a in actions:
            s2, r = discrete_transition(dynamics, disc, s, a, xd, Q, R, 0.1, method="rk4")
            x2 = torch.tensor(
                state_to_continuous(s2, disc), dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                v2 = V_net(x2).item()
            qs.append(r + 0.99 * v2)
        if (max(qs) - min(qs)) < tie_eps:
            tied += 1

    print(f"\nStates with all actions tied (within {tie_eps:g}): "
          f"{tied} / {len(states)}  ({100.0 * tied / len(states):.2f}%)")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_policy_x_theta(
        V_net, disc, actions, xd, Q, R,
        dt=0.1, method="rk4", gamma=0.99,
        save_path="figures/vi_policy_x_theta.png",
    )
