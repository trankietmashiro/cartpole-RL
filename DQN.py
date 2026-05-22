"""
Deep Q-Learning trainer.

Architecture and training pipeline are the deep_qlearning.py design:
  * Q(s) -> vector of Q-values, one per discrete action
  * hard target-net sync every P steps
  * pre-allocated replay buffer; minibatches drawn uniformly
  * persistent Adam state on the main net (moments survive across TD updates)

Reusability: the training code never references a specific dynamics. To run
on a new system, supply three callables and a few numbers, then call
`train_dqn`. The cartpole entry point at the bottom shows the pattern.

Costs are MINIMIZED (cost-to-go form). Epsilon-greedy picks the action with
the LOWEST predicted Q.
"""

import copy
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np

import neural_nets as nn


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class DQNConfig:
    # --- Network ---
    hidden:      int             = 64
    activations: Tuple[str, ...] = ("relu", "relu")
    lr:          float           = 0.01
    batch_size:  int             = 1024     # minibatch size inside learn_q_factor
    max_epochs:  int             = 100      # epochs per learn_q_factor call

    # --- Bellman ---
    gamma: float = 0.8      # discount
    alpha: float = 1.0      # TD blend (1 = full Bellman backup)

    # --- Exploration ---
    # epsilon = eps_min + (1 - eps_min) * exp(-eps_decay * episode_index)
    eps_min:   float = 0.01
    eps_decay: float = 0.01

    # --- Outer loop ---
    num_episodes:          int = 300
    max_steps_per_episode: int = 200
    warmup_steps:          int = 1000   # skip training until buffer has this many transitions
    train_every:           int = 20     # train main net every N env steps (post-warmup)
    target_sync_every:     int = 100    # hard-copy main -> target every P env steps (post-warmup)
    sample_size:           int = 500    # random transitions drawn per TD update

    # --- Bookkeeping ---
    seed:           Optional[int]      = None
    log_progress:   bool               = True
    on_target_sync: Optional[Callable] = None   # called as fn(Qt, step_count)


# ==============================================================================
# Q-model: weights + persistent Adam state + action set
# ==============================================================================
class QModel:
    """
    Holds the network weights, a persistent Adam state, and the discrete
    action set. Two instances are used: Qm (main, trained) and Qt (target,
    periodically hard-synced from Qm). The target's Adam state is allocated
    but never updated.
    """
    def __init__(self, state_dim: int, actions: np.ndarray,
                 cfg: DQNConfig, seed: Optional[int] = None):
        self.a           = np.asarray(actions)
        self.num_actions = len(self.a)
        self.layers      = list(cfg.activations)

        dims = [state_dim] + [cfg.hidden] * len(self.layers) + [self.num_actions]
        self.weights = nn.init_NN(dims, layers=self.layers, seed=seed)

        # Persistent Adam state — survives between learn_q_factor calls.
        self.adam_state = nn.init_adam(self.weights)
        self.t          = 0

        self.lr         = cfg.lr
        self.batch_size = cfg.batch_size
        self.max_epochs = cfg.max_epochs

    def copy_weights_from(self, other: "QModel"):
        """Hard-copy weights other -> self. Adam state intentionally not copied."""
        self.weights = copy.deepcopy(other.weights)

    def predict(self, states: np.ndarray) -> np.ndarray:
        """states: (state_dim, N) -> (N, num_actions)."""
        y_pred, _, _ = nn.forward(states, self.weights, self.layers)
        return y_pred.T

    def save(self, path: str):
        Wx, Bx, Wy, By = self.weights
        m, v = self.adam_state
        np.savez(
            path,
            Wy=Wy, By=By,
            m_Wy=m["Wy"], v_Wy=v["Wy"],
            m_By=m["By"], v_By=v["By"],
            **{f"Wx_{i}":   w  for i, w  in enumerate(Wx)},
            **{f"Bx_{i}":   b  for i, b  in enumerate(Bx)},
            **{f"m_Wx_{i}": mw for i, mw in enumerate(m["Wx"])},
            **{f"v_Wx_{i}": vw for i, vw in enumerate(v["Wx"])},
            **{f"m_Bx_{i}": mb for i, mb in enumerate(m["Bx"])},
            **{f"v_Bx_{i}": vb for i, vb in enumerate(v["Bx"])},
            actions=self.a,
            t=np.array(self.t),
            n_hidden=np.array(len(Wx)),
        )

    def load(self, path: str):
        d = np.load(path)
        n = int(d["n_hidden"])
        Wx = [d[f"Wx_{i}"] for i in range(n)]
        Bx = [d[f"Bx_{i}"] for i in range(n)]
        self.weights = (Wx, Bx, d["Wy"], d["By"])
        self.adam_state = (
            {"Wx": [d[f"m_Wx_{i}"] for i in range(n)],
             "Bx": [d[f"m_Bx_{i}"] for i in range(n)],
             "Wy": d["m_Wy"], "By": d["m_By"]},
            {"Wx": [d[f"v_Wx_{i}"] for i in range(n)],
             "Bx": [d[f"v_Bx_{i}"] for i in range(n)],
             "Wy": d["v_Wy"], "By": d["v_By"]},
        )
        self.t = int(d["t"])


# ==============================================================================
# Q-factor helpers (operate on (N, num_actions) tables)
# ==============================================================================
def get_Qstar(Qx: np.ndarray) -> np.ndarray:
    """Min Q over actions, per sample."""
    return np.min(Qx, axis=1)


def get_q_factor(a_inds: np.ndarray, Qx: np.ndarray) -> np.ndarray:
    """Pick the Q-value at the chosen action index for each row."""
    return Qx[np.arange(len(a_inds)), a_inds]


def set_q_factor(a_inds: np.ndarray, Q0: np.ndarray, Qx: np.ndarray) -> np.ndarray:
    """Return a copy of Qx with Qx[i, a_inds[i]] := Q0[i]."""
    out = Qx.copy()
    out[np.arange(len(a_inds)), a_inds] = Q0
    return out


# ==============================================================================
# Epsilon-greedy action selection
# ==============================================================================
def epsilon_greedy(Q: QModel, x: np.ndarray, epsilon: float, state_dim: int) -> int:
    """With prob epsilon explore; else pick the action with the LOWEST Q."""
    if np.random.rand() < epsilon:
        return int(np.random.randint(Q.num_actions))
    Qx = Q.predict(x.reshape(state_dim, 1))
    return int(np.argmin(Qx[0]))


# ==============================================================================
# TD update (vectorized over a minibatch)
# ==============================================================================
def td_update(x0: np.ndarray, a_inds: np.ndarray, x1: np.ndarray,
              Qt: QModel, cost_fn: Callable, target: np.ndarray,
              cfg: DQNConfig) -> np.ndarray:
    """
    One-step TD targets for a minibatch.
        x0     : (state_dim, N)
        a_inds : (N,) action indices actually taken
        x1     : (state_dim, N)
    Returns (N, num_actions): Qt's current outputs with the chosen-action
    column overwritten by the Bellman backup. The other columns are intact,
    so the regression only nudges the value of the sampled action.
    """
    Qx  = Qt.predict(x0)
    Qx1 = Qt.predict(x1)

    Qhat0 = get_q_factor(a_inds, Qx)
    Qhat1 = get_Qstar(Qx1)

    us = Qt.a[a_inds]                              # (N,)
    c  = cost_fn(x0, us.reshape(1, -1), target)    # (N,)

    Qnew = Qhat0 + cfg.alpha * (c + cfg.gamma * Qhat1 - Qhat0)
    return set_q_factor(a_inds, Qnew, Qx)


# ==============================================================================
# Supervised regression onto TD targets (preserves persistent Adam state)
# ==============================================================================
def learn_q_factor(xs: np.ndarray, Qtargets: np.ndarray, Q: QModel) -> QModel:
    """
    Train Q.weights to regress xs -> Qtargets.
        xs       : (state_dim, N)
        Qtargets : (N, num_actions)

    Runs Q.max_epochs passes with minibatches of size Q.batch_size. We update
    Adam directly (not via nn.train) so Q.adam_state and Q.t persist across
    calls — the running moments matter between TD updates.
    """
    Y = Qtargets.T                                 # (num_actions, N)
    N = xs.shape[1]
    batch_size = min(Q.batch_size, N)

    for _ in range(Q.max_epochs):
        perm = np.random.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = xs[:, idx], Y[:, idx]

            y_pred, Z, H = nn.forward(xb, Q.weights, Q.layers)
            grads        = nn.backward(xb, Q.weights, y_pred, Z, H, Q.layers, yb)
            Q.t += 1
            Q.weights, Q.adam_state = nn.update_adam(
                Q.weights, grads, Q.adam_state, Q.t, lr=Q.lr,
            )
    return Q


# ==============================================================================
# One training step: sample a random minibatch from the buffer, compute TD
# targets, fit the main net.
# ==============================================================================
def train_network(x0k: np.ndarray, a0k: np.ndarray, x1k: np.ndarray,
                  Qt: QModel, cost_fn: Callable, target: np.ndarray,
                  cfg: DQNConfig, Qm: QModel) -> QModel:
    N = x0k.shape[1]
    inds = np.random.randint(0, N, size=cfg.sample_size)
    xs, aas, x1s = x0k[:, inds], a0k[inds], x1k[:, inds]

    Qj = td_update(xs, aas, x1s, Qt, cost_fn, target, cfg)
    return learn_q_factor(xs, Qj, Qm)


# ==============================================================================
# Convenience: cost-to-go = min_a Q(s, a)
# ==============================================================================
def cost_to_go(Q: QModel, s: np.ndarray) -> np.ndarray:
    """s: (state_dim, N) -> (N,) cost-to-go estimate."""
    return np.min(Q.predict(s), axis=1)


# ==============================================================================
# Main training loop
# ==============================================================================
def train_dqn(*, state_dim: int,
              actions:   np.ndarray,
              target:    np.ndarray,
              reset_fn:  Callable[[], np.ndarray],
              step_fn:   Callable[[np.ndarray, float], Tuple[np.ndarray, bool]],
              cost_fn:   Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
              cfg:       Optional[DQNConfig] = None
              ) -> Tuple[QModel, QModel]:
    """
    Train DQN on whatever dynamics you wire in. Returns (Qm, Qt).

    state_dim : length of the state vector
    actions   : 1-D array of discrete actions
    target    : (state_dim,) desired state; passed through to cost_fn

    reset_fn() -> (state_dim,)             : returns the initial state
    step_fn(x, u) -> (x_next, done)        : x is (state_dim,), u is a scalar
    cost_fn(X, U, target) -> (N,)          : VECTORIZED.
                                             X is (state_dim, N), U is (1, N)
    """
    cfg = cfg if cfg is not None else DQNConfig()
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    Qm = QModel(state_dim, actions, cfg, seed=cfg.seed)
    Qt = QModel(state_dim, actions, cfg, seed=cfg.seed)
    Qt.copy_weights_from(Qm)

    # Pre-allocated replay buffer (worst-case size).
    buf_size = cfg.num_episodes * cfg.max_steps_per_episode
    x0k = np.zeros((state_dim, buf_size))
    a0k = np.zeros(buf_size, dtype=int)
    x1k = np.zeros((state_dim, buf_size))

    k = 0  # global step counter / write index into the buffer

    for i in range(cfg.num_episodes):
        x0      = reset_fn()
        done    = False
        epsilon = cfg.eps_min + (1 - cfg.eps_min) * np.exp(-cfg.eps_decay * i)
        j       = 0

        while not done:
            if cfg.log_progress:
                print(f"\repisode {i+1:4d}/{cfg.num_episodes}  |  "
                      f"step in ep {j+1:4d}  |  total steps {k+1:6d}",
                      end="", flush=True)

            # ── act ─────────────────────────────────────────────────────────
            x0k[:, k] = x0
            a_ind     = epsilon_greedy(Qm, x0, epsilon, state_dim)
            a0k[k]    = a_ind
            u         = float(actions[a_ind])

            x1, term  = step_fn(x0, u)
            j += 1
            if term or j >= cfg.max_steps_per_episode:
                done = True

            x1k[:, k] = x1
            x0 = x1

            # ── learn main net ──────────────────────────────────────────────
            if k > cfg.warmup_steps and k % cfg.train_every == 0:
                Qm = train_network(
                    x0k[:, :k+1], a0k[:k+1], x1k[:, :k+1],
                    Qt, cost_fn, target, cfg, Qm,
                )

            k += 1

            # ── sync target net ─────────────────────────────────────────────
            if k > cfg.warmup_steps and k % cfg.target_sync_every == 0:
                Qt.copy_weights_from(Qm)
                if cfg.on_target_sync is not None:
                    cfg.on_target_sync(Qt, k)

    if cfg.log_progress:
        print("\nTraining complete.")

    return Qm, Qt
