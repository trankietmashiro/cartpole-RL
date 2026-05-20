"""
Deep Q-Learning for Pendulum Swing-Up
NumPy version using a hand-rolled neural network (see neural_nets.py).

Differences from the PyTorch version:
  * QModel wraps the (Wx, Bx, Wy, By) weight tuple plus a persistent
    Adam state (self.t carries the bias-correction step count across
    training calls, so the optimizer's running moments survive between
    learn_q_factor() invocations -- this is important).
  * Boundary transposes: the rest of the script keeps the MATLAB-style
    (features, batch) layout; only QModel.predict() flips to
    (batch, features) on the way out, matching the original API.
  * Target-network sync uses copy.deepcopy on the weight tuple. Do NOT
    just assign -- that would alias arrays and the target would track
    the main net in real time.

Note: this WILL be much slower than the PyTorch version. For a quick
smoke-test, lower M (episodes) to ~30 and max_epochs to ~20.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

import neural_nets as nn

# ──────────────────────────────────────────────────────────────────────────────
# Global timestep
# ──────────────────────────────────────────────────────────────────────────────
DT = 0.1


# ==============================================================================
# Q-model container  (mirrors the Qm / Qt structs in the PyTorch version)
# ==============================================================================
class QModel:
    """
    Holds the network weights, a persistent Adam state, and the discrete
    action set. Two instances are used: Qm (main, trained) and Qt (target,
    periodically synced from Qm).
    """
    def __init__(self, actions: np.ndarray, hidden: int = 64,
                 lr: float = 0.01, batch_size: int = 1024,
                 max_epochs: int = 100, seed: int | None = None):
        self.a = actions
        self.num_actions = len(actions)

        # Architecture: 2 -> hidden -> hidden -> num_actions  (two ReLU layers)
        self.layers  = ["relu", "relu"]
        self.weights = nn.init_NN([2, hidden, hidden, self.num_actions],
                                  layers=self.layers, seed=seed)

        # Persistent Adam state (moments + step count).
        self.adam_state = nn.init_adam(self.weights)
        self.t          = 0

        self.lr         = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def copy_weights_from(self, other: "QModel"):
        """Copy weights main -> target. Adam state is NOT copied: the target
        net never trains, so its Adam buffers stay at zero forever."""
        self.weights = copy.deepcopy(other.weights)

    def predict(self, states: np.ndarray) -> np.ndarray:
        """
        states : (2, N)  column-major
        returns: (N, num_actions)
        """
        y_pred, _, _ = nn.forward(states, self.weights, self.layers)
        return y_pred.T          # (num_actions, N) -> (N, num_actions)
    
    def save(self, path):
        """Save weights, Adam state, and metadata to a .npz file."""
        Wx, Bx, Wy, By = self.weights
        m, v = self.adam_state
        np.savez(
            path,
            # output layer
            Wy=Wy, By=By,
            m_Wy=m["Wy"], v_Wy=v["Wy"],
            m_By=m["By"], v_By=v["By"],
            # hidden layers (variable count -> indexed keys)
            **{f"Wx_{i}": w  for i, w  in enumerate(Wx)},
            **{f"Bx_{i}": b  for i, b  in enumerate(Bx)},
            **{f"m_Wx_{i}": mw for i, mw in enumerate(m["Wx"])},
            **{f"v_Wx_{i}": vw for i, vw in enumerate(v["Wx"])},
            **{f"m_Bx_{i}": mb for i, mb in enumerate(m["Bx"])},
            **{f"v_Bx_{i}": vb for i, vb in enumerate(v["Bx"])},
            # bookkeeping
            actions=self.a,
            t=np.array(self.t),
            n_hidden=np.array(len(Wx)),
        )

    def load(self, path):
        """Load weights and Adam state from a .npz file."""
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
# Pendulum dynamics
# ==============================================================================
def dynamics(x: np.ndarray, u: float | np.ndarray) -> np.ndarray:
    """
    x    : (2, N) – [theta; theta_dot]
    u    : scalar or (1, N)
    returns xdot (2, N)
    """
    m, l, b, lc = 1.0, 0.5, 0.1, 0.5
    I, g = 0.25, 9.8
    xdot = np.vstack([x[1, :],
                      (u - m * g * l * np.sin(x[0, :]) - b * x[1, :]) / I])
    return xdot


# ==============================================================================
# Instantaneous cost  g(x, u)
# ==============================================================================
def get_QR():
    Q = np.diag([10.0, 1.0]) * DT
    R = 1.0 * DT
    return Q, R


def cost_function(X: np.ndarray, u: np.ndarray, Xd: np.ndarray) -> np.ndarray:
    """
    X  : (2, N)
    u  : (1, N) or scalar
    returns: (N,)
    """
    Q, R = get_QR()
    cost = (Q[0, 0] * (X[0, :] - np.pi) ** 2
            + Q[1, 1] * X[1, :] ** 2
            + R * u.ravel() ** 2)
    return cost


# ==============================================================================
# Q-factor helpers
# ==============================================================================
def get_Qstar(Qx: np.ndarray) -> np.ndarray:
    """Best (minimum cost) Q-value over all actions. Qx: (N, num_actions)."""
    return np.min(Qx, axis=1)


def get_q_factor(a_inds: np.ndarray, Qx: np.ndarray) -> np.ndarray:
    """Pick Q-value for the chosen action at each sample."""
    N = len(a_inds)
    return Qx[np.arange(N), a_inds]


def set_q_factor(a_inds: np.ndarray, Q0: np.ndarray,
                 Qx: np.ndarray) -> np.ndarray:
    """Write updated Q-values back into a copy of the Q-table."""
    Qx_new = Qx.copy()
    N = len(a_inds)
    Qx_new[np.arange(N), a_inds] = Q0
    return Qx_new


# ==============================================================================
# Epsilon-greedy action selection
# ==============================================================================
def get_action(Q: QModel, x: np.ndarray, epsilon: float) -> int:
    """
    Epsilon-greedy on cost: with prob epsilon pick random action, else
    pick the action with the LOWEST predicted Q (cost-to-go).
    """
    if np.random.rand() < epsilon:
        return np.random.randint(Q.num_actions)
    Qx = Q.predict(x.reshape(2, 1))      # (1, num_actions)
    return int(np.argmin(Qx[0]))


# ==============================================================================
# Temporal-difference target computation
# ==============================================================================
def td_update(x0: np.ndarray, a_inds: np.ndarray, x1: np.ndarray,
              Qt: QModel, gamma: float, alpha: float,
              xd: np.ndarray) -> np.ndarray:
    """Compute TD targets for a mini-batch. Returns (N, num_actions)."""
    Qx  = Qt.predict(x0)              # (N, num_actions) – baseline
    Qx1 = Qt.predict(x1)              # (N, num_actions) – next-state values

    Qhat0 = get_q_factor(a_inds, Qx)   # (N,) value of taken action
    Qhat1 = get_Qstar(Qx1)             # (N,) best next-state value

    us = Qt.a[a_inds]                                  # (N,)
    c  = cost_function(x0, us.reshape(1, -1), xd)      # (N,)

    # One-step Bellman backup (cost-to-go form: we want to MINIMISE).
    Qnew  = Qhat0 + alpha * (c + gamma * Qhat1 - Qhat0)
    Qxnew = set_q_factor(a_inds, Qnew, Qx)
    return Qxnew


# ==============================================================================
# Network training  (supervised regression on TD targets)
# ==============================================================================
def learn_q_factor(xs: np.ndarray, Qtargets: np.ndarray,
                   Q: QModel) -> QModel:
    """
    Train Q.weights to regress xs -> Qtargets.
    xs       : (2, N)
    Qtargets : (N, num_actions)

    Runs Q.max_epochs passes over the data, with mini-batches of size
    Q.batch_size. The Adam state on Q persists across calls -- we never
    reset it, so the running moments stay meaningful between TD updates.
    """
    # Flip targets to match our (features, batch) convention.
    Y = Qtargets.T                                     # (num_actions, N)
    N = xs.shape[1]
    batch_size = min(Q.batch_size, N)

    for _ in range(Q.max_epochs):
        perm = np.random.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb  = xs[:, idx]
            yb  = Y[:, idx]

            y_pred, Z, H = nn.forward(xb, Q.weights, Q.layers)
            grads        = nn.backward(xb, Q.weights, y_pred, Z, H,
                                       Q.layers, yb)
            Q.t += 1
            Q.weights, Q.adam_state = nn.update_adam(
                Q.weights, grads, Q.adam_state, Q.t, lr=Q.lr,
            )
    return Q


# ==============================================================================
# Train the main network on a random mini-batch drawn from the replay buffer
# ==============================================================================
def train_network(x0k: np.ndarray, a0k: np.ndarray, x1k: np.ndarray,
                  Qt: QModel, gamma: float, alpha: float,
                  xd: np.ndarray, Qm: QModel) -> QModel:
    N = x0k.shape[1]
    inds = np.random.randint(0, N, size=500)
    xs  = x0k[:, inds]
    aas = a0k[inds]
    x1s = x1k[:, inds]

    Qj = td_update(xs, aas, x1s, Qt, gamma, alpha, xd)
    Qm = learn_q_factor(xs, Qj, Qm)
    return Qm


# ==============================================================================
# Cost-to-go  (min_a Q(s, a))
# ==============================================================================
def cost_to_go(Q: QModel, s: np.ndarray) -> np.ndarray:
    Qx = Q.predict(s)             # (N, num_actions)
    return np.min(Qx, axis=1)     # (N,)


# ==============================================================================
# State normalisation / angle wrapping
# ==============================================================================
def normalize_state(s: np.ndarray, q_bins: np.ndarray,
                    qdot_bins: np.ndarray) -> np.ndarray:
    s = s.copy()
    s[0, :] = np.mod(s[0, :], 2 * np.pi)
    s[0, :] = np.clip(s[0, :], q_bins[0],    q_bins[-1])
    s[1, :] = np.clip(s[1, :], qdot_bins[0], qdot_bins[-1])
    return s


# ==============================================================================
# Visualisation helpers
# ==============================================================================
def vi_plot(J: np.ndarray, q_bins: np.ndarray, qdot_bins: np.ndarray,
            ax: plt.Axes):
    n1, n2 = len(q_bins), len(qdot_bins)
    ax.cla()
    ax.imshow(J.reshape(n1, n2).T,
              origin='lower',
              extent=[q_bins[0], q_bins[-1], qdot_bins[0], qdot_bins[-1]],
              aspect='auto', cmap='viridis')
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title('Cost-to-go')
    plt.pause(0.01)


def draw_pendulum(x: np.ndarray, ax: plt.Axes):
    theta = x[0]
    l = 0.75
    ax.cla()
    px = l * np.sin(theta)
    py = -l * np.cos(theta)
    ax.plot([0, px], [0, py], 'r-', linewidth=3)
    ax.plot(px, py, 'ko', markersize=12)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f'θ = {theta:.2f} rad')
    plt.pause(0.001)


# ==============================================================================
# Main training loop
# ==============================================================================
def deep_qlearning():
    np.random.seed(42)

    xd = np.array([np.pi, 0.0])     # desired state (upright)
    gamma = 0.8                     # discount factor

    # Discrete actions
    actions = np.linspace(-2, 2, 11)

    # State-space grid for visualisation
    q_bins    = np.linspace(0,   2 * np.pi, 51)
    qdot_bins = np.linspace(-10, 10,        51)
    qq, qqd   = np.meshgrid(q_bins, qdot_bins, indexing='ij')
    s_grid    = np.vstack([qq.ravel(), qqd.ravel()])   # (2, 51*51)

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    alpha    = 1      # TD blend (1 = full Bellman backup)
    M        = 300    # episodes
    N        = 20     # train every N simulation steps
    Ninit    = 1000   # warm-up steps before training begins
    P        = 100    # copy weights to target every P steps
    Nepisode = 200    # max steps per episode

    # ── Initialise networks ───────────────────────────────────────────────────
    Qm = QModel(actions, hidden=64, lr=0.01, batch_size=1024,
                max_epochs=100, seed=0)
    Qt = QModel(actions, hidden=64, lr=0.01, batch_size=1024,
                max_epochs=100, seed=0)
    Qt.copy_weights_from(Qm)

    # Replay buffer (pre-allocated)
    buf_size = M * Nepisode
    x0k = np.zeros((2, buf_size))
    a0k = np.zeros(buf_size, dtype=int)
    x1k = np.zeros((2, buf_size))

    # Visualisation
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_pend, ax_ctg, ax_conv = axes
    fig.tight_layout()

    Jlast   = np.zeros(s_grid.shape[1])
    dJnorm  = []
    kJ_hist = []

    k = 0   # global simulation step counter

    # ── Episode loop ──────────────────────────────────────────────────────────
    for i in range(M):
        x0      = np.array([0.0, 0.0])
        done    = False
        epsilon = 0.01 + (1 - 0.01) * np.exp(-0.01 * i)
        j = 0

        while not done:
            print(f'\repisode {i+1:3d}/{M}  |  step in ep {j+1:3d}  |  '
                  f'total steps {k+1:5d}', end='', flush=True)

            x0k[:, k] = x0
            a_ind     = get_action(Qm, x0, epsilon)
            a0k[k]    = a_ind
            u         = actions[a_ind]

            x1 = x0.reshape(2, 1) + dynamics(x0.reshape(2, 1), u) * DT
            x1 = x1.ravel()

            # Termination
            if x1[1] > qdot_bins[-1] or x1[1] < qdot_bins[0]:
                done = True
            elif j >= Nepisode:
                done = True

            j += 1
            x1 = normalize_state(x1.reshape(2, 1), q_bins,
                                 qdot_bins).ravel()
            x1k[:, k] = x1
            x0 = x1

            # Train main network
            if k % N == 0 and k > Ninit:
                Qm = train_network(x0k[:, :k+1], a0k[:k+1],
                                   x1k[:, :k+1],
                                   Qt, gamma, alpha, xd, Qm)

            k += 1

            # Sync target net & visualise
            if k % P == 0 and k > Ninit:
                Qt.copy_weights_from(Qm)
                J = cost_to_go(Qt, s_grid)
                dJnorm.append(np.linalg.norm(J - Jlast))
                kJ_hist.append(k)
                Jlast = J

                vi_plot(J, q_bins, qdot_bins, ax_ctg)
                ax_conv.cla()
                ax_conv.plot(kJ_hist, dJnorm, 'b-')
                ax_conv.set_xlabel('Simulation step')
                ax_conv.set_ylabel('‖ΔJ‖')
                ax_conv.set_title('Convergence')
                plt.pause(0.01)
    Qt.save("pendulum_qnet.npz")
    print('\nTraining complete.')

    # ── Final visualisation ────────────────────────────────────────────────────
    J = cost_to_go(Qt, s_grid)
    vi_plot(J, q_bins, qdot_bins, ax_ctg)
    plt.ioff()

    # ── Roll out the learned policy ────────────────────────────────────────────
    x0      = np.array([0.0, 0.0])
    epsilon = 0.0
    xsave   = np.zeros((2, 100))

    for k in range(100):
        draw_pendulum(x0, ax_pend)
        xsave[:, k] = x0
        a_ind = get_action(Qt, x0, epsilon)
        u     = actions[a_ind]
        x1    = x0.reshape(2, 1) + dynamics(x0.reshape(2, 1), u) * DT
        x1    = normalize_state(x1, q_bins, qdot_bins).ravel()
        x0    = x1

    # Trajectory in state space
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.imshow(J.reshape(len(q_bins), len(qdot_bins)).T,
               origin='lower',
               extent=[q_bins[0], q_bins[-1], qdot_bins[0], qdot_bins[-1]],
               aspect='auto', cmap='viridis')
    ax2.plot(xsave[0, :], xsave[1, :], '*-g', markersize=4,
             label='Learned policy')
    ax2.set_xlabel('θ (rad)'); ax2.set_ylabel('θ̇ (rad/s)')
    ax2.set_title('Policy trajectory on cost-to-go')
    ax2.legend()
    plt.tight_layout()
    plt.show()

# ==============================================================================
if __name__ == '__main__':
    deep_qlearning()
