"""
Deep Q-Learning for Cart-Pole Swing-Up
NumPy version using a hand-rolled neural network (see neural_nets.py)
and the cart-pole dynamics in cartpole.py.

Adapted from the pendulum version. Key differences:

  * 4D dynamics from cartpole.py, integrated with RK4 via get_next_state.
  * Pole angle is encoded as (sin θ, cos θ) so the network never sees a
    discontinuity at the ±π wrap.  Network input is therefore 5D:
        features = [x, sin θ, cos θ, ẋ, θ̇]
    The replay buffer stores RAW 4D states; we encode at the network
    boundary inside QModel.predict().  This keeps the cost function
    (which is most natural on the raw angle) simple.
  * Target state is upright: θ = π, matching the cartpole.py sign
    convention where θ = 0 is hanging down.  In encoded coordinates the
    target is [0, 0, -1, 0, 0].
  * Angle cost uses (1 + cos θ), which is 0 at θ = π and 2 at θ = 0 --
    no wrapping, smooth everywhere.
  * Termination: cart leaves ±X_BOUND or pole rate exceeds ±RATE_BOUND.
  * Cost-to-go visualisation slices through (x = 0, ẋ = 0) so the heatmap
    is still 2D over (θ, θ̇), and the final rollout is animated with
    cartpole.animate_cartpole.

This will be slow.  For a quick smoke-test set M ≈ 50, Nepisode ≈ 200.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

import neural_nets as nn
import cartpole as cp


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DT          = 0.05     # 50 ms -- RK4 stays accurate
STATE_D     = 4        # raw state dim:    [x, θ, ẋ, θ̇]
FEAT_D      = 5        # encoded feat dim: [x, sinθ, cosθ, ẋ, θ̇]

X_BOUND     = 2.4      # |x| > X_BOUND  -> terminate
RATE_BOUND  = 20.0     # |θ̇| > RATE_BOUND -> terminate
VEL_BOUND   = 10.0     # |ẋ| > VEL_BOUND  -> terminate

THETA_TARGET = np.pi   # upright


# ──────────────────────────────────────────────────────────────────────────────
# State encoding: raw 4D -> 5D features with (sin θ, cos θ)
# ──────────────────────────────────────────────────────────────────────────────
def encode_state(x: np.ndarray) -> np.ndarray:
    """
    x : (4,) or (4, N) raw state  [x, θ, ẋ, θ̇]
    returns (5, N) features      [x, sinθ, cosθ, ẋ, θ̇]
    """
    x = np.asarray(x, dtype=float).reshape(STATE_D, -1)
    return np.vstack([
        x[0:1, :],
        np.sin(x[1:2, :]),
        np.cos(x[1:2, :]),
        x[2:3, :],
        x[3:4, :],
    ])


# ==============================================================================
# Q-model container
# ==============================================================================
class QModel:
    """
    Holds network weights, persistent Adam state, and the discrete action set.
    Two instances are used: Qm (main, trained) and Qt (target, periodically
    synced from Qm).
    """
    def __init__(self, actions: np.ndarray, hidden: int = 128,
                 lr: float = 1e-3, batch_size: int = 1024,
                 max_epochs: int = 50, seed: int | None = None):
        self.a = actions
        self.num_actions = len(actions)

        # Architecture: 5 -> hidden -> hidden -> num_actions   (two ReLU layers)
        self.layers  = ["relu", "relu"]
        self.weights = nn.init_NN([FEAT_D, hidden, hidden, self.num_actions],
                                  layers=self.layers, seed=seed)

        # Persistent Adam state (moments + step count).
        self.adam_state = nn.init_adam(self.weights)
        self.t          = 0

        self.lr         = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def copy_weights_from(self, other: "QModel"):
        """Copy weights main -> target.  Adam state is NOT copied: the target
        net never trains, so its Adam buffers stay at zero forever."""
        self.weights = copy.deepcopy(other.weights)

    def predict(self, states_raw: np.ndarray) -> np.ndarray:
        """
        states_raw : (4,) or (4, N) RAW states.  Encoding to 5D features
                     happens inside this call.
        returns    : (N, num_actions)
        """
        feats = encode_state(states_raw)
        y_pred, _, _ = nn.forward(feats, self.weights, self.layers)
        return y_pred.T          # (num_actions, N) -> (N, num_actions)

    def save(self, path):
        """Save weights, Adam state, and metadata to a .npz file."""
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

    def load(self, path):
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
# Dynamics wrapper -- RK4 from cartpole.py, then wrap angle (already in get_next_state)
# ==============================================================================
def step(x: np.ndarray, u: float, dt: float = DT) -> np.ndarray:
    """One simulation step.  x : (4,), u : scalar.  Returns (4,)."""
    return cp.get_next_state(x, u, dt=dt, method="rk4")


def is_terminal(x: np.ndarray) -> bool:
    """Episode ends if cart leaves the track or velocities blow up."""
    return (abs(x[0]) > X_BOUND
            or abs(x[2]) > VEL_BOUND
            or abs(x[3]) > RATE_BOUND)


# ==============================================================================
# Instantaneous cost  g(x, u)
# ==============================================================================
def get_QR():
    """Per-step cost weights.  Tweak these if learning stalls.

    The angle term dominates so the policy prioritises getting the pole up;
    the position term is light to allow large cart excursions during swing-up;
    the velocity / rate terms are small regularisers."""
    Q_pos   = 1.0   * DT     # cart position penalty
    Q_ang   = 100.0 * DT     # pole angle penalty (multiplies 1 + cos θ ∈ [0, 2])
    Q_vel   = 0.1   * DT     # cart velocity penalty
    Q_rate  = 0.1   * DT     # pole rate penalty
    R       = 0.01  * DT     # control penalty
    return Q_pos, Q_ang, Q_vel, Q_rate, R


def cost_function(X: np.ndarray, u: np.ndarray, Xd=None) -> np.ndarray:
    """
    X : (4, N) raw state
    u : (1, N) or scalar
    returns (N,)

    Angle term uses (1 + cos θ), which is 0 at θ = π and 2 at θ = 0 — no
    wrap needed.  Xd is accepted for API parity with the pendulum version
    but unused (the target [0, π, 0, 0] is baked in).
    """
    Q_pos, Q_ang, Q_vel, Q_rate, R = get_QR()
    u = np.asarray(u, dtype=float).ravel()

    pos_cost  = Q_pos  * X[0, :] ** 2
    ang_cost  = Q_ang  * (1.0 + np.cos(X[1, :]))   # min at θ = π
    vel_cost  = Q_vel  * X[2, :] ** 2
    rate_cost = Q_rate * X[3, :] ** 2
    u_cost    = R      * u ** 2
    return pos_cost + ang_cost + vel_cost + rate_cost + u_cost


# ==============================================================================
# Q-factor helpers  (unchanged from pendulum version)
# ==============================================================================
def get_Qstar(Qx: np.ndarray) -> np.ndarray:
    """Best (minimum cost) Q-value over all actions.  Qx: (N, num_actions)."""
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
    Epsilon-greedy on cost: with prob epsilon pick random action, else pick
    the action with the LOWEST predicted Q (cost-to-go).
    """
    if np.random.rand() < epsilon:
        return np.random.randint(Q.num_actions)
    Qx = Q.predict(x.reshape(STATE_D, 1))     # (1, num_actions)
    return int(np.argmin(Qx[0]))


# ==============================================================================
# Temporal-difference target computation
# ==============================================================================
def td_update(x0: np.ndarray, a_inds: np.ndarray, x1: np.ndarray,
              Qt: QModel, gamma: float, alpha: float,
              xd=None) -> np.ndarray:
    """Compute TD targets for a mini-batch.  Returns (N, num_actions)."""
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
# Network training (supervised regression on TD targets)
# ==============================================================================
def learn_q_factor(xs_raw: np.ndarray, Qtargets: np.ndarray,
                   Q: QModel) -> QModel:
    """
    Train Q.weights to regress xs_raw -> Qtargets.

    xs_raw   : (4, N)  RAW states (we encode to 5D below)
    Qtargets : (N, num_actions)

    Runs Q.max_epochs passes over the data, with mini-batches of size
    Q.batch_size.  The Adam state on Q persists across calls — we never
    reset it, so the running moments stay meaningful between TD updates.
    """
    feats = encode_state(xs_raw)                       # (5, N)
    Y     = Qtargets.T                                 # (num_actions, N)
    N     = feats.shape[1]
    batch_size = min(Q.batch_size, N)

    for _ in range(Q.max_epochs):
        perm = np.random.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb  = feats[:, idx]
            yb  = Y[:, idx]

            y_pred, Z, H = nn.forward(xb, Q.weights, Q.layers)
            grads        = nn.backward(xb, Q.weights, y_pred, Z, H,
                                       Q.layers, yb)
            Q.t += 1
            Q.weights, Q.adam_state = nn.update_adam(
                Q.weights, grads, Q.adam_state, Q.t, lr=Q.lr,
            )
    return Q


def train_network(x0k: np.ndarray, a0k: np.ndarray, x1k: np.ndarray,
                  Qt: QModel, gamma: float, alpha: float,
                  xd, Qm: QModel, sample_size: int = 500) -> QModel:
    """Sample a random mini-batch from the replay buffer and do one TD pass."""
    N = x0k.shape[1]
    inds = np.random.randint(0, N, size=min(sample_size, N))
    xs  = x0k[:, inds]
    aas = a0k[inds]
    x1s = x1k[:, inds]

    Qj = td_update(xs, aas, x1s, Qt, gamma, alpha, xd)
    Qm = learn_q_factor(xs, Qj, Qm)
    return Qm


# ==============================================================================
# Cost-to-go  (min_a Q(s, a))
# ==============================================================================
def cost_to_go(Q: QModel, s_raw: np.ndarray) -> np.ndarray:
    Qx = Q.predict(s_raw)         # (N, num_actions)
    return np.min(Qx, axis=1)     # (N,)


# ==============================================================================
# Visualisation helpers
# ==============================================================================
def make_vis_grid(theta_bins: np.ndarray,
                  thetadot_bins: np.ndarray) -> np.ndarray:
    """
    Build a (4, n_θ*n_θ̇) grid of RAW states with x = ẋ = 0 held fixed,
    suitable for feeding to cost_to_go().
    """
    tt, ttd = np.meshgrid(theta_bins, thetadot_bins, indexing='ij')
    n = tt.size
    grid = np.zeros((STATE_D, n))
    grid[0, :] = 0.0          # x
    grid[1, :] = tt.ravel()   # θ
    grid[2, :] = 0.0          # ẋ
    grid[3, :] = ttd.ravel()  # θ̇
    return grid


def vi_plot(J: np.ndarray, theta_bins: np.ndarray,
            thetadot_bins: np.ndarray, ax: plt.Axes):
    n1, n2 = len(theta_bins), len(thetadot_bins)
    ax.cla()
    ax.imshow(J.reshape(n1, n2).T,
              origin='lower',
              extent=[theta_bins[0], theta_bins[-1],
                      thetadot_bins[0], thetadot_bins[-1]],
              aspect='auto', cmap='viridis')
    ax.axvline(THETA_TARGET, color='r', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title('Cost-to-go  (slice at x=0, ẋ=0)')
    plt.pause(0.01)


def draw_cartpole(x: np.ndarray, ax: plt.Axes, l: float = 0.5):
    """Lightweight per-step cart-pole drawing (used during live rollout)."""
    cart_x, theta = x[0], x[1]
    ax.cla()
    # Cart
    ax.plot([cart_x - 0.2, cart_x + 0.2], [0, 0], 'k-', lw=8)
    # Pole (note: pole tip at (cart_x + l*sin θ, -l*cos θ) to match
    # cartpole.animate_cartpole's convention)
    bx = cart_x + l * np.sin(theta)
    by = -l * np.cos(theta)
    ax.plot([cart_x, bx], [0, by], 'r-', lw=3)
    ax.plot(bx, by, 'ko', ms=10)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(-X_BOUND, color='gray', ls=':', lw=0.5)
    ax.axvline( X_BOUND, color='gray', ls=':', lw=0.5)
    ax.set_xlim(-X_BOUND - 0.3, X_BOUND + 0.3)
    ax.set_ylim(-1.5 * l, 1.5 * l)
    ax.set_aspect('equal')
    ax.set_title(f'x={cart_x:+.2f}  θ={theta:+.2f}')
    plt.pause(0.001)


# ==============================================================================
# Main training loop
# ==============================================================================
def deep_qlearning():
    np.random.seed(42)

    xd    = np.array([0.0, THETA_TARGET, 0.0, 0.0])  # desired state (upright)
    gamma = 0.95                                     # discount factor

    # Discrete actions: 11 evenly spaced forces in [-15, 15] N.
    # ±15 N is comfortably enough for swing-up with the default cartpole.py
    # parameters (mc=1.0, mp=0.1, l=0.5).
    actions = np.linspace(-5.0, 5.0, 21)

    # State-space grid for visualisation (slice at x=0, ẋ=0)
    theta_bins    = np.linspace(0,   2 * np.pi, 51)
    thetadot_bins = np.linspace(-RATE_BOUND, RATE_BOUND, 51)
    s_grid        = make_vis_grid(theta_bins, thetadot_bins)   # (4, 51*51)

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    alpha    = 1        # TD blend (1 = full Bellman backup)
    M        = 1000      # episodes
    N        = 50       # train every N simulation steps
    Ninit    = 2000     # warm-up steps before training begins
    P        = 200      # copy weights to target every P steps
    Nepisode = 500      # max steps per episode (= 25 s of sim at DT=0.05)

    # ── Initialise networks ───────────────────────────────────────────────────
    Qm = QModel(actions, hidden=128, lr=1e-3, batch_size=1024,
                max_epochs=50, seed=0)
    Qt = QModel(actions, hidden=128, lr=1e-3, batch_size=1024,
                max_epochs=50, seed=0)
    Qt.copy_weights_from(Qm)

    # Replay buffer (pre-allocated, RAW 4D states)
    buf_size = M * Nepisode
    x0k = np.zeros((STATE_D, buf_size))
    a0k = np.zeros(buf_size, dtype=int)
    x1k = np.zeros((STATE_D, buf_size))

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
        # Start hanging down with a small random perturbation (helps exploration)
        x0 = np.array([
            np.random.uniform(-0.1,  0.1),     # cart pos
            np.random.uniform(-0.1,  0.1),     # angle (small jitter from 0)
            0.0,                                # cart vel
            0.0,                                # pole rate
        ])
        done    = False
        epsilon = 0.01 + (1 - 0.01) * np.exp(-0.005 * i)
        j = 0

        while not done:
            print(f'\repisode {i+1:3d}/{M}  |  step in ep {j+1:3d}  |  '
                  f'total steps {k+1:6d}  |  ε={epsilon:.3f}',
                  end='', flush=True)

            x0k[:, k] = x0
            a_ind     = get_action(Qm, x0, epsilon)
            a0k[k]    = a_ind
            u         = actions[a_ind]

            x1 = step(x0, u, DT)        # RK4 integration + angle wrap

            # Termination
            if is_terminal(x1) or j >= Nepisode - 1:
                done = True

            j += 1
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

                vi_plot(J, theta_bins, thetadot_bins, ax_ctg)
                ax_conv.cla()
                ax_conv.plot(kJ_hist, dJnorm, 'b-')
                ax_conv.set_xlabel('Simulation step')
                ax_conv.set_ylabel('‖ΔJ‖')
                ax_conv.set_title('Convergence')
                plt.pause(0.01)

    Qt.save("cartpole_qnet.npz")
    print('\nTraining complete.')

    # ── Final cost-to-go ──────────────────────────────────────────────────────
    J = cost_to_go(Qt, s_grid)
    vi_plot(J, theta_bins, thetadot_bins, ax_ctg)
    plt.ioff()

    # ── Roll out the learned policy ───────────────────────────────────────────
    rollout_steps = 600                  # 30 s
    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # hanging down
    epsilon = 0.0
    xsave = np.zeros((STATE_D, rollout_steps))
    tsave = np.arange(rollout_steps) * DT

    for k in range(rollout_steps):
        draw_cartpole(x0, ax_pend)
        xsave[:, k] = x0
        a_ind = get_action(Qt, x0, epsilon)
        u     = actions[a_ind]
        x0    = step(x0, u, DT)

    # State-space trajectory on cost-to-go (θ, θ̇ slice)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.imshow(J.reshape(len(theta_bins), len(thetadot_bins)).T,
               origin='lower',
               extent=[theta_bins[0], theta_bins[-1],
                       thetadot_bins[0], thetadot_bins[-1]],
               aspect='auto', cmap='viridis')
    # Wrap θ into [0, 2π] so it lands inside the heatmap extent
    th_plot = np.mod(xsave[1, :], 2 * np.pi)
    ax2.plot(th_plot, xsave[3, :], '*-g', markersize=4,
             label='Learned policy')
    ax2.axvline(THETA_TARGET, color='r', ls='--', lw=1, alpha=0.7)
    ax2.set_xlabel('θ (rad)'); ax2.set_ylabel('θ̇ (rad/s)')
    ax2.set_title('Policy trajectory on cost-to-go  (x=0, ẋ=0 slice)')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # ── Animation with the proper cart-pole visualiser ────────────────────────
    cp.animate_cartpole(tsave, xsave, cp.DEFAULT_PARAM)


# ==============================================================================
if __name__ == '__main__':
    deep_qlearning()
