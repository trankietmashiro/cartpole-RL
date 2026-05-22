"""
Deep Q-Learning for Pendulum Swing-Up.

Only the pendulum-specific pieces live here:
  * dynamics + Euler integration
  * quadratic cost around the upright equilibrium
  * angle wrapping / velocity clipping
  * visualization (live pendulum, cost-to-go heatmap, convergence trace,
    final policy rollout overlay)

The training pipeline itself is in DQN.py and is environment-agnostic --
this file just wires the four required callables into DQN.train_dqn.
"""

import numpy as np
import matplotlib.pyplot as plt

import DQN


# ──────────────────────────────────────────────────────────────────────────────
# Global timestep
# ──────────────────────────────────────────────────────────────────────────────
DT = 0.1


# ==============================================================================
# Pendulum dynamics
# ==============================================================================
def dynamics(x: np.ndarray, u) -> np.ndarray:
    """
    x : (2, N) – [theta; theta_dot]
    u : scalar or (1, N)
    Returns xdot (2, N).
    """
    m, l, b = 1.0, 0.5, 0.1
    I, g    = 0.25, 9.8
    return np.vstack([
        x[1, :],
        (u - m * g * l * np.sin(x[0, :]) - b * x[1, :]) / I,
    ])


# ==============================================================================
# Instantaneous cost  g(x, u)
# ==============================================================================
def get_QR():
    Q = np.diag([10.0, 1.0]) * DT
    R = 1.0 * DT
    return Q, R


def cost_function(X: np.ndarray, u: np.ndarray, Xd: np.ndarray) -> np.ndarray:
    """
    Quadratic penalty around the upright. Matches the signature DQN.train_dqn
    expects: vectorized over the batch dimension.
        X  : (2, N)
        u  : (1, N)
        Xd : (2,)  desired state
    Returns (N,).
    """
    Q, R = get_QR()
    return (Q[0, 0] * (X[0, :] - Xd[0]) ** 2
            + Q[1, 1] * (X[1, :] - Xd[1]) ** 2
            + R * u.ravel() ** 2)


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
# Main entry point
# ==============================================================================
def deep_qlearning():
    np.random.seed(42)

    # ── Problem setup ────────────────────────────────────────────────────────
    xd      = np.array([np.pi, 0.0])           # desired state (upright)
    actions = np.linspace(-2, 2, 11)           # discrete torques

    # State-space grid (used for both normalization bounds and visualization)
    q_bins    = np.linspace(0,   2 * np.pi, 51)
    qdot_bins = np.linspace(-10, 10,        51)
    qq, qqd   = np.meshgrid(q_bins, qdot_bins, indexing='ij')
    s_grid    = np.vstack([qq.ravel(), qqd.ravel()])      # (2, 51*51)

    # ── Step function expected by the trainer ────────────────────────────────
    # Closes over q_bins / qdot_bins so termination and normalization stay in
    # sync. The trainer calls this as step_fn(x_1d, u_scalar).
    def step_fn(x: np.ndarray, u: float):
        x1 = x.reshape(2, 1) + dynamics(x.reshape(2, 1), u) * DT
        x1 = x1.ravel()
        done = bool(x1[1] > qdot_bins[-1] or x1[1] < qdot_bins[0])
        x1 = normalize_state(x1.reshape(2, 1), q_bins, qdot_bins).ravel()
        return x1, done

    # ── Live visualisation (pendulum | cost-to-go | convergence) ─────────────
    plt.ion()
    fig, (ax_pend, ax_ctg, ax_conv) = plt.subplots(1, 3, figsize=(15, 4))
    fig.tight_layout()

    # List-cell so the inner callback can rebind without needing `nonlocal`.
    Jlast   = [np.zeros(s_grid.shape[1])]
    dJnorm  = []
    kJ_hist = []

    def on_sync(Qt: DQN.QModel, k: int):
        """Called by the trainer after every target-net hard sync."""
        J = DQN.cost_to_go(Qt, s_grid)
        dJnorm.append(np.linalg.norm(J - Jlast[0]))
        kJ_hist.append(k)
        Jlast[0] = J

        vi_plot(J, q_bins, qdot_bins, ax_ctg)
        ax_conv.cla()
        ax_conv.plot(kJ_hist, dJnorm, 'b-')
        ax_conv.set_xlabel('Simulation step')
        ax_conv.set_ylabel('‖ΔJ‖')
        ax_conv.set_title('Convergence')
        plt.pause(0.01)

    # ── Hyper-parameters & training ──────────────────────────────────────────
    cfg = DQN.DQNConfig(
        hidden                = 64,
        activations           = ("relu", "relu"),
        lr                    = 0.01,
        batch_size            = 1024,
        max_epochs            = 100,
        gamma                 = 0.8,
        alpha                 = 1.0,
        eps_min               = 0.01,
        eps_decay             = 0.01,
        num_episodes          = 300,
        max_steps_per_episode = 200,
        warmup_steps          = 1000,
        train_every           = 20,
        target_sync_every     = 100,
        sample_size           = 500,
        seed                  = 42,
        on_target_sync        = on_sync,
    )

    Qm, Qt = DQN.train_dqn(
        state_dim = 2,
        actions   = actions,
        target    = xd,
        reset_fn  = lambda: np.array([0.0, 0.0]),
        step_fn   = step_fn,
        cost_fn   = cost_function,
        cfg       = cfg,
    )
    Qt.save("pendulum_qnet.npz")
    print('Training complete.')

    # ── Final cost-to-go ─────────────────────────────────────────────────────
    J = DQN.cost_to_go(Qt, s_grid)
    vi_plot(J, q_bins, qdot_bins, ax_ctg)
    plt.ioff()

    # ── Roll out the learned policy ──────────────────────────────────────────
    x0    = np.array([0.0, 0.0])
    xsave = np.zeros((2, 100))
    for k in range(100):
        draw_pendulum(x0, ax_pend)
        xsave[:, k] = x0
        a_ind = DQN.epsilon_greedy(Qt, x0, epsilon=0.0, state_dim=2)
        u     = float(actions[a_ind])
        x1    = x0.reshape(2, 1) + dynamics(x0.reshape(2, 1), u) * DT
        x0    = normalize_state(x1, q_bins, qdot_bins).ravel()

    # Trajectory in state space, overlaid on cost-to-go
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
