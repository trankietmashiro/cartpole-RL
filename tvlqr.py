"""
tvlqr.py  —  General-purpose Time-Varying LQR framework.

Builds on ilqr.py: takes an ILQRSolution (or any nominal trajectory) and
a DynamicsModel, runs a backward Riccati sweep, then simulates closed-loop.

Pipeline
--------
1.  (Optional) Run ILQR.solve() to produce a nominal (x*, u*).
2.  Instantiate TVLQR(dynamics, Q, R, Qf).
3.  Call .gains(solution)     → TVLQRGains
4.  Call .simulate(x0, gains, solution)  → TVLQRResult
5.  Plot / animate as needed.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ilqr import DynamicsModel, ILQRSolution


# =========================================================================
#  Data containers
# =========================================================================

@dataclass
class TVLQRGains:
    """Feedback gains K_k at each time step."""
    Klist : list          # list of (nU, nX) arrays, length N-1
    dt    : float
    N     : int


@dataclass
class TVLQRResult:
    """Closed-loop trajectory produced by simulate()."""
    x_cl  : np.ndarray   # (nX, N)
    u_cl  : np.ndarray   # (nU, N-1)
    t_vec : np.ndarray   # (N,)


# =========================================================================
#  TVLQR solver
# =========================================================================

class TVLQR:
    """
    Time-Varying LQR tracker.

    Parameters
    ----------
    dynamics : DynamicsModel
        Must implement discrete_jacobians() — the same object used for iLQR.
    Q  : (nX, nX)  running state-error cost
    R  : (nU, nU)  running input-error cost
    Qf : (nX, nX)  terminal state-error cost
    """

    def __init__(
        self,
        dynamics : DynamicsModel,
        Q        : np.ndarray,
        R        : np.ndarray,
        Qf       : np.ndarray,
    ):
        self.dynamics = dynamics
        self.Q  = Q
        self.R  = R
        self.Qf = Qf

    # ------------------------------------------------------------------
    def gains(self, nominal: ILQRSolution, dt: float) -> TVLQRGains:
        """
        Backward Riccati sweep along the nominal trajectory.

        Discrete-time LQR at each step:
            K_k  = -(R + B_k' P_{k+1} B_k)^{-1} B_k' P_{k+1} A_k
            P_k  = Q + A_cl' P_{k+1} A_cl + K_k' R K_k
                   where  A_cl = A_k + B_k K_k
        """
        xtraj, utraj = nominal.xtraj, nominal.utraj
        N = xtraj.shape[1]

        P     = self.Qf.copy()
        Klist = [None] * (N - 1)

        for k in range(N - 2, -1, -1):
            A, B = self.dynamics.discrete_jacobians(
                k * dt, xtraj[:, [k]], utraj[:, [k]], dt
            )
            S        = self.R + B.T @ P @ B
            K        = -np.linalg.solve(S, B.T @ P @ A)   # (nU, nX)
            Klist[k] = K

            Acl = A + B @ K
            P   = self.Q + K.T @ self.R @ K + Acl.T @ P @ Acl

        return TVLQRGains(Klist, dt, N)

    # ------------------------------------------------------------------
    def simulate(
        self,
        x0      : np.ndarray,        # (nX, 1) initial state (may be perturbed)
        gains   : TVLQRGains,
        nominal : ILQRSolution,
    ) -> TVLQRResult:
        """
        Roll out the nonlinear dynamics under TV-LQR feedback:
            u_k = u*_k + K_k (x_k − x*_k)
        """
        xtraj, utraj = nominal.xtraj, nominal.utraj
        N  = gains.N
        dt = gains.dt
        nX = xtraj.shape[0]
        nU = utraj.shape[0]

        x_cl = np.zeros((nX, N))
        u_cl = np.zeros((nU, N - 1))
        x    = np.asarray(x0).reshape(nX, 1)
        x_cl[:, [0]] = x

        for k in range(N - 1):
            dx           = x - xtraj[:, [k]]
            u            = utraj[:, [k]] + gains.Klist[k] @ dx
            u_cl[:, [k]] = u
            x            = self.dynamics.integrate(k * dt, x, u, dt)
            x_cl[:, [k + 1]] = x

        t_vec = np.arange(N) * dt
        return TVLQRResult(x_cl, u_cl, t_vec)


# =========================================================================
#  Plotting helper
# =========================================================================

def plot_comparison(
    t_vec  : np.ndarray,
    nominal: ILQRSolution,
    result : TVLQRResult,
    xd     : np.ndarray,
    state_labels: list[str] | None = None,
):
    """
    Six-panel comparison: 4 state traces, control input, tracking error.
    Works for any system; pass state_labels to customise axis titles.
    """
    import matplotlib.pyplot as plt

    nX = nominal.xtraj.shape[0]
    if state_labels is None:
        state_labels = [f"x[{i}]" for i in range(nX)]

    t_u  = t_vec[:-1]
    N    = len(t_vec)
    ncols = 2
    nrows = (nX + 2 + ncols - 1) // ncols   # states + control + error

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    fig.suptitle("iLQR nominal  vs  TV-LQR closed-loop", fontsize=13)
    axes = axes.flatten()

    for i in range(nX):
        ax = axes[i]
        ax.plot(t_vec, nominal.xtraj[i, :], 'k--', lw=1.5, label='iLQR nominal')
        ax.plot(t_vec, result.x_cl[i, :],   'b-',  lw=1.5, label='TV-LQR c/l')
        ax.axhline(float(xd.flat[i]), color='r', ls=':', lw=1, label='target')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(state_labels[i])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # control input(s)
    ax_u = axes[nX]
    for j in range(nominal.utraj.shape[0]):
        ax_u.plot(t_u, nominal.utraj[j, :], 'k--', lw=1.5, label=f'nominal u[{j}]')
        ax_u.plot(t_u, result.u_cl[j, :],   'b-',  lw=1.5, label=f'TV-LQR u[{j}]')
    ax_u.set_xlabel("Time [s]")
    ax_u.set_ylabel("Control input")
    ax_u.legend(fontsize=7)
    ax_u.grid(True, alpha=0.3)

    # tracking error
    ax_e = axes[nX + 1]
    err  = np.linalg.norm(result.x_cl - nominal.xtraj, axis=0)
    ax_e.plot(t_vec, err, 'r-', lw=1.5)
    ax_e.set_xlabel("Time [s]")
    ax_e.set_ylabel("‖x_cl − x*‖")
    ax_e.set_title("State tracking error")
    ax_e.grid(True, alpha=0.3)

    # hide any spare panels
    for ax in axes[nX + 2:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
