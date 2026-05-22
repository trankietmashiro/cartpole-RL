"""
TV-LQR trajectory tracking for a cart-pole swing-up.

Pipeline
--------
1. Run iLQR to produce a nominal (x*, u*) swing-up trajectory.
2. Linearise the dynamics along that trajectory to get (A_k, B_k).
3. Solve the time-varying discrete Riccati equation *backwards* along the
   trajectory → time-varying feedback gains K_k.
4. Simulate the closed-loop system from a perturbed initial condition:
       u_k = u*_k  +  K_k (x_k − x*_k)
5. Plot and animate both the nominal and closed-loop trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from cartpole import cartpole_dynamics, cartpole_grads, animate_cartpole
from ilqr_cartpole import ilqr


# =========================================================================
#   discretise: ZOH linearisation at a single (x, u) point
# =========================================================================
def discretise(t, x, u, param, dt):
    """
    Returns discrete-time Jacobians (A, B) via Euler zero-order hold:
        A = I + dt * df/dx
        B =     dt * df/du
    """
    Ac, Bc = cartpole_grads(t, x, u, param)
    A = np.eye(4) + dt * Ac
    B = dt * Bc
    return A, B


# =========================================================================
#   tvlqr_gains: backward Riccati sweep along the nominal trajectory
# =========================================================================
def tvlqr_gains(xtraj, utraj, param, dt, Q, R, Qf):
    """
    Solve the discrete-time LQR Riccati equation backwards along (xtraj, utraj).

    At each step the local LQR problem is:
        min  sum  delta_x^T Q delta_x + delta_u^T R delta_u  +  delta_xN^T Qf delta_xN
        s.t. delta_x_{k+1} = A_k delta_x_k + B_k delta_u_k

    Optimal gain:
        K_k   = -(R + B_k^T P_{k+1} B_k)^{-1} B_k^T P_{k+1} A_k
        P_k   = Q + A_cl^T P_{k+1} A_cl + K_k^T R K_k     (A_cl = A_k + B_k K_k)

    Parameters
    ----------
    xtraj : (nX, N)   nominal state trajectory from iLQR
    utraj : (nU, N-1) nominal input trajectory from iLQR
    param : dict      cartpole physical parameters
    dt    : float     timestep
    Q     : (nX, nX)  running state cost
    R     : (nU, nU)  running input cost
    Qf    : (nX, nX)  terminal state cost

    Returns
    -------
    Klist : list of (nU, nX) arrays, length N-1
        Klist[k] is the feedback gain at step k.
    """
    nX, N = xtraj.shape

    P = Qf.copy()              # initialise with terminal cost  P_N = Qf
    Klist = [None] * (N - 1)

    for k in range(N - 2, -1, -1):
        A, B = discretise(k * dt, xtraj[:, [k]], utraj[:, [k]], param, dt)

        # optimal gain
        S       = R + B.T @ P @ B
        K       = -np.linalg.solve(S, B.T @ P @ A)   # (nU, nX)
        Klist[k] = K

        # Riccati recursion
        Acl = A + B @ K
        P   = Q + K.T @ R @ K + Acl.T @ P @ Acl

    return Klist


# =========================================================================
#   simulate_tvlqr: closed-loop rollout under TV-LQR
# =========================================================================
def simulate_tvlqr(x0_cl, xtraj, utraj, Klist, param, dt):
    """
    Roll out the nonlinear cart-pole dynamics under TV-LQR feedback.

    Control law at each step:
        u_k = u*_k  +  K_k (x_k − x*_k)

    Parameters
    ----------
    x0_cl  : (4, 1)   (possibly perturbed) initial state
    xtraj  : (nX, N)  nominal state trajectory
    utraj  : (nU, N-1) nominal input trajectory
    Klist  : list of (nU, nX) gains from tvlqr_gains
    param  : dict
    dt     : float

    Returns
    -------
    x_cl : (nX, N)   closed-loop state trajectory
    u_cl : (nU, N-1) closed-loop input trajectory
    """
    nX, N = xtraj.shape
    nU    = utraj.shape[0]

    x_cl  = np.zeros((nX, N))
    u_cl  = np.zeros((nU, N - 1))
    x     = np.asarray(x0_cl).reshape(nX, 1)
    x_cl[:, [0]] = x

    t = 0.0
    for k in range(N - 1):
        delta_x = x - xtraj[:, [k]]
        u       = utraj[:, [k]] + Klist[k] @ delta_x
        u_cl[:, [k]] = u

        xdot = np.asarray(cartpole_dynamics(t, x, u, param)).reshape(nX, 1)
        x    = x + dt * xdot
        x_cl[:, [k + 1]] = x
        t   += dt

    return x_cl, u_cl


# =========================================================================
#   plot_results
# =========================================================================
def plot_results(t_vec, xtraj, utraj, x_cl, u_cl, xd):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle("iLQR nominal  vs  TV-LQR closed-loop", fontsize=13)

    state_info = [
        ("Cart position  [m]",   0),
        ("Pole angle  [rad]",    1),
        ("Cart velocity  [m/s]", 2),
        ("Pole ang-vel [rad/s]", 3),
    ]
    N   = xtraj.shape[1]
    t_u = t_vec[:N - 1]

    for idx, (label, si) in enumerate(state_info):
        ax = axes[idx // 2][idx % 2]
        ax.plot(t_vec, xtraj[si, :], 'k--', lw=1.5, label='iLQR nominal')
        ax.plot(t_vec, x_cl[si, :],  'b-',  lw=1.5, label='TV-LQR c/l')
        ax.axhline(float(xd.flat[si]), color='r', ls=':', lw=1, label='target')
        ax.set_xlabel("Time  [s]")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax_u = axes[2][0]
    ax_u.plot(t_u, utraj[0, :], 'k--', lw=1.5, label='iLQR nominal')
    ax_u.plot(t_u, u_cl[0, :],  'b-',  lw=1.5, label='TV-LQR c/l')
    ax_u.set_xlabel("Time  [s]")
    ax_u.set_ylabel("Force  [N]")
    ax_u.set_title("Control input")
    ax_u.legend(fontsize=7)
    ax_u.grid(True, alpha=0.3)

    ax_e = axes[2][1]
    err  = np.linalg.norm(x_cl - xtraj, axis=0)
    ax_e.plot(t_vec, err, 'r-', lw=1.5)
    ax_e.set_xlabel("Time  [s]")
    ax_e.set_ylabel("||x_cl − x*||")
    ax_e.set_title("State tracking error")
    ax_e.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================================================================
#   main
# =========================================================================
def main():
    # ── shared problem setup ──────────────────────────────────────────────
    T  = 2.5
    dt = 0.05
    N  = int(T / dt)
    nX = 4
    nU = 1

    true_param = {
        'mc': 10.0, 'mp': 2.0, 'l': 0.5,
        'g':   9.8, 'b':  0.1, 'd': 0.1,
    }

    x0 = np.array([[0.0], [0.0],    [0.0], [0.0]])
    xd = np.array([[0.0], [np.pi],  [0.0], [0.0]])

    # iLQR cost matrices (unchanged from ilqr_cartpole.py)
    Q_ilqr  = 0.0   * np.eye(nX)
    Qf_ilqr = 1.0e4 * np.eye(nX)
    R_ilqr  = np.array([[0.01]])

    # ── Step 1: iLQR → nominal plan ───────────────────────────────────────
    print("Running iLQR ...")
    xtraj, utraj, ktraj, Ktraj = ilqr(
        x0,
        np.zeros((nX, N)),
        np.zeros((nU, N - 1)),
        np.zeros((nU, N)),
        np.zeros((nU, nX, N)),
        N, dt, true_param, Q_ilqr, R_ilqr, Qf_ilqr, xd,
    )
    print("iLQR done.")

    # ── Step 2-3: TV-LQR Riccati sweep ───────────────────────────────────
    # Q_lqr / R_lqr tune tracking aggressiveness independently of iLQR costs.
    # Heavier Q_lqr → tighter state tracking; heavier R_lqr → softer inputs.
    Q_lqr  = 10.0  * np.eye(nX)
    Qf_lqr = 1.0e4 * np.eye(nX)
    R_lqr  = np.array([[0.1]])

    print("Computing TV-LQR gains ...")
    Klist = tvlqr_gains(xtraj, utraj, true_param, dt, Q_lqr, R_lqr, Qf_lqr)
    print("TV-LQR gains ready.")

    # ── Step 4: closed-loop sim from a perturbed x0 ───────────────────────
    # Small perturbation in cart position and pole angle to test rejection.
    x0_cl = x0 + np.array([[0.1], [0.05], [0.0], [0.0]])
    print(f"Simulating TV-LQR from x0 = {x0_cl.flatten()} ...")
    x_cl, u_cl = simulate_tvlqr(x0_cl, xtraj, utraj, Klist, true_param, dt)
    print("Closed-loop simulation done.")

    # ── Step 5: results ───────────────────────────────────────────────────
    t_vec = np.linspace(0, T, N)
    plot_results(t_vec, xtraj, utraj, x_cl, u_cl, xd)

    print("\nAnimating nominal trajectory ...")
    animate_cartpole(t_vec, xtraj, true_param)

    print("Animating closed-loop trajectory ...")
    animate_cartpole(t_vec, x_cl, true_param)


if __name__ == "__main__":
    main()
