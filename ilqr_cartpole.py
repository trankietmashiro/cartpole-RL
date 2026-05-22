"""
Iterative LQR (iLQR) swing-up of a cart-pole.

Python translation of the MATLAB script `iterative_lqr_cart_pole.m`.
Mirrors the original structure (ilqr -> forward_pass / backward_pass,
analytic dynamics + analytic Jacobians, terminal-cost-only formulation
for the swing-up).

Run:
    python ilqr_cartpole.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
from cartpole import cartpole_dynamics, cartpole_grads, animate_cartpole


# =========================================================================
#   main
# =========================================================================
def main():
    T = 2.5            # time horizon (s)
    dt = 0.05          # time step (s)
    N = int(T / dt)    # number of time steps
    nX = 4             # number of states
    nU = 1             # number of inputs

    # cartpole physical parameters
    true_param = {
        'mc': 10.0,    # cart mass
        'mp': 2.0,     # pole mass
        'l':  0.5,     # pole length
        'g':  9.8,     # gravity
        'b':  0.1,     # cart viscous friction
        'd':  0.1,     # pole viscous friction
    }

    # initial conditions (column vectors throughout)
    x0 = np.array([[0.0], [0.0], [0.0], [0.0]])

    xtraj = np.zeros((nX, N))           # state trajectory
    utraj = np.zeros((nU, N - 1))       # input trajectory
    ktraj = np.zeros((nU, N))           # feedforward gains
    Ktraj = np.zeros((nU, nX, N))       # feedback gains

    Q  = 0.0    * np.eye(nX)            # running state cost
    Qf = 1.0e4  * np.eye(nX)            # final state cost
    R  = np.array([[0.01]])             # input cost
    xd = np.array([[0.0], [np.pi], [0.0], [0.0]])   # desired (upright)

    # call iterative LQR
    xtraj, utraj, ktraj, Ktraj = ilqr(
        x0, xtraj, utraj, ktraj, Ktraj, N, dt, true_param, Q, R, Qf, xd
    )

    # simulate the optimized open-loop trajectory
    t = np.zeros(N)
    x = np.zeros((nX, N))
    x[:, [0]] = x0
    for k in range(N - 1):
        xdot = cartpole_dynamics(t[k], x[:, [k]], utraj[:, [k]], true_param).reshape(-1, 1)
        x[:, [k + 1]] = x[:, [k]] + xdot * dt
        t[k + 1] = t[k] + dt

    # animate
    animate_cartpole(t, x, true_param)


# =========================================================================
#   iterative LQR
# =========================================================================
def ilqr(x0, xtraj, utraj, ktraj, Ktraj, N, dt, param, Q, R, Qf, xd):
    J = 1e6
    Jlast = J
    xtrajprev = xtraj
    utrajprev = utraj
    for i in range(1000):
        xtraj, utraj, J = forward_pass(
            x0, xtraj, utraj, ktraj, Ktraj, N, dt, param, Q, R, Qf, xd, J, xtrajprev, utrajprev
        )
        Ktraj, ktraj = backward_pass(
            xtraj, utraj, ktraj, Ktraj, Q, R, Qf, xd, param, N, dt, xtrajprev, utrajprev
        )
        if abs(J - Jlast) < 1e-6:
            break
        Jlast = J

    return xtraj, utraj, ktraj, Ktraj


# =========================================================================
#   Q-terms
# =========================================================================
def Q_terms(gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx):
    rho = 0
    Vxx_reg = Vxx + rho * np.eye(Vxx.shape[0])

    Qx  = gx  + fx.T @ Vx
    Qu  = gu  + fu.T @ Vx
    Qxx = gxx + fx.T @ Vxx @ fx
    Qux = gux + fu.T @ Vxx @ fx
    Quu = guu + fu.T @ Vxx @ fu

    Quxbar = gux + fu.T @ Vxx_reg @ fx
    Quubar = guu + fu.T @ Vxx_reg @ fu

    return Qx, Qu, Qxx, Qux, Quu, Quxbar, Quubar


# =========================================================================
#   gains
# =========================================================================
def get_gains(Qx, Qu, Qxx, Qux, Quu):

    rho = 1e-4
    Quu_reg = Quu + rho * np.eye(Quu.shape[0])
    Quu_inv = np.linalg.inv(Quu_reg)
    v = -Quu_inv @ Qu       # feedforward correction
    K = -Quu_inv @ Qux      # feedback gain
    return K, v


# =========================================================================
#   V-terms
# =========================================================================
def V_terms(Qx, Qu, Qxx, Qux, Quu, K, v):
    Vx  = Qx + K.T @ Qu + Qux.T @ v + K.T @ Quu @ v
    Vxx = Qxx + K.T @ Qux + Qux.T @ K + K.T @ Quu @ K
    return Vx, Vxx


# =========================================================================
#   backward pass
# =========================================================================
def backward_pass(xtraj, utraj, ktraj, Ktraj, Q, R, Qf, xd, param, N, dt, xtrajprev, utrajprev):
    nX = xtraj.shape[0]

    # terminal V from final-cost gradients
    xN = xtraj[:, [N - 1]]
    uN = utraj[:, [N - 2]]
    gxN, _, gxxN, _, _ = final_cost_gradients(xN, uN, xd, Qf)
    Vx  = gxN
    Vxx = gxxN

    # iterate from N-2 down to 0
    for i in range(N - 2, -1, -1):
        gx, gu, gxx, gux, guu = cost_gradients(
            xtraj[:, [i]], xtrajprev[:, [i]], utraj[:, [i]], utrajprev[:, [i]], xd, Q, R
        )
        fx, fu = cartpole_grads(0.0, xtraj[:, [i]], utraj[:, [i]], param)
        fu = fu * dt
        fx = fx * dt + np.eye(nX)

        Qx, Qu, Qxx, Qux, Quu, Quxbar, Quubar = Q_terms(
            gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx
        )
        K, v = get_gains(Qx, Qu, Qxx, Quxbar, Quubar)
        Ktraj[:, :, i] = K
        ktraj[:, [i]]  = v
        Vx, Vxx = V_terms(Qx, Qu, Qxx, Qux, Quu, K, v)

    return Ktraj, ktraj


# =========================================================================
#   running cost
# =========================================================================
def cost(x, x0, u, u0, Q, R):
    return float((0.5 * (x-x0).T @ Q @ (x-x0) + 0.5 * (u-u0).T @ R @ (u-u0)).item())


# =========================================================================
#   final cost
# =========================================================================
def final_cost(x, u, xd, Qf):
    e = x - xd
    return float((0.5 * e.T @ Qf @ e).item())


# =========================================================================
#   final cost gradients
# =========================================================================
def final_cost_gradients(x, u, xd, Qf):
    nX = x.shape[0]
    nU = u.shape[0]
    gx  = Qf @ (x - xd)
    gu  = np.zeros((nU, 1))
    gxx = Qf
    gux = np.zeros((nU, nX))
    guu = np.zeros((nU, nU))
    return gx, gu, gxx, gux, guu


# =========================================================================
#   running cost gradients
# =========================================================================
def cost_gradients(x, x0, u, u0, xd, Q, R):
    nX = x.shape[0]
    nU = u.shape[0]
    gx  = Q @ (x-x0)
    gu  = R @ (u-u0)
    gxx = Q
    gux = np.zeros((nU, nX))
    guu = R
    return gx, gu, gxx, gux, guu


# =========================================================================
#   forward pass with backtracking line search on alpha
# =========================================================================
def forward_pass(x0, xtraj0, utraj0, ktraj, Ktraj, N, dt,
                 param, Q, R, Qf, xd, J0, xtrajprev, utrajprev):
    nX = x0.shape[0]
    nU = utraj0.shape[0]

    J = 1e7
    alpha = 1e0
    xtraj = np.zeros((nX, N))
    utraj = np.zeros((nU, N - 1))

    while J0 < J:
        xtraj = np.zeros((nX, N))
        utraj = np.zeros((nU, N - 1))
        t = 0.0
        x = x0.copy()
        J = 0.0
        for i in range(N - 1):
            xtraj[:, [i]] = x
            u = (utraj0[:, [i]]
                 + alpha * ktraj[:, [i]]
                 + Ktraj[:, :, i] @ (x - xtraj0[:, [i]]))
            utraj[:, [i]] = u
            J += cost(x, xtrajprev[:, [i]], u, utrajprev[:, [i]], Q, R)
            xdot = cartpole_dynamics(t, x, u, param).reshape(x.shape)
            t += dt
            x = x + xdot * dt
        xtraj[:, [N - 1]] = x
        J += final_cost(x, utraj[:, [N - 2]], xd, Qf)
        alpha /= 2.0
        if alpha < 1e-10:    # safety: avoid an unbounded line search
            break

    return xtraj, utraj, J

if __name__ == "__main__":
    main()
