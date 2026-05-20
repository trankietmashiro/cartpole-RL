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
        xdot = cartpole_dynamics(t[k], x[:, [k]], utraj[:, [k]], true_param)
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
            xdot = cartpole_dynamics(t, x, u, param)
            t += dt
            x = x + xdot * dt
        xtraj[:, [N - 1]] = x
        J += final_cost(x, utraj[:, [N - 2]], xd, Qf)
        alpha /= 2.0
        if alpha < 1e-10:    # safety: avoid an unbounded line search
            break

    return xtraj, utraj, J


# =========================================================================
#   cart pole dynamics
# =========================================================================
def cartpole_dynamics(t, x, u, param):
    mc, mp, l = param['mc'], param['mp'], param['l']
    g, b, d   = param['g'],  param['b'],  param['d']

    xf = x.flatten()
    x1, x2, x3, x4 = xf[0], xf[1], xf[2], xf[3]
    u_val = float(np.asarray(u).flatten()[0])

    s = np.sin(x2)
    c = np.cos(x2)

    xdot = np.zeros((4, 1))
    xdot[0, 0] = x3
    xdot[1, 0] = x4
    xdot[2, 0] = (u_val - b * x3 + d * x4 * c / l
                  + mp * s * (l * x4**2 + g * c)) / (mc + mp * s**2)
    xdot[3, 0] = (-u_val * c + b * x3 * c
                  - d * (mc + mp) * x4 / (mp * l)
                  - mp * l * x4**2 * c * s
                  - (mc + mp) * g * s / (mp * l)) / (l * (mc + mp * s**2))
    return xdot


# =========================================================================
#   cart pole gradients (analytic Jacobians from the MATLAB version)
# =========================================================================
def cartpole_grads(t, x, u, param):
    mc, mp, l = param['mc'], param['mp'], param['l']
    g, b, d   = param['g'],  param['b'],  param['d']

    xf = x.flatten()
    x1, x2, x3, x4 = xf[0], xf[1], xf[2], xf[3]
    u_val = float(np.asarray(u).flatten()[0])

    s = np.sin(x2)
    c = np.cos(x2)

    dfdx = np.zeros((4, 4))
    dfdx[0, 2] = 1.0
    dfdx[1, 3] = 1.0

    df3dx2 = (
        -(g * mp * s**2 - mp * c * (l * x4**2 + g * c) + (d * x4 * s) / l)
        / (mp * s**2 + mc)
        - (2 * mp * c * s
           * (u_val - b * x3 + mp * s * (l * x4**2 + g * c) + (d * x4 * c) / l))
        / (mp * s**2 + mc) ** 2
    )
    df3dx3 = -b / (mp * s**2 + mc)
    df3dx4 = ((d * c) / l + 2 * l * mp * x4 * s) / (mp * s**2 + mc)

    df4dx2 = (
        (2 * mp * c * s
         * (l * mp * c * s * x4**2 + (d * (mc + mp) * x4) / (l * mp)
            + u_val * c - b * x3 * c + (g * s * (mc + mp)) / (l * mp)))
        / (l * (mp * s**2 + mc) ** 2)
        - (b * x3 * s - u_val * s
           + l * mp * x4**2 * c**2 - l * mp * x4**2 * s**2
           + (g * c * (mc + mp)) / (l * mp))
        / (l * (mp * s**2 + mc))
    )
    df4dx3 = (b * c) / (l * (mc - mp * (c**2 - 1)))
    df4dx4 = -((d * (mc + mp)) / (l * mp) + l * mp * x4 * np.sin(2 * x2)) \
             / (l * (mp * s**2 + mc))

    dfdx[2, 0] = 0.0
    dfdx[2, 1] = df3dx2
    dfdx[2, 2] = df3dx3
    dfdx[2, 3] = df3dx4

    dfdx[3, 0] = 0.0
    dfdx[3, 1] = df4dx2
    dfdx[3, 2] = df4dx3
    dfdx[3, 3] = df4dx4

    df3du =  1.0 / (mp * s**2 + mc)
    df4du = -c   / (l * (mc - mp * (c**2 - 1)))
    dfdu = np.array([[0.0], [0.0], [df3du], [df4du]])

    return dfdx, dfdu


# =========================================================================
#   cart-pole animation (matplotlib)
# =========================================================================
def animate_cartpole(t, x, param):
    l = param['l']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5 * l, 2.5 * l)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    wb, hb, wheelr = 0.3, 0.15, 0.05

    cart   = Rectangle((-wb / 2, -hb / 2), wb, hb,
                       facecolor=(0.3, 0.6, 0.4), edgecolor='k')
    lwheel = Circle((0, 0), wheelr, color='k')
    rwheel = Circle((0, 0), wheelr, color='k')
    ax.add_patch(cart); ax.add_patch(lwheel); ax.add_patch(rwheel)

    pole_line, = ax.plot([], [], color=(0.9, 0.1, 0.0), lw=4, solid_capstyle='round')
    bob = Circle((0, 0), 0.06, facecolor='b', edgecolor='k', zorder=5)
    ax.add_patch(bob)

    title = ax.set_title('')

    def update(i):
        xi  = x[0, i]
        thi = x[1, i]
        cart.set_xy((xi - wb / 2, -hb / 2))
        lwheel.center = (xi - wb / 2 + wheelr, -hb - wheelr)
        rwheel.center = (xi + wb / 2 - wheelr, -hb - wheelr)
        bx = xi + l * np.sin(thi)
        by = -l * np.cos(thi)
        pole_line.set_data([xi, bx], [0, by])
        bob.center = (bx, by)
        title.set_text(f't = {t[i]:.2f} sec')
        return cart, lwheel, rwheel, pole_line, bob, title

    anim = FuncAnimation(fig, update, frames=len(t),
                         interval=200, blit=False, repeat=True)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
