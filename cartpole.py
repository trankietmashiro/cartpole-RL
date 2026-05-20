import numpy as np

# =========================
# Default physical parameters
# =========================
DEFAULT_PARAM = {
    'mc': 1.0,    # cart mass      [kg]
    'mp': 0.05,    # pole mass      [kg]
    'l':  0.5,    # half-pole len  [m]
    'g':  9.81,   # gravity        [m/s^2]
    'b':  0.0,    # cart friction  [N·s/m]
    'd':  0.0,    # pole damping   [N·m·s/rad]
}

def wrap_to_pi(angle_radians):
    """Wraps angle in radians to [-pi, pi]."""
    return np.arctan2(np.sin(angle_radians), np.cos(angle_radians))

# =========================
# Integrators: x_{t+1} from x_t
# =========================
def step_euler(dyn, x, u, dt):
    return x + dt * dyn(x, u)

def step_rk4(dyn, x, u, dt):
    k1 = dyn(x, u)
    k2 = dyn(x + 0.5 * dt * k1, u)
    k3 = dyn(x + 0.5 * dt * k2, u)
    k4 = dyn(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def get_next_state(x, u, dt=0.01, method="rk4", param=None):
    """
    Integrate one step of the cart-pole dynamics.

    x      : (4,) array  [cart_pos, pole_angle, cart_vel, pole_rate]
    u      : scalar      horizontal force on cart  [N]
    dt     : timestep    [s]
    method : 'euler' or 'rk4'
    param  : physical parameter dict (defaults to DEFAULT_PARAM)

    Returns x_next : (4,) array with pole angle wrapped to [-π, π].
    """
    if param is None:
        param = DEFAULT_PARAM

    # Bind param so the integrators see dyn(x, u)
    def dyn(xf, uf):
        return cartpole_dynamics(0, xf, uf, param)

    x = np.asarray(x, dtype=float)
    m = method.lower()
    if m == "euler":
        x_next = step_euler(dyn, x, u, dt)
    elif m == "rk4":
        x_next = step_rk4(dyn, x, u, dt)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'euler' or 'rk4'.")

    x_next = np.asarray(x_next, dtype=float)
    x_next[1] = wrap_to_pi(x_next[1])   # index 1 = pole angle
    return x_next

# =========================
# Reward (continuous)
# =========================
def get_cost(xnext, u, xd, Q=None, R=0.001, lows=None, highs=None, oob_penalty=1e6):
    """
      r = [(x-xd)^T Q (x-xd) + u^T R u] + (oob_penalty if x is OOB)
    """
    if Q is None:
        Q = np.eye(4)

    x  = np.asarray(xnext, dtype=float)
    xd = np.asarray(xd, dtype=float)

    dx = x - xd
    dx[1] = wrap_to_pi(dx[1])  # wrap pole-angle error  (index 1)

    Q = np.asarray(Q, dtype=float)
    u = float(u)

    if np.isscalar(R):
        u_cost = float(R) * (u ** 2)
    else:
        R = np.asarray(R, dtype=float)
        u_cost = float(np.array([u]) @ R @ np.array([u]))

    r = float(dx @ Q @ dx + u_cost)

    # FIX: apply out-of-bounds penalty (was accepted but never used)
    if lows is not None and highs is not None:
        if np.any(x < np.asarray(lows)) or np.any(x > np.asarray(highs)):
            r += oob_penalty

    return float(r)


# =========================================================================
#   cart-pole dynamics  (state: x = [x, theta, x_dot, theta_dot]^T)
# =========================================================================
def cartpole_dynamics(t, x, u, param):
    """Continuous-time dynamics xdot = f(t, x, u).

    x : (4, 1) column vector  [cart pos, pole angle, cart vel, pole rate]
    u : scalar or (1, 1)      horizontal force on cart
    param : dict with keys mc, mp, l, g, b, d
    Returns xdot as a (4, 1) column vector.
    """
    mc, mp, l = param['mc'], param['mp'], param['l']
    g, b, d   = param['g'],  param['b'],  param['d']

    xf = np.asarray(x).flatten()
    x1, x2, x3, x4 = xf[0], xf[1], xf[2], xf[3]
    u_val = float(np.asarray(u).flatten()[0])

    s = np.sin(x2)
    c = np.cos(x2)

    xdot = np.zeros(4)
    xdot[0] = x3
    xdot[1] = x4
    xdot[2] = (u_val - b * x3 + d * x4 * c / l
               + mp * s * (l * x4**2 + g * c)) / (mc + mp * s**2)
    xdot[3] = (-u_val * c + b * x3 * c
               - d * (mc + mp) * x4 / (mp * l)
               - mp * l * x4**2 * c * s
               - (mc + mp) * g * s / (mp * l)) / (l * (mc + mp * s**2))
    return xdot


# =========================================================================
#   cart-pole analytic Jacobians  (df/dx, df/du)
# =========================================================================
def cartpole_grads(t, x, u, param):
    """Analytic Jacobians of cartpole_dynamics at (x, u)."""
    mc, mp, l = param['mc'], param['mp'], param['l']
    g, b, d   = param['g'],  param['b'],  param['d']

    xf = np.asarray(x).flatten()
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
    df4dx3 = (b * c) / (l * (mp * s**2 + mc))
    df4dx4 = -((d * (mc + mp)) / (l * mp) + l * mp * x4 * np.sin(2 * x2)) \
             / (l * (mp * s**2 + mc))

    dfdx[2, 1] = df3dx2
    dfdx[2, 2] = df3dx3
    dfdx[2, 3] = df3dx4
    dfdx[3, 1] = df4dx2
    dfdx[3, 2] = df4dx3
    dfdx[3, 3] = df4dx4

    df3du =  1.0 / (mp * s**2 + mc)
    df4du = -c   / (l * (mp * s**2 + mc))
    dfdu = np.array([[0.0], [0.0], [df3du], [df4du]])

    return dfdx, dfdu


# =========================================================================
#   cart-pole animation (matplotlib).  Lazy-imports matplotlib so importing
#   this module for dynamics-only use stays cheap.
# =========================================================================
def animate_cartpole(t, x, param):
    """Animate a precomputed trajectory.

    t : (N,) array of times
    x : (4, N) array of states
    param : dict with key 'l' (pole length)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.animation import FuncAnimation

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

    pole_line, = ax.plot([], [], color=(0.9, 0.1, 0.0),
                         lw=4, solid_capstyle='round')
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
                         interval=50, blit=False, repeat=True)
    plt.show()
    return anim
