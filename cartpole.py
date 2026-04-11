import numpy as np

# =========================
# Cartpole physical constants
# =========================
M_C = 1.0      # M
M_P = 0.1      # m
L   = 0.5      # l (half-length / COM distance)
G   = 9.81

MU_C = 0.0     # μ_c (cart Coulomb friction)
MU_P = 0.0     # μ_p (pole damping/friction)

def sgn_deadband(v, eps=1e-6):
    # paper uses sgn(xdot); this avoids chatter near 0
    if v > eps:  return 1.0
    if v < -eps: return -1.0
    return 0.0

def dynamics(x, F):
    """
    State x = [p, p_dot, theta, theta_dot]
    theta = 0 upright (this model makes theta=0 unstable due to +g*sin(theta))
    """
    p, p_dot, th, th_dot = x
    F = float(F)

    s = np.sin(th)
    c = np.cos(th)

    # Coulomb friction on cart
    Fc = MU_C * sgn_deadband(p_dot)

    M = M_C
    m = M_P
    l = L

    # Common intermediate from the paper-style equations
    # temp = (F + m*l*th_dot^2*sin(th) - μ_c*sgn(x_dot)) / (M + m)
    temp = (F + m*l*(th_dot**2)*s - Fc) / (M + m)

    # theta_ddot:
    # ( g*sin(th) - cos(th)*temp - (μ_p*th_dot)/(m*l) ) / ( l*(4/3 - (m*cos^2(th))/(M+m)) )
    denom = l * (4.0/3.0 - (m * c * c) / (M + m))
    th_ddot = (G*s - c*temp - (MU_P * th_dot) / (m*l)) / denom

    # x_ddot:
    # (F + m*l*(th_dot^2*sin(th) - th_ddot*cos(th)) - μ_c*sgn(x_dot)) / (M+m)
    p_ddot = (F + m*l*((th_dot**2)*s - th_ddot*c) - Fc) / (M + m)

    return np.array([p_dot, p_ddot, th_dot, th_ddot], dtype=float)


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

def wrap_to_pi(angle_radians):
    """Wraps angle in radians to [-pi, pi]."""
    return np.arctan2(np.sin(angle_radians), np.cos(angle_radians))

def get_next_state_continuous(dyn, x, u, dt, method="rk4"):
    m = method.lower()
    if m == "euler":
        x_next = step_euler(dyn, x, u, dt)
    elif m == "rk4":
        x_next = step_rk4(dyn, x, u, dt)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'euler' or 'rk4'.")

    x_next = np.asarray(x_next, dtype=float)
    x_next[2] = wrap_to_pi(x_next[2])
    return x_next

# =========================
# Reward (continuous)
# =========================
def get_cost(x, u, xd, Q, R, lows=None, highs=None, oob_penalty=1e6):
    """
      r = [(x-xd)^T Q (x-xd) + u^T R u] + (oob_penalty if x is OOB)
    """
    x  = np.asarray(x, dtype=float)
    xd = np.asarray(xd, dtype=float)

    dx = x - xd
    dx[2] = wrap_to_pi(dx[2])  # wrap angle error

    Q = np.asarray(Q, dtype=float)
    u = float(u)

    if np.isscalar(R):
        u_cost = float(R) * (u ** 2)
    else:
        R = np.asarray(R, dtype=float)
        u_cost = float(np.array([u]) @ R @ np.array([u]))

    r = float(dx @ Q @ dx + u_cost)

    if lows is not None and highs is not None:
        if np.any(x < lows) or np.any(x > highs):
            r += float(oob_penalty)

    return float(r)


# =========================
# Continuous transition (no Discretizer needed)
# =========================
def continuous_transition(
    dyn, x, a, xd, Q, R, dt,
    lows=None, highs=None,
    method="rk4",
    oob_penalty=1e6,
):
    """
    Roll out one step from a continuous state x under action a.

    Parameters
    ----------
    dyn   : dynamics function f(x, a) → x_dot
    x     : current continuous state (np.ndarray)
    a     : scalar action
    xd    : target state
    Q, R  : cost weights
    dt    : time step
    lows, highs : state-space bounds for OOB penalty (optional)
    method      : integration method ('euler' or 'rk4')
    oob_penalty : reward penalty added when x_next is out of bounds

    Returns
    -------
    x_next : np.ndarray  – next continuous state
    r      : float       – immediate reward
    """
    x_next = get_next_state_continuous(dyn, x, a, dt, method=method)
    r = get_cost(x_next, a, xd, Q, R, lows=lows, highs=highs,
                   oob_penalty=oob_penalty)
    return x_next, r
