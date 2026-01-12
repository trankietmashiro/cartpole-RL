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
def get_reward(x, u, xd, Q, R, disc=None, oob_penalty=-1e6):
    """
    Quadratic reward:
      r = -[(x-xd)^T Q (x-xd) + u^T R u] + (oob_penalty if x is OOB)
    """
    x = np.asarray(x, dtype=float)
    xd = np.asarray(xd, dtype=float)

    dx = x - xd
    dx[2] = wrap_to_pi(dx[2])  # wrap angle error

    Q = np.asarray(Q, dtype=float)
    u = float(u)

    # allow scalar R
    if np.isscalar(R):
        u_cost = float(R) * (u ** 2)
    else:
        R = np.asarray(R, dtype=float)
        u_cost = float(np.array([u]) @ R @ np.array([u]))

    cost = float(dx @ Q @ dx + u_cost)
    r = -cost

    if disc is not None:
        oob = np.any(x < disc.lows) or np.any(x > disc.highs)
        if oob:
            r += float(oob_penalty)  # oob_penalty is negative by default

    return float(r)

# =========================
# State discretization
# =========================
class Discretizer:
    """
    Discretizes continuous state x into a tuple of bin indices s=(i,j,k,l),
    and maps s back to a representative continuous state using bin centers.
    """
    def __init__(self, lows, highs, n_bins):
        self.lows  = np.asarray(lows, dtype=float)
        self.highs = np.asarray(highs, dtype=float)
        self.n_bins = np.asarray(n_bins, dtype=int)

        assert self.lows.shape == self.highs.shape == self.n_bins.shape
        assert np.all(self.highs > self.lows)
        assert np.all(self.n_bins >= 2)

        # spacing between neighboring bin centers (uniform)
        self.width = (self.highs - self.lows) / (self.n_bins - 1)

    def grid_edges(self):
        """
        Continuous bounds implied by the discretization grid (bin edges),
        assuming self.lows/self.highs are bin centers.
        """
        edge_lows  = self.lows  - 0.5 * self.width
        edge_highs = self.highs + 0.5 * self.width
        return edge_lows, edge_highs

    def is_oob(self, x):
        """True if x lies outside the grid edges."""
        x = np.asarray(x, dtype=float).reshape(-1)
        lo, hi = self.grid_edges()
        return bool(np.any(x < lo) or np.any(x > hi))

    def discretize(self, x):
        """
        x (continuous) -> s (tuple of bin indices)
        """
        x = np.asarray(x, dtype=float)
        x = np.clip(x, self.lows, self.highs)
        idx = np.rint((x - self.lows) / self.width).astype(int)
        idx = np.clip(idx, 0, self.n_bins - 1)
        return tuple(idx.tolist())

    def bin_center(self, s):
        """
        s (tuple of bin indices) -> representative continuous state (bin center)
        """
        s = np.asarray(s, dtype=float)
        return self.lows + s * self.width


# =========================
# Discrete MDP wrapper
# =========================
def discrete_transition(dyn, disc, s, a, xd, Q, R, dt, oob_penalty=-1e6, method="rk4"):
    """
    s : discrete state (tuple of bin indices)
    returns:
      s_next : discrete next state (tuple of bin indices)
      r      : reward (float)
    """

    # 1) representative continuous state for dynamics rollout
    x_center = disc.bin_center(s)

    # 2) propagate dynamics from representative state
    x_next = get_next_state_continuous(dyn, x_center, a, dt, method=method)

    # 3) next discrete state
    s_next = disc.discretize(x_next)

    # 4) reward: use OOB check from discretizer edges; evaluate quadratic at bin center (stable for tabular DP)
    r = get_reward(x_next, a, xd, Q, R, disc=disc, oob_penalty=oob_penalty)

    return s_next, float(r)

