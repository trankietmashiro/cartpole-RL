import numpy as np
from cartpole import dynamics, get_next_state_continuous
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Model
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# Tensor helpers
# ============================================================
def to_state_tensor(x, device):
    """
    Accepts:
      - numpy (4, N) or (N, 4) or (4,) or (N,)
      - torch (4, N) or (N, 4) or (4,)
    Returns torch float32 (N, 4)
    """
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
    elif not torch.is_tensor(x):
        raise TypeError(f"Unsupported type for x: {type(x)}")

    if x.ndim == 1:
        x = x.unsqueeze(0)

    # If (4, N) -> (N, 4)
    if x.shape[0] == 4 and x.shape[1] != 4:
        x = x.transpose(0, 1)

    if x.shape[1] != 4:
        raise ValueError(f"State must have 4 dims, got shape {tuple(x.shape)}")

    return x


def predict_Q(Qnet, x):
    device = next(Qnet.parameters()).device
    x_t = to_state_tensor(x, device)
    return Qnet(x_t)  # (N, A)


# ============================================================
# Cost / reward (you are using cost as r)
# ============================================================
def get_cost_weights(dt):
    Q_cost = np.diag([10.0, 1.0, 10.0, 1.0]) * dt   # example weights
    R_cost = 1.0 * dt
    return Q_cost, R_cost


def cost_function(X, u, Xd, dt):
    """
    X, Xd: (4,) or (4,N)
    u: scalar or (N,) or (N,1)
    Returns: (N,) cost
    """
    Q_cost, R_cost = get_cost_weights(dt)

    X  = np.asarray(X, dtype=float)
    Xd = np.asarray(Xd, dtype=float)
    u  = np.asarray(u, dtype=float)

    if X.ndim == 1:  X = X.reshape(4, 1)
    if Xd.ndim == 1: Xd = Xd.reshape(4, 1)

    if X.shape[0] != 4 or Xd.shape[0] != 4:
        raise ValueError(f"X and Xd must have 4 rows, got {X.shape}, {Xd.shape}")

    if Xd.shape[1] == 1 and X.shape[1] > 1:
        Xd = np.repeat(Xd, X.shape[1], axis=1)

    N = X.shape[1]
    if u.ndim == 0:
        u = np.full((N,), float(u))
    else:
        u = u.reshape(-1)
        if u.size == 1 and N > 1:
            u = np.full((N,), float(u[0]))
        if u.size != N:
            raise ValueError(f"u has size {u.size}, but batch N={N}")

    E = X - Xd
    C_state = np.sum(E * (Q_cost @ E), axis=0)     # (N,)
    C_ctrl  = R_cost * (u ** 2)                    # (N,)
    return C_state + C_ctrl


# ============================================================
# DQN utilities
# ============================================================
def get_Qstar(Qx):
    # Qx: (batch, num_actions)
    return torch.max(Qx, dim=1).values  # (batch,)


def get_q_factor(actions, Qx):
    # actions: (batch,) long
    return Qx.gather(1, actions.unsqueeze(1)).squeeze(1)


def set_q_factor(a, Qxt, Qx):
    """
    a:   (batch,) long
    Qxt: (batch,) float
    Qx:  (batch, num_actions) float
    """
    Qj = Qx.clone()
    idx = torch.arange(Qj.shape[0], device=Qj.device)
    Qj[idx, a] = Qxt
    return Qj


def td_update(x, a, x1, Qm, Qt, gamma, alpha, xd, dt):
    """
    Returns Q_target: a full (batch, A) tensor where only the chosen actions
    are replaced with the TD-updated targets.

    x, x1: numpy or torch; (4,N) or (N,4) or (4,)
    a:     numpy or torch; (N,) action indices
    """
    device = next(Qm.parameters()).device

    # States -> tensors (N,4)
    x_t  = to_state_tensor(x, device)
    x1_t = to_state_tensor(x1, device)

    # Actions -> long tensor (N,)
    if isinstance(a, np.ndarray):
        a_t = torch.as_tensor(a.reshape(-1), dtype=torch.long, device=device)
    elif torch.is_tensor(a):
        a_t = a.to(device=device, dtype=torch.long).reshape(-1)
    else:
        raise TypeError(f"Unsupported type for a: {type(a)}")

    # Predict Q(s,·) using main net (common; keeps targets aligned with training head)
    Qx  = Qm(x_t)     # (N, A)

    # Next-state Q(s',·) using target net
    with torch.no_grad():
        Qx1 = Qt(x1_t)            # (N, A)
        Qhat1 = get_Qstar(Qx1)    # (N,)

    # Current Q(s,a)
    Qhat0 = get_q_factor(a_t, Qx) # (N,)

    # Cost as "r" (your design choice)
    # cost_function expects (4,N) for batch; convert x1 back to (4,N)
    x1_np_4N = x1_t.detach().cpu().numpy().T  # (4,N)
    a_np = a_t.detach().cpu().numpy()         # (N,)

    r_np = cost_function(x1_np_4N, a_np, xd, dt)  # (N,)
    r = torch.as_tensor(r_np, dtype=Qx.dtype, device=device)

    # TD update for chosen action only
    Qxt = Qhat0 + alpha * (r + gamma * Qhat1 - Qhat0)  # (N,)

    # Full target matrix
    Q_target = set_q_factor(a_t, Qxt, Qx)  # (N, A)
    return Q_target


def learn_q_factor(x, Q_target, Qm):
    if not hasattr(Qm, "optimizer"):
        raise AttributeError("Attach optimizer to Qm as Qm.optimizer")

    device = next(Qm.parameters()).device
    x_t = to_state_tensor(x, device)

    Qm.train()
    pred = Qm(x_t)  # (N,A)

    if pred.shape != Q_target.shape:
        raise ValueError(f"pred {pred.shape} vs target {Q_target.shape}")

    loss = torch.mean((pred - Q_target) ** 2)

    Qm.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(Qm.parameters(), max_norm=10.0)
    Qm.optimizer.step()
    return loss.item()


def get_action(x, Qm, epsilon, num_actions):
    # x can be (4,) or (4,1) or (N,4) but you return ONE action, so assume single state.
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    Qx = predict_Q(Qm, x)  # (1, A) if single state
    return int(Qx.argmax(dim=1)[0].item())


def train(xk, ak, x1k, Qm, Qt, gamma, alpha, xd, dt, batch_size=500, rng=None):
    """
    Assumes replay buffers:
      xk, x1k: (4, N)
      ak: (N,) or (1,N) or (N,1)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = xk.shape[1]  # (4, N)
    if N == 0:
        return None

    inds = rng.integers(low=0, high=N, size=min(batch_size, N))
    xs  = xk[:, inds]      # (4, B)
    x1s = x1k[:, inds]     # (4, B)

    if ak.ndim == 2:
        as_ = ak[:, inds].reshape(-1)
    else:
        as_ = ak[inds].reshape(-1)

    Q_target = td_update(xs, as_, x1s, Qm, Qt, gamma, alpha, xd, dt)
    loss = learn_q_factor(xs, Q_target, Qm)
    return loss


# ============================================================
# Setup
# ============================================================
actions_range = (-2.0, 2.0)
num_actions = 5
actions_bin = np.linspace(actions_range[0], actions_range[1], num_actions)

num_states = 4

Qt = MLP(input_dim=num_states, hidden_dim1=256, hidden_dim2=256, output_dim=num_actions)
Qm = MLP(input_dim=num_states, hidden_dim1=256, hidden_dim2=256, output_dim=num_actions)

dt = 0.1

max_episodes = 500
max_steps_per_episode = 200

steps_train = 20
gamma = 0.99
alpha = 0.1

x0 = np.array([0.0, 0.0, np.pi, 0.0])
xd = np.array([0.0, 0.0, 0.0, 0.0])

x0k = np.zeros((num_states, steps_train))
ak = np.zeros((1, steps_train))
Qmk = np.zeros((steps_train, num_actions))
Qm = learn_q_factor(x0k, torch.as_tensor(Qmk, dtype=torch.float32), Qm)
Qt = learn_q_factor(x0k, torch.as_tensor(Qmk, dtype=torch.float32), Qt)

for i in range(max_episodes):
    x = x0.copy()
    epsilon = 0.01 + 0.9 * np.exp(-i * 0.01)
    for t in range(max_steps_per_episode):
        a_idx = get_action(x, Qm, epsilon, num_actions=num_actions)
        a = actions_bin[a_idx]

        x1 = get_next_state_continuous(x, a, dt)
        x = x1.copy()

        # break if linear (x1[1]) or angular velocity (x1[3]) exceed bounds
        if np.any(np.abs(x1[[1, 3]]) > 1.0):
            break
    
    print(f"Episode {i}: steps = {t+1}, epsilon = {epsilon:.3f}")
