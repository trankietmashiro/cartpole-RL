"""
PILCO - simple pendulum swing-up
Goal: swing pendulum from hanging down (θ=π) to upright (θ=0)

PILCO idea: instead of trying thousands of real experiments,
learn a model of the dynamics (GP), then optimize the policy
against that model. Only a few real experiments needed!
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


# ════════════════════════════════════════════
# STEP 1: THE REAL WORLD (pendulum physics)
# We only use this to collect real data.
# PILCO tries to minimize how often it calls this.
# ════════════════════════════════════════════

class Pendulum:
    def __init__(self):
        self.dt    = 0.05   # time step
        self.b     = 0.1    # friction
        self.u_max = 2.0    # max torque
        self.theta  = np.pi  # angle (π = hanging down)
        self.dtheta = 0.0    # angular velocity

    def reset(self):
        # start hanging down, with a tiny bit of noise
        self.theta  = np.pi + rng.normal(0, 0.01)
        self.dtheta = rng.normal(0, 0.01)
        return self._state()

    def step(self, u):
        # clamp action, then apply physics: θ'' = -sin(θ) - b·θ' + u
        u = np.clip(u, -self.u_max, self.u_max)
        self.dtheta += (-np.sin(self.theta) - self.b * self.dtheta + u) * self.dt
        self.theta  += self.dtheta * self.dt
        return self._state()

    def _state(self):
        # use [sin θ, cos θ, θ̇] instead of raw θ to avoid angle-wrap issues
        return np.array([np.sin(self.theta),
                         np.cos(self.theta),
                         self.dtheta])


# ════════════════════════════════════════════
# STEP 2: COST FUNCTION
# How bad is a given state?
# 0 = perfect (upright), 1 = worst
# ════════════════════════════════════════════

def cost(state):
    # goal state: upright = [sin0, cos0, 0] = [0, 1, 0]
    goal = np.array([0.0, 1.0, 0.0])
    dist_squared = np.sum((state - goal) ** 2)
    # saturating cost: close to goal → ~0, far away → ~1
    return 1.0 - np.exp(-0.5 * dist_squared / 0.25)


# ════════════════════════════════════════════
# STEP 3: GAUSSIAN PROCESS DYNAMICS MODEL
#
# A GP is a flexible model that also tells us
# HOW UNCERTAIN it is about its predictions.
#
# We train one GP per state dimension to predict
# how the state will change: Δs = s_{t+1} - s_t
# ════════════════════════════════════════════

class GP:
    def __init__(self):
        # these are kernel hyperparameters (kept fixed for simplicity)
        self.signal_std = 1.0    # how much the function can vary
        self.length_scale = 1.0  # how quickly it varies
        self.noise_std = 0.1     # observation noise
        # training data (set when we call fit)
        self.X = None  # inputs:  [state, action]
        self.Y = None  # targets: Δstate

    def _kernel(self, A, B):
        # squared-exponential (RBF) kernel: k(a,b) = σ² exp(-||a-b||² / 2l²)
        # measures "similarity" between two input points
        diff = A[:, None, :] - B[None, :, :]        # shape (n, m, d)
        sq_dist = np.sum(diff ** 2, axis=-1)         # shape (n, m)
        return self.signal_std**2 * np.exp(-sq_dist / (2 * self.length_scale**2))

    def fit(self, X, Y):
        # store training data and pre-compute the kernel inverse
        # (this is the expensive part: O(n³), but we only do it once per iteration)
        self.X = X
        self.Y = Y
        n = len(X)
        K = self._kernel(X, X) + self.noise_std**2 * np.eye(n)
        # alpha = K⁻¹ Y  (precomputed for fast predictions later)
        self.alpha = np.linalg.solve(K, Y)
        # L is the Cholesky factor, used to compute predictive variance
        self.L = np.linalg.cholesky(K)

    def predict(self, X_test):
        # GP prediction at new points X_test
        Ks = self._kernel(X_test, self.X)       # covariance: test vs train
        mean = Ks @ self.alpha                   # predictive mean
        v = np.linalg.solve(self.L, Ks.T)       # for variance computation
        # predictive variance: how uncertain is the GP here?
        # high variance = GP has not seen data near this point
        var = self.signal_std**2 - np.sum(v**2, axis=0)
        return mean, np.maximum(var, 1e-8)       # clip to stay positive


# ════════════════════════════════════════════
# STEP 4: POLICY (what action to take)
#
# A simple linear policy: u = tanh(W · s) * u_max
# We will optimize W to minimize expected cost.
# ════════════════════════════════════════════

class LinearPolicy:
    def __init__(self, state_dim=3, u_max=2.0):
        self.u_max = u_max
        # random initial weights (state_dim inputs → 1 action)
        self.W = rng.normal(0, 0.1, state_dim)

    def __call__(self, state):
        # squash with tanh so output stays in [-u_max, +u_max]
        return self.u_max * np.tanh(self.W @ state)

    def get_params(self):
        return self.W.copy()

    def set_params(self, w):
        self.W = w.copy()


# ════════════════════════════════════════════
# STEP 5: COLLECT REAL DATA
# Run the real pendulum for T steps and
# record every (state, action, next_state) triple.
# ════════════════════════════════════════════

def collect_data(env, policy, T=40):
    states, actions = [], []
    s = env.reset()
    total_cost = 0.0

    for _ in range(T):
        u = policy(s)
        states.append(s)
        actions.append(u)
        total_cost += cost(s)
        s = env.step(u)

    # removed: states.append(s)  ← this was causing the length mismatch
    return np.array(states), np.array(actions), total_cost


# ════════════════════════════════════════════
# STEP 6: FIT GP MODELS TO COLLECTED DATA
#
# Input:  z = [state_t, action_t]
# Target: Δstate = state_{t+1} - state_t
#
# One GP per state dimension (3 GPs total).
# ════════════════════════════════════════════

def fit_gp_models(all_states, all_actions):
    all_Z, all_dS = [], []

    for states, actions in zip(all_states, all_actions):
        # states: (T, 3),  actions: (T,)
        # Δs[t] = states[t+1] - states[t], so we only have T-1 pairs
        Z  = np.column_stack([states[:-1], actions[:-1]])  # (T-1, 4)
        dS = states[1:] - states[:-1]                      # (T-1, 3)
        all_Z.append(Z)
        all_dS.append(dS)

    Z  = np.vstack(all_Z)
    dS = np.vstack(all_dS)

    models = []
    for d in range(3):
        gp = GP()
        gp.fit(Z, dS[:, d])
        models.append(gp)

    print(f"  GP trained on {len(Z)} transitions")
    return models


# ════════════════════════════════════════════
# STEP 7: SIMULATE INSIDE THE GP MODEL
#
# This is the core of PILCO:
# instead of running the real pendulum,
# we imagine rollouts using the GP model.
#
# The GP also gives us uncertainty, so we
# sample multiple possible futures.
# ════════════════════════════════════════════

def simulate_in_model(policy, gp_models, s0, T=40, n_samples=30):
    total_cost = 0.0

    for _ in range(n_samples):
        s = s0.copy()
        for _ in range(T):
            u = policy(s)
            # build the input the GP expects: [state, action]
            z = np.append(s, u).reshape(1, -1)

            # ask each GP: "what will Δs be?"
            # GP gives mean + uncertainty → sample from it
            ds = np.zeros(3)
            for d, gp in enumerate(gp_models):
                mean, var = gp.predict(z)
                # sample the next state change (incorporates uncertainty!)
                ds[d] = rng.normal(mean[0], np.sqrt(var[0]))

            s = s + ds
            total_cost += cost(s)

    # return average cost across all imagined futures
    return total_cost / n_samples


# ════════════════════════════════════════════
# STEP 8: OPTIMIZE THE POLICY
#
# Use finite-difference gradients to find
# policy weights W that minimize J(π).
#
# All of this happens inside the GP model —
# no real pendulum runs needed here!
# ════════════════════════════════════════════

def optimize_policy(policy, gp_models, s0, n_iters=40, lr=0.05):
    costs = []
    eps = 1e-3  # finite difference step size

    for i in range(n_iters):
        w0 = policy.get_params()
        J0 = simulate_in_model(policy, gp_models, s0)  # current cost

        # compute gradient: how does cost change if we nudge each weight?
        grad = np.zeros_like(w0)
        for k in range(len(w0)):
            w_plus = w0.copy()
            w_plus[k] += eps
            policy.set_params(w_plus)
            J_plus = simulate_in_model(policy, gp_models, s0, n_samples=10)
            grad[k] = (J_plus - J0) / eps  # ∂J/∂w_k

        # gradient descent step: move weights in direction of lower cost
        policy.set_params(w0 - lr * grad)
        costs.append(J0)

        if i % 10 == 0:
            print(f"  policy iter {i:3d}  model cost = {J0:.3f}")

    return costs


# ════════════════════════════════════════════
# MAIN PILCO LOOP
#
# The full algorithm:
# 1. Collect real data (with random / current policy)
# 2. Fit GP models to all data so far
# 3. Optimize policy against GP model
# 4. Deploy policy, collect more real data
# 5. Repeat (usually converges in 3-10 episodes!)
# ════════════════════════════════════════════

def main():
    env    = Pendulum()
    policy = LinearPolicy(state_dim=3, u_max=2.0)

    all_states  = []
    all_actions = []
    real_costs  = []

    print("=" * 50)
    print("  PILCO — pendulum swing-up")
    print("=" * 50)

    # --- seed with one random episode so GP has some data to start ---
    print("\n[init] Random rollout to seed the GP...")
    random_policy = lambda s: rng.uniform(-2, 2)
    states, actions, _ = collect_data(env, random_policy, T=40)
    all_states.append(states)
    all_actions.append(actions)

    # --- PILCO outer loop ---
    for iteration in range(6):
        print(f"\n{'─'*50}")
        print(f"[iter {iteration+1}] Fitting GP dynamics model...")
        gp_models = fit_gp_models(all_states, all_actions)

        print(f"[iter {iteration+1}] Optimizing policy inside GP model...")
        s0 = env.reset()
        optimize_policy(policy, gp_models, s0, n_iters=40, lr=0.05)

        print(f"[iter {iteration+1}] Running optimized policy on real pendulum...")
        states, actions, J = collect_data(env, policy, T=40)
        all_states.append(states)
        all_actions.append(actions)
        real_costs.append(J)
        print(f"  → Real cumulative cost: {J:.3f}")

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # learning curve
    axes[0].plot(range(1, len(real_costs)+1), real_costs, 'o-',
                 color='#185FA5', linewidth=2, markersize=8)
    axes[0].set_xlabel("PILCO iteration (real episodes used)")
    axes[0].set_ylabel("Cumulative cost")
    axes[0].set_title("PILCO learns fast — only a few real episodes!")
    axes[0].grid(True, alpha=0.3)

    # final episode trajectory
    final_states = all_states[-1]
    t = np.arange(len(final_states)) * 0.05
    axes[1].plot(t, final_states[:, 0], label='sin θ')
    axes[1].plot(t, final_states[:, 1], label='cos θ')
    axes[1].plot(t, final_states[:, 2], label='θ̇')
    axes[1].axhline(1.0, color='green', linestyle='--', alpha=0.4, label='goal (cosθ=1)')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Final episode state trajectory")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pilco_result.png", dpi=130)
    plt.show()
    print("\nDone! Plot saved to pilco_result.png")


if __name__ == "__main__":
    main()
