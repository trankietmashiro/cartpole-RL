import numpy as np
from cartpole import cartpole_dynamics, cartpole_grads, animate_cartpole
from ilqr import ILQR, DynamicsModel, QuadraticCost
from tvlqr import TVLQR, plot_comparison

param = {
    'mc': 10.0, 'mp': 2.0, 'l': 0.5,
    'g':   9.8, 'b':  0.1, 'd': 0.1,
}

class CartpoleDynamics(DynamicsModel):
    def __init__(self, p): self.p = p
    def step(self, t, x, u):
        return cartpole_dynamics(t, x, u, self.p).reshape(-1, 1)
    def jacobians(self, t, x, u):
        return cartpole_grads(t, x, u, self.p)

nX, nU = 4, 1
T,  dt = 2.5, 0.05
N       = int(T / dt)

x0 = np.zeros((nX, 1))
xd = np.array([[0.0], [np.pi], [0.0], [0.0]])

dyn = CartpoleDynamics(param)

# ── Step 1: iLQR nominal plan ─────────────────────────────────────────
print("Running iLQR ...")
ilqr_cost = QuadraticCost(
    Q  = np.zeros((nX, nX)),
    R  = 0.01 * np.eye(nU),
    Qf = 1e4  * np.eye(nX),
)
nominal = ILQR(dyn, ilqr_cost).solve(x0, xd, N, dt, nU)
print(f"iLQR done in {nominal.iters} iters | cost = {nominal.cost:.4f}")

# ── Step 2-3: TV-LQR gains ───────────────────────────────────────────
print("Computing TV-LQR gains ...")
tvlqr = TVLQR(
    dynamics = dyn,
    Q  = 10.0  * np.eye(nX),
    R  = 0.1   * np.eye(nU),
    Qf = 1e4   * np.eye(nX),
)
gains = tvlqr.gains(nominal, dt)
print("Gains ready.")

# ── Step 4: closed-loop sim from perturbed x0 ────────────────────────
x0_cl  = x0 + np.array([[0.1], [0.05], [0.0], [0.0]])
print(f"Simulating from x0_cl = {x0_cl.flatten()} ...")
result = tvlqr.simulate(x0_cl, gains, nominal)
print("Done.")

# ── Step 5: plot + animate ────────────────────────────────────────────
t_vec  = np.arange(N) * dt
labels = ["Cart pos [m]", "Pole angle [rad]", "Cart vel [m/s]", "Pole ω [rad/s]"]
plot_comparison(t_vec, nominal, result, xd, labels)

animate_cartpole(t_vec, nominal.xtraj, param)
animate_cartpole(t_vec, result.x_cl,  param)