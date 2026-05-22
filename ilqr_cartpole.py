import numpy as np
from cartpole import cartpole_dynamics, cartpole_grads, animate_cartpole
from ilqr import ILQR, DynamicsModel, QuadraticCost

param = {
    'mc': 10.0, 'mp': 2.0, 'l': 0.5,
    'g':  9.8,  'b':  0.1, 'd': 0.1,
}

class CartpoleDynamics(DynamicsModel):
    def __init__(self, p): self.p = p
    def step(self, t, x, u):
        return cartpole_dynamics(t, x, u, self.p).reshape(-1, 1)
    def jacobians(self, t, x, u):
        return cartpole_grads(t, x, u, self.p)

# ----- cost matrices -----
nX, nU = 4, 1
Q  = np.zeros((nX, nX))
R  = 0.01 * np.eye(nU)
Qf = 1e4  * np.eye(nX)

x0 = np.zeros((nX, 1))
xd = np.array([[0.0], [np.pi], [0.0], [0.0]])

T, dt = 2.5, 0.05
N     = int(T / dt)

# ----- solve -----
solver   = ILQR(CartpoleDynamics(param), QuadraticCost(Q, R, Qf))
solution = solver.solve(x0, xd, N, dt, nU)

print(f"Converged in {solution.iters} iterations  |  cost = {solution.cost:.4f}")

# ----- replay open-loop -----
nX_ = x0.shape[0]
x   = np.zeros((nX_, N))
t   = np.zeros(N)
x[:, [0]] = x0
dyn = CartpoleDynamics(param)
for k in range(N - 1):
    x[:, [k + 1]] = dyn.integrate(t[k], x[:, [k]], solution.utraj[:, [k]], dt)
    t[k + 1] = t[k] + dt

animate_cartpole(t, x, param)