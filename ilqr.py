"""
ilqr.py  —  General-purpose Iterative LQR framework.

Usage
-----
1.  Subclass `DynamicsModel` and implement `step()` and `jacobians()`.
2.  Subclass `CostFunction` and implement `running()`, `running_grads()`,
    `terminal()`, and `terminal_grads()`.
3.  Instantiate `ILQR(dynamics, cost)` and call `.solve()`.

Example (cartpole swing-up) at the bottom of the file.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# =========================================================================
#  Protocols / abstract base classes
# =========================================================================

class DynamicsModel(ABC):
    """
    Continuous-time dynamics x_dot = f(t, x, u).
    The iLQR solver integrates with Euler by default; override
    `integrate()` if you want a different integrator (RK4, etc.).
    """

    @abstractmethod
    def step(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Returns x_dot  (shape: (nX, 1)).
        x : (nX, 1)
        u : (nU, 1)
        """

    @abstractmethod
    def jacobians(self, t: float, x: np.ndarray, u: np.ndarray):
        """
        Returns (fx, fu) — continuous-time Jacobians.
        fx : (nX, nX)   df/dx
        fu : (nX, nU)   df/du
        """

    def integrate(self, t: float, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Euler integration (override for RK4, etc.)."""
        return x + self.step(t, x, u).reshape(x.shape) * dt

    def discrete_jacobians(self, t: float, x: np.ndarray, u: np.ndarray, dt: float):
        """
        Discretise continuous Jacobians with the Euler approximation.
        Override if you have analytic discrete Jacobians.
        """
        fx_c, fu_c = self.jacobians(t, x, u)
        nX = x.shape[0]
        fx = fx_c * dt + np.eye(nX)
        fu = fu_c * dt
        return fx, fu


class CostFunction(ABC):
    """
    Quadratic-ish cost with analytic gradients.

    Running cost:   l(x, x_ref, u, u_ref)
    Terminal cost:  lf(x, xd)
    """

    @abstractmethod
    def running(self, x: np.ndarray, x_ref: np.ndarray,
                u: np.ndarray, u_ref: np.ndarray) -> float:
        """Scalar running cost."""

    @abstractmethod
    def running_grads(self, x: np.ndarray, x_ref: np.ndarray,
                      u: np.ndarray, u_ref: np.ndarray):
        """
        Returns (gx, gu, gxx, gux, guu).
        gx  : (nX, 1)
        gu  : (nU, 1)
        gxx : (nX, nX)
        gux : (nU, nX)
        guu : (nU, nU)
        """

    @abstractmethod
    def terminal(self, x: np.ndarray, xd: np.ndarray) -> float:
        """Scalar terminal cost."""

    @abstractmethod
    def terminal_grads(self, x: np.ndarray, xd: np.ndarray):
        """
        Returns (gx, gu, gxx, gux, guu) at the terminal step.
        gu/gux/guu are zero-filled by convention.
        """


# =========================================================================
#  Convenience: standard quadratic cost
# =========================================================================

class QuadraticCost(CostFunction):
    """
    l  = 0.5 (x-x_ref)' Q (x-x_ref) + 0.5 (u-u_ref)' R (u-u_ref)
    lf = 0.5 (x-xd)' Qf (x-xd)
    """

    def __init__(self, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray):
        self.Q  = Q
        self.R  = R
        self.Qf = Qf
        self._nX = Q.shape[0]
        self._nU = R.shape[0]

    def running(self, x, x_ref, u, u_ref):
        ex = x - x_ref
        eu = u - u_ref
        cost = ex.T @ self.Q @ ex + eu.T @ self.R @ eu
        return 0.5 * cost.item()

    def running_grads(self, x, x_ref, u, u_ref):
        ex = x - x_ref
        eu = u - u_ref
        gx  = self.Q @ ex
        gu  = self.R @ eu
        gxx = self.Q
        gux = np.zeros((self._nU, self._nX))
        guu = self.R
        return gx, gu, gxx, gux, guu

    def terminal(self, x, xd):
        e = x - xd
        cost = e.T @ self.Qf @ e
        return 0.5 * cost.item()

    def terminal_grads(self, x, xd):
        nX, nU = self._nX, self._nU
        gx  = self.Qf @ (x - xd)
        gu  = np.zeros((nU, 1))
        gxx = self.Qf
        gux = np.zeros((nU, nX))
        guu = np.zeros((nU, nU))
        return gx, gu, gxx, gux, guu


# =========================================================================
#  Core iLQR solver
# =========================================================================

@dataclass
class ILQRSolution:
    xtraj  : np.ndarray          # (nX, N)
    utraj  : np.ndarray          # (nU, N-1)
    ktraj  : np.ndarray          # (nU, N)   feedforward gains
    Ktraj  : np.ndarray          # (nU, nX, N) feedback gains
    cost   : float
    iters  : int


class ILQR:
    """
    Iterative LQR solver.

    Parameters
    ----------
    dynamics : DynamicsModel
    cost     : CostFunction
    max_iter : int
    tol      : float    convergence threshold (|ΔJ|)
    rho      : float    regularisation added to Quu during gain computation
    """

    def __init__(
        self,
        dynamics : DynamicsModel,
        cost     : CostFunction,
        max_iter : int   = 1000,
        tol      : float = 1e-6,
        rho      : float = 1e-4,
    ):
        self.dynamics = dynamics
        self.cost     = cost
        self.max_iter = max_iter
        self.tol      = tol
        self.rho      = rho

    # ------------------------------------------------------------------
    def solve(
        self,
        x0    : np.ndarray,
        xd    : np.ndarray,
        N     : int,
        dt    : float,
        nU    : int,
        utraj0: Optional[np.ndarray] = None,
    ) -> ILQRSolution:
        """Run iLQR and return the optimised trajectory."""
        nX = x0.shape[0]
        if utraj0 is not None:
            nU = utraj0.shape[0]

        xtraj = np.zeros((nX, N))
        utraj = np.zeros((nU, N - 1)) if utraj0 is None else utraj0.copy()
        ktraj = np.zeros((nU, N))
        Ktraj = np.zeros((nU, nX, N))

        # reference trajectories (updated each iteration)
        xtraj_ref = xtraj.copy()
        utraj_ref = utraj.copy()

        J_prev = 1e12
        J      = 1e12

        for iteration in range(self.max_iter):
            xtraj, utraj, J = self._forward_pass(
                x0, xtraj, utraj, ktraj, Ktraj, N, dt, xd, J,
                xtraj_ref, utraj_ref,
            )
            Ktraj, ktraj = self._backward_pass(
                xtraj, utraj, N, dt, xd, xtraj_ref, utraj_ref
            )

            if abs(J - J_prev) < self.tol:
                break
            J_prev      = J
            xtraj_ref   = xtraj.copy()
            utraj_ref   = utraj.copy()

        return ILQRSolution(xtraj, utraj, ktraj, Ktraj, J, iteration + 1)

    # ------------------------------------------------------------------
    #  Backward pass
    # ------------------------------------------------------------------
    def _backward_pass(self, xtraj, utraj, N, dt, xd, xtraj_ref, utraj_ref):
        nX = xtraj.shape[0]
        nU = utraj.shape[0]
        ktraj = np.zeros((nU, N))
        Ktraj = np.zeros((nU, nX, N))

        # terminal boundary conditions
        xN = xtraj[:, [N - 1]]
        uN = utraj[:, [N - 2]]
        gx, gu, gxx, gux, guu = self.cost.terminal_grads(xN, xd)
        Vx  = gx
        Vxx = gxx

        for i in range(N - 2, -1, -1):
            x_i    = xtraj[:, [i]]
            u_i    = utraj[:, [i]]
            xr_i   = xtraj_ref[:, [i]]
            ur_i   = utraj_ref[:, [i]]

            gx, gu, gxx, gux, guu = self.cost.running_grads(x_i, xr_i, u_i, ur_i)
            fx, fu = self.dynamics.discrete_jacobians(i * dt, x_i, u_i, dt)

            Qx, Qu, Qxx, Qux, Quu, Quxbar, Quubar = self._Q_terms(
                gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx
            )
            K, v = self._get_gains(Qu, Qux, Quu, Quxbar, Quubar)

            Ktraj[:, :, i] = K
            ktraj[:, [i]]  = v
            Vx, Vxx = self._V_terms(Qx, Qu, Qxx, Qux, Quu, K, v)

        return Ktraj, ktraj

    # ------------------------------------------------------------------
    #  Forward pass with backtracking line search
    # ------------------------------------------------------------------
    def _forward_pass(self, x0, xtraj0, utraj0, ktraj, Ktraj, N, dt,
                      xd, J0, xtraj_ref, utraj_ref):
        nX = x0.shape[0]
        nU = utraj0.shape[0]

        J     = np.inf
        alpha = 1.0
        xtraj = np.zeros((nX, N))
        utraj = np.zeros((nU, N - 1))

        while J0 < J:
            xtraj = np.zeros((nX, N))
            utraj = np.zeros((nU, N - 1))
            x = x0.copy()
            t = 0.0
            J = 0.0

            for i in range(N - 1):
                xtraj[:, [i]] = x
                dx = x - xtraj0[:, [i]]
                u  = utraj0[:, [i]] + alpha * ktraj[:, [i]] + Ktraj[:, :, i] @ dx
                utraj[:, [i]] = u

                J += self.cost.running(x, xtraj_ref[:, [i]], u, utraj_ref[:, [i]])
                x  = self.dynamics.integrate(t, x, u, dt)
                t += dt

            xtraj[:, [N - 1]] = x
            J += self.cost.terminal(x, xd)

            alpha /= 2.0
            if alpha < 1e-10:
                break

        return xtraj, utraj, J

    # ------------------------------------------------------------------
    #  Q / V helpers  (same maths as original, just tidied up)
    # ------------------------------------------------------------------
    def _Q_terms(self, gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx):
        rho     = self.rho
        Vxx_reg = Vxx + rho * np.eye(Vxx.shape[0])

        Qx  = gx  + fx.T @ Vx
        Qu  = gu  + fu.T @ Vx
        Qxx = gxx + fx.T @ Vxx     @ fx
        Qux = gux + fu.T @ Vxx     @ fx
        Quu = guu + fu.T @ Vxx     @ fu

        Quxbar = gux + fu.T @ Vxx_reg @ fx
        Quubar = guu + fu.T @ Vxx_reg @ fu

        return Qx, Qu, Qxx, Qux, Quu, Quxbar, Quubar

    def _get_gains(self, Qu, Qux, Quu, Quxbar, Quubar):
        Quu_reg = Quubar + self.rho * np.eye(Quu.shape[0])
        v = -np.linalg.solve(Quu_reg, Qu)
        K = -np.linalg.solve(Quu_reg, Quxbar)
        return K, v

    @staticmethod
    def _V_terms(Qx, Qu, Qxx, Qux, Quu, K, v):
        Vx  = Qx  + K.T @ Qu + Qux.T @ v + K.T @ Quu @ v
        Vxx = Qxx + K.T @ Qux + Qux.T @ K + K.T @ Quu @ K
        Vxx = 0.5 * (Vxx + Vxx.T)
        return Vx, Vxx


# =========================================================================
#  Example: cartpole swing-up (mirrors ilqr_cartpole.py)
# =========================================================================

if __name__ == "__main__":
    # ----- plug in the cartpole dynamics -----
    from cartpole import cartpole_dynamics, cartpole_grads, animate_cartpole

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
