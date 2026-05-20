import numpy as np
from cartpole import cartpole_dynamics, cartpole_grads, animate_cartpole, regressor
from ilqr_cartpole import ilqr
import matplotlib.pyplot as plt

def init_params():
    param = {
        'mc': 1.0,     # cart mass
        'mp': 1.0,     # pole mass
        'l':  0.5,     # pole length
        'g':  9.8,     # gravity
        'b':  0.1,     # cart viscous friction
        'd':  0.1,     # pole viscous friction \
            }

    return param


def main():
    T = 2.5
    dt = 0.05
    N = int(T / dt)
    nX, nU = 4, 1

    true_param = {'mc': 10.0, 'mp': 2.0, 'l': 0.5, 'g': 9.8, 'b': 0.1, 'd': 0.1}
    sim_param  = init_params()

    x0 = np.array([[0.0], [0.0], [0.0], [0.0]])

    xtraj = np.zeros((nX, N))
    utraj = np.zeros((nU, N - 1))
    ktraj = np.zeros((nU, N))
    Ktraj = np.zeros((nU, nX, N))

    Q  = 0   * np.eye(nX)
    Qf = 1.0e2 * np.eye(nX)
    R  = np.array([[0.01]])
    xd = np.array([[0.0], [np.pi], [0.0], [0.0]])

    xdata, udata, xdot_data = [], [], []
    Jsave = []

    num_iterations = 100

    for i in range(num_iterations):
        # plan with current model
        xtraj, utraj, ktraj, Ktraj = ilqr(
            x0, xtraj, utraj, ktraj, Ktraj, N, dt, sim_param, Q, R, Qf, xd
        )

        # roll out on the TRUE system with feedback
        t = np.zeros(N)
        x = np.zeros((nX, N))
        u = np.zeros((nU, N - 1))
        x[:, [0]] = x0

        for k in range(N - 1):
            u[:, [k]] = (utraj[:, [k]]
                        + Ktraj[:, :, k] @ (x[:, [k]] - xtraj[:, [k]]))
            xdot = cartpole_dynamics(t[k], x[:, [k]], u[:, [k]], true_param)
            x[:, [k + 1]] = x[:, [k]] + xdot * dt
            t[k + 1] = t[k] + dt

        J = final_cost(x[:, [-1]], u[:, [-1]], xd, Qf)
        Jsave.append(J)

        # add measurement noise
        xf = x
        x = x + 0.01 * np.random.randn(nX, N)

        # accumulate dataset across iterations
        xdata.append(x[:, 0:N-1])
        u_aug = np.vstack([
            u[:, 0:N-1],
            np.zeros((1, N-1))
        ])
        udata.append(u_aug)
        xdot_data.append((x[:, 1:N] - x[:, 0:N-1]) / dt)

        # concatenate everything
        X    = np.hstack(xdata)          # (4, M)
        Xdot = np.hstack(xdot_data)      # (4, M)
        U    = np.hstack(udata)          # (2, M)

        # least squares -> [m_c, m_p]
        theta_hat = least_squares(X, Xdot, U)

        # update the planner's model
        sim_param['mc'] = float(theta_hat[0])
        sim_param['mp'] = float(theta_hat[1])

        print(f"iter {i:2d}:  m_c_hat = {theta_hat[0]:.3f},  "
              f"m_p_hat = {theta_hat[1]:.3f},  J = {J:.2f}")
   
    # phase plots after convergence
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(xtraj[0, :], xtraj[2, :])
    axes[0].set_xlabel('cart position x  (m)')
    axes[0].set_ylabel(r'cart velocity $\dot x$  (m/s)')
    axes[0].set_title('Cart phase plot')
    axes[0].grid(True)

    axes[1].plot(xtraj[1, :], xtraj[3, :])
    axes[1].set_xlabel(r'pole angle $\theta$  (rad)')
    axes[1].set_ylabel(r'pole rate $\dot\theta$  (rad/s)')
    axes[1].set_title('Pole phase plot')
    axes[1].grid(True)
    plt.tight_layout()

    animate_cartpole(t, xf, true_param)

def final_cost(x, u, xd, Qf):
    e = x - xd
    return float((0.5 * e.T @ Qf @ e).item())

def least_squares(X, Xdot, u_data, gamma=1e-3):
    N = X.shape[1]

    Phi = np.zeros((2*N, 2))
    y   = np.zeros(2*N)

    for i in range(N):

        q      = X[0:2, i]
        q_dot  = X[2:4, i]
        q_ddot = Xdot[2:4, i]

        Y_i = regressor(q, q_dot, q_dot, q_ddot)   # (2,2)

        Phi[2*i:2*i+2, :] = Y_i

        y[2*i:2*i+2] = u_data[:, i]

    # SVD
    U, S, VT = np.linalg.svd(Phi, full_matrices=False)

    # Invert singular values safely
    S_inv = np.zeros_like(S)

    tol = 1e-8
    for i in range(len(S)):
        if S[i] > tol:
            S_inv[i] = 1.0 / S[i]

    # Pseudoinverse solution
    theta_hat = VT.T @ np.diag(S_inv) @ U.T @ y

    return theta_hat


if __name__ == "__main__":
    main()
