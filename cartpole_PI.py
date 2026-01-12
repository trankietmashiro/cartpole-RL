import numpy as np
from cartpole import dynamics, Discretizer, discrete_transition
import matplotlib.pyplot as plt
from collections import Counter

def get_action(state, policy):
    return policy[state]

def policy_evaluation(V, policy, disc, states, actions, target_state, Q, R, dt, method, alpha, gamma,
                      theta=1e-6, max_sweeps=200):
    """
    TD(0) evaluation for deterministic discrete MDP:
      s' = T(s,a)
      r  = R(s,a)
    """
    for _ in range(max_sweeps):
        delta = 0.0
        for s in states:
            v_old = V[s]
            a = get_action(s, policy)

            s2, r = discrete_transition(dynamics, disc, s, a, target_state, Q, R, dt, method=method)

            # td_target = r + gamma * V[s2]
            # V[s] = v_old + alpha * (td_target - v_old)
            V[s] = r + gamma * V[s2]

            delta = max(delta, abs(V[s] - v_old))

        if delta < theta:
            break
    return V

def policy_improvement(V, policy, disc, states, actions, target_state, Q, R, dt, method, gamma):
    stable = True
    for s in states:
        old_a = get_action(s, policy)

        best_a = old_a
        best_q = -np.inf

        for a in actions:
            s2, r = discrete_transition(dynamics, disc, s, a, target_state, Q, R, dt, method=method)
            q  = r + gamma * V[s2]

            if q > best_q:
                best_q = q
                best_a = a

        policy[s] = best_a
        if best_a != old_a:
            stable = False

    return policy, stable

def policy_iteration(disc, states, actions, target, Q, R, dt=0.02, method="rk4",
                     alpha=0.1, gamma=0.99, iterations=50):
    # Initialize V and pi
    V = {s: 0.0 for s in states}
    pi = {s: actions[3] for s in states}

    for i in range(iterations):
        print(f"\n=== PI iter {i} ===")
        V = policy_evaluation(V, pi, disc, states, actions, target, Q, R, dt, method, alpha, gamma)
        pi, stable = policy_improvement(V, pi, disc, states, actions, target, Q, R, dt, method, gamma)
        counts = Counter(pi.values())
        N = len(pi)

        print("Action distribution in policy (pi):")
        for a in actions:
            c = counts.get(a, 0)
            print(f"  action {a:>4}: {c:>8} states  ({100.0 * c / N:6.2f}%)")  
        if stable:
            break
    return V, pi

def enumerate_states(disc):
    grids = [range(n) for n in disc.n_bins.tolist()]
    # meshgrid -> list of tuples
    return [tuple(idx) for idx in np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(grids), -1).T]

if __name__ == "__main__":
    # -------------------------
    # Discretization
    # -------------------------
    disc = Discretizer(
        lows=[-1.0, -1.0, -np.pi/12, -1.0],
        highs=[ 1.0,  1.0,  np.pi/12,  1.0],
        n_bins=[5, 5, 5, 5],
    )
    states = enumerate_states(disc)

    # -------------------------
    # Discrete action set
    # -------------------------
    actions = [-2.0, -1.0, -0.2, -0.1, 0.0, 0.1, 0.2, 1.0, 2.0]

    # Target state (xd)
    xd = np.zeros(4)

    # -------------------------
    # Quadratic weights
    # -------------------------
    # Penalize angle most, then position, then velocities
    Q = np.diag([5.0, 0.5, 5.0, 0.5])
    R = 0

    # -------------------------
    # Train
    # -------------------------
    dt = 0.1
    gamma = 0.99
    alpha = 0.1

    V, pi = policy_iteration(disc, states, actions, xd, Q, R, dt=dt, method="rk4", alpha=alpha, gamma=gamma, iterations=20)

    # -------------------------
    # Optional: how many states have (almost) all actions tied in 1-step greedy Q?
    # -------------------------
    tie_eps = 1e-12
    tied_states = 0
    for s in states:
        qs = []
        for a in actions:
            s2, r = discrete_transition(dynamics, disc, s, a, xd, Q, R, dt, method="rk4")
            qs.append(r + gamma * V[s2])
        if (max(qs) - min(qs)) < tie_eps:
            tied_states += 1

    print(f"\nStates with all actions tied (within {tie_eps:g}): "
          f"{tied_states} / {len(states)}  ({100.0*tied_states/len(states):.2f}%)")

