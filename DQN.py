import numpy as np
import neural_nets as nn
import cartpole

layers           = ["tanh", "tanh"]
layer_dimensions = [5, 64, 64, 1]

Qt = nn.init_NN(layer_dimensions)
Qd = nn.init_NN(layer_dimensions)

def greedy(x, Qd, layers, actions, epsilon=0.5):
    eps0 = np.random.rand()
    if eps0 < epsilon:                          # fix #1: explore when rand < epsilon
        return np.random.choice(actions)
    else:
        Qmin = 1e6
        amin = actions[0]
        for a in actions:
            input = np.concatenate([x, [a]]).reshape(-1, 1)
            Qx, _, _ = nn.forward(input, Qd, layers)
            if Qx < Qmin:
                Qmin = Qx
                amin = a
        return amin

def Qd_train(x0, actions, xd, Qt, Qd, layers, alpha = 1, gamma = 0.9, iteration = 200, episode = 20):
    x = x0
    scollect = []
    Qcollect = []
    for ep in range(episode):
        for i in range(iteration):
            action = greedy(x, Qd, layers, actions)
            x_next = cartpole.get_next_state(x, action)
            r = cartpole.get_cost(x_next, action, xd)
            Qnextmin = 1e8
            for a in actions:
                input = np.concatenate([x_next, [a]]).reshape(-1, 1)
                Qnext, _, _ = nn.forward(input, Qt, layers)
                if Qnext < Qnextmin:
                    Qnextmin = Qnext
            
            s = np.concatenate([x, [action]]).reshape(-1, 1)
            Qxa_current, _, _ = nn.forward(s, Qd, layers)             # fix #3 (format, noted)
            Qxa_new = Qxa_current + alpha * (r + gamma * Qnextmin - Qxa_current)

            scollect.append(s)                        # fix #2: collect taken action
            Qcollect.append(Qxa_new)

            x = x_next
    
    X_batch = np.hstack(scollect)                            # (5, N)
    Y_batch = np.array(Qcollect).reshape(1, -1)             # (1, N)
    Qd = nn.train(X_batch, Y_batch, Qd, layers)

    return Qd

def Qt_train(Qt, Qd, tau=0.005):
    Wx_t, Bx_t, Wy_t, By_t = Qt
    Wx_d, Bx_d, Wy_d, By_d = Qd
    new_Wx = [tau*wd + (1-tau)*wt for wd, wt in zip(Wx_d, Wx_t)]
    new_Bx = [tau*bd + (1-tau)*bt for bd, bt in zip(Bx_d, Bx_t)]
    return (new_Wx, new_Bx, tau*Wy_d + (1-tau)*Wy_t, tau*By_d + (1-tau)*By_t)  # fix #4

layers           = ["tanh", "tanh"]
layer_dimensions = [5, 64, 64, 1]

Qt = nn.init_NN(layer_dimensions)
Qd = nn.init_NN(layer_dimensions)
x0 = np.array([0, 0, np.pi/12, 0])
actions = np.linspace(-10, 10, 10)
xd = np.array([0, 0, 0, 0])

iteration = 10

for i in range(iteration):
    Qd = Qd_train(x0, actions, xd, Qt, Qd, layers)
    Qt = Qt_train(Qt, Qd)                          
