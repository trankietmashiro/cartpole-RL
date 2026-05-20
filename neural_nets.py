import numpy as np
import activation as actv


# ── Activation dispatch ────────────────────────────────────────────────────
# Extend these dicts whenever a new activation is added to activation.py.
# softmax is intentionally excluded: it produces a Jacobian, not a scalar
# derivative, so it requires special handling in the backward pass.

_ACTV_DISPATCH = {
    "sigmoid":    actv.sigmoid,
    "tanh":       actv.tanh,
    "relu":       actv.relu,
    "leaky_relu": actv.leaky_relu,
    "elu":        actv.elu,
    "swish":      actv.swish,
    "gelu":       actv.gelu,
    "softplus":   actv.softplus,
}

_DERV_DISPATCH = {
    "sigmoid":    actv.sigmoid_derivative,
    "tanh":       actv.tanh_derivative,
    "relu":       actv.relu_derivative,
    "leaky_relu": actv.leaky_relu_derivative,
    "elu":        actv.elu_derivative,
    "swish":      actv.swish_derivative,
    "gelu":       actv.gelu_derivative,
    "softplus":   actv.softplus_derivative,
}

# Activations that benefit from He init (gain = sqrt(2)). All others get
# Xavier/Glorot (gain = 1.0). Tanh technically wants gain ≈ 5/3 but 1.0
# is the standard simplification and works fine in practice.
_HE_ACTIVATIONS = {"relu", "leaky_relu", "elu", "swish", "gelu"}


def apply_actv(z, layer_type):
    if layer_type not in _ACTV_DISPATCH:
        raise ValueError(
            f"Unknown activation '{layer_type}'. "
            f"Choose from: {sorted(_ACTV_DISPATCH)}."
        )
    return _ACTV_DISPATCH[layer_type](z)


def derv_actv(z, layer_type):
    if layer_type not in _DERV_DISPATCH:
        raise ValueError(
            f"Unknown activation '{layer_type}'. "
            f"Choose from: {sorted(_DERV_DISPATCH)}."
        )
    return _DERV_DISPATCH[layer_type](z)


# ── Initialization ─────────────────────────────────────────────────────────

def _init_gain(actv_name):
    """He gain for relu-family, Xavier gain otherwise."""
    return np.sqrt(2.0) if actv_name in _HE_ACTIVATIONS else 1.0


def init_NN(dimensions, layers=None, seed=None):
    """
    dimensions: list of layer sizes, e.g. [2, 256, 64, 1]
    layers:     list of activation names, one per hidden layer. If provided,
                each hidden weight matrix is scaled with He (relu-family) or
                Xavier (everything else). If None, falls back to Xavier
                everywhere. Output layer is always Xavier (it's linear).
    seed:       optional int for reproducibility.

    Returns (Wx, Bx, Wy, By).
    """
    if len(dimensions) < 2:
        raise ValueError("dimensions must have at least 2 entries (input + output).")
    if seed is not None:
        np.random.seed(seed)

    Wx, Bx = [], []
    for n in range(len(dimensions) - 2):
        fan_in = dimensions[n]
        gain   = _init_gain(layers[n]) if layers is not None else 1.0
        scale  = gain / np.sqrt(fan_in)
        w = np.random.randn(dimensions[n+1], dimensions[n]) * scale
        b = np.zeros((dimensions[n+1], 1))
        Wx.append(w)
        Bx.append(b)

    Wy = np.random.randn(dimensions[-1], dimensions[-2]) / np.sqrt(dimensions[-2])
    By = np.zeros((dimensions[-1], 1))
    return (Wx, Bx, Wy, By)


def _validate_layers(layers, Wx):
    """Raise a clear error when the layers list does not match the network."""
    n_hidden = len(Wx)
    if len(layers) != n_hidden:
        raise ValueError(
            f"`layers` has {len(layers)} activation(s) but the network has "
            f"{n_hidden} hidden layer(s). They must match.\n"
            f"  layers   = {layers}\n"
            f"  expected = {n_hidden} activation name(s)\n"
            f"  Hint: len(layers) == len(dimensions) - 2"
        )


# ── Forward pass ───────────────────────────────────────────────────────────

def forward(x, weights, layers):
    """
    x:       (input_dim, batch_size)
    layers:  list of activation names for hidden layers, e.g. ["relu", "tanh"]
    Returns:
        y_pred: (output_dim, batch_size)  -- linear output, no activation
        Z:      list of pre-activation values per hidden layer
        H:      list of post-activation values per hidden layer
    """
    Wx, Bx, Wy, By = weights
    _validate_layers(layers, Wx)
    Z, H = [], []
    h = x
    for i in range(len(Wx)):
        z = Wx[i] @ h + Bx[i]
        h = apply_actv(z, layers[i])
        Z.append(z)
        H.append(h)
    y_pred = Wy @ h + By
    return y_pred, Z, H


# ── Loss ───────────────────────────────────────────────────────────────────

def mse_loss(y_pred, y):
    """Mean squared error: (1/2N) * ||y_pred - y||^2_F"""
    N = y.shape[1]
    return 0.5 / N * np.linalg.norm(y_pred - y) ** 2


# ── Backward pass ──────────────────────────────────────────────────────────

def backward(x, weights, y_pred, Z, H, layers, y):
    Wx, Bx, Wy, By = weights
    _validate_layers(layers, Wx)
    N = y.shape[1]

    dLdy  = (y_pred - y) / N
    dLdWy = dLdy @ H[-1].T
    dLdBy = np.sum(dLdy, axis=1, keepdims=True)
    dLdh  = Wy.T @ dLdy

    dLdWx = [None] * len(Wx)
    dLdBx = [None] * len(Bx)

    for i in reversed(range(len(Wx))):
        h_prev   = H[i-1] if i > 0 else x
        dLdz     = dLdh * derv_actv(Z[i], layers[i])
        dLdWx[i] = dLdz @ h_prev.T
        dLdBx[i] = np.sum(dLdz, axis=1, keepdims=True)
        dLdh     = Wx[i].T @ dLdz

    return dLdWx, dLdBx, dLdWy, dLdBy


# ── SGD update ─────────────────────────────────────────────────────────────

def update(weights, grads, lr):
    Wx, Bx, Wy, By = weights
    dWx, dBx, dWy, dBy = grads
    for i in range(len(Wx)):
        Wx[i] -= lr * dWx[i]
        Bx[i] -= lr * dBx[i]
    Wy -= lr * dWy
    By -= lr * dBy
    return (Wx, Bx, Wy, By)


# ── Adam ───────────────────────────────────────────────────────────────────

def init_adam(weights):
    """Allocate first/second moment buffers matching the weight shapes."""
    Wx, Bx, Wy, By = weights
    m = {
        "Wx": [np.zeros_like(w) for w in Wx],
        "Bx": [np.zeros_like(b) for b in Bx],
        "Wy": np.zeros_like(Wy),
        "By": np.zeros_like(By),
    }
    v = {
        "Wx": [np.zeros_like(w) for w in Wx],
        "Bx": [np.zeros_like(b) for b in Bx],
        "Wy": np.zeros_like(Wy),
        "By": np.zeros_like(By),
    }
    return m, v


def update_adam(weights, grads, state, t,
                lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """
    Standard Adam with bias correction. Optional L2 weight decay is applied
    to weight matrices only (not biases), which is the usual convention.
    `t` is the global step count (1-indexed).
    """
    Wx, Bx, Wy, By = weights
    dWx, dBx, dWy, dBy = grads
    m, v = state

    bc1 = 1 - beta1 ** t
    bc2 = 1 - beta2 ** t

    def step(p, g, m_p, v_p, decay):
        if weight_decay > 0.0 and decay:
            g = g + weight_decay * p
        m_p[...] = beta1 * m_p + (1 - beta1) * g
        v_p[...] = beta2 * v_p + (1 - beta2) * (g * g)
        m_hat = m_p / bc1
        v_hat = v_p / bc2
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    for i in range(len(Wx)):
        step(Wx[i], dWx[i], m["Wx"][i], v["Wx"][i], decay=True)
        step(Bx[i], dBx[i], m["Bx"][i], v["Bx"][i], decay=False)
    step(Wy, dWy, m["Wy"], v["Wy"], decay=True)
    step(By, dBy, m["By"], v["By"], decay=False)

    return weights, (m, v)


# ── Training loop ──────────────────────────────────────────────────────────

def train(x, y, weights, layers,
          lr=1e-3, epochs=1000, batch_size=None,
          optimizer="adam", weight_decay=0.0,
          log_every=10, seed=None):
    """
    x: (input_dim,  N)
    y: (output_dim, N)
    layers: list of activation names, one per hidden layer.

    batch_size: None or >= N for full batch; otherwise mini-batch SGD/Adam
                with a fresh shuffle each epoch.
    optimizer:  "adam" or "sgd".
    weight_decay: L2 penalty on weight matrices (not biases). 0 to disable.
    seed:       seeds the per-epoch shuffle for reproducibility.

    Returns (weights, history) where history is the list of per-epoch
    mean losses (averaged across mini-batches in that epoch).
    """
    _validate_layers(layers, weights[0])
    if seed is not None:
        np.random.seed(seed)

    N = x.shape[1]
    if batch_size is None or batch_size >= N:
        batch_size = N

    state = init_adam(weights) if optimizer == "adam" else None
    t = 0
    history = []

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[:, idx], y[:, idx]

            y_pred, Z, H = forward(xb, weights, layers)
            loss         = mse_loss(y_pred, yb)
            grads        = backward(xb, weights, y_pred, Z, H, layers, yb)

            if optimizer == "adam":
                t += 1
                weights, state = update_adam(
                    weights, grads, state, t,
                    lr=lr, weight_decay=weight_decay,
                )
            elif optimizer == "sgd":
                weights = update(weights, grads, lr)
            else:
                raise ValueError(f"Unknown optimizer '{optimizer}'.")

            epoch_loss += loss
            n_batches  += 1

        avg = epoch_loss / n_batches
        history.append(avg)
        if epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg:.6f}")

    return weights, history


# ── Gradient check ─────────────────────────────────────────────────────────

def gradient_check(x, y, weights, layers, eps=1e-6):
    """
    Compare analytical gradients against centered finite differences.
    Returns a dict {param_name: max_relative_error}. Healthy values are
    typically < 1e-6 (with eps=1e-6) -- if anything is > 1e-4, there's a
    bug somewhere in forward/backward.

    Use a small input (e.g. 5-10 samples) and a small network; this scales
    with the total number of parameters.
    """
    Wx, Bx, Wy, By = weights

    y_pred, Z, H = forward(x, weights, layers)
    dWx, dBx, dWy, dBy = backward(x, weights, y_pred, Z, H, layers, y)

    def numerical_grad(param):
        num = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            l_plus  = mse_loss(forward(x, weights, layers)[0], y)
            param[idx] = orig - eps
            l_minus = mse_loss(forward(x, weights, layers)[0], y)
            param[idx] = orig
            num[idx] = (l_plus - l_minus) / (2 * eps)
            it.iternext()
        return num

    def rel_err(a, n):
        denom = np.abs(a) + np.abs(n) + 1e-12
        return float(np.max(np.abs(a - n) / denom))

    errors = {
        "Wy": rel_err(dWy, numerical_grad(Wy)),
        "By": rel_err(dBy, numerical_grad(By)),
    }
    for i, (w, b, dw, db) in enumerate(zip(Wx, Bx, dWx, dBx)):
        errors[f"Wx[{i}]"] = rel_err(dw, numerical_grad(w))
        errors[f"Bx[{i}]"] = rel_err(db, numerical_grad(b))
    return errors
