import numpy as np

# ── Sigmoid ──────────────────────────────────────────────
def sigmoid(x):
    # Numerically stable: 0.5 * (1 + tanh(x/2)) is algebraically
    # identical to 1 / (1 + exp(-x)) but uses numpy's stable tanh,
    # so it doesn't overflow for very negative x.
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ── Tanh ─────────────────────────────────────────────────
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# ── ReLU ─────────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# ── Leaky ReLU ───────────────────────────────────────────
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# ── ELU ──────────────────────────────────────────────────
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

# ── Swish ────────────────────────────────────────────────
def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

# ── GELU ─────────────────────────────────────────────────
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    t = np.tanh(tanh_arg)
    dt = (1 - t**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
    return 0.5 * (1 + t) + 0.5 * x * dt

# ── Softplus ─────────────────────────────────────────────
def softplus(x):
    # np.logaddexp(0, x) == log(1 + exp(x)) but stable for large x.
    return np.logaddexp(0.0, x)

def softplus_derivative(x):
    return sigmoid(x)  # derivative of softplus is sigmoid

# ── Softmax ──────────────────────────────────────────────
def softmax(x):
    # Subtract max for numerical stability (avoids exp overflow)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def softmax_derivative(x):
    # Returns the full Jacobian matrix (n, n) for a 1-D input of length n.
    # For a batch, each sample's Jacobian is computed independently.
    s = softmax(x)
    # Jacobian: diag(s) - s @ s.T
    return np.diagflat(s) - np.outer(s, s)
