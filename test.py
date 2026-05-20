"""
Smoke test / demo for the neural network module.

Runs in two parts:
  1. Gradient check on a small random net to confirm forward and
     backward agree to ~1e-7 relative error.
  2. Train a small MLP to fit y = sin(2π·x1) · cos(2π·x2) and report
     loss curve plus a few sample predictions.
"""

import numpy as np
import neural_nets as nn


def gradient_check_demo():
    print("=" * 60)
    print("1) Gradient check")
    print("=" * 60)

    # Small inputs keep the finite-difference loop fast.
    np.random.seed(0)
    x = np.random.randn(3, 8)   # input_dim=3, batch=8
    y = np.random.randn(2, 8)   # output_dim=2

    # Mix activation families so both He and Xavier paths are exercised.
    layers  = ["tanh", "relu"]
    weights = nn.init_NN([3, 6, 4, 2], layers=layers, seed=0)

    errs = nn.gradient_check(x, y, weights, layers, eps=1e-6)

    print("Max relative error per parameter (want < 1e-4):")
    all_ok = True
    for name, e in errs.items():
        ok = e < 1e-4
        all_ok &= ok
        flag = "ok" if ok else "FAIL"
        print(f"  {name:8s}  {e:.2e}  [{flag}]")
    print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}\n")


def training_demo():
    print("=" * 60)
    print("2) Training: fit y = sin(2π·x1) · cos(2π·x2)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    N   = 1000
    x   = rng.uniform(-1.0, 1.0, size=(2, N))
    y   = (np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))[None, :]
    y  += 0.05 * rng.standard_normal(y.shape)   # mild noise

    layers  = ["relu", "relu"]
    weights = nn.init_NN([2, 64, 64, 1], layers=layers, seed=0)

    # Baseline loss before any training.
    y0, _, _ = nn.forward(x, weights, layers)
    print(f"Initial loss: {nn.mse_loss(y0, y):.4f}\n")

    weights, hist = nn.train(
        x, y, weights, layers,
        lr=1e-3, epochs=200, batch_size=64,
        optimizer="adam", weight_decay=0.0,
        log_every=20, seed=0,
    )

    print(f"\nLoss: {hist[0]:.4f}  →  {hist[-1]:.6f}")
    print(f"Reduction factor: {hist[0] / max(hist[-1], 1e-12):.1f}x\n")

    # Spot check on fresh (held-out) samples.
    print("Sample predictions on fresh points:")
    x_test = rng.uniform(-1.0, 1.0, size=(2, 5))
    y_true = np.sin(2 * np.pi * x_test[0]) * np.cos(2 * np.pi * x_test[1])
    y_pred, _, _ = nn.forward(x_test, weights, layers)

    print(f"  {'x1':>7s} {'x2':>7s} {'true':>9s} {'pred':>9s} {'|err|':>9s}")
    for i in range(x_test.shape[1]):
        print(f"  {x_test[0, i]:7.3f} {x_test[1, i]:7.3f} "
              f"{y_true[i]:9.4f} {y_pred[0, i]:9.4f} "
              f"{abs(y_true[i] - y_pred[0, i]):9.4f}")


def sgd_vs_adam_demo():
    print("\n" + "=" * 60)
    print("3) Sanity: Adam should beat plain SGD on the same problem")
    print("=" * 60)

    rng = np.random.default_rng(1)
    N   = 500
    x   = rng.uniform(-1.0, 1.0, size=(2, N))
    y   = (np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))[None, :]

    layers = ["relu", "relu"]

    # Identical init for a fair comparison.
    w_sgd  = nn.init_NN([2, 32, 32, 1], layers=layers, seed=42)
    w_adam = nn.init_NN([2, 32, 32, 1], layers=layers, seed=42)

    print("\n-- SGD (lr=0.05) --")
    _, hist_sgd = nn.train(
        x, y, w_sgd, layers,
        lr=0.05, epochs=100, batch_size=64,
        optimizer="sgd", log_every=25, seed=7,
    )

    print("\n-- Adam (lr=1e-3) --")
    _, hist_adam = nn.train(
        x, y, w_adam, layers,
        lr=1e-3, epochs=100, batch_size=64,
        optimizer="adam", log_every=25, seed=7,
    )

    print(f"\nFinal loss  | SGD: {hist_sgd[-1]:.5f}   Adam: {hist_adam[-1]:.5f}")


if __name__ == "__main__":
    gradient_check_demo()
    training_demo()
    sgd_vs_adam_demo()
