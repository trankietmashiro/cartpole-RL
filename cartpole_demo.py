"""
cartpole_demo.py
================
Simulate the CartPole system under the learned greedy policy and visualise
the closed-loop trajectory from a chosen start state to the goal.

Usage
-----
    python cartpole_demo.py --algo vi
    python cartpole_demo.py --algo pi
    python cartpole_demo.py --algo vi --start 0.0 0.0 0.2 0.0
    python cartpole_demo.py --algo vi --steps 200 --model-path models/my_vi.pth

State convention
----------------
    x = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    goal: x = [0, 0, 0, 0]  (pole upright, cart centred, stationary)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import torch

from cartpole import dynamics, continuous_transition, wrap_to_pi
from cartpole_run import build_cartpole_solver


# ──────────────────────────────────────────────────────────────────────────────
# Rollout
# ──────────────────────────────────────────────────────────────────────────────

def rollout(solver, x0: np.ndarray, n_steps: int, dt: float):
    """
    Simulate the system under the greedy policy for n_steps timesteps.

    Returns
    -------
    xs : (n_steps+1, 4)  – state trajectory  [p, p_dot, th, th_dot]
    us : (n_steps,)      – applied actions
    rs : (n_steps,)      – immediate rewards
    """
    xs = np.zeros((n_steps + 1, len(x0)), dtype=np.float32)
    us = np.zeros(n_steps, dtype=np.float32)
    rs = np.zeros(n_steps, dtype=np.float32)

    xs[0] = x0
    for t in range(n_steps):
        x   = xs[t]
        a   = solver.greedy_action(x)
        x_next, r = continuous_transition(
            dynamics, x, a,
            solver.target_state, solver.Q, solver.R, dt,
            lows=solver.lows, highs=solver.highs,
            method=solver.method,
        )
        xs[t + 1] = x_next
        us[t]     = a
        rs[t]     = r

    return xs, us, rs


# ──────────────────────────────────────────────────────────────────────────────
# Static trajectory plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory(xs, us, rs, dt, save_path=None):
    """Four-panel time-series plot of the state, action, and reward."""
    T    = len(us)
    time = np.arange(T + 1) * dt

    labels = ["Cart position (m)", "Cart velocity (m/s)",
              "Pole angle (rad)",  "Pole ang. vel. (rad/s)"]

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

    for i, (ax, label) in enumerate(zip(axes[:4], labels)):
        ax.plot(time, xs[:, i], linewidth=1.5)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[4].step(time[:-1], us, where="post", linewidth=1.5, color="tab:orange")
    axes[4].axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    axes[4].set_ylabel("Action (N)", fontsize=9)
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(time[:-1], rs, linewidth=1.5, color="tab:green")
    axes[5].set_ylabel("Reward", fontsize=9)
    axes[5].set_xlabel("Time (s)", fontsize=9)
    axes[5].grid(True, alpha=0.3)

    fig.suptitle("CartPole closed-loop trajectory", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Trajectory plot saved → {save_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Animation
# ──────────────────────────────────────────────────────────────────────────────

def animate(xs, us, dt, save_path=None):
    """
    Render an animated CartPole visualisation.

    Cart is a rectangle on a track; pole is a line from the cart centre.
    """
    CART_W, CART_H = 0.3, 0.15
    POLE_L         = 0.5 * 2          # full pole length (2 * L from cartpole.py)
    WHEEL_R        = 0.06
    TRACK_Y        = 0.0

    p_min  = xs[:, 0].min() - 0.8
    p_max  = xs[:, 0].max() + 0.8

    fig, (ax_anim, ax_action) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Animation axis ────────────────────────────────────────────────────────
    ax_anim.set_xlim(p_min, p_max)
    ax_anim.set_ylim(-0.4, POLE_L + 0.3)
    ax_anim.set_aspect("equal")
    ax_anim.axhline(TRACK_Y, color="k", linewidth=1.5)          # track
    ax_anim.axvline(0, color="gray", linewidth=0.8,
                    linestyle="--", alpha=0.5)                    # goal centre
    ax_anim.set_title("CartPole Animation", fontsize=11)
    ax_anim.set_xlabel("Cart position (m)")
    ax_anim.axis("off")

    cart_patch = patches.Rectangle(
        (-CART_W / 2, TRACK_Y + WHEEL_R),
        CART_W, CART_H,
        linewidth=1.5, edgecolor="k", facecolor="steelblue",
    )
    ax_anim.add_patch(cart_patch)

    wheel_l = plt.Circle((-CART_W / 4, TRACK_Y + WHEEL_R), WHEEL_R,
                          color="dimgray", zorder=5)
    wheel_r = plt.Circle(( CART_W / 4, TRACK_Y + WHEEL_R), WHEEL_R,
                          color="dimgray", zorder=5)
    ax_anim.add_patch(wheel_l)
    ax_anim.add_patch(wheel_r)

    pole_line, = ax_anim.plot([], [], linewidth=5, color="saddlebrown",
                              solid_capstyle="round")
    bob        = plt.Circle((0, 0), 0.05, color="firebrick", zorder=6)
    ax_anim.add_patch(bob)

    time_text  = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes,
                               fontsize=9, verticalalignment="top")
    force_text = ax_anim.text(0.02, 0.88, "", transform=ax_anim.transAxes,
                               fontsize=9, verticalalignment="top", color="tab:orange")

    # ── Action bar axis ───────────────────────────────────────────────────────
    T      = len(us)
    time_v = np.arange(T) * dt
    ax_action.step(time_v, us, where="post", linewidth=1.2, color="tab:orange")
    ax_action.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_action.set_ylabel("Action (N)", fontsize=8)
    ax_action.set_xlabel("Time (s)",   fontsize=8)
    ax_action.grid(True, alpha=0.3)
    ax_action.set_xlim(time_v[0], time_v[-1])
    vline = ax_action.axvline(0, color="red", linewidth=1.0)

    plt.tight_layout()

    # ── Update function ───────────────────────────────────────────────────────
    cart_pivot_y = TRACK_Y + WHEEL_R + CART_H / 2   # vertical centre of cart

    def update(frame):
        p, _, th, _ = xs[frame]

        # Cart
        cart_patch.set_xy((p - CART_W / 2, TRACK_Y + WHEEL_R))
        wheel_l.center = (p - CART_W / 4, TRACK_Y + WHEEL_R)
        wheel_r.center = (p + CART_W / 4, TRACK_Y + WHEEL_R)

        # Pole (angle=0 → upright; positive angle → tilts right)
        px0 = p
        py0 = cart_pivot_y + CART_H / 2
        px1 = px0 + POLE_L * np.sin(th)
        py1 = py0 + POLE_L * np.cos(th)
        pole_line.set_data([px0, px1], [py0, py1])
        bob.center = (px1, py1)

        # Text
        t_now = frame * dt
        time_text.set_text(f"t = {t_now:.2f} s")
        u_now = us[min(frame, T - 1)]
        force_text.set_text(f"F = {u_now:+.2f} N")
        vline.set_xdata([t_now, t_now])

        return (cart_patch, wheel_l, wheel_r,
                pole_line, bob, time_text, force_text, vline)

    interval_ms = max(20, int(dt * 1000))
    anim = animation.FuncAnimation(
        fig, update,
        frames=len(xs),
        interval=interval_ms,
        blit=True,
    )

    if save_path:
        writer = animation.FFMpegWriter(fps=int(1 / dt), bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Animation saved → {save_path}")
    else:
        plt.show()

    return anim


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo: run learned CartPole policy from start to goal."
    )
    parser.add_argument(
        "--algo", choices=["vi", "pi"], default="vi",
        help="Which algorithm's model to load.",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Override default checkpoint path.",
    )
    parser.add_argument(
        "--start", nargs=4, type=float,
        default=[0.0, 0.0, np.pi/15, 0.0],
        metavar=("p", "p_dot", "theta", "theta_dot"),
        help="Initial state (default: slight pole tilt of 0.15 rad).",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of simulation steps (default: 200).",
    )
    parser.add_argument(
        "--save-plot", default=None,
        help="Save trajectory plot to this path (e.g. figures/traj.png).",
    )
    parser.add_argument(
        "--save-anim", default=None,
        help="Save animation to this path (e.g. figures/demo.mp4). Requires ffmpeg.",
    )
    args = parser.parse_args()

    model_path = args.model_path or f"models/nn_{args.algo}_weights.pth"

    # ── Build solver and load checkpoint ─────────────────────────────────────
    solver = build_cartpole_solver(args.algo)
    loaded = solver.load_checkpoint(model_path)
    if not loaded:
        raise FileNotFoundError(
            f"No checkpoint found at '{model_path}'. "
            f"Run cartpole_run.py --algo {args.algo} first."
        )

    # ── Rollout ───────────────────────────────────────────────────────────────
    x0 = np.array(args.start, dtype=np.float32)
    print(f"\nStart state : p={x0[0]:.3f}  ṗ={x0[1]:.3f}  "
          f"θ={x0[2]:.3f} rad  θ̇={x0[3]:.3f}")
    print(f"Goal state  : p=0.000  ṗ=0.000  θ=0.000 rad  θ̇=0.000")
    print(f"Simulating {args.steps} steps at dt={solver.dt}s …\n")

    xs, us, rs = rollout(solver, x0, args.steps, solver.dt)

    # ── Summary ───────────────────────────────────────────────────────────────
    final = xs[-1]
    print(f"Final state : p={final[0]:.3f}  ṗ={final[1]:.3f}  "
          f"θ={final[2]:.3f} rad  θ̇={final[3]:.3f}")
    print(f"Total reward: {rs.sum():.2f}")
    print(f"Angle error at end: {abs(wrap_to_pi(final[2])):.4f} rad "
          f"({np.degrees(abs(wrap_to_pi(final[2]))):.2f}°)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_trajectory(xs, us, rs, solver.dt, save_path=args.save_plot)
    animate(xs, us, solver.dt, save_path=args.save_anim)

