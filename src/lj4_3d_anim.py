#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Animate the 3D dynamics of the 4-body Lennard-Jones system."""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from lj4_core import available_configs, simulate_trajectory


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        choices=available_configs(),
        default="triangle_center",
        help="equilibrium configuration to animate",
    )
    ap.add_argument(
        "--mode-displacement",
        type=float,
        default=0.02,
        help="initial displacement amplitude along the first stable mode",
    )
    ap.add_argument(
        "--mode-velocity",
        type=float,
        default=0.05,
        help="initial velocity amplitude along the first stable mode",
    )
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument(
        "--fps",
        type=int,
        default=30,
        help="frames per second for interactive playback",
    )
    ap.add_argument(
        "--thin", type=int, default=5, help="store every Nth integrator step"
    )
    ap.add_argument("--outfile", type=str, default="lj4_anim.mp4")
    ap.add_argument(
        "--center_mass",
        type=float,
        default=1.0,
        help="mass assigned to the central particle in the triangle+center case",
    )
    ap.add_argument(
        "--trace-index",
        type=int,
        default=None,
        help="optional particle index to trace (defaults to config-specific choice)",
    )
    return ap.parse_args()


def update_plot(ax, result, trace_index):
    fig = ax.get_figure()
    (pts,) = ax.plot([], [], [], "o", markersize=6)
    edge_lines = [ax.plot([], [], [], "-", linewidth=1.0)[0] for _ in result.spec.edges]
    (trace,) = ax.plot([], [], [], "-", linewidth=1.0)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    all_xyz = result.positions.reshape(-1, 3)
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    span = maxs - mins
    pad = 0.2 * (span + 1e-6)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{result.spec.label} — first stable mode ω²={result.omega2:.3f}")

    trace_points = result.positions[:, trace_index, :]
    current_frame: list[int] = [0]

    def show_frame(frame_idx: int) -> None:
        frame_idx = max(0, min(frame_idx, len(result.positions) - 1))
        X = result.positions[frame_idx]
        pts.set_data(X[:, 0], X[:, 1])
        pts.set_3d_properties(X[:, 2])
        for line, (i, j) in zip(edge_lines, result.spec.edges):
            line.set_data([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])
            line.set_3d_properties([X[i, 2], X[j, 2]])
        history = trace_points[: frame_idx + 1]
        trace.set_data(history[:, 0], history[:, 1])
        trace.set_3d_properties(history[:, 2])
        time_text.set_text(f"t = {result.times[frame_idx]:.2f}")
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    frame_slider = Slider(
        ax=slider_ax,
        label="time",
        valmin=float(result.times[0]),
        valmax=float(result.times[-1]),
        valinit=float(result.times[0]),
        valstep=result.times,
    )

    def on_slider_change(val: float) -> None:
        new_frame = int(np.searchsorted(result.times, val, side="left"))
        if new_frame >= len(result.times) or not math.isclose(result.times[new_frame], val):
            new_frame = max(0, min(len(result.times) - 1, new_frame - 1))
        if new_frame != current_frame[0]:
            current_frame[0] = new_frame
            show_frame(new_frame)

    frame_slider.on_changed(on_slider_change)

    def on_key_press(event) -> None:
        if event.key in {"right", "d"}:
            next_frame = min(current_frame[0] + 1, len(result.times) - 1)
            frame_slider.set_val(result.times[next_frame])
        elif event.key in {"left", "a"}:
            prev_frame = max(current_frame[0] - 1, 0)
            frame_slider.set_val(result.times[prev_frame])

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    show_frame(0)


def simulate(
    config: str,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
):
    result = simulate_trajectory(
        config=config,
        mode_displacement=mode_displacement,
        mode_velocity=mode_velocity,
        dt=dt,
        total_time=total_time,
        save_stride=save_stride,
        center_mass=center_mass,
    )
    return result.spec, result.omega2, result.times, result.positions


def main() -> None:
    args = parse_args()
    if args.thin < 1:
        raise ValueError("--thin must be >= 1")
    if args.center_mass <= 0.0:
        raise ValueError("--center_mass must be positive")

    result = simulate_trajectory(
        args.config,
        args.mode_displacement,
        args.mode_velocity,
        args.dt,
        args.T,
        args.thin,
        args.center_mass,
    )

    trace_index = args.trace_index if args.trace_index is not None else result.spec.trace_index
    if trace_index < 0 or trace_index >= result.positions.shape[1]:
        raise ValueError("trace index out of bounds for this configuration")

    print(f"Configuration: {result.spec.label}")
    print(f"First stable mode ω² = {result.omega2:.6f}")
    print(
        f"Initial conditions: displacement={args.mode_displacement:.4f}, "
        f"velocity={args.mode_velocity:.4f}"
    )

    fig = plt.figure(figsize=(6.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    update_plot(ax, result, trace_index)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
