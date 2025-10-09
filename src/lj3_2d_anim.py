#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D animation for the 3-body Lennard-Jones (6, 12) system used in lj3_2d.py.
Shows the motion of the two outer particles plus the central particle and records
its path. Optionally saves an MP4 (via ffmpeg) or GIF fallback if ffmpeg is
unavailable.

Examples
--------
python lj3_2d_anim.py --x0 1.24 --vb 0.20 --y0 0.02 --dt 0.002 --T 60 \\
  --fps 30 --thin 5 --outfile lj3_anim.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from lj3_2d import step_yoshida4


def simulate(
    x0: float,
    vb: float,
    y0: float,
    dt: float,
    T: float,
    save_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the lj3_2d dynamics and collect positions every ``save_stride`` steps."""
    if save_stride <= 0:
        raise ValueError("save_stride must be a positive integer")
    X = np.array([[-x0, 0.0], [0.0, y0], [x0, 0.0]], float)
    V = np.array([[+vb, 0.0], [0.0, 0.0], [-vb, 0.0]], float)

    nsteps = int(T / dt)
    if nsteps <= 0:
        raise ValueError("T must be greater than dt to perform at least one step")

    snaps = []
    times = []
    Xc = X.copy()
    Vc = V.copy()
    for s in range(nsteps):
        Xc, Vc = step_yoshida4(Xc, Vc, dt)
        if (s + 1) % save_stride == 0:
            snaps.append(Xc.copy())
            times.append((s + 1) * dt)
    return np.array(times), np.array(snaps)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Animate the 2D 3-body Lennard-Jones simulation from lj3_2d.py"
    )
    ap.add_argument("--x0", type=float, default=1.24, help="outer half-separation")
    ap.add_argument(
        "--vb", type=float, default=0.20, help="outer opposite speed (+vb and -vb)"
    )
    ap.add_argument("--y0", type=float, default=0.02, help="initial y of center")
    ap.add_argument("--dt", type=float, default=0.002, help="time step")
    ap.add_argument("--T", type=float, default=60.0, help="total simulated time")
    ap.add_argument(
        "--thin",
        type=int,
        default=5,
        metavar="N",
        help="store every Nth integrator step (controls animation FPS)",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=30,
        help="frames per second for saving animations",
    )
    ap.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="optional path to save the animation (mp4/gif)",
    )
    ap.add_argument(
        "--no_show",
        action="store_true",
        help="skip displaying the interactive window (useful when only saving)",
    )
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import animation

    times, snaps = simulate(args.x0, args.vb, args.y0, args.dt, args.T, args.thin)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    (pts,) = ax.plot([], [], "o", markersize=7)
    edges = [ax.plot([], [], "-", linewidth=1.2)[0] for _ in range(2)]
    (trace,) = ax.plot([], [], "-", linewidth=1.0)

    all_xy = snaps.reshape(-1, 2)
    mins = all_xy.min(axis=0)
    maxs = all_xy.max(axis=0)
    spans = maxs - mins + 1e-6
    pad = 0.2 * spans
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])

    x_span = spans[0]
    y_span = spans[1]
    y_pad = max(pad[1], 0.15 * x_span)
    y_mid = 0.5 * (maxs[1] + mins[1])
    y_half = 0.5 * y_span + y_pad
    ax.set_ylim(y_mid - y_half, y_mid + y_half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D LJ 3-body")

    cx, cy = [], []

    def init():
        pts.set_data([], [])
        for e in edges:
            e.set_data([], [])
        trace.set_data([], [])
        cx.clear()
        cy.clear()
        return [pts, trace, *edges]

    connections = [(0, 1), (1, 2)]

    def update(frame_index: int):
        X = snaps[frame_index]
        pts.set_data(X[:, 0], X[:, 1])
        for e, (i, j) in zip(edges, connections):
            e.set_data([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])
        cx.append(X[1, 0])
        cy.append(X[1, 1])
        trace.set_data(cx, cy)
        return [pts, trace, *edges]

    frame_interval_ms = 1000.0 * args.dt * args.thin
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(snaps),
        interval=frame_interval_ms,
        blit=True,
    )

    outfile = Path(args.outfile) if args.outfile else None
    if outfile is not None:
        try:
            writer = animation.FFMpegWriter(
                fps=args.fps,
                metadata={"artist": "lj3_2d_anim"},
                bitrate=1800,
            )
            anim.save(str(outfile), writer=writer)
            print(f"Saved animation: {outfile}")
        except Exception as ffmpeg_err:  # pragma: no cover - best effort fallback
            print("FFmpeg unavailable or failed, falling back to GIF.")
            print("Reason:", ffmpeg_err)
            try:
                from matplotlib.animation import PillowWriter

                gif_out = outfile.with_suffix(".gif")
                anim.save(str(gif_out), writer=PillowWriter(fps=args.fps))
                print(f"Saved animation: {gif_out}")
            except Exception as gif_err:
                print("GIF fallback failed:", gif_err)
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
