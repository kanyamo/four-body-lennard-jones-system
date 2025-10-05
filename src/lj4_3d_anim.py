#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D animation for the 4-body (triangle + center) LJ system.
Saves an MP4 (ffmpeg) or falls back to a GIF (Pillow) if ffmpeg is unavailable.

Examples
--------
python lj4_3d_anim.py --side_scale 1.03 --vb 0.22 --z0 0.005 --dt 0.002 --T 60 \
  --fps 30 --thin 5 --outfile tri_anim.mp4
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def lj_force_pair_3d(r_vec):
    r2 = float(r_vec @ r_vec)
    if r2 < 1e-16:
        return np.zeros(3)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    factor = 24.0 * (2.0 * inv_r12 * inv_r2 - inv_r6 * inv_r2)
    return factor * r_vec


def lj_total_forces_3d(X):
    n = X.shape[0]
    F = np.zeros_like(X)
    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = X[j] - X[i]
            fij = lj_force_pair_3d(rij)
            F[i] -= fij
            F[j] += fij
    return F


cbrt2 = 2.0 ** (1 / 3)
w1 = 1.0 / (2.0 - cbrt2)
w0 = -cbrt2 / (2.0 - cbrt2)
YS = (w1, w0, w1)


def step_yoshida4(X, V, dt, masses):
    inv_m = 1.0 / masses[:, None]
    for w in YS:
        F = lj_total_forces_3d(X)
        V = V + 0.5 * (w * dt) * (F * inv_m)
        X = X + (w * dt) * V
        F = lj_total_forces_3d(X)
        V = V + 0.5 * (w * dt) * (F * inv_m)
    return X, V


def triangle_vertices_3d(side):
    R = side / math.sqrt(3.0)
    ang = np.deg2rad([0.0, 120.0, 240.0])
    verts = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros_like(ang)], axis=1)
    return verts - verts.mean(axis=0, keepdims=True)


def simulate(side_scale, vb, z0, dt, T, seed=24, save_stride=2, center_mass=1.0):
    r_star = (2.0 * (1.0 + 1.0 / (3**6)) / (1.0 + 1.0 / (3**3))) ** (1.0 / 6.0)
    side = math.sqrt(3.0) * r_star * side_scale
    rng = np.random.default_rng(seed)
    verts = triangle_vertices_3d(side)
    center = np.array([[0.0, 0.0, z0]], float)
    X = np.vstack([verts, center])
    masses = np.concatenate([np.ones(3), np.array([center_mass], dtype=float)])
    V = np.zeros_like(X)
    for i in range(3):
        r2 = X[i, :2]
        rn = np.linalg.norm(r2)
        u = r2 / rn if rn > 1e-12 else np.array([1.0, 0.0])
        sign = +1.0 if True else -1.0
        V[i, :2] = sign * vb * u
    # V[:, 2] += rng.normal(scale=0.002 * vb + 6e-4, size=4)
    # V[:3, 1] += rng.normal(scale=8e-4, size=3)
    v_com = np.sum(V * masses[:, None], axis=0) / masses.sum()
    V -= v_com

    nsteps = int(T / dt)
    snaps = []
    times = []
    Xc = X.copy()
    Vc = V.copy()
    for s in range(nsteps):
        Xc, Vc = step_yoshida4(Xc, Vc, dt, masses)
        if s % save_stride == 0:
            snaps.append(Xc.copy())
            times.append((s + 1) * dt)
    return np.array(times), np.array(snaps)  # (F, 4, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side_scale", type=float, default=1.00)
    ap.add_argument("--vb", type=float, default=0.20)
    ap.add_argument("--z0", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument(
        "--fps", type=int, default=30, help="frames per second for the video"
    )
    ap.add_argument(
        "--thin", type=int, default=5, help="store every Nth integrator step"
    )
    ap.add_argument("--outfile", type=str, default="tri_anim.mp4")
    ap.add_argument(
        "--center_mass",
        type=float,
        default=1.0,
        help="relative mass of the central particle (vertices have mass 1)",
    )
    args = ap.parse_args()

    times, snaps = simulate(
        args.side_scale,
        args.vb,
        args.z0,
        args.dt,
        args.T,
        args.seed,
        save_stride=args.thin,
        center_mass=args.center_mass,
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    (pts,) = ax.plot([], [], [], "o", markersize=6)
    edges = [ax.plot([], [], [], "-", linewidth=1.0)[0] for _ in range(3)]
    (trace,) = ax.plot([], [], [], "-", linewidth=1.0)

    all_xyz = snaps.reshape(-1, 3)
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    pad = 0.2 * (maxs - mins + 1e-6)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D LJ triangle+center")

    cx = []
    cy = []
    cz = []

    def init():
        pts.set_data([], [])
        pts.set_3d_properties([])
        for e in edges:
            e.set_data([], [])
            e.set_3d_properties([])
        trace.set_data([], [])
        trace.set_3d_properties([])
        return [pts, trace] + edges

    def update(f):
        X = snaps[f]
        pts.set_data(X[:, 0], X[:, 1])
        pts.set_3d_properties(X[:, 2])
        conn = [(0, 1), (1, 2), (2, 0)]
        for e, (i, j) in zip(edges, conn):
            e.set_data([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])
            e.set_3d_properties([X[i, 2], X[j, 2]])
        cx.append(X[3, 0])
        cy.append(X[3, 1])
        cz.append(X[3, 2])
        trace.set_data(cx, cy)
        trace.set_3d_properties(cz)
        return [pts, trace] + edges

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(snaps),
        interval=1000 * args.dt,
        blit=True,
    )
    plt.show()
    # try:
    #     Writer = animation.FFMpegWriter
    #     writer = Writer(fps=args.fps, metadata=dict(artist="lj4_3d_anim"), bitrate=1800)
    #     anim.save(args.outfile, writer=writer)
    #     print(f"Saved animation: {args.outfile}")
    # except Exception as e:
    #     print("FFmpeg not available or failed, falling back to GIF. Reason:", e)
    #     try:
    #         gif_out = args.outfile.rsplit(".", 1)[0] + ".gif"
    #         from matplotlib.animation import PillowWriter

    #         anim.save(gif_out, writer=PillowWriter(fps=args.fps))
    #         print(f"Saved animation: {gif_out}")
    #     except Exception as e2:
    #         print("GIF fallback failed:", e2)
    plt.close(fig)


if __name__ == "__main__":
    main()
