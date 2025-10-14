#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animate the 3D dynamics of the 4-body Lennard-Jones system for any of the
five known equilibria.  The initial condition is prepared by displacing the
equilibrium along its first stable normal mode and optionally giving a velocity
kick along the same direction.

Examples
--------
python lj4_3d_anim.py --config triangle_center --mode-displacement 0.02 \
  --mode-velocity 0.10 --dt 0.002 --T 60 --thin 5 --outfile tri_anim.mp4
"""

import argparse
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


@dataclass(frozen=True)
class EquilibriumSpec:
    """Geometry and plotting metadata for a 4-body equilibrium."""

    key: str
    label: str
    positions: np.ndarray
    edges: tuple[tuple[int, int], ...]
    trace_index: int


def _regular_tetrahedron() -> np.ndarray:
    edge = 2.0 ** (1.0 / 6.0)
    base = np.array(
        [
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        ],
        dtype=float,
    )
    scale = edge / (2.0 * math.sqrt(2.0))
    return base * scale


def _square_planar() -> np.ndarray:
    side = 1.1126198392
    half = 0.5 * side
    return np.array(
        [
            (-half, -half, 0.0),
            (half, -half, 0.0),
            (half, half, 0.0),
            (-half, half, 0.0),
        ],
        dtype=float,
    )


def _rhombus_planar() -> np.ndarray:
    positions = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.1202309526, 0.0, 0.0),
            (-0.5555370480, 0.9727774540, 0.0),
            (0.5646939046, 0.9727774540, 0.0),
        ],
        dtype=float,
    )
    positions -= positions.mean(axis=0, keepdims=True)
    return positions


def _triangle_plus_center() -> np.ndarray:
    r_star = (2.0 * (1.0 + 1.0 / (3.0**6)) / (1.0 + 1.0 / (3.0**3))) ** (1.0 / 6.0)
    angles = np.deg2rad([0.0, 120.0, 240.0])
    ring = np.stack(
        [r_star * np.cos(angles), r_star * np.sin(angles), np.zeros_like(angles)],
        axis=1,
    )
    center = np.zeros((1, 3), dtype=float)
    coords = np.vstack([ring, center])
    coords -= coords.mean(axis=0, keepdims=True)
    return coords


def _isosceles_with_interior() -> np.ndarray:
    base = 1.1230225004
    apex_y = 2.0885491154
    interior_y = 0.9695654105
    coords = np.array(
        [
            (-0.5 * base, 0.0, 0.0),
            (0.5 * base, 0.0, 0.0),
            (0.0, apex_y, 0.0),
            (0.0, interior_y, 0.0),
        ],
        dtype=float,
    )
    coords -= coords.mean(axis=0, keepdims=True)
    return coords


EQUILIBRIA: dict[str, EquilibriumSpec] = {}


def _register_equilibria() -> None:
    global EQUILIBRIA
    EQUILIBRIA = {
        "tetrahedron": EquilibriumSpec(
            key="tetrahedron",
            label="tetrahedron (T_d)",
            positions=_regular_tetrahedron(),
            edges=tuple(combinations(range(4), 2)),
            trace_index=0,
        ),
        "rhombus": EquilibriumSpec(
            key="rhombus",
            label="rhombus (θ≈60.27°)",
            positions=_rhombus_planar(),
            edges=((0, 1), (1, 2), (2, 3), (3, 0)),
            trace_index=0,
        ),
        "square": EquilibriumSpec(
            key="square",
            label="square (D_4h)",
            positions=_square_planar(),
            edges=((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)),
            trace_index=0,
        ),
        "triangle_center": EquilibriumSpec(
            key="triangle_center",
            label="triangle + center (C_3v)",
            positions=_triangle_plus_center(),
            edges=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
            trace_index=3,
        ),
        "isosceles_interior": EquilibriumSpec(
            key="isosceles_interior",
            label="isosceles triangle + interior (C_s)",
            positions=_isosceles_with_interior(),
            edges=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
            trace_index=3,
        ),
    }


_register_equilibria()


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


def pair_hessian(r_vec: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(r_vec))
    if r < 1e-12:
        raise ValueError("Particles overlap; Hessian undefined.")
    e = r_vec / r
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    v1 = 4.0 * (-12.0 * inv_r12 * inv_r + 6.0 * inv_r6 * inv_r)
    v2 = 4.0 * (156.0 * inv_r12 * inv_r2 - 42.0 * inv_r6 * inv_r2)
    outer = np.outer(e, e)
    return (v2 - v1 * inv_r) * outer + (v1 * inv_r) * np.eye(3)


def build_hessian(positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    H = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            kij = pair_hessian(rij)
            sl_i = slice(3 * i, 3 * i + 3)
            sl_j = slice(3 * j, 3 * j + 3)
            H[sl_i, sl_i] += kij
            H[sl_j, sl_j] += kij
            H[sl_i, sl_j] -= kij
            H[sl_j, sl_i] -= kij
    return H


def recenter(points: np.ndarray, masses: np.ndarray) -> np.ndarray:
    com = np.average(points, axis=0, weights=masses)
    return points - com


def first_stable_mode(
    positions: np.ndarray, masses: np.ndarray, tol: float = 1e-8
) -> tuple[np.ndarray, float]:
    H = build_hessian(positions)
    weights = np.repeat(np.sqrt(masses), 3)
    Hmw = H / weights[:, None]
    Hmw = Hmw / weights[None, :]
    eigvals, eigvecs = np.linalg.eigh(Hmw)
    for idx, lam in enumerate(eigvals):
        if lam > tol:
            vec = eigvecs[:, idx] / weights
            coords = vec.reshape(-1, 3)
            coords = recenter(coords, masses)
            norm = np.linalg.norm(coords)
            if norm > 0.0:
                coords /= norm
            return coords, float(lam)
    raise RuntimeError("No stable mode located above tolerance")


def prepare_equilibrium(
    config: str, center_mass: float
) -> tuple[EquilibriumSpec, np.ndarray, np.ndarray]:
    if config not in EQUILIBRIA:
        raise KeyError(f"Unknown configuration '{config}'")
    spec = EQUILIBRIA[config]
    base = np.array(spec.positions, copy=True)
    masses = np.ones(base.shape[0], dtype=float)
    if config == "triangle_center":
        masses[:3] = 1.0
        masses[3] = center_mass
    base = recenter(base, masses)
    return spec, base, masses


def simulate(
    config: str,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
) -> tuple[EquilibriumSpec, float, np.ndarray, np.ndarray]:
    spec, equilibrium, masses = prepare_equilibrium(config, center_mass)
    mode_shape, omega2 = first_stable_mode(equilibrium, masses)

    X0 = equilibrium + mode_displacement * mode_shape
    V0 = mode_velocity * mode_shape
    X0 = recenter(X0, masses)
    v_com = np.sum(V0 * masses[:, None], axis=0) / masses.sum()
    V0 -= v_com

    nsteps = int(total_time / dt)
    snaps = [X0.copy()]
    times = [0.0]
    Xc = X0.copy()
    Vc = V0.copy()
    for step in range(nsteps):
        Xc, Vc = step_yoshida4(Xc, Vc, dt, masses)
        if (step + 1) % save_stride == 0:
            snaps.append(Xc.copy())
            times.append((step + 1) * dt)
    return spec, omega2, np.array(times), np.array(snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        choices=sorted(EQUILIBRIA.keys()),
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
    args = ap.parse_args()

    if args.thin < 1:
        raise ValueError("--thin must be >= 1")
    if args.center_mass <= 0.0:
        raise ValueError("--center_mass must be positive")

    spec, omega2, times, snaps = simulate(
        args.config,
        args.mode_displacement,
        args.mode_velocity,
        args.dt,
        args.T,
        args.thin,
        args.center_mass,
    )

    trace_index = args.trace_index if args.trace_index is not None else spec.trace_index
    if trace_index < 0 or trace_index >= snaps.shape[1]:
        raise ValueError("trace index out of bounds for this configuration")

    print(f"Configuration: {spec.label}")
    print(f"First stable mode ω² = {omega2:.6f}")
    print(
        f"Initial conditions: displacement={args.mode_displacement:.4f}, "
        f"velocity={args.mode_velocity:.4f}"
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    (pts,) = ax.plot([], [], [], "o", markersize=6)
    edge_lines = [ax.plot([], [], [], "-", linewidth=1.0)[0] for _ in spec.edges]
    (trace,) = ax.plot([], [], [], "-", linewidth=1.0)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    all_xyz = snaps.reshape(-1, 3)
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
    ax.set_title(f"{spec.label} — first stable mode ω²={omega2:.3f}")

    trace_x: list[float] = []
    trace_y: list[float] = []
    trace_z: list[float] = []

    def init():
        trace_x.clear()
        trace_y.clear()
        trace_z.clear()
        pts.set_data([], [])
        pts.set_3d_properties([])
        for line in edge_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        trace.set_data([], [])
        trace.set_3d_properties([])
        time_text.set_text("")
        return [pts, trace, time_text, *edge_lines]

    def update(frame: int):
        X = snaps[frame]
        pts.set_data(X[:, 0], X[:, 1])
        pts.set_3d_properties(X[:, 2])
        for line, (i, j) in zip(edge_lines, spec.edges):
            line.set_data([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])
            line.set_3d_properties([X[i, 2], X[j, 2]])
        trace_x.append(X[trace_index, 0])
        trace_y.append(X[trace_index, 1])
        trace_z.append(X[trace_index, 2])
        trace.set_data(trace_x, trace_y)
        trace.set_3d_properties(trace_z)
        time_text.set_text(f"t = {times[frame]:.2f}")
        return [pts, trace, time_text, *edge_lines]

    frame_interval = 1000.0 * args.dt * args.thin
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(snaps),
        interval=frame_interval,
        blit=True,
    )
    plt.show()
    # Saving support can be re-enabled if desired:
    # Writer = animation.FFMpegWriter
    # writer = Writer(fps=args.fps, metadata=dict(artist="lj4_3d_anim"), bitrate=1800)
    # anim.save(args.outfile, writer=writer)
    plt.close(fig)


if __name__ == "__main__":
    main()
