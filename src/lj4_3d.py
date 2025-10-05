#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D, 4-body (equilateral triangle + central particle), all LJ pairs, equal masses.
Time integration: 4th-order symplectic (Yoshida composition of velocity-Verlet).
We test stabilization of the central particle's distance from the triangle plane.

Outputs
-------
- CSV with time, center (x,y,z), rho(t)  (if --save_traj)
- Metrics JSON (if --save_metrics): rho_rms, rho_max, frac(rho<thr)
- Optional PNG plots (--plot)

Usage (examples)
----------------
python lj4_3d.py --side_scale 1.00 --vb 0.20 --z0 0.02 --dt 0.002 --T 80 --save_traj tri_run.csv --save_metrics tri_metrics.json --plot tri.png
# light sweep over vb:
python lj4_3d.py --side_scale 1.05 --z0 0.005 --sweep_vb 0.16 0.26 7 --out_prefix tri_sweep
"""

import argparse
import math
import json
import numpy as np

import matplotlib.pyplot as plt


# -------------- LJ 3D --------------
def lj_force_pair_3d(r_vec: np.ndarray) -> np.ndarray:
    r2 = float(r_vec @ r_vec)
    if r2 < 1e-16:
        return np.zeros(3)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    factor = 24.0 * (2.0 * inv_r12 * inv_r2 - inv_r6 * inv_r2)  # ε=σ=1
    return factor * r_vec


def lj_total_forces_3d(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    F = np.zeros_like(X)
    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = X[j] - X[i]
            fij = lj_force_pair_3d(rij)
            F[i] -= fij
            F[j] += fij
    return F


# -------------- Yoshida 4th --------------
cbrt2 = 2.0 ** (1.0 / 3.0)
w1 = 1.0 / (2.0 - cbrt2)
w0 = -cbrt2 / (2.0 - cbrt2)
YS = (w1, w0, w1)


def step_yoshida4(
    X: np.ndarray, V: np.ndarray, dt: float, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inv_m = 1.0 / masses[:, None]
    for w in YS:
        F = lj_total_forces_3d(X)
        V = V + 0.5 * (w * dt) * (F * inv_m)
        X = X + (w * dt) * V
        F = lj_total_forces_3d(X)
        V = V + 0.5 * (w * dt) * (F * inv_m)
    return X, V


# -------------- Geometry & run --------------
def triangle_vertices_3d(side: float) -> np.ndarray:
    # equilateral triangle in the xy-plane with centroid at origin
    R = side / math.sqrt(3.0)  # circumradius
    ang = np.deg2rad([0.0, 120.0, 240.0])
    verts = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros_like(ang)], axis=1)
    verts -= verts.mean(axis=0, keepdims=True)
    return verts


def metrics_from_center_traj(center_traj: np.ndarray) -> tuple[dict, np.ndarray]:
    rho = np.linalg.norm(center_traj, axis=1)  # distance from origin
    return {
        "rho_rms": float(np.sqrt(np.mean(rho**2))),
        "rho_max": float(np.max(np.abs(rho))),
        "frac(rho<0.02)": float(np.mean(rho < 0.02)),
        "rho_last": float(rho[-1]),
    }, rho


def run_case(
    side_scale: float,
    vb: float,
    z0: float = 0.02,
    dt: float = 0.002,
    T: float = 80.0,
    save_every: int = 10,
    seed: int = 0,
    center_mass: float = 1.0,
):
    r_star = (2.0 * (1.0 + 1.0 / (3**6)) / (1.0 + 1.0 / (3**3))) ** (1.0 / 6.0)
    side = math.sqrt(3.0) * r_star * side_scale
    # rng = np.random.default_rng(seed)

    verts = triangle_vertices_3d(side)
    center = np.array([[0.0, 0.0, z0]], float)
    X = np.vstack([verts, center])
    masses = np.concatenate([np.ones(3), np.array([center_mass], dtype=float)])

    # in-plane alternating breathing (+vb,-vb,+vb)
    V = np.zeros_like(X)
    for i in range(3):
        r2 = X[i, :2]
        norm = np.linalg.norm(r2)
        u = r2 / norm if norm > 1e-12 else np.array([1.0, 0.0])
        sign = +1.0 if True else -1.0
        V[i, :2] = sign * vb * u

    # tiny symmetry-breaking noise (y,z)
    # V[:, 2] += rng.normal(scale=0.002 * vb + 6e-4, size=4)
    # V[:3, 1] += rng.normal(scale=8e-4, size=3)
    v_com = np.sum(V * masses[:, None], axis=0) / masses.sum()
    V -= v_com

    nsteps = int(T / dt)
    Xc, Vc = X.copy(), V.copy()
    times_list, center_traj_list = [], []

    for s in range(nsteps):
        Xc, Vc = step_yoshida4(Xc, Vc, dt, masses)
        if (s + 1) % save_every == 0:
            times_list.append((s + 1) * dt)
            center_traj_list.append(Xc[3].copy())

    times = np.array(times_list)
    center_traj = np.array(center_traj_list)
    metrics, rho = metrics_from_center_traj(center_traj)
    metrics.update(
        {
            "side_scale": side_scale,
            "vb": vb,
            "z0": z0,
            "dt": dt,
            "T": T,
            "save_every": save_every,
            "seed": seed,
            "center_mass": center_mass,
        }
    )
    return times, center_traj, rho, metrics


def maybe_plot(times, rho, out_png: str | None, title: str):
    if out_png is None:
        return

    plt.figure(figsize=(7, 3))
    plt.plot(times, rho)
    plt.xlabel("time")
    plt.ylabel("rho (center)")
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="3D, 4-body (triangle+center) LJ with Yoshida 4th-order integrator"
    )
    ap.add_argument(
        "--side_scale",
        type=float,
        default=1.00,
        help="triangle side / (sqrt(3)*r*) (1.0 makes center-vertex ≈ r*)",
    )
    ap.add_argument(
        "--vb",
        type=float,
        default=0.20,
        help="in-plane alternating breathing speed at vertices",
    )
    ap.add_argument("--z0", type=float, default=0.02, help="initial z-offset of center")
    ap.add_argument("--dt", type=float, default=0.002, help="time step")
    ap.add_argument("--T", type=float, default=80.0, help="total simulated time")
    ap.add_argument(
        "--save_every", type=int, default=10, help="subsampling interval for outputs"
    )
    ap.add_argument(
        "--seed", type=int, default=24, help="rng seed for tiny symmetry-breaking noise"
    )
    ap.add_argument(
        "--center_mass",
        type=float,
        default=1.0,
        help="relative mass of the central particle (vertices have mass 1)",
    )
    ap.add_argument(
        "--save_traj", type=str, default=None, help="CSV path to save t,x,y,z,rho"
    )
    ap.add_argument(
        "--save_metrics", type=str, default=None, help="JSON path to save metrics"
    )
    ap.add_argument(
        "--plot", type=str, default=None, help="PNG path to save rho(t) plot"
    )
    # Optional sweep over vb
    ap.add_argument(
        "--sweep_vb",
        nargs=3,
        type=float,
        default=None,
        metavar=("VB_MIN", "VB_MAX", "N"),
        help="optional sweep over vb; writes multiple files with --out_prefix",
    )
    ap.add_argument(
        "--out_prefix", type=str, default="tri", help="prefix for sweep outputs"
    )
    args = ap.parse_args()

    if args.sweep_vb is None:
        t, ctr, rho, m = run_case(
            args.side_scale,
            args.vb,
            args.z0,
            args.dt,
            args.T,
            args.save_every,
            args.seed,
            args.center_mass,
        )
        if args.save_traj:
            np.savetxt(
                args.save_traj,
                np.c_[t, ctr, rho],
                delimiter=",",
                header="t,x,y,z,rho",
                comments="",
            )
            print(f"saved: {args.save_traj}")
        if args.save_metrics:
            with open(args.save_metrics, "w") as f:
                json.dump(m, f, indent=2)
            print(f"saved: {args.save_metrics}")
        maybe_plot(t, rho, args.plot, "3D 4-body (triangle+center) — rho(t)")
    else:
        vb_min, vb_max, n = args.sweep_vb
        n = int(n)
        vbs = np.linspace(vb_min, vb_max, n)
        all_metrics = []
        for i, vb in enumerate(vbs):
            t, ctr, rho, m = run_case(
                args.side_scale,
                float(vb),
                args.z0,
                args.dt,
                args.T,
                args.save_every,
                args.seed,
                args.center_mass,
            )
            base = f"{args.out_prefix}_vb{i:02d}"
            np.savetxt(
                base + ".csv",
                np.c_[t, ctr, rho],
                delimiter=",",
                header="t,x,y,z,rho",
                comments="",
            )
            with open(base + ".json", "w") as f:
                json.dump(m, f, indent=2)
            maybe_plot(
                t,
                rho,
                base + ".png" if args.plot is not None else None,
                f"triangle+center — vb={vb:.3f}",
            )
            all_metrics.append(m)
            print(f"vb={vb:.4f} -> saved {base}.csv/.json")
        with open(args.out_prefix + "_all_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"saved sweep metrics: {args.out_prefix}_all_metrics.json")


if __name__ == "__main__":
    main()
