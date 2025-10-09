#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D, 3-body Lennard-Jones (6,12), equal masses, all pairs interact (autonomous).
Time integration: 4th-order symplectic (Yoshida composition of velocity-Verlet).
Goal: test Kapitza-like stabilization of the center when outers start in a breathing mode.

Outputs
-------
- CSV with time, center y(t), energy(t)  (if --save_traj)
- One-line metrics JSON (if --save_metrics): y_maxabs, y_rms, frac(|y|<thr), energy_rel_drift
- Optional PNG plots (--plot)

Usage (examples)
----------------
python lj3_2d.py --x0 1.24 --vb 0.20 --y0 0.02 --dt 0.002 --T 120 --save_traj run.csv --save_metrics metrics.json --plot run.png
python lj3_2d.py --x0 1.20 --vb 0.18 --sweep_vb 0.14 0.28 8 --out_prefix sweep_case
"""

import argparse
import json
import numpy as np

import matplotlib.pyplot as plt


# ---------------- LJ (6,12) 2D ----------------
def lj_force_pair_2d(r_vec: np.ndarray) -> np.ndarray:
    r2 = float(r_vec @ r_vec)
    if r2 < 1e-16:
        return np.zeros(2)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    factor = 24.0 * (2.0 * inv_r12 * inv_r2 - inv_r6 * inv_r2)  # ε=σ=1
    return factor * r_vec


def lj_total_forces_2d(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    F = np.zeros_like(X)
    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = X[j] - X[i]
            fij = lj_force_pair_2d(rij)
            F[i] -= fij
            F[j] += fij
    return F


def lj_potential_2d(X: np.ndarray) -> float:
    n = X.shape[0]
    U = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(X[j] - X[i]))
            if r < 1e-14:
                continue
            U += 4.0 * ((1.0 / r) ** 12 - (1.0 / r) ** 6)
    return U


def kinetic_2d(V: np.ndarray) -> float:
    return 0.5 * np.sum(V**2)


# -------------- Yoshida 4th-order --------------
cbrt2 = 2.0 ** (1.0 / 3.0)
w1 = 1.0 / (2.0 - cbrt2)
w0 = -cbrt2 / (2.0 - cbrt2)
YS = (w1, w0, w1)


def step_yoshida4(
    X: np.ndarray, V: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    for w in YS:
        F = lj_total_forces_2d(X)
        V = V + 0.5 * (w * dt) * F
        X = X + (w * dt) * V
        F = lj_total_forces_2d(X)
        V = V + 0.5 * (w * dt) * F
    return X, V


# -------------- One simulation --------------
def run_case(
    x0: float,
    vb: float,
    y0: float = 0.02,
    dt: float = 0.002,
    T: float = 120.0,
    save_every: int = 10,
    seed: int = 0,
):
    X = np.array([[-x0, 0.0], [0.0, y0], [x0, 0.0]], float)
    V = np.array([[+vb, 0.0], [0.0, 0.0], [-vb, 0.0]], float)

    nsteps = int(T / dt)
    Xc, Vc = X.copy(), V.copy()
    times_list, y_ser_list, energy_list = [], [], []
    E0 = kinetic_2d(Vc) + lj_potential_2d(Xc)

    for s in range(nsteps):
        Xc, Vc = step_yoshida4(Xc, Vc, dt)
        if (s + 1) % save_every == 0:
            t = (s + 1) * dt
            times_list.append(t)
            y_ser_list.append(float(Xc[1, 1]))
            energy_list.append(kinetic_2d(Vc) + lj_potential_2d(Xc))

    times = np.array(times_list)
    y_ser = np.array(y_ser_list)
    energy = np.array(energy_list)
    # metrics
    yabs = np.abs(y_ser)
    metrics = {
        "x0": x0,
        "vb": vb,
        "y0": y0,
        "dt": dt,
        "T": T,
        "save_every": save_every,
        "seed": seed,
        "y_maxabs": float(np.max(yabs)),
        "y_rms": float(np.sqrt(np.mean(y_ser**2))),
        "frac(|y|<0.02)": float(np.mean(yabs < 0.02)),
        "energy_rel_drift": float(
            (np.max(energy) - np.min(energy)) / (abs(E0) + 1e-12)
        ),
    }
    return times, y_ser, energy, metrics


def maybe_plot(times, y_ser, out_png: str | None):
    if out_png is None:
        return
    plt.figure(figsize=(7, 3))
    plt.plot(times, y_ser)
    plt.xlabel("time")
    plt.ylabel("center y")
    plt.title("2D 3-body LJ — y_center(t)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="2D, 3-body LJ (autonomous) with Yoshida 4th-order integrator"
    )
    ap.add_argument("--x0", type=float, default=1.24, help="outer half-separation")
    ap.add_argument(
        "--vb", type=float, default=0.20, help="outer opposite speed (+vb and -vb)"
    )
    ap.add_argument("--y0", type=float, default=0.02, help="initial y of center")
    ap.add_argument("--dt", type=float, default=0.002, help="time step")
    ap.add_argument("--T", type=float, default=120.0, help="total time")
    ap.add_argument(
        "--save_every", type=int, default=10, help="subsampling interval for output"
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="rng seed for tiny symmetry-breaking noise"
    )
    ap.add_argument(
        "--save_traj", type=str, default=None, help="CSV path to save t,y,energy"
    )
    ap.add_argument(
        "--save_metrics", type=str, default=None, help="JSON path to save metrics"
    )
    ap.add_argument("--plot", type=str, default=None, help="PNG path to save y(t)")
    # simple sweep over vb if requested
    ap.add_argument(
        "--sweep_vb",
        nargs=3,
        type=float,
        default=None,
        metavar=("VB_MIN", "VB_MAX", "N"),
        help="optional sweep over vb; writes multiple files with --out_prefix",
    )
    ap.add_argument(
        "--out_prefix", type=str, default="run", help="prefix for sweep outputs"
    )
    args = ap.parse_args()

    if args.sweep_vb is None:
        t, y, E, m = run_case(
            args.x0, args.vb, args.y0, args.dt, args.T, args.save_every, args.seed
        )
        if args.save_traj:
            import numpy as np

            np.savetxt(
                args.save_traj,
                np.c_[t, y, E],
                delimiter=",",
                header="t,y,energy",
                comments="",
            )
            print(f"saved: {args.save_traj}")
        if args.save_metrics:
            with open(args.save_metrics, "w") as f:
                json.dump(m, f, indent=2)
            print(f"saved: {args.save_metrics}")
        maybe_plot(t, y, args.plot)
    else:
        vb_min, vb_max, n = args.sweep_vb
        n = int(n)
        vbs = np.linspace(vb_min, vb_max, n)
        all_metrics = []
        for i, vb in enumerate(vbs):
            t, y, E, m = run_case(
                args.x0, float(vb), args.y0, args.dt, args.T, args.save_every, args.seed
            )
            base = f"{args.out_prefix}_vb{i:02d}"
            np.savetxt(
                base + ".csv",
                np.c_[t, y, E],
                delimiter=",",
                header="t,y,energy",
                comments="",
            )
            with open(base + ".json", "w") as f:
                json.dump(m, f, indent=2)
            maybe_plot(t, y, base + ".png" if args.plot is not None else None)
            all_metrics.append(m)
            print(f"vb={vb:.4f} -> saved {base}.csv/.json")
        with open(args.out_prefix + "_all_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"saved sweep metrics: {args.out_prefix}_all_metrics.json")


if __name__ == "__main__":
    main()
