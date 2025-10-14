#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time integration helper for the 3D, 4-body Lennard-Jones system.

The script prepares any of the five equilibria discovered by
``lj4_equilibrium_search.py`` and perturbs it along its first stable normal mode.
The perturbation is controlled by an initial displacement and velocity along
that mode, matching the semantics introduced in ``lj4_3d_anim.py``.

Each run integrates the equations of motion with the fourth-order Yoshida
splitting of velocity-Verlet and optionally saves the sampled trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


# --- Geometry catalogue ----------------------------------------------------


@dataclass(frozen=True)
class EquilibriumSpec:
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


# --- Lennard-Jones forces ---------------------------------------------------


def lj_force_pair_3d(r_vec: np.ndarray) -> np.ndarray:
    r2 = float(r_vec @ r_vec)
    if r2 < 1e-16:
        return np.zeros(3)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    factor = 24.0 * (2.0 * inv_r12 * inv_r2 - inv_r6 * inv_r2)
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


def lj_total_potential(X: np.ndarray) -> float:
    energy = 0.0
    n = X.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = np.linalg.norm(X[j] - X[i])
            inv_r6 = (1.0 / r) ** 6
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy


def kinetic_energy(V: np.ndarray, masses: np.ndarray) -> float:
    return 0.5 * np.sum(masses[:, None] * (V**2))


def total_energy(X: np.ndarray, V: np.ndarray, masses: np.ndarray) -> float:
    return kinetic_energy(V, masses) + lj_total_potential(X)


# --- Yoshida 4 --------------------------------------------------------------

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


# --- Normal mode prep -------------------------------------------------------


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


# --- Time integration -------------------------------------------------------


def simulate(
    config: str,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
) -> tuple[
    EquilibriumSpec,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    if save_stride < 1:
        raise ValueError("save_stride must be >= 1")
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
    kin_vals = [kinetic_energy(V0, masses)]
    pot_vals = [lj_total_potential(X0)]
    tot_vals = [kin_vals[0] + pot_vals[0]]
    Xc = X0.copy()
    Vc = V0.copy()
    for step in range(nsteps):
        Xc, Vc = step_yoshida4(Xc, Vc, dt, masses)
        if (step + 1) % save_stride == 0:
            snaps.append(Xc.copy())
            times.append((step + 1) * dt)
            kin = kinetic_energy(Vc, masses)
            pot = lj_total_potential(Xc)
            kin_vals.append(kin)
            pot_vals.append(pot)
            tot_vals.append(kin + pot)
    energy_initial = tot_vals[0]
    energy_final = tot_vals[-1]
    return (
        spec,
        omega2,
        mode_shape,
        masses,
        np.array(times),
        np.array(snaps),
        np.array(kin_vals),
        np.array(pot_vals),
        np.array(tot_vals),
        energy_initial,
        energy_final,
    )


# --- Output helpers ---------------------------------------------------------


def save_trajectory_csv(path: Path, times: np.ndarray, positions: np.ndarray) -> None:
    n_particles = positions.shape[1]
    header_cols = ["t"]
    for pid in range(n_particles):
        header_cols.extend([f"x{pid}", f"y{pid}", f"z{pid}"])
    header = ",".join(header_cols)
    data = np.column_stack([times, positions.reshape(len(times), -1)])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_summary_json(
    path: Path,
    spec: EquilibriumSpec,
    omega2: float,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    masses: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
    mode_shape: np.ndarray,
    kinetic_series: np.ndarray,
    potential_series: np.ndarray,
    total_series: np.ndarray,
    energy_initial: float,
    energy_final: float,
) -> None:
    summary = {
        "config": spec.key,
        "label": spec.label,
        "omega2_first_stable": omega2,
        "mode_displacement": mode_displacement,
        "mode_velocity": mode_velocity,
        "dt": dt,
        "total_time": total_time,
        "save_stride": save_stride,
        "frames": int(len(times)),
        "n_particles": int(positions.shape[1]),
        "masses": masses.tolist(),
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "mode_shape": mode_shape.tolist(),
        "times": times.tolist(),
        "energies": {
            "kinetic": kinetic_series.tolist(),
            "potential": potential_series.tolist(),
            "total": total_series.tolist(),
        },
        "energy_initial": energy_initial,
        "energy_final": energy_final,
    }
    summary_path = Path(path)
    summary_path.write_text(json.dumps(summary, indent=2))


def plot_energy_series(
    path: Path,
    times: np.ndarray,
    kinetic_series: np.ndarray,
    potential_series: np.ndarray,
    total_series: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3.5))
    plt.plot(times, kinetic_series, label="kinetic")
    plt.plot(times, potential_series, label="potential")
    if total_series is not None:
        plt.plot(times, total_series, label="total", linestyle="--", linewidth=1.0)
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.title("Energy budget vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# --- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Integrate the 4-body LJ system from a first-mode perturbation",
    )
    ap.add_argument(
        "--config",
        choices=sorted(EQUILIBRIA.keys()),
        default="triangle_center",
        help="equilibrium configuration to integrate",
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
        default=0.00,
        help="initial velocity amplitude along the first stable mode",
    )
    ap.add_argument("--dt", type=float, default=0.002, help="integrator time step")
    ap.add_argument("--T", type=float, default=80.0, help="total simulated time")
    ap.add_argument(
        "--thin",
        type=int,
        default=10,
        help="store every Nth integrator step (>=1)",
    )
    ap.add_argument(
        "--center-mass",
        type=float,
        default=1.0,
        help="mass assigned to the central particle in the triangle+center case",
    )
    ap.add_argument(
        "--save-traj",
        type=Path,
        default=None,
        help="optional CSV path for the sampled trajectory",
    )
    ap.add_argument(
        "--save-summary",
        type=Path,
        default=None,
        help="optional JSON path with run metadata",
    )
    ap.add_argument(
        "--plot-energies",
        type=Path,
        default=None,
        help="optional PNG path plotting kinetic/potential/total energies",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.center_mass <= 0.0:
        raise ValueError("--center-mass must be positive")

    (
        spec,
        omega2,
        mode_shape,
        masses,
        times,
        positions,
        kinetic_series,
        potential_series,
        total_series,
        energy_initial,
        energy_final,
    ) = simulate(
        args.config,
        args.mode_displacement,
        args.mode_velocity,
        args.dt,
        args.T,
        args.thin,
        args.center_mass,
    )

    print(f"Configuration: {spec.label}")
    print(f"First stable mode ω² = {omega2:.6f}")
    print(
        f"Initial conditions: displacement={args.mode_displacement:.4f}, "
        f"velocity={args.mode_velocity:.4f}"
    )
    print(f"Saved frames: {len(times)} (every {args.thin} steps)")
    print(f"Time span: {times[0]:.3f} → {times[-1]:.3f}")
    print(f"Kinetic energy: {kinetic_series[0]:.8f} → {kinetic_series[-1]:.8f}")
    print(f"Potential energy: {potential_series[0]:.8f} → {potential_series[-1]:.8f}")
    print(f"Total energy: {total_series[0]:.8f} → {total_series[-1]:.8f}")

    if args.save_traj is not None:
        save_trajectory_csv(args.save_traj, times, positions)
        print(f"Trajectory saved to {args.save_traj}")

    if args.save_summary is not None:
        save_summary_json(
            args.save_summary,
            spec,
            omega2,
            args.mode_displacement,
            args.mode_velocity,
            args.dt,
            args.T,
            args.thin,
            masses,
            times,
            positions,
            mode_shape,
            kinetic_series,
            potential_series,
            total_series,
            energy_initial,
            energy_final,
        )
        print(f"Summary saved to {args.save_summary}")

    if args.plot_energies is not None:
        plot_energy_series(
            args.plot_energies,
            times,
            kinetic_series,
            potential_series,
            total_series,
        )
        print(f"Energy plot saved to {args.plot_energies}")


if __name__ == "__main__":
    main()
