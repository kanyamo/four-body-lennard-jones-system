#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time integration helper for the 3D, 4-body Lennard-Jones system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lj4_core import (
    EquilibriumSpec,
    SimulationResult,
    available_configs,
    simulate_trajectory,
)


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
    Path(path).write_text(json.dumps(summary, indent=2))


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Integrate the 4-body LJ system from a first-mode perturbation",
    )
    ap.add_argument(
        "--config",
        choices=available_configs(),
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


def integrate(
    config: str,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
) -> SimulationResult:
    return simulate_trajectory(
        config=config,
        mode_displacement=mode_displacement,
        mode_velocity=mode_velocity,
        dt=dt,
        total_time=total_time,
        save_stride=save_stride,
        center_mass=center_mass,
        record_energies=True,
    )


def simulate(
    config: str,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
):
    result = integrate(
        config,
        mode_displacement,
        mode_velocity,
        dt,
        total_time,
        save_stride,
        center_mass,
    )
    return (
        result.spec,
        result.omega2,
        result.mode_shape,
        result.masses,
        result.times,
        result.positions,
        result.kinetic,
        result.potential,
        result.total,
        result.energy_initial,
        result.energy_final,
    )


def main() -> None:
    args = parse_args()
    if args.center_mass <= 0.0:
        raise ValueError("--center-mass must be positive")

    result = integrate(
        args.config,
        args.mode_displacement,
        args.mode_velocity,
        args.dt,
        args.T,
        args.thin,
        args.center_mass,
    )

    print(f"Configuration: {result.spec.label}")
    print(f"First stable mode ω² = {result.omega2:.6f}")
    print(
        f"Initial conditions: displacement={args.mode_displacement:.4f}, "
        f"velocity={args.mode_velocity:.4f}"
    )
    print(f"Saved frames: {len(result.times)} (every {args.thin} steps)")
    print(f"Time span: {result.times[0]:.3f} → {result.times[-1]:.3f}")
    if result.kinetic is not None and result.potential is not None and result.total is not None:
        print(
            f"Kinetic energy: {result.kinetic[0]:.8f} → {result.kinetic[-1]:.8f}"
        )
        print(
            f"Potential energy: {result.potential[0]:.8f} → {result.potential[-1]:.8f}"
        )
        print(f"Total energy: {result.total[0]:.8f} → {result.total[-1]:.8f}")

    if args.save_traj is not None:
        save_trajectory_csv(args.save_traj, result.times, result.positions)
        print(f"Trajectory saved to {args.save_traj}")

    if args.save_summary is not None and result.kinetic is not None and result.potential is not None and result.total is not None and result.energy_initial is not None and result.energy_final is not None:
        save_summary_json(
            args.save_summary,
            result.spec,
            result.omega2,
            args.mode_displacement,
            args.mode_velocity,
            args.dt,
            args.T,
            args.thin,
            result.masses,
            result.times,
            result.positions,
            result.mode_shape,
            result.kinetic,
            result.potential,
            result.total,
            result.energy_initial,
            result.energy_final,
        )
        print(f"Summary saved to {args.save_summary}")

    if args.plot_energies is not None and result.kinetic is not None and result.potential is not None:
        plot_energy_series(
            args.plot_energies,
            result.times,
            result.kinetic,
            result.potential,
            result.total,
        )
        print(f"Energy plot saved to {args.plot_energies}")


if __name__ == "__main__":
    main()
