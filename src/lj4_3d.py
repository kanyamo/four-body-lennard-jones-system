#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time integration helper for the 3D, 4-body Lennard-Jones system."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence as Seq
from pathlib import Path

import numpy as np

from lj4_core import (
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
    result: SimulationResult,
    dt: float,
    total_time: float,
    save_stride: int,
) -> None:
    summary = {
        "config": result.spec.key,
        "label": result.spec.label,
        "primary_omega2": result.omega2,
        "dt": dt,
        "total_time": total_time,
        "save_stride": save_stride,
        "frames": int(len(result.times)),
        "n_particles": int(result.positions.shape[1]),
        "masses": result.masses.tolist(),
        "time_start": float(result.times[0]),
        "time_end": float(result.times[-1]),
        "times": result.times.tolist(),
        "mode_selection": {
            "indices": list(result.mode_indices),
            "eigenvalues": list(result.mode_eigenvalues),
            "displacement_coeffs": list(result.displacement_coeffs),
            "velocity_coeffs": list(result.velocity_coeffs),
        },
        "initial_displacement": result.mode_shape.tolist(),
        "initial_velocity": result.initial_velocity.tolist(),
        "energies": {
            "kinetic": result.kinetic.tolist(),
            "potential": result.potential.tolist(),
            "total": result.total.tolist(),
        },
        "energy_initial": result.energy_initial,
        "energy_final": result.energy_final,
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
        type=str,
        default="0.02",
        help="comma-separated displacement amplitudes for selected modes",
    )
    ap.add_argument(
        "--mode-velocity",
        type=str,
        default="0.00",
        help="comma-separated velocity amplitudes for selected modes",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="0",
        help="comma-separated stable-mode indices (0 = lowest positive eigenvalue)",
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
    ap.add_argument(
        "--random-kick-energy",
        type=float,
        default=0.01,
        help="additive random kinetic energy injected to break symmetry (set 0 to disable)",
    )
    ap.add_argument(
        "--random-kick-seed",
        type=int,
        default=12345,
        help="seed for the random kick (use -1 for nondeterministic)",
    )
    return ap.parse_args()


def _ensure_float_list(value: float | Seq[float]) -> list[float]:
    if isinstance(value, Seq) and not isinstance(value, (str, bytes)):
        return [float(v) for v in value]
    return [float(value)]


def integrate(
    config: str,
    mode_displacement: float | Seq[float],
    mode_velocity: float | Seq[float],
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
    mode_indices: Seq[int] | None = None,
    random_kick_energy: float = 0.01,
    random_kick_seed: int | None = 12345,
) -> SimulationResult:
    disp_list = _ensure_float_list(mode_displacement)
    vel_list = _ensure_float_list(mode_velocity)
    seed = random_kick_seed if random_kick_seed != -1 else None
    return simulate_trajectory(
        config=config,
        mode_indices=mode_indices,
        mode_displacements=disp_list,
        mode_velocities=vel_list,
        dt=dt,
        total_time=total_time,
        save_stride=save_stride,
        center_mass=center_mass,
        record_energies=True,
        random_kick_energy=random_kick_energy,
        random_seed=seed,
    )


def simulate(
    config: str,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
    mode_indices: Seq[int] | None = None,
    mode_displacements: Seq[float] | None = None,
    mode_velocities: Seq[float] | None = None,
    random_kick_energy: float = 0.01,
    random_kick_seed: int | None = 12345,
):
    result = integrate(
        config,
        mode_displacements if mode_displacements is not None else mode_displacement,
        mode_velocities if mode_velocities is not None else mode_velocity,
        dt,
        total_time,
        save_stride,
        center_mass,
        mode_indices=mode_indices,
        random_kick_energy=random_kick_energy,
        random_kick_seed=random_kick_seed,
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

    def parse_float_list(raw: str, name: str) -> list[float]:
        parts = [item.strip() for item in str(raw).split(",") if item.strip()]
        if not parts:
            raise ValueError(f"{name} must contain at least one value")
        return [float(item) for item in parts]

    def parse_int_list(raw: str, name: str) -> list[int]:
        parts = [item.strip() for item in str(raw).split(",") if item.strip()]
        if not parts:
            raise ValueError(f"{name} must contain at least one value")
        return [int(item) for item in parts]

    mode_indices = tuple(parse_int_list(args.modes, "--modes"))
    mode_displacements = parse_float_list(args.mode_displacement, "--mode-displacement")
    mode_velocities = parse_float_list(args.mode_velocity, "--mode-velocity")

    result = integrate(
        args.config,
        mode_displacements,
        mode_velocities,
        args.dt,
        args.T,
        args.thin,
        args.center_mass,
        mode_indices=mode_indices,
        random_kick_energy=args.random_kick_energy,
        random_kick_seed=args.random_kick_seed,
    )

    print(f"Configuration: {result.spec.label}")
    mode_info = ", ".join(
        f"{idx} (ω²={eig:.6f}, disp={disp:.4f}, vel={vel:.4f})"
        for idx, eig, disp, vel in zip(
            result.mode_indices,
            result.mode_eigenvalues,
            result.displacement_coeffs,
            result.velocity_coeffs,
        )
    )
    print(f"Modes: {mode_info}")
    print(f"Displacement coeffs: {', '.join(f'{v:.4f}' for v in mode_displacements)}")
    print(f"Velocity coeffs: {', '.join(f'{v:.4f}' for v in mode_velocities)}")
    print(f"Saved frames: {len(result.times)} (every {args.thin} steps)")
    print(f"Time span: {result.times[0]:.3f} → {result.times[-1]:.3f}")
    if (
        result.kinetic is not None
        and result.potential is not None
        and result.total is not None
    ):
        print(f"Kinetic energy: {result.kinetic[0]:.8f} → {result.kinetic[-1]:.8f}")
        print(
            f"Potential energy: {result.potential[0]:.8f} → {result.potential[-1]:.8f}"
        )
        print(f"Total energy: {result.total[0]:.8f} → {result.total[-1]:.8f}")

    if args.save_traj is not None:
        save_trajectory_csv(args.save_traj, result.times, result.positions)
        print(f"Trajectory saved to {args.save_traj}")

    if (
        args.save_summary is not None
        and result.kinetic is not None
        and result.potential is not None
        and result.total is not None
        and result.energy_initial is not None
        and result.energy_final is not None
    ):
        save_summary_json(
            args.save_summary,
            result,
            args.dt,
            args.T,
            args.thin,
        )
        print(f"Summary saved to {args.save_summary}")

    if (
        args.plot_energies is not None
        and result.kinetic is not None
        and result.potential is not None
    ):
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
