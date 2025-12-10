#!/usr/bin/env python3
"""Time integration helper for the 3D, 4-body spring rhombus system."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from spring4_core import (
    compute_modal_basis,
    kinetic_energy,
    simulate_trajectory,
    total_potential,
)
from utils import DIHEDRAL_EDGES, dihedral_angles, recenter


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Integrate the 4-body spring rhombus from a mode perturbation",
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
    ap.add_argument("--T", type=float, default=40.0, help="total simulated time")
    ap.add_argument(
        "--thin",
        type=int,
        default=10,
        help="store every Nth integrator step (>=1)",
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
        "--plot-modal",
        type=Path,
        default=None,
        help="optional PNG path plotting modal coordinates/velocities/energies",
    )
    ap.add_argument(
        "--plot-modal-categories",
        type=str,
        default="stable",
        help="comma-separated classes to plot in modal figure (stable,unstable,zero)",
    )
    ap.add_argument(
        "--plot-dihedral",
        type=Path,
        default=None,
        help="optional PNG path plotting dihedral angles/gaps",
    )
    ap.add_argument(
        "--kick-energy",
        type=float,
        default=0.0,
        help="total kinetic energy injected as ±z kick (T,B: +v; R,L: -v)",
    )
    return ap.parse_args()


def save_trajectory_csv(path: Path, times: np.ndarray, positions: np.ndarray) -> None:
    n_particles = positions.shape[1]
    header_cols = ["t"]
    for pid in range(n_particles):
        header_cols.extend([f"x{pid}", f"y{pid}", f"z{pid}"])
    header = ",".join(header_cols)
    data = np.column_stack([times, positions.reshape(len(times), -1)])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def compute_modal_projections(
    result, basis_vecs: np.ndarray, align: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    masses = result.masses
    mass_diag = np.repeat(masses, 3)
    basis_T = basis_vecs.T
    eq_flat = result.spec.positions.reshape(-1)

    coords = np.zeros((len(result.times), eq_flat.size))
    velocities = np.zeros_like(coords)
    for idx, (pos, vel) in enumerate(zip(result.positions, result.velocities)):
        if align:
            centered = recenter(pos, masses)
            delta = centered.reshape(-1) - recenter(
                result.spec.positions, masses
            ).reshape(-1)
            v_flat = vel.reshape(-1)
        else:
            delta = pos.reshape(-1) - eq_flat
            v_flat = vel.reshape(-1)
        coords[idx] = basis_T @ (mass_diag * delta)
        velocities[idx] = basis_T @ (mass_diag * v_flat)
    return coords, velocities


def main() -> None:
    args = parse_args()
    if args.thin < 1:
        raise ValueError("--thin must be >= 1")

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

    kick_velocity = None
    if args.kick_energy != 0.0:
        v_mag = math.sqrt(args.kick_energy / 2.0)
        kick_velocity = np.zeros((4, 3), dtype=float)
        # Particle order: R(0), T(1), L(2), B(3)
        kick_velocity[0, 2] = -v_mag
        kick_velocity[1, 2] = +v_mag
        kick_velocity[2, 2] = -v_mag
        kick_velocity[3, 2] = +v_mag

    result = simulate_trajectory(
        mode_indices=mode_indices,
        mode_displacements=mode_displacements,
        mode_velocities=mode_velocities,
        dt=args.dt,
        total_time=args.T,
        save_stride=args.thin,
        kick_velocity=kick_velocity,
    )

    kinetic = np.array([kinetic_energy(v, result.masses) for v in result.velocities])
    potential = np.array([total_potential(x) for x in result.positions])
    total = kinetic + potential

    print(f"Config: {result.spec.label}")
    print(f"Mode indices: {result.mode_indices}")
    print(f"Displacements: {result.displacement_coeffs}")
    print(f"Velocities: {result.velocity_coeffs}")
    print(
        f"dt={result.dt:.4f}, total_time={result.total_time}, frames={len(result.times)}"
    )
    print(f"Total energy: {total[0]:.6f} -> {total[-1]:.6f}")

    if args.save_traj:
        save_trajectory_csv(args.save_traj, result.times, result.positions)
        print(f"Saved trajectory CSV: {args.save_traj}")

    if args.save_summary:
        eigvals, _, classes = compute_modal_basis(result.spec.positions, result.masses)
        summary = {
            "mode_indices": list(result.mode_indices),
            "mode_eigenvalues": list(result.mode_eigenvalues),
            "displacement_coeffs": list(result.displacement_coeffs),
            "velocity_coeffs": list(result.velocity_coeffs),
            "dt": result.dt,
            "total_time": result.total_time,
            "save_stride": result.save_stride,
            "energies": {
                "kinetic": kinetic.tolist(),
                "potential": potential.tolist(),
                "total": total.tolist(),
            },
            "modal_spectrum": {
                "eigenvalues": eigvals.tolist(),
                "classifications": list(classes),
            },
        }
        args.save_summary.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary JSON: {args.save_summary}")

    if args.plot_energies:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3.5))
        plt.plot(result.times, kinetic, label="kinetic")
        plt.plot(result.times, potential, label="potential")
        plt.plot(result.times, total, label="total", linestyle="--", linewidth=1.0)
        plt.xlabel("time")
        plt.ylabel("energy")
        plt.title("Energy budget vs time (spring)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_energies, dpi=160)
        plt.close()
        print(f"Saved energy plot: {args.plot_energies}")

    if args.plot_modal or args.plot_dihedral:
        basis_vals, basis_vecs, basis_classes = compute_modal_basis(
            result.spec.positions, result.masses
        )
        coords, vels = compute_modal_projections(result, basis_vecs, align=True)

        if args.plot_modal:
            import matplotlib.pyplot as plt

            classes_to_plot = [
                c.strip() for c in args.plot_modal_categories.split(",") if c.strip()
            ]
            indices = [
                i for i, cls in enumerate(basis_classes) if cls in classes_to_plot
            ]
            labels = [f"{basis_classes[i]}_{i}" for i in indices]
            modal_energy = 0.5 * (vels**2) + 0.5 * coords * (
                coords * basis_vals[None, :]
            )

            fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
            ax_q, ax_v, ax_e, ax_meta = axes.flatten()

            def plot_lines(ax, data, ylabel: str) -> None:
                for idx, label in zip(indices, labels):
                    ax.plot(result.times, np.abs(data[:, idx]), label=label)
                ax.set_xlabel("time")
                ax.set_ylabel(ylabel)
                ax.set_yscale("log")
                ax.grid(True, linestyle="--", alpha=0.3)

            plot_lines(ax_q, coords, "modal |q|")
            ax_q.set_title("Modal coordinates")
            plot_lines(ax_v, vels, "modal |q_dot|")
            ax_v.set_title("Modal velocities")
            plot_lines(ax_e, modal_energy, "modal energy")
            ax_e.set_title("Modal energies")

            ax_meta.axis("off")
            ax_meta.text(
                0.02,
                0.98,
                f"classes={','.join(classes_to_plot)}",
                ha="left",
                va="top",
                fontsize="small",
                wrap=True,
            )

            handles, legend_labels = ax_q.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles, legend_labels, loc="upper center", ncol=4, fontsize="small"
                )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            plt.savefig(args.plot_modal, dpi=170)
            plt.close(fig)
            print(f"Saved modal plot: {args.plot_modal}")

        if args.plot_dihedral:
            import matplotlib.pyplot as plt

            angles = np.array([dihedral_angles(pos) for pos in result.positions])
            gaps = math.pi - angles
            plt.figure(figsize=(6, 3.5))
            for idx, edge in enumerate(DIHEDRAL_EDGES):
                plt.plot(result.times, gaps[:, idx], label=f"{edge[0]}-{edge[1]}")
            plt.xlabel("time")
            plt.ylabel("dihedral gap (rad from pi)")
            plt.yscale("log")
            plt.title("Planarity gap vs time (spring)")
            plt.legend(loc="best", fontsize="small")
            plt.tight_layout()
            plt.savefig(args.plot_dihedral, dpi=160)
            plt.close()
            print(f"Saved dihedral plot: {args.plot_dihedral}")


if __name__ == "__main__":
    main()
