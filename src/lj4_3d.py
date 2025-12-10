#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time integration helper for the 3D, 4-body Lennard-Jones system."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence as Seq
from pathlib import Path
from typing import Any

import numpy as np

from lj4_storage import (
    CACHE_DEFAULT_DIRNAME,
    bundle_exists,
    compute_bundle_dir,
    load_simulation_bundle,
    save_simulation_bundle,
)

from lj4_core import (
    SimulationResult,
    available_configs,
    compute_dihedral_series,
    compute_energy_series,
    compute_modal_energies,
    compute_modal_projections,
    plot_dihedral_series,
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
    modal_coords, modal_vels, modal_basis = compute_modal_projections(result)
    dihedral_angles, dihedral_gap = compute_dihedral_series(result)
    kinetic, potential, total = compute_energy_series(result)
    modal_ke, modal_pe, modal_te, _ = compute_modal_energies(
        result, modal_basis=modal_basis
    )

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
        "repulsive_exp": result.repulsive_exp,
        "attractive_exp": result.attractive_exp,
        "time_start": float(result.times[0]),
        "time_end": float(result.times[-1]),
        "times": result.times.tolist(),
        "mode_selection": {
            "indices": list(result.mode_indices),
            "eigenvalues": list(result.mode_eigenvalues),
            "displacement_coeffs": list(result.displacement_coeffs),
            "velocity_coeffs": list(result.velocity_coeffs),
        },
        "initial_displacement": result.initial_displacement.tolist(),
        "initial_velocity": result.initial_velocity.tolist(),
        "modal_basis": {
            "eigenvalues": modal_basis.eigenvalues.tolist(),
            "classifications": list(modal_basis.classifications),
            "labels": list(modal_basis.labels),
            "vectors": modal_basis.vectors.tolist(),
        },
        "modal_coordinates": modal_coords.tolist(),
        "modal_velocities": modal_vels.tolist(),
        "modal_energy": {
            "kinetic": modal_ke.tolist(),
            "potential": modal_pe.tolist(),
            "total": modal_te.tolist(),
        },
        "dihedral_edges": [list(edge) for edge in result.dihedral_edges],
        "dihedral_angles": dihedral_angles.tolist(),
        "dihedral_planarity_gap": dihedral_gap.tolist(),
        "energies": {
            "kinetic": kinetic.tolist(),
            "potential": potential.tolist(),
            "total": total.tolist(),
        },
        "energy_initial": float(total[0]),
        "energy_final": float(total[-1]),
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


def plot_modal_series(
    path: Path,
    times: np.ndarray,
    modal_coords: np.ndarray,
    modal_vels: np.ndarray,
    modal_energy: np.ndarray,
    indices: list[int],
    labels: Seq[str],
    params_text: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
    ax_q, ax_v, ax_e, ax_meta = axes.flatten()

    def plot_lines(ax, data, ylabel: str) -> None:
        for idx in indices:
            ax.plot(times, np.abs(data[:, idx]), label=labels[idx])
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.3)

    plot_lines(ax_q, modal_coords, "modal |q|")
    ax_q.set_title("Modal coordinates")

    plot_lines(ax_v, modal_vels, "modal |q_dot|")
    ax_v.set_title("Modal velocities")

    plot_lines(ax_e, modal_energy, "modal energy")
    ax_e.set_title("Modal energies")

    ax_meta.axis("off")
    ax_meta.text(
        0.02,
        0.98,
        params_text,
        ha="left",
        va="top",
        fontsize="small",
        wrap=True,
    )

    handles, legend_labels = ax_q.get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=4, fontsize="small")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(path, dpi=170)
    plt.close(fig)


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
        "--repulsive-exp",
        type=int,
        default=12,
        help="repulsive exponent p in the (p,q)-LJ potential (triangle_center only)",
    )
    ap.add_argument(
        "--attractive-exp",
        type=int,
        default=6,
        help="attractive exponent q in the (p,q)-LJ potential (triangle_center only)",
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
        "--modal-report",
        type=str,
        default="unstable",
        help="comma-separated modal classes to print (stable,unstable,zero,all,none)",
    )
    ap.add_argument(
        "--plot-modal",
        type=Path,
        default=None,
        help="optional PNG path plotting modal coordinates for selected classes",
    )
    ap.add_argument(
        "--plot-modal-categories",
        type=str,
        default=None,
        help="comma-separated modal classes to plot (default: same as --modal-report)",
    )
    ap.add_argument(
        "--plot-dihedral",
        type=Path,
        default=None,
        help="optional PNG path plotting dihedral evolution over time",
    )
    ap.add_argument(
        "--dihedral-quantity",
        choices=("gap", "angle"),
        default="gap",
        help="plot 180°-angle gap (gap) or the raw dihedral angle (angle)",
    )
    ap.add_argument(
        "--modal-kick-energy",
        type=float,
        default=0.01,
        help="deterministic kinetic energy injected along the leading unstable mode (0 to disable)",
    )
    ap.add_argument(
        "--random-displacement",
        type=float,
        default=0.0,
        help="ランダム方向の初期変位の大きさ（0で無効）",
    )
    ap.add_argument(
        "--random-kick-energy",
        type=float,
        default=0.0,
        help="ランダム方向の初期運動エネルギー（0で無効）",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="ランダム初期化のシード（未指定ならRNGデフォルト）",
    )
    ap.add_argument(
        "--save-bundle",
        type=Path,
        default=None,
        help="保存用バンドルのディレクトリ (metadata.json + series.npz)",
    )
    ap.add_argument(
        "--load-bundle",
        type=Path,
        default=None,
        help="既存バンドルを読み込んで再計算せずに出力を生成する",
    )
    ap.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="パラメータから決まる場所に自動で保存・再利用する (デフォルト: 有効)",
    )
    ap.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="キャッシュを無効化する",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(CACHE_DEFAULT_DIRNAME),
        help="キャッシュ保存のベースディレクトリ",
    )
    return ap.parse_args()


def _ensure_float_list(value: float | Seq[float]) -> list[float]:
    if isinstance(value, Seq) and not isinstance(value, (str, bytes)):
        return [float(v) for v in value]
    return [float(value)]


def _parse_modal_categories(raw: str, arg_name: str) -> tuple[str, ...]:
    normalized = raw.strip().lower()
    if normalized == "none":
        return ()
    if normalized == "all":
        return ("unstable", "stable", "zero")
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = {"unstable", "stable", "zero"}
    invalid = [item for item in items if item not in valid]
    if invalid:
        raise ValueError(
            f"{arg_name} contains invalid modal class(es): {', '.join(invalid)}"
        )
    # preserve user order but drop duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _collect_modal_indices(
    classifications: Seq[str],
    target_categories: Seq[str],
) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {cat: [] for cat in target_categories}
    for idx, cls in enumerate(classifications):
        for category in target_categories:
            if cls == category:
                mapping[category].append(idx)
                break
    return mapping


def _build_run_parameters(
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
    repulsive_exp: int,
    attractive_exp: int,
    mode_indices: Seq[int],
    mode_displacements: Seq[float],
    mode_velocities: Seq[float],
    modal_kick_energy: float,
    random_displacement: float,
    random_kick_energy: float,
    random_seed: int | None,
) -> dict[str, float | int | list[int] | list[float] | None]:
    return {
        "dt": dt,
        "total_time": total_time,
        "save_stride": save_stride,
        "center_mass": center_mass,
        "repulsive_exp": repulsive_exp,
        "attractive_exp": attractive_exp,
        "mode_indices": list(mode_indices),
        "mode_displacements": list(mode_displacements),
        "mode_velocities": list(mode_velocities),
        "modal_kick_energy": modal_kick_energy,
        "random_displacement": random_displacement,
        "random_kick_energy": random_kick_energy,
        "random_seed": random_seed,
    }


def integrate(
    config: str,
    mode_displacement: float | Seq[float],
    mode_velocity: float | Seq[float],
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
    repulsive_exp: int,
    attractive_exp: int,
    mode_indices: Seq[int] | None = None,
    modal_kick_energy: float = 0.01,
    random_displacement: float = 0.0,
    random_kick_energy: float = 0.0,
    random_seed: int | None = None,
) -> SimulationResult:
    disp_list = _ensure_float_list(mode_displacement)
    vel_list = _ensure_float_list(mode_velocity)
    return simulate_trajectory(
        config=config,
        mode_indices=mode_indices,
        mode_displacements=disp_list,
        mode_velocities=vel_list,
        dt=dt,
        total_time=total_time,
        save_stride=save_stride,
        center_mass=center_mass,
        repulsive_exp=repulsive_exp,
        attractive_exp=attractive_exp,
        modal_kick_energy=modal_kick_energy,
        random_displacement=random_displacement,
        random_kick_energy=random_kick_energy,
        random_seed=random_seed,
    )


def simulate(
    config: str,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
    repulsive_exp: int,
    attractive_exp: int,
    mode_indices: Seq[int] | None = None,
    mode_displacements: Seq[float] | None = None,
    mode_velocities: Seq[float] | None = None,
    modal_kick_energy: float = 0.01,
    random_displacement: float = 0.0,
    random_kick_energy: float = 0.0,
    random_seed: int | None = None,
):
    result = integrate(
        config,
        mode_displacements if mode_displacements is not None else mode_displacement,
        mode_velocities if mode_velocities is not None else mode_velocity,
        dt,
        total_time,
        save_stride,
        center_mass,
        repulsive_exp=repulsive_exp,
        attractive_exp=attractive_exp,
        mode_indices=mode_indices,
        modal_kick_energy=modal_kick_energy,
        random_displacement=random_displacement,
        random_kick_energy=random_kick_energy,
        random_seed=random_seed,
    )
    kinetic, potential, total = compute_energy_series(result)
    return (
        result.spec,
        result.omega2,
        result.initial_displacement,
        result.masses,
        result.times,
        result.positions,
        kinetic,
        potential,
        total,
        float(total[0]),
        float(total[-1]),
    )


def main() -> None:
    args = parse_args()
    loading_bundle = args.load_bundle is not None
    random_used = args.random_displacement > 0.0 or args.random_kick_energy > 0.0
    if random_used and args.use_cache:
        raise ValueError(
            "--no-cache を指定してください（ランダム初期化を使うときはキャッシュ不可）"
        )
    if random_used and loading_bundle:
        raise ValueError("--load-bundle とランダム初期化は同時に使えません")
    if not loading_bundle:
        if args.center_mass <= 0.0:
            raise ValueError("--center-mass must be positive")
        if args.repulsive_exp <= args.attractive_exp or args.attractive_exp <= 0:
            raise ValueError("repulsive-exp must be > attractive-exp > 0")
    if args.use_cache and loading_bundle:
        print("注意: --load-bundle が指定されたため --use-cache は無視します。")
        args.use_cache = False

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

    modal_report_categories = _parse_modal_categories(
        args.modal_report, "--modal-report"
    )
    if args.plot_modal_categories is None:
        modal_plot_categories = modal_report_categories
    else:
        modal_plot_categories = _parse_modal_categories(
            args.plot_modal_categories, "--plot-modal-categories"
        )

    if loading_bundle:
        loaded = load_simulation_bundle(args.load_bundle)
        result = loaded.result
        run_parameters = loaded.metadata.get("run_parameters", {})
        print(f"バンドルを読み込みました: {args.load_bundle}")
        if not run_parameters:
            print(
                "注意: run_parameters が見つからなかったため、CLI引数を参考表示に使います。"
            )
        mode_indices = tuple(result.mode_indices)
        mode_displacements = list(result.displacement_coeffs)
        mode_velocities = list(result.velocity_coeffs)
    else:
        mode_indices = tuple(parse_int_list(args.modes, "--modes"))
        mode_displacements = parse_float_list(
            args.mode_displacement, "--mode-displacement"
        )
        mode_velocities = parse_float_list(args.mode_velocity, "--mode-velocity")
        run_parameters = _build_run_parameters(
            args.dt,
            args.T,
            args.thin,
            args.center_mass,
            args.repulsive_exp,
            args.attractive_exp,
            mode_indices,
            mode_displacements,
            mode_velocities,
            args.modal_kick_energy,
            args.random_displacement,
            args.random_kick_energy,
            args.random_seed,
        )
        if args.use_cache:
            cache_dir, cache_key = compute_bundle_dir(
                args.cache_dir, args.config, run_parameters
            )
            if bundle_exists(cache_dir):
                loaded = load_simulation_bundle(cache_dir)
                result = loaded.result
                run_parameters = loaded.metadata.get("run_parameters", run_parameters)
                print(f"キャッシュヒット: {cache_dir} (key={cache_key})")
            else:
                result = integrate(
                    args.config,
                    mode_displacements,
                    mode_velocities,
                    args.dt,
                    args.T,
                    args.thin,
                    args.center_mass,
                    repulsive_exp=args.repulsive_exp,
                    attractive_exp=args.attractive_exp,
                    mode_indices=mode_indices,
                    modal_kick_energy=args.modal_kick_energy,
                    random_displacement=args.random_displacement,
                    random_kick_energy=args.random_kick_energy,
                    random_seed=args.random_seed,
                )
                save_simulation_bundle(cache_dir, result, run_parameters)
                print(f"キャッシュ保存: {cache_dir} (key={cache_key})")
            if args.save_bundle is not None and Path(args.save_bundle) != cache_dir:
                save_simulation_bundle(args.save_bundle, result, run_parameters)
                print(f"バンドルを書き出しました: {args.save_bundle}")
        else:
            result = integrate(
                args.config,
                mode_displacements,
                mode_velocities,
                args.dt,
                args.T,
                args.thin,
                args.center_mass,
                repulsive_exp=args.repulsive_exp,
                attractive_exp=args.attractive_exp,
                mode_indices=mode_indices,
                modal_kick_energy=args.modal_kick_energy,
                random_displacement=args.random_displacement,
                random_kick_energy=args.random_kick_energy,
                random_seed=args.random_seed,
            )
            if args.save_bundle is not None:
                save_simulation_bundle(args.save_bundle, result, run_parameters)
                print(f"バンドルを書き出しました: {args.save_bundle}")

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
    run_dt = float(run_parameters.get("dt", args.dt))
    run_T = float(run_parameters.get("total_time", args.T))
    run_stride = int(run_parameters.get("save_stride", args.thin))
    print(f"Integrator dt: {run_dt:.5f}, total_time: {run_T}")
    print(f"Saved frames: {len(result.times)} (every {run_stride} steps)")
    print(f"Time span: {result.times[0]:.3f} → {result.times[-1]:.3f}")
    energy_series: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    modal_data: tuple[np.ndarray, np.ndarray, Any] | None = None
    modal_energy_data: tuple[np.ndarray, np.ndarray, np.ndarray, Any] | None = None
    dihedral_data: tuple[np.ndarray, np.ndarray] | None = None

    def get_energy_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal energy_series
        if energy_series is None:
            energy_series = compute_energy_series(result)
        return energy_series

    def get_modal_data() -> tuple[np.ndarray, np.ndarray, Any]:
        nonlocal modal_data
        if modal_data is None:
            modal_data = compute_modal_projections(result)
        return modal_data

    def get_modal_energy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        nonlocal modal_energy_data
        if modal_energy_data is None:
            modal_energy_data = compute_modal_energies(result)
        return modal_energy_data

    def get_dihedral_data() -> tuple[np.ndarray, np.ndarray]:
        nonlocal dihedral_data
        if dihedral_data is None:
            dihedral_data = compute_dihedral_series(result)
        return dihedral_data

    kinetic, potential, total = get_energy_series()
    print(f"Kinetic energy: {kinetic[0]:.8f} → {kinetic[-1]:.8f}")
    print(f"Potential energy: {potential[0]:.8f} → {potential[-1]:.8f}")
    print(f"Total energy: {total[0]:.8f} → {total[-1]:.8f}")

    if modal_report_categories:
        modal_coords, _, modal_basis = get_modal_data()
        modal_groups = _collect_modal_indices(
            modal_basis.classifications, modal_report_categories
        )
        for category in modal_report_categories:
            indices = modal_groups.get(category, [])
            if not indices:
                continue
            label_prefix = category.capitalize()
            entries = []
            for idx in indices:
                label = modal_basis.labels[idx]
                initial_coeff = modal_coords[0, idx]
                final_coeff = modal_coords[-1, idx]
                entries.append(f"{label}: {initial_coeff:.6f} → {final_coeff:.6f}")
            print(f"{label_prefix} modal coordinates:", "; ".join(entries))

    if args.save_traj is not None:
        save_trajectory_csv(args.save_traj, result.times, result.positions)
        print(f"Trajectory saved to {args.save_traj}")

    if args.save_summary is not None:
        save_summary_json(
            args.save_summary,
            result,
            run_dt,
            run_T,
            run_stride,
        )
        print(f"Summary saved to {args.save_summary}")

    if args.plot_energies is not None:
        kinetic, potential, total = get_energy_series()
        plot_energy_series(
            args.plot_energies,
            result.times,
            kinetic,
            potential,
            total,
        )
        print(f"Energy plot saved to {args.plot_energies}")

    if args.plot_modal is not None and modal_plot_categories:
        modal_coords, modal_vels, modal_basis = get_modal_data()
        modal_ke, modal_pe, modal_te, _ = get_modal_energy_data()
        modal_energy_monotone = 0.5 * modal_vels**2 + 0.5 * (
            np.abs(modal_basis.eigenvalues)[None, :]
        ) * (modal_coords**2)
        modal_groups = _collect_modal_indices(
            modal_basis.classifications, modal_plot_categories
        )
        plot_indices: list[int] = []
        for category in modal_plot_categories:
            plot_indices.extend(modal_groups.get(category, []))
        if plot_indices:
            plot_modal_series(
                args.plot_modal,
                result.times,
                modal_coords,
                modal_vels,
                modal_energy_monotone,
                plot_indices,
                modal_basis.labels,
                params_text=(
                    f"config: {result.spec.label}\n"
                    f"modes: {', '.join(str(i) for i in result.mode_indices)}\n"
                    f"disp: {', '.join(f'{v:.4f}' for v in result.displacement_coeffs)}\n"
                    f"vel: {', '.join(f'{v:.4f}' for v in result.velocity_coeffs)}\n"
                    f"dt={run_dt}, T={run_T}, stride={run_stride}\n"
                    f"center_mass={run_parameters.get('center_mass', args.center_mass)}, kick={run_parameters.get('modal_kick_energy', args.modal_kick_energy)}"
                ),
            )
            print(
                f"Modal coordinate plot saved to {args.plot_modal} "
                f"(classes: {', '.join(modal_plot_categories)})"
            )
        else:
            print(
                "No modal indices matched the requested plot categories; "
                "plot was not produced."
            )
    if args.plot_dihedral is not None:
        dihedral_angles, _ = get_dihedral_data()
        plot_dihedral_series(
            args.plot_dihedral,
            result.times,
            dihedral_angles,
            edges=result.dihedral_edges,
            quantity=args.dihedral_quantity,
        )
        print(
            f"Dihedral plot saved to {args.plot_dihedral} "
            f"(quantity: {args.dihedral_quantity})"
        )


if __name__ == "__main__":
    main()
