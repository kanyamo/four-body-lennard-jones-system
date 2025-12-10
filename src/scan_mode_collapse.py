#!/usr/bin/env python3
"""Scan collapse times for mode-driven LJ4 trajectories.

与えられた設定・モードに対し、初期運動エネルギー（モード速度係数）を走査し、
シミュレーション中に不安定モード座標のいずれかが指定閾値を超える最初の時刻を
「崩壊時間」として記録します。結果は CSV に保存でき、オプションでプロットも作成できます。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from lj4_core import (
    available_configs,
    compute_energy_series,
    compute_modal_projections,
    kinetic_energy,
    DEFAULT_REPULSIVE_EXP,
    DEFAULT_ATTRACTIVE_EXP,
    prepare_equilibrium,
    simulate_trajectory,
    stable_modes,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=available_configs(), default="rhombus")
    ap.add_argument(
        "--mode-index", type=int, default=0, help="primary normal mode index"
    )
    ap.add_argument(
        "--secondary-mode",
        type=int,
        default=None,
        help="optional secondary mode index for mixing scans",
    )
    ap.add_argument(
        "--energy-min",
        type=float,
        default=0.0,
        help="minimum initial kinetic energy for the selected mode",
    )
    ap.add_argument(
        "--energy-max",
        type=float,
        default=1.0,
        help="maximum initial kinetic energy for the selected mode",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=8,
        help="number of evenly spaced samples (>=2 for a range, 1 for single point)",
    )
    ap.add_argument(
        "--unstable-threshold",
        type=float,
        default=0.1,
        help="absolute modal coordinate threshold for unstable modes",
    )
    ap.add_argument(
        "--energy-total",
        type=float,
        default=None,
        help="(mixing mode) total initial kinetic energy to enforce",
    )
    ap.add_argument("--dt", type=float, default=0.002, help="integrator time step")
    ap.add_argument("--T", type=float, default=220.0, help="integration horizon")
    ap.add_argument(
        "--thin",
        type=int,
        default=20,
        help="store every Nth integrator step (>=1)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/collapse_scan"),
        help="directory for CSV / artefacts",
    )
    ap.add_argument(
        "--csv-name",
        type=str,
        default=None,
        help="optional CSV filename (defaults to auto-generated)",
    )
    ap.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="optional path for collapse-time plot (PNG)",
    )
    ap.add_argument(
        "--mode-displacement",
        type=float,
        default=0.0,
        help="initial displacement coefficient for the selected mode",
    )
    ap.add_argument(
        "--mix-samples",
        type=int,
        default=11,
        help="number of ratio samples when using two modes (>=2 recommended)",
    )
    ap.add_argument(
        "--mix-min",
        type=float,
        default=0.0,
        help="lower bound for the energy-ratio parameter (two-mode scans)",
    )
    ap.add_argument(
        "--mix-max",
        type=float,
        default=1.0,
        help="upper bound for the energy-ratio parameter (two-mode scans)",
    )
    ap.add_argument(
        "--center-mass",
        type=float,
        default=1.0,
        help="mass of the central particle (triangle+center config only)",
    )
    ap.add_argument(
        "--modal-kick-energy",
        type=float,
        default=0.01,
        help="deterministic kinetic energy injected along the leading unstable mode (0 to disable)",
    )
    return ap.parse_args()


def collapse_time_modal(
    times: np.ndarray, modal_coords: np.ndarray, unstable_indices: list[int], threshold: float
) -> float | None:
    if not unstable_indices:
        return None
    unstable_slice = modal_coords[:, unstable_indices]
    envelope = np.max(np.abs(unstable_slice), axis=1)
    mask = envelope >= threshold
    if not np.any(mask):
        return None
    idx = int(np.argmax(mask))
    return float(times[idx])


def kinetic_energy_to_velocity(energy: float) -> float:
    if energy < 0.0:
        raise ValueError("kinetic energy must be non-negative")
    return float(np.sqrt(2.0 * energy))


def sample_energies(min_val: float, max_val: float, samples: int) -> Iterable[float]:
    if samples <= 1:
        yield float(min_val)
        return
    for value in np.linspace(min_val, max_val, samples):
        yield float(value)


def main() -> None:
    args = parse_args()

    if args.energy_min < 0.0:
        raise SystemExit("--energy-min must be non-negative")
    if args.energy_max < args.energy_min:
        raise SystemExit("--energy-max must be >= --energy-min")
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.mix_samples < 1:
        raise SystemExit("--mix-samples must be >= 1")
    if not (0.0 <= args.mix_min <= args.mix_max <= 1.0):
        raise SystemExit("--mix-min/max must satisfy 0 <= min <= max <= 1")

    _, equilibrium, masses = prepare_equilibrium(
        args.config,
        args.center_mass,
        repulsive_exp=DEFAULT_REPULSIVE_EXP,
        attractive_exp=DEFAULT_ATTRACTIVE_EXP,
    )
    all_modes = stable_modes(
        equilibrium,
        masses,
        repulsive_exp=DEFAULT_REPULSIVE_EXP,
        attractive_exp=DEFAULT_ATTRACTIVE_EXP,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | int | str | None]] = []

    def run_simulation(
        mode_indices: list[int], vel_coeffs: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, float, float]:
        result = simulate_trajectory(
            config=args.config,
            mode_indices=mode_indices,
            mode_displacements=[args.mode_displacement] * len(mode_indices),
            mode_velocities=vel_coeffs,
            dt=args.dt,
            total_time=args.T,
            save_stride=args.thin,
            center_mass=args.center_mass,
            modal_kick_energy=args.modal_kick_energy,
            random_displacement=0.0,
            random_kick_energy=0.0,
            random_seed=None,
            repulsive_exp=DEFAULT_REPULSIVE_EXP,
            attractive_exp=DEFAULT_ATTRACTIVE_EXP,
        )
        times = np.asarray(result.times)
        kinetic, potential, _ = compute_energy_series(result)
        modal_coords, _, modal_basis = compute_modal_projections(result)
        unstable_indices = [
            idx for idx, cls in enumerate(modal_basis.classifications) if cls == "unstable"
        ]
        collapse = collapse_time_modal(
            times, modal_coords, unstable_indices, args.unstable_threshold
        )
        if unstable_indices:
            envelope = np.max(np.abs(modal_coords[:, unstable_indices]), axis=1)
            envelope_max = float(np.max(envelope))
        else:
            envelope_max = float("nan")
        pmin = float(potential.min())
        return times, kinetic, potential, collapse, pmin, envelope_max

    secondary = args.secondary_mode

    if secondary is None:
        mode_indices = [args.mode_index]
        for energy in sample_energies(args.energy_min, args.energy_max, args.samples):
            velocity = kinetic_energy_to_velocity(energy)
            _, kinetic, potential, collapse, pmin, env_max = run_simulation(
                mode_indices, [velocity]
            )
            k0 = float(kinetic[0])
            records.append(
                {
                    "mode_primary": args.mode_index,
                    "mode_secondary": "",
                    "mix_ratio": np.nan,
                    "energy_target": energy,
                    "kinetic_initial": k0,
                    "collapse_time": collapse,
                    "potential_min": pmin,
                    "unstable_envelope_max": env_max,
                    "unstable_threshold": args.unstable_threshold,
                }
            )
            if collapse is None:
                status = f"K0={k0:.4f}, v={velocity:.4f}, collapse=none"
            else:
                status = f"K0={k0:.4f}, v={velocity:.4f}, collapse={collapse:.3f}"
            status += f", max|q_unstable|={env_max:.3f}"
            print(status)
    else:
        if args.energy_total is None:
            raise SystemExit("--energy-total must be provided when mixing two modes")
        if args.energy_total < 0.0:
            raise SystemExit("--energy-total must be non-negative")
        mode_indices = [args.mode_index, secondary]
        vec_primary, _ = all_modes[args.mode_index]
        vec_secondary, _ = all_modes[secondary]
        ratios = (
            np.linspace(args.mix_min, args.mix_max, args.mix_samples)
            if args.mix_samples > 1
            else np.array([args.mix_min])
        )
        for alpha in ratios:
            alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
            weight_primary = np.sqrt(alpha_clamped)
            weight_secondary = np.sqrt(max(0.0, 1.0 - alpha_clamped))
            raw_velocity = (
                weight_primary * vec_primary + weight_secondary * vec_secondary
            )
            raw_energy = kinetic_energy(raw_velocity, masses)
            if args.energy_total == 0.0:
                scale = 0.0
            else:
                if raw_energy <= 0.0:
                    raise RuntimeError(
                        "Degenerate combination produced zero kinetic energy; adjust mix parameters"
                    )
                scale = float(np.sqrt(args.energy_total / raw_energy))
            vel_coeffs = [scale * weight_primary, scale * weight_secondary]
            _, kinetic, potential, collapse, pmin, env_max = run_simulation(
                mode_indices, vel_coeffs
            )
            k0 = float(kinetic[0])
            records.append(
                {
                    "mode_primary": args.mode_index,
                    "mode_secondary": secondary,
                    "mix_ratio": float(alpha_clamped),
                    "energy_target": args.energy_total,
                    "kinetic_initial": k0,
                    "collapse_time": collapse,
                    "potential_min": pmin,
                    "unstable_envelope_max": env_max,
                    "unstable_threshold": args.unstable_threshold,
                }
            )
            if collapse is None:
                status = f"ratio={alpha:.3f}, scale={scale:.4f}, collapse=none"
            else:
                status = f"ratio={alpha:.3f}, scale={scale:.4f}, collapse={collapse:.3f}"
            status += f", max|q_unstable|={env_max:.3f}"
            print(status)

    csv_name = args.csv_name or (
        f"{args.config}_mode{args.mode_index}"
        + (f"_mix{secondary}" if secondary is not None else "")
        + "_collapse.csv"
    )
    csv_path = args.output_dir / csv_name
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "mode_primary",
                "mode_secondary",
                "mix_ratio",
                "energy_target",
                "kinetic_initial",
                "collapse_time",
                "potential_min",
                "unstable_envelope_max",
                "unstable_threshold",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    if args.plot is not None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f"Matplotlib import failed: {exc}")

        xs = [row["kinetic_initial"] for row in records]
        ys = [
            row["collapse_time"] if row["collapse_time"] is not None else np.nan
            for row in records
        ]
        plt.figure(figsize=(4.8, 3.4))
        if secondary is None:
            x_vals = xs
            plt.plot(x_vals, ys, marker="o")
            plt.xlabel("Initial kinetic energy (actual)")
        else:
            x_vals = [row["mix_ratio"] for row in records]
            plt.plot(x_vals, ys, marker="o")
            plt.xlabel(f"Energy ratio (0→mode {secondary}, 1→mode {args.mode_index})")
        plt.axhline(0.0, color="0.7", linewidth=0.8)
        plt.ylabel(
            f"Collapse time (first |q_unstable| >= {args.unstable_threshold:g})"
        )
        title = f"Collapse scan: {args.config}, mode {args.mode_index}"
        if secondary is not None:
            title += f" + {secondary}"
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
