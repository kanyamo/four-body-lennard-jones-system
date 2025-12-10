#!/usr/bin/env python3
"""Scan collapse time versus initial mode displacement for 4-body LJ equilibria."""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from lj4_core import (
    available_configs,
    dihedral_angles,
    simulate_trajectory,
)
from lj4_storage import CACHE_DEFAULT_DIRNAME, simulate_with_cache


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=available_configs(), default="rhombus")
    ap.add_argument("--mode-index", type=int, default=0)
    ap.add_argument("--disp-min", type=float, default=0.0)
    ap.add_argument("--disp-max", type=float, default=0.2)
    ap.add_argument("--disp-samples", type=int, default=10)
    ap.add_argument("--modal-kick-energy", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--center-mass", type=float, default=1.0)
    ap.add_argument("--repulsive-exp", type=int, default=12)
    ap.add_argument("--attractive-exp", type=int, default=6)
    ap.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.5,
        help="threshold on |pi - dihedral| to declare collapse",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory to write plot/csv/json outputs (default: results/displacement_collapse/YYYYMMDD_HHMMSS)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(CACHE_DEFAULT_DIRNAME),
    )
    ap.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        default=True,
    )
    return ap.parse_args()


def collapse_time_from_dihedral(
    times: np.ndarray, positions: np.ndarray, threshold: float
) -> float | None:
    for t, pos in zip(times, positions):
        gaps = np.abs(math.pi - dihedral_angles(pos))
        if float(np.max(gaps)) > threshold:
            return float(t)
    return None


def linspace_samples(min_val: float, max_val: float, samples: int) -> Iterable[float]:
    if samples <= 1:
        yield float(min_val)
        return
    for v in np.linspace(min_val, max_val, samples):
        yield float(v)


def main() -> None:
    args = parse_args()
    if args.disp_max < args.disp_min:
        raise SystemExit("--disp-max must be >= --disp-min")
    if args.disp_samples < 1:
        raise SystemExit("--disp-samples must be >= 1")
    if args.thin < 1:
        raise SystemExit("--thin must be >= 1")
    if args.center_mass <= 0.0:
        raise SystemExit("--center-mass must be positive")
    if args.repulsive_exp <= args.attractive_exp or args.attractive_exp <= 0:
        raise SystemExit("repulsive-exp must be > attractive-exp > 0")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("results/displacement_collapse") / stamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    displacements: list[float] = []
    collapse_times: list[float | None] = []
    from_cache_flags: list[bool] = []

    for disp in linspace_samples(args.disp_min, args.disp_max, args.disp_samples):
        run_parameters: dict[str, Any] = {
            "mode_displacements": [disp],
            "mode_velocities": [0.0],
            "dt": args.dt,
            "total_time": args.T,
            "save_stride": args.thin,
            "center_mass": args.center_mass,
            "repulsive_exp": args.repulsive_exp,
            "attractive_exp": args.attractive_exp,
            "mode_indices": [args.mode_index],
            "modal_kick_energy": args.modal_kick_energy,
            "random_displacement": 0.0,
            "random_kick_energy": 0.0,
            "random_seed": None,
        }

        result, run_params_out, cache_dir, cache_key, from_cache = simulate_with_cache(
            args.config,
            run_parameters,
            cache_root=args.cache_dir,
            use_cache=args.use_cache,
            simulate_fn=lambda **kw: simulate_trajectory(args.config, **kw),
        )

        t_collapse = collapse_time_from_dihedral(
            np.asarray(result.times),
            np.asarray(result.positions),
            args.collapse_threshold,
        )
        displacements.append(disp)
        collapse_times.append(t_collapse)
        from_cache_flags.append(from_cache)

        status = "cache" if from_cache else "run"
        t_disp = "none" if t_collapse is None else f"{t_collapse:.3f}"
        print(f"[{status}] disp={disp:.6f} collapse_time={t_disp} key={cache_key}")

    # Save CSV
    csv_path = args.output_dir / "displacement_collapse.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode_displacement", "collapse_time", "from_cache"])
        for d, t, cflag in zip(displacements, collapse_times, from_cache_flags):
            writer.writerow([d, "" if t is None else t, int(cflag)])

    # Plot
    plot_path = args.output_dir / "displacement_collapse.png"
    plt.figure(figsize=(6, 4))
    times_numeric = [np.nan if t is None else t for t in collapse_times]
    plt.plot(displacements, times_numeric, marker="o")
    plt.xlabel("mode displacement")
    plt.ylabel("collapse time")
    plt.title(f"Collapse time vs displacement ({args.config})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=170)
    plt.close()
    print(f"CSV saved to {csv_path}")
    print(f"Plot saved to {plot_path}")

    # Save parameters for reproducibility
    params_out = {
        "config": args.config,
        "mode_index": args.mode_index,
        "disp_min": args.disp_min,
        "disp_max": args.disp_max,
        "disp_samples": args.disp_samples,
        "collapse_threshold": args.collapse_threshold,
        "dt": args.dt,
        "T": args.T,
        "thin": args.thin,
        "center_mass": args.center_mass,
        "repulsive_exp": args.repulsive_exp,
        "attractive_exp": args.attractive_exp,
        "modal_kick_energy": args.modal_kick_energy,
        "use_cache": args.use_cache,
        "cache_dir": str(args.cache_dir),
    }
    params_path = args.output_dir / "parameters.json"
    params_path.write_text(__import__("json").dumps(params_out, indent=2))
    print(f"Parameters saved to {params_path}")


if __name__ == "__main__":
    main()
