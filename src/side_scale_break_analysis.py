#!/usr/bin/env python3
"""Sweep side_scale and estimate how long the triangle+center structure stays intact."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

import lj4_3d


def initialize_state(
    side_scale: float, vb: float, z0: float, center_mass: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return initial positions, velocities, and masses for the configuration."""
    r_star = (2.0 * (1.0 + 1.0 / (3**6)) / (1.0 + 1.0 / (3**3))) ** (1.0 / 6.0)
    side = math.sqrt(3.0) * r_star * side_scale

    verts = lj4_3d.triangle_vertices_3d(side)
    center = np.array([[0.0, 0.0, z0]], dtype=float)
    X0 = np.vstack([verts, center])
    masses = np.concatenate([np.ones(3), np.array([center_mass], dtype=float)])

    V0 = np.zeros_like(X0)
    for i in range(3):
        r2 = X0[i, :2]
        norm = np.linalg.norm(r2)
        u = r2 / norm if norm > 1e-12 else np.array([1.0, 0.0])
        V0[i, :2] = vb * u  # alternating signs are redundant for symmetry, keep +vb
    v_com = np.sum(V0 * masses[:, None], axis=0) / masses.sum()
    V0 -= v_com
    return X0, V0, masses


def pairwise_distances(
    X: np.ndarray, pairs: Sequence[tuple[int, int]] | None = None
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Compute the distances for all requested particle index pairs."""
    if pairs is None:
        n = X.shape[0]
        pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    dists = np.array([np.linalg.norm(X[j] - X[i]) for i, j in pairs], dtype=float)
    return dists, list(pairs)


def measure_breaks(
    side_scale: float,
    vb: float,
    z0: float,
    dt: float,
    T: float,
    thresholds: Sequence[float],
    check_every: int,
    center_mass: float,
) -> list[dict[str, float | str | None]]:
    """Integrate once and record break times for all provided *thresholds*."""
    if not thresholds:
        raise ValueError("measure_breaks() requires at least one threshold")

    thresholds_arr = np.array(sorted(thresholds), dtype=float)
    X, V, masses = initialize_state(side_scale, vb, z0, center_mass)
    d0, pairs = pairwise_distances(X)

    nsteps = int(T / dt)
    max_rel_change = 0.0
    break_steps: list[int | None] = [None] * len(thresholds_arr)
    break_pair_idx: list[int | None] = [None] * len(thresholds_arr)

    for step in range(nsteps):
        X, V = lj4_3d.step_yoshida4(X, V, dt, masses)
        if (step + 1) % check_every != 0:
            continue
        d, _ = pairwise_distances(X, pairs)
        rel_change = np.abs(d - d0) / d0
        cur_max = float(np.max(rel_change))
        if cur_max > max_rel_change:
            max_rel_change = cur_max
        for idx, thr in enumerate(thresholds_arr):
            if break_steps[idx] is not None:
                continue
            if np.any(rel_change > thr):
                break_steps[idx] = step + 1
                break_pair_idx[idx] = int(np.argmax(rel_change))

    results: list[dict[str, float | str | None]] = []
    for idx, thr in enumerate(thresholds_arr):
        step_hit = break_steps[idx]
        if step_hit is None:
            break_time = None
            pair_str: str | None = None
        else:
            break_time = step_hit * dt
            pair = (
                pairs[break_pair_idx[idx]] if break_pair_idx[idx] is not None else None
            )
            pair_str = None if pair is None else f"{pair[0]}-{pair[1]}"
        results.append(
            {
                "side_scale": side_scale,
                "threshold": float(thr),
                "break_time": break_time,
                "max_rel_change": max_rel_change,
                "break_pair": pair_str,
            }
        )
    return results


def sweep_side_scales(
    side_scales: Iterable[float],
    vb: float,
    z0: float,
    dt: float,
    T: float,
    thresholds: Sequence[float],
    check_every: int,
    center_mass: float,
) -> list[dict[str, float | str | None]]:
    """Evaluate break times for each side_scale in *side_scales* across all thresholds."""
    results: list[dict[str, float | str | None]] = []
    for scale in tqdm(side_scales, desc="Sweeping side scales"):
        res = measure_breaks(
            scale, vb, z0, dt, T, thresholds, check_every, center_mass
        )
        results.extend(res)
    return results


def plot_results(
    results: Sequence[dict[str, float | str | None]],
    T: float,
    thresholds: Sequence[float],
    out_path: Path,
    center_mass: float,
) -> None:
    """Create a scatter/line plot of break times vs side_scale."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    grouped: dict[float, list[dict[str, float | str | None]]] = {
        float(thr): [] for thr in thresholds
    }
    for r in results:
        grouped[float(r["threshold"])].append(r)

    for thr in sorted(grouped):
        entries = grouped[thr]
        if not entries:
            continue
        entries.sort(key=lambda item: float(item["side_scale"]))
        scales = np.array([float(item["side_scale"]) for item in entries])
        times = np.array(
            [
                T if item["break_time"] is None else float(item["break_time"])
                for item in entries
            ]
        )
        broke_mask = np.array([item["break_time"] is not None for item in entries])
        plt.plot(scales, times, label=f"threshold={thr:.3f}", linewidth=1.5)
    plt.axhline(T, color="#bbbbbb", linestyle="--", linewidth=1.0, label=f"T={T}")
    plt.xlabel("side_scale")
    plt.ylabel("time to break")
    plt.title(f"Time until structure break vs side_scale (center_mass={center_mass:.2f})")
    plt.xlim(0.8, 1.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_csv(
    results: Sequence[dict[str, float | str | None]],
    out_path: Path,
    T: float,
    thresholds: Sequence[float],
    center_mass: float,
) -> None:
    """Persist the numeric summary to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "side_scale,threshold,break_time_or_T,max_rel_change,break_pair"
    thresholds_str = ",".join(
        f"{thr:.6f}" for thr in sorted({float(t) for t in thresholds})
    )
    lines = [
        f"# center_mass={center_mass:.6f}",
        f"# break_thresholds={thresholds_str}",
        header,
    ]
    for r in results:
        side_scale = float(r["side_scale"])  # typed for mypy
        threshold = float(r["threshold"])
        break_time = r["break_time"]
        bt_or_T = T if break_time is None else float(break_time)
        max_rel = float(r["max_rel_change"])
        pair = r["break_pair"] or ""
        lines.append(
            f"{side_scale:.8f},{threshold:.6f},{bt_or_T:.8f},{max_rel:.8f},{pair}"
        )
    out_path.write_text("\n".join(lines))


def parse_side_scales(args: argparse.Namespace) -> list[float]:
    if args.side_scales:
        return list(args.side_scales)
    if args.side_scale_min is not None and args.side_scale_max is not None:
        n = args.n_side_scales if args.n_side_scales is not None else 6
        return list(np.linspace(args.side_scale_min, args.side_scale_max, n))
    return list(np.linspace(0.95, 1.10, 7))


def _fmt_float_for_dir(value: float, digits: int = 6) -> str:
    s = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    if "." not in s:
        s += ".0"
    return s


def build_output_dir(
    root: Path,
    args: argparse.Namespace,
    side_scales: Sequence[float],
    thresholds: Sequence[float],
) -> Path:
    parts: list[str] = [f"center_mass={_fmt_float_for_dir(args.center_mass)}"]
    if args.side_scale_min is not None:
        parts.append(f"side_scale_min={_fmt_float_for_dir(args.side_scale_min)}")
    if args.side_scale_max is not None:
        parts.append(f"side_scale_max={_fmt_float_for_dir(args.side_scale_max)}")
    if args.n_side_scales is not None:
        parts.append(f"n_side_scales={args.n_side_scales}")
    else:
        parts.append(f"n_side_scales={len(side_scales)}")
    if args.side_scale_min is None and side_scales:
        parts.append(f"side_scale_min={_fmt_float_for_dir(min(side_scales))}")
    if args.side_scale_max is None and side_scales:
        parts.append(f"side_scale_max={_fmt_float_for_dir(max(side_scales))}")
    parts.append(f"vb={_fmt_float_for_dir(args.vb)}")
    parts.append(f"z0={_fmt_float_for_dir(args.z0)}")
    parts.append(f"dt={_fmt_float_for_dir(args.dt)}")
    parts.append(f"T={_fmt_float_for_dir(args.T)}")
    if thresholds:
        thr_str = "-".join(_fmt_float_for_dir(t) for t in thresholds)
        parts.append(f"thresholds={thr_str}")
    return root / "__".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep side_scale and record break times."
    )
    parser.add_argument(
        "--side-scales", type=float, nargs="*", help="explicit side_scale values to use"
    )
    parser.add_argument(
        "--side-scale-min",
        type=float,
        help="linspace start (used if explicit list omitted)",
    )
    parser.add_argument(
        "--side-scale-max",
        type=float,
        help="linspace end (used if explicit list omitted)",
    )
    parser.add_argument(
        "--n-side-scales",
        type=int,
        help="linspace count (used with --side-scale-min/max)",
    )
    parser.add_argument(
        "--center-mass",
        type=float,
        default=1.0,
        help="relative mass assigned to the central particle",
    )
    parser.add_argument(
        "--vb", type=float, default=0.20, help="in-plane breathing velocity magnitude"
    )
    parser.add_argument(
        "--z0",
        type=float,
        default=0.02,
        help="initial z-offset for the central particle",
    )
    parser.add_argument("--dt", type=float, default=0.002, help="time step size")
    parser.add_argument("--T", type=float, default=120.0, help="total integration time")
    parser.add_argument(
        "--break-threshold",
        type=float,
        default=0.30,
        help="default relative distance change threshold (used when --break-thresholds absent)",
    )
    parser.add_argument(
        "--break-thresholds",
        type=float,
        nargs="+",
        help="explicit list of thresholds; overrides --break-threshold",
    )
    parser.add_argument(
        "--check-every",
        type=int,
        default=1,
        help="evaluate the break condition every N integrator steps",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="where to store the time-to-break plot",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="where to store numeric results",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/side_scale_break"),
        help="base directory for automatic output organization",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip creating a plot (useful for headless runs)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="skip writing the CSV summary",
    )
    args = parser.parse_args()

    if args.check_every < 1:
        raise ValueError("--check-every must be >= 1")

    side_scales = parse_side_scales(args)
    thresholds = (
        args.break_thresholds
        if args.break_thresholds is not None
        else [args.break_threshold]
    )

    output_dir = build_output_dir(args.output_root, args, side_scales, thresholds)

    plot_out = args.plot_out or output_dir / "side_scale_break.png"
    csv_out = args.csv_out or output_dir / "side_scale_break.csv"

    results = sweep_side_scales(
        side_scales,
        args.vb,
        args.z0,
        args.dt,
        args.T,
        thresholds,
        args.check_every,
        args.center_mass,
    )

    if not args.no_plot:
        plot_results(results, args.T, thresholds, plot_out, args.center_mass)
    if not args.no_csv:
        save_csv(results, csv_out, args.T, thresholds, args.center_mass)

    print("side_scale threshold break_time_or_None max_rel_change break_pair")
    for r in results:
        print(
            f"{r['side_scale']:.5f}"
            f" {float(r['threshold']):.3f}"
            f" {r['break_time'] if r['break_time'] is not None else 'None'}"
            f" {r['max_rel_change']:.4f}"
            f" {r['break_pair'] or '-'}"
        )

    if not args.no_plot or not args.no_csv:
        print(f"outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()
