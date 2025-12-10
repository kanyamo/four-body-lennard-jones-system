#!/usr/bin/env python3
"""Animate the 3D dynamics of the 4-body spring rhombus system."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence as Seq
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from lj4_storage import load_simulation_bundle, save_simulation_bundle
from spring4_core import simulate_trajectory


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode-displacement",
        type=str,
        default="0.02",
        help="comma-separated displacement amplitudes for selected modes",
    )
    ap.add_argument(
        "--mode-velocity",
        type=str,
        default="0.05",
        help="comma-separated velocity amplitudes for selected modes",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="0",
        help="comma-separated stable-mode indices (0 = lowest positive eigenvalue)",
    )
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument(
        "--fps",
        type=int,
        default=30,
        help="frames per second for interactive playback",
    )
    ap.add_argument(
        "--thin", type=int, default=5, help="store every Nth integrator step"
    )
    ap.add_argument("--outfile", type=str, default="spring4_anim.mp4")
    ap.add_argument(
        "--trace-index",
        type=int,
        default=None,
        help="optional particle index to trace (defaults to config-specific choice)",
    )
    ap.add_argument(
        "--kick-energy",
        type=float,
        default=0.0,
        help="add velocity ±z (T,B:+v; R,L:-v) with this total kinetic energy",
    )
    ap.add_argument(
        "--save-bundle",
        type=str,
        default=None,
        help="保存用バンドルのディレクトリ (metadata.json + series.npz)",
    )
    ap.add_argument(
        "--load-bundle",
        type=str,
        default=None,
        help="既存バンドルを読み込んで再計算せずにアニメーション表示する",
    )
    return ap.parse_args()


def update_plot(ax, result, trace_index):
    fig = ax.get_figure()
    (pts,) = ax.plot([], [], [], "o", markersize=6)
    edge_lines = [ax.plot([], [], [], "-", linewidth=1.0)[0] for _ in result.spec.edges]
    (trace,) = ax.plot([], [], [], "-", linewidth=1.0)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    all_xyz = result.positions.reshape(-1, 3)
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    span = maxs - mins
    max_span = float(span.max())
    pad = 0.2 * (max_span + 1e-6)
    centers = (maxs + mins) * 0.5
    ax.set_xlim(centers[0] - 0.5 * max_span - pad, centers[0] + 0.5 * max_span + pad)
    ax.set_ylim(centers[1] - 0.5 * max_span - pad, centers[1] + 0.5 * max_span + pad)
    ax.set_zlim(centers[2] - 0.5 * max_span - pad, centers[2] + 0.5 * max_span + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    mode_list = ", ".join(str(idx) for idx in result.mode_indices)
    ax.set_title(f"{result.spec.label} — modes {mode_list}")

    trace_points = result.positions[:, trace_index, :]
    current_frame: list[int] = [0]

    def show_frame(frame_idx: int) -> None:
        frame_idx = max(0, min(frame_idx, len(result.positions) - 1))
        X = result.positions[frame_idx]
        pts.set_data(X[:, 0], X[:, 1])
        pts.set_3d_properties(X[:, 2])
        for line, (i, j) in zip(edge_lines, result.spec.edges):
            line.set_data([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])
            line.set_3d_properties([X[i, 2], X[j, 2]])
        history = trace_points[: frame_idx + 1]
        trace.set_data(history[:, 0], history[:, 1])
        trace.set_3d_properties(history[:, 2])
        time_text.set_text(f"t = {result.times[frame_idx]:.2f}")
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    frame_slider = Slider(
        ax=slider_ax,
        label="time",
        valmin=float(result.times[0]),
        valmax=float(result.times[-1]),
        valinit=float(result.times[0]),
        valstep=result.times,
    )

    def on_slider_change(val: float) -> None:
        new_frame = int(np.searchsorted(result.times, val, side="left"))
        if new_frame >= len(result.times) or not math.isclose(
            result.times[new_frame], val
        ):
            new_frame = max(0, min(len(result.times) - 1, new_frame - 1))
        if new_frame != current_frame[0]:
            current_frame[0] = new_frame
            show_frame(new_frame)

    frame_slider.on_changed(on_slider_change)

    def on_key_press(event) -> None:
        if event.key in {"right", "d"}:
            next_frame = min(current_frame[0] + 1, len(result.times) - 1)
            frame_slider.set_val(result.times[next_frame])
        elif event.key in {"left", "a"}:
            prev_frame = max(current_frame[0] - 1, 0)
            frame_slider.set_val(result.times[prev_frame])

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    show_frame(0)


def _ensure_float_list(value: float | Seq[float]) -> list[float]:
    if isinstance(value, Seq) and not isinstance(value, (str, bytes)):
        return [float(v) for v in value]
    return [float(value)]


def _build_kick_velocity(kick_energy: float) -> np.ndarray | None:
    if kick_energy <= 0.0:
        return None
    v_mag = math.sqrt(kick_energy / 2.0)
    kick_velocity = np.zeros((4, 3), dtype=float)
    # Particle order: R(0), T(1), L(2), B(3)
    kick_velocity[0, 2] = -v_mag
    kick_velocity[1, 2] = +v_mag
    kick_velocity[2, 2] = -v_mag
    kick_velocity[3, 2] = +v_mag
    return kick_velocity


def simulate(
    mode_displacement: float | Seq[float],
    mode_velocity: float | Seq[float],
    dt: float,
    total_time: float,
    save_stride: int,
    kick_energy: float,
    mode_indices: Seq[int] | None = None,
):
    disp = _ensure_float_list(mode_displacement)
    vel = _ensure_float_list(mode_velocity)
    result = simulate_trajectory(
        mode_indices=mode_indices,
        mode_displacements=disp,
        mode_velocities=vel,
        dt=dt,
        total_time=total_time,
        save_stride=save_stride,
        kick_velocity=_build_kick_velocity(kick_energy),
    )
    return result.spec, result.mode_eigenvalues, result.times, result.positions, result


def main() -> None:
    args = parse_args()
    loading_bundle = args.load_bundle is not None
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

    if loading_bundle:
        loaded = load_simulation_bundle(args.load_bundle)
        result = loaded.result
        run_parameters = loaded.metadata.get("run_parameters", {})
        mode_indices = tuple(result.mode_indices)
        mode_displacements = list(result.displacement_coeffs)
        mode_velocities = list(result.velocity_coeffs)
        print(f"バンドルを読み込みました: {args.load_bundle}")
    else:
        mode_indices = tuple(parse_int_list(args.modes, "--modes"))
        mode_displacements = parse_float_list(
            args.mode_displacement, "--mode-displacement"
        )
        mode_velocities = parse_float_list(args.mode_velocity, "--mode-velocity")
        run_parameters = {
            "dt": args.dt,
            "total_time": args.T,
            "save_stride": args.thin,
            "mode_indices": list(mode_indices),
            "mode_displacements": list(mode_displacements),
            "mode_velocities": list(mode_velocities),
            "kick_energy": args.kick_energy,
        }
        result = simulate_trajectory(
            mode_indices=mode_indices,
            mode_displacements=mode_displacements,
            mode_velocities=mode_velocities,
            dt=args.dt,
            total_time=args.T,
            save_stride=args.thin,
            kick_velocity=_build_kick_velocity(args.kick_energy),
        )
        if args.save_bundle is not None:
            out_dir = Path(args.save_bundle)
            save_simulation_bundle(out_dir, result, run_parameters)
            print(f"バンドルを書き出しました: {args.save_bundle}")

    trace_index = (
        args.trace_index if args.trace_index is not None else result.spec.trace_index
    )
    if trace_index < 0 or trace_index >= result.positions.shape[1]:
        raise ValueError("trace index out of bounds for this configuration")

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
    if args.kick_energy > 0.0:
        print(f"Kick energy: {args.kick_energy:.4f}")

    fig = plt.figure(figsize=(6.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    update_plot(ax, result, trace_index)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
