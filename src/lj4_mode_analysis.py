#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal-mode analysis for the 4-body (triangle + center) Lennard-Jones system.

The script reconstructs the Hessian of the LJ potential at the symmetric
equilateral configuration, optionally rescales the triangle radius, and solves
for the eigenvalues/eigenvectors of both the raw Hessian and the mass-weighted
"dynamical" matrix.  It also reports the six zero modes (three translations and
three rotations) explicitly so they can be checked against symmetry arguments.

Examples
--------
python lj4_mode_analysis.py --center-mass 1.0 --outer-mass 1.0 --digits 6
python lj4_mode_analysis.py --side-scale 1.05 --print-vectors unstable
python lj4_mode_analysis.py --save-json results/lj4_modes.json --digits 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np


LJ_COEFF = 4.0  # ε = σ = 1 in reduced units


def equilibrium_radius() -> float:
    """Return the equilibrium center–vertex distance r* for the 4-body setup."""

    num = 2.0 * (1.0 + 1.0 / (3.0**6))
    den = 1.0 + 1.0 / (3.0**3)
    return (num / den) ** (1.0 / 6.0)


def triangle_vertices(radius: float) -> np.ndarray:
    """Vertices of an equilateral triangle (centroid at origin) in the xy-plane."""

    angles = np.deg2rad([0.0, 120.0, 240.0])
    verts = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)],
        axis=1,
    )
    verts -= verts.mean(axis=0, keepdims=True)
    return verts


def pair_hessian(r_vec: np.ndarray) -> np.ndarray:
    """LJ Hessian block contribution for a given displacement vector."""

    r = float(np.linalg.norm(r_vec))
    if r < 1e-12:
        raise ValueError("Particles overlap; Hessian undefined.")
    e = r_vec / r
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    v1 = LJ_COEFF * (-12.0 * inv_r12 * inv_r + 6.0 * inv_r6 * inv_r)
    v2 = LJ_COEFF * (156.0 * inv_r12 * inv_r2 - 42.0 * inv_r6 * inv_r2)
    outer = np.outer(e, e)
    return (v2 - v1 * inv_r) * outer + (v1 * inv_r) * np.eye(3)


def build_hessian(positions: np.ndarray) -> np.ndarray:
    """Construct the 12×12 Hessian matrix for the supplied configuration."""

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


def mass_weighted_hessian(H: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return M^{-1/2} H M^{-1/2} for positive masses."""

    if np.any(masses <= 0):
        raise ValueError("Masses must be positive for mode analysis.")
    weights = np.repeat(np.sqrt(masses), 3)
    Hmw = H / weights[:, None]
    Hmw = Hmw / weights[None, :]
    return Hmw


@dataclass
class Mode:
    index: int
    omega2: float
    omega: complex
    vector: np.ndarray
    classification: str
    overlap: dict[str, float]


def classify_modes(
    omega2: np.ndarray,
    vectors: np.ndarray,
    tol: float,
    translations: np.ndarray,
    rotations: np.ndarray,
) -> list[Mode]:
    """Tag each mode as stable/unstable/zero and record overlaps with symmetries."""

    modes: list[Mode] = []
    trans = translations.reshape(3, -1)
    rots = rotations.reshape(3, -1)
    for idx, lam in enumerate(omega2):
        if lam > tol:
            omega = np.sqrt(lam)
            cls = "stable"
        elif lam < -tol:
            omega = 1j * np.sqrt(-lam)
            cls = "unstable"
        else:
            omega = 0.0
            cls = "zero"
        vec = vectors[:, idx]
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:
            vec = vec / vec_norm
        # overlaps with translations and rotations
        overlaps: dict[str, float] = {}
        overlaps["T"] = float(np.max(np.abs(trans @ vec)))
        overlaps["R"] = float(np.max(np.abs(rots @ vec)))
        modes.append(
            Mode(
                index=idx,
                omega2=float(lam),
                omega=omega,
                vector=vec,
                classification=cls,
                overlap=overlaps,
            )
        )
    return modes


def symmetry_basis(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal bases for translations and rotations."""

    n = positions.shape[0]
    trans = []
    for axis in range(3):
        vec = np.zeros((n, 3))
        vec[:, axis] = 1.0
        vec = vec.reshape(-1)
        vec /= np.linalg.norm(vec)
        trans.append(vec)

    rots = []
    for axis in range(3):
        vec = np.zeros((n, 3))
        for i, (x, y, z) in enumerate(positions):
            if axis == 0:  # x
                vec[i] = (0.0, -z, y)
            elif axis == 1:  # y
                vec[i] = (z, 0.0, -x)
            else:  # z
                vec[i] = (-y, x, 0.0)
        flat = vec.reshape(-1)
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat /= norm
        rots.append(flat)

    return np.array(trans), np.array(rots)


def flatten_modes(vecs: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Convert mass-weighted eigenvectors to coordinate-space vectors."""

    weights = np.repeat(np.sqrt(masses), 3)
    coords = vecs / weights[:, None]
    # re-normalize each column for readability
    norms = np.linalg.norm(coords, axis=0)
    norms[norms == 0.0] = 1.0
    return coords / norms


def format_complex(value: complex, digits: int) -> str:
    if isinstance(value, complex) and abs(value.imag) > 1e-12:
        return f"{value.real:.{digits}f}+{value.imag:.{digits}f}j"
    return f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Normal-mode analysis for the LJ triangle+center setup",
    )
    ap.add_argument(
        "--side-scale",
        type=float,
        default=1.0,
        help="radius scaling w.r.t. equilibrium r* (1.0 uses the stationary point)",
    )
    ap.add_argument(
        "--outer-mass",
        type=float,
        default=1.0,
        help="mass of each vertex particle",
    )
    ap.add_argument(
        "--center-mass",
        type=float,
        default=1.0,
        help="mass of the central particle",
    )
    ap.add_argument(
        "--digits",
        type=int,
        default=6,
        help="number of digits to print for eigenvalues",
    )
    ap.add_argument(
        "--zero-tol",
        type=float,
        default=1e-8,
        help="threshold on |ω^2| treated as zero",
    )
    ap.add_argument(
        "--print-vectors",
        choices=["none", "zero", "unstable", "stable", "all"],
        default="none",
        help="print eigenvectors belonging to the selected class",
    )
    ap.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="optional path to dump eigen data as JSON",
    )
    return ap.parse_args()


def select_indices(modes: Iterable[Mode], target: str) -> list[int]:
    if target == "all":
        return [m.index for m in modes]
    return [m.index for m in modes if m.classification == target]


def main() -> None:
    args = parse_args()

    r_star = equilibrium_radius()
    radius = args.side_scale * r_star

    # positions: central particle first, then triangle vertices
    positions = np.vstack([np.zeros((1, 3)), triangle_vertices(radius)])

    H = build_hessian(positions)
    masses = np.array([args.center_mass] + [args.outer_mass] * 3, dtype=float)
    H_mw = mass_weighted_hessian(H, masses)

    lam_h, vec_h = np.linalg.eigh(H)
    lam_dyn, vec_dyn_mass = np.linalg.eigh(H_mw)
    vec_dyn = flatten_modes(vec_dyn_mass, masses)

    translations, rotations = symmetry_basis(positions)
    modes = classify_modes(lam_dyn, vec_dyn, args.zero_tol, translations, rotations)

    print("LJ 4-body (triangle + center) mode analysis")
    print(f"  r* = {r_star:.8f}, radius = {radius:.8f} (scale={args.side_scale})")
    print(
        "  masses = [center={:.4f}, vertices={:.4f}]".format(
            args.center_mass, args.outer_mass
        )
    )
    print(f"  zero tolerance = {args.zero_tol:.1e}")
    print()

    print("Hessian eigenvalues λ (potential curvature):")
    for val in lam_h:
        print(f"  {val:.{args.digits}f}")
    print()

    print("Mass-weighted eigenvalues ω² and classifications:")
    for mode in modes:
        omega_str = format_complex(mode.omega, args.digits)
        print(
            f"  idx={mode.index:2d}  ω²={mode.omega2:.{args.digits}f}"
            f"  ω={omega_str:<12}  class={mode.classification:<8}"
            f"  |T|={mode.overlap['T']:.3f}  |R|={mode.overlap['R']:.3f}"
        )

    if args.print_vectors != "none":
        targets = select_indices(modes, args.print_vectors)
        if targets:
            print()
            print(f"Eigenvectors ({args.print_vectors}):")
        for idx in targets:
            vec = vec_dyn[:, idx]
            print(f"mode {idx} (ω²={lam_dyn[idx]:.{args.digits}f})")
            reshaped = vec.reshape(-1, 3)
            for pid, comps in enumerate(reshaped):
                label = "P0" if pid == 0 else f"P{pid}"
                print(
                    f"  {label}: ({comps[0]:+.6f}, {comps[1]:+.6f}, {comps[2]:+.6f})"
                )
            print()

    if args.save_json:
        data = {
            "radius": radius,
            "equilibrium_radius": r_star,
            "side_scale": args.side_scale,
            "masses": {
                "center": args.center_mass,
                "outer": args.outer_mass,
            },
            "hessian_eigenvalues": lam_h.tolist(),
            "modes": [
                {
                    "index": m.index,
                    "omega2": m.omega2,
                    "omega": [float(np.real(m.omega)), float(np.imag(m.omega))],
                    "classification": m.classification,
                    "overlap_T": m.overlap["T"],
                    "overlap_R": m.overlap["R"],
                    "vector": vec_dyn[:, m.index].reshape(-1, 3).tolist(),
                }
                for m in modes
            ],
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
