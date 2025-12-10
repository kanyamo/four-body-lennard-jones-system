#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal-mode analysis for selected 4-body Lennard-Jones equilibria.

The original version of this script handled the ``C3v`` configuration consisting
of one central particle and three particles at the vertices of an equilateral
triangle.  In addition to that case we now support the planar ``D2h`` rhombus
equilibria that arise from the simultaneous force-balance conditions derived in
``reports/rhombus_equilibrium.md``.  The script reconstructs the Hessian of the
LJ potential for the requested configuration (optionally applying a uniform
scale to the edge length) and solves for the eigenvalues/eigenvectors of both
the raw Hessian and the mass-weighted "dynamical" matrix.  It also reports the
six zero modes (three translations and three rotations) explicitly so they can
be checked against symmetry arguments.

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


def lj_force_derivative(r: float) -> float:
    """Return V'(r) for the reduced (6,12) Lennard-Jones potential."""

    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    return LJ_COEFF * (6.0 * inv_r6 * inv_r - 12.0 * inv_r12 * inv_r)


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
        description="Normal-mode analysis for selected 4-body Lennard-Jones equilibria",
    )
    ap.add_argument(
        "--config",
        choices=["triangle_center", "rhombus", "square"],
        default="triangle_center",
        help="equilibrium family to analyse (triangle_center, rhombus, square)",
    )
    ap.add_argument(
        "--side-scale",
        type=float,
        default=1.0,
        help="uniform scaling relative to the equilibrium edge length",
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


# --- Rhombus helpers -----------------------------------------------------


def rhombus_equilibrium_angles(tol: float = 1e-10) -> list[float]:
    """Return all interior angles θ in (0, π) satisfying the rhombus equilibrium.

    The polynomial constraint F(c) = (2c^2 - 1) P(c) = 0 is solved for
    c = cos(θ/2).  Only real roots with 0 < c < 1 are kept.
    """

    coeffs_p = [
        32768.0,
        0.0,
        -131072.0,
        0.0,
        213248.0,
        0.0,
        -180992.0,
        0.0,
        84224.0,
        0.0,
        -19712.0,
        0.0,
        2817.0,
        0.0,
        -1281.0,
        0.0,
        257.0,
    ]

    roots = list(np.roots(coeffs_p))
    roots.extend([np.sqrt(0.5), -np.sqrt(0.5)])

    angles: list[float] = []
    for root in roots:
        if abs(root.imag) > tol:
            continue
        c = float(root.real)
        if not (0.0 < c < 1.0):
            continue
        theta = 2.0 * float(np.arccos(c))
        if not (0.0 < theta < np.pi):
            continue
        # Deduplicate within tolerance
        if all(abs(theta - existing) > 1e-8 for existing in angles):
            angles.append(theta)

    angles.sort()
    return angles


def rhombus_edge_length(theta: float) -> float:
    """Return equilibrium edge length a(θ) from the analytic formula."""

    c = float(np.cos(theta / 2.0))
    numerator = 1.0 + (2.0**14) * (c**14)
    denominator = (2.0**5) * (c**6) * (1.0 + (2.0**8) * (c**8))
    if denominator <= 0.0:
        raise ValueError("Invalid angle: cannot construct rhombus edge length")
    return (numerator / denominator) ** (1.0 / 6.0)


def rhombus_vertices(a: float, theta: float) -> np.ndarray:
    """Return a planar rhombus with side length a and interior angle θ.

    The vertices are placed so that the shorter対角線 (2 a sin θ/2) lies on the x 軸
    and the長い対角線 (2 a cos θ/2) lies on the y 軸, matching the orientation used
    in reports/_rhombus_planar(). The centroid is at the origin.
    """

    short_diag = 2.0 * a * np.sin(theta / 2.0)
    long_diag = 2.0 * a * np.cos(theta / 2.0)
    verts = np.array(
        [
            (-0.5 * short_diag, 0.0, 0.0),
            (0.0, -0.5 * long_diag, 0.0),
            (0.0, 0.5 * long_diag, 0.0),
            (0.5 * short_diag, 0.0, 0.0),
        ],
        dtype=float,
    )
    return verts


def validate_rhombus(theta: float, a: float, tol: float = 1e-8) -> None:
    """Ensure both rhombus force-balance conditions are satisfied."""

    c = float(np.cos(theta / 2.0))
    s = float(np.sin(theta / 2.0))
    eq1 = 2.0 * c * lj_force_derivative(a) + lj_force_derivative(2.0 * a * c)
    eq2 = 2.0 * s * lj_force_derivative(a) + lj_force_derivative(2.0 * a * s)
    if abs(eq1) > tol or abs(eq2) > tol:
        raise ValueError(
            "Angle {:.6f} rad ({:.6f}°) is not an LJ rhombus equilibrium".format(
                theta, np.degrees(theta)
            )
        )


def format_angle(theta: float, digits: int = 10) -> str:
    return f"{np.degrees(theta):.{digits}f}"


def main() -> None:
    args = parse_args()

    if args.config == "triangle_center":
        r_star = equilibrium_radius()
        radius = args.side_scale * r_star
        positions = np.vstack([np.zeros((1, 3)), triangle_vertices(radius)])
        masses = np.array([args.center_mass] + [args.outer_mass] * 3, dtype=float)
        config_label = "triangle + center (C3v)"
        info_lines = [
            f"r* = {r_star:.8f}",
            f"radius = {radius:.8f} (scale={args.side_scale})",
        ]
        save_metadata = {
            "radius": radius,
            "equilibrium_radius": r_star,
        }
    else:
        available = rhombus_equilibrium_angles()
        if args.config == "rhombus":
            theta = min(available)
            config_label = "planar rhombus (D2h)"
        else:
            target = np.pi / 2.0
            theta = min(available, key=lambda ang: abs(ang - target))
            config_label = "square (D4h)"

        a_eq = rhombus_edge_length(theta)
        validate_rhombus(theta, a_eq, tol=1e-8)
        a = args.side_scale * a_eq
        positions = rhombus_vertices(a, theta)
        masses = np.full(4, args.outer_mass, dtype=float)
        short_diag = 2.0 * a * np.cos(theta / 2.0)
        long_diag = 2.0 * a * np.sin(theta / 2.0)
        info_lines = [
            f"θ = {np.degrees(theta):.9f}°",
            f"a_eq = {a_eq:.8f}",
            f"a = {a:.8f} (scale={args.side_scale})",
            f"short diag = {short_diag:.8f}",
            f"long diag = {long_diag:.8f}",
        ]
        save_metadata = {
            "theta_deg": float(np.degrees(theta)),
            "a_equilibrium": a_eq,
            "a": a,
            "short_diagonal": short_diag,
            "long_diagonal": long_diag,
        }

    H = build_hessian(positions)
    H_mw = mass_weighted_hessian(H, masses)

    lam_h, vec_h = np.linalg.eigh(H)
    lam_dyn, vec_dyn_mass = np.linalg.eigh(H_mw)
    vec_dyn = flatten_modes(vec_dyn_mass, masses)

    translations, rotations = symmetry_basis(positions)
    modes = classify_modes(lam_dyn, vec_dyn, args.zero_tol, translations, rotations)

    print("LJ 4-body mode analysis")
    print(f"  configuration = {config_label}")
    for line in info_lines:
        print(f"  {line}")
    if args.config == "triangle_center":
        print(
            "  masses = [center={:.4f}, vertices={:.4f}]".format(
                args.center_mass, args.outer_mass
            )
        )
    else:
        print(f"  masses = vertices={args.outer_mass:.4f}")
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
            "configuration": config_label,
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
        data.update(save_metadata)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
