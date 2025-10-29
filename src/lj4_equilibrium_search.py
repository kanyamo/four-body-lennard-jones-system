#!/usr/bin/env python3
"""Multi-start Newton search for 4-body Lennard-Jones equilibria.

The translational and rotational gauge freedom is fixed by pinning particle 0 at
the origin, particle 1 on the x-axis, and particle 2 inside the xy-plane.  The
remaining coordinates (x1, x2, y2, x3, y3, z3) are treated as independent
variables.  For each random initial guess the script applies a damped Newton
iteration to drive the residual forces on those free degrees of freedom to
zero.  Unique equilibria are deduplicated by their sorted inter-particle
distances and summarised together with their energies and Hessian spectra.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np


LJ_COEFF = 4.0

KNOWN_CONFIGS: dict[tuple[int, ...], str] = {
    (112246, 112246, 112246, 112246, 112246, 112246): "tetrahedron (T_d)",
    (112023, 112023, 112023, 112023, 112480, 193765): "rhombus (θ≈60.27°, C_s)",
    (111262, 111262, 111262, 111262, 157348, 157348): "square (D_{4h})",
    (111593, 111593, 111593, 193285, 193285, 193285): "triangle + center (C_{3v})",
    (
        111898,
        112042,
        112042,
        112302,
        216271,
        216271,
    ): "isosceles triangle + interior (C_s)",
    (
        111954,
        112094,
        112094,
        224048,
        224048,
        336142,
    ): "linear chain (D_{\infty h})",
}


def lj_potential(r: float) -> float:
    inv_r6 = (1.0 / r) ** 6
    inv_r12 = inv_r6 * inv_r6
    return LJ_COEFF * (inv_r12 - inv_r6)


def pair_force(r_vec: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(r_vec)
    if r < 1e-10:
        raise ValueError("Particles overlap; force undefined.")
    inv_r = 1.0 / r
    v_prime = 24.0 * (r**6 - 2.0) / (r**13)
    return -v_prime * r_vec * inv_r


def total_potential(positions: np.ndarray) -> float:
    energy = 0.0
    n = positions.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[j] - positions[i])
            energy += lj_potential(r)
    return energy


def all_forces(positions: np.ndarray) -> np.ndarray:
    forces = np.zeros_like(positions)
    n = positions.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            fij = pair_force(rij)
            forces[i] += fij
            forces[j] -= fij
    return forces


def params_to_positions(params: np.ndarray) -> np.ndarray:
    x1, x2, y2, x3, y3, z3 = params
    positions = np.array(
        [
            (0.0, 0.0, 0.0),
            (x1, 0.0, 0.0),
            (x2, y2, 0.0),
            (x3, y3, z3),
        ],
        dtype=float,
    )
    return positions


def residual(params: np.ndarray) -> np.ndarray:
    pos = params_to_positions(params)
    try:
        forces = all_forces(pos)
    except ValueError:
        return np.full(6, 1e6)
    # Free variables correspond to (x1, x2, y2, x3, y3, z3)
    return np.array(
        [
            forces[1, 0],
            forces[2, 0],
            forces[2, 1],
            forces[3, 0],
            forces[3, 1],
            forces[3, 2],
        ]
    )


def numerical_jacobian(params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    base = residual(params)
    jac = np.zeros((base.size, params.size))
    for i in range(params.size):
        step = eps * max(1.0, abs(params[i]))
        if step == 0.0:
            step = eps
        plus = params.copy()
        plus[i] += step
        minus = params.copy()
        minus[i] -= step
        jac[:, i] = (residual(plus) - residual(minus)) / (2.0 * step)
    return jac


def damped_newton(
    start: np.ndarray,
    max_iter: int = 40,
    tol: float = 1e-10,
    jac_eps: float = 1e-6,
) -> tuple[np.ndarray | None, dict[str, float]]:
    x = start.copy()
    info: dict[str, float] = {}
    last_norm = None
    for iteration in range(1, max_iter + 1):
        r = residual(x)
        r_norm = float(np.linalg.norm(r, ord=2))
        if r_norm < tol:
            info.update({"iterations": iteration, "residual": r_norm})
            return x, info
        J = numerical_jacobian(x, eps=jac_eps)
        try:
            delta = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        step = 1.0
        success = False
        for _ in range(8):
            candidate = x + step * delta
            if candidate[0] <= 0.0:
                step *= 0.5
                continue
            r_candidate = residual(candidate)
            cand_norm = float(np.linalg.norm(r_candidate, ord=2))
            if cand_norm < r_norm:
                x = candidate
                r_norm = cand_norm
                success = True
                break
            step *= 0.5
        if not success:
            x = x + delta
        if last_norm is not None and abs(last_norm - r_norm) < 1e-14:
            break
        last_norm = r_norm
    info.update(
        {"iterations": iteration, "residual": float(np.linalg.norm(residual(x)))}
    )
    return None, info


def canonicalise(params: np.ndarray) -> np.ndarray:
    canon = params.copy()
    if canon[2] < 0.0:  # enforce y2 >= 0 by flipping across the xz-plane
        canon[2] *= -1
        canon[4] *= -1
        canon[5] *= -1
    return canon


def pair_distances(positions: np.ndarray) -> list[float]:
    dists: list[float] = []
    n = positions.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(positions[j] - positions[i]))
    return dists


def pair_distance_signature(
    positions: np.ndarray, tol: float = 1e-6
) -> tuple[int, ...]:
    dists = sorted(pair_distances(positions))
    return tuple(int(round(d / tol)) for d in dists)


def build_hessian(positions: np.ndarray) -> np.ndarray:
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


def pair_hessian(r_vec: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(r_vec))
    if r < 1e-10:
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


@dataclass
class Equilibrium:
    params: np.ndarray
    positions: np.ndarray
    energy: float
    force_norm: float
    signature: tuple[int, ...]
    eigenvalues: np.ndarray


def analyse_solution(params: np.ndarray, max_distance: float) -> Equilibrium | None:
    canon = canonicalise(params)
    pos = params_to_positions(canon)
    forces = all_forces(pos)
    energy = total_potential(pos)
    dists = pair_distances(pos)
    if max(dists) > max_distance:
        return None
    signature = pair_distance_signature(pos, tol=1e-5)
    H = build_hessian(pos)
    eigvals = np.linalg.eigvalsh(H)
    return Equilibrium(
        params=canon,
        positions=pos,
        energy=energy,
        force_norm=float(np.linalg.norm(forces)),
        signature=signature,
        eigenvalues=eigvals,
    )


def multi_start(samples: int, seed: int, max_distance: float) -> list[Equilibrium]:
    rng = random.Random(seed)
    results: dict[tuple[int, ...], Equilibrium] = {}

    guesses: list[np.ndarray] = []

    # Include hand-crafted seeds near known configurations
    guesses.append(
        np.array(
            [
                1.1224620483,
                0.5612310242,
                0.9720806497,
                0.5612310242,
                0.3240268832,
                0.9158734010,
            ]
        )
    )  # tetrahedron
    guesses.append(
        np.array([1.1126198392, 0.0, 1.1126198392, 1.1126198392, 1.1126198392, 0.0])
    )  # square
    guesses.append(
        np.array(
            [1.1202309526, -0.5555370480, 0.9727774540, 0.5646939046, 0.9727774540, 0.0]
        )
    )  # rhombus
    guesses.append(
        np.array(
            [
                1.1230225004,
                0.5615112500,
                -2.0885491154,
                0.5615112500,
                -0.9695654105,
                0.0,
            ]
        )
    )  # isosceles + interior
    guesses.append(
        np.array(
            [1.9328543867, 0.9664271933, 1.6739010010, 0.9664271933, 0.5579670003, 0.0]
        )
    )  # triangle+center
    guesses.append(
        np.array(
            [
                1.1209389078,
                2.2404785421,
                1e-3,
                3.3614174499,
                0.0,
                1e-3,
            ]
        )
    )  # near-linear chain (tiny transverse offset avoids gauge singularity)

    for _ in range(samples):
        x1 = rng.uniform(0.6, 2.0)
        x2 = rng.uniform(-1.2, 1.2)
        y2 = rng.uniform(0.2, 1.8)
        x3 = rng.uniform(-1.5, 1.5)
        y3 = rng.uniform(-1.5, 1.5)
        z3 = rng.uniform(-1.8, 1.8)
        guesses.append(np.array([x1, x2, y2, x3, y3, z3]))

    for guess in tqdm(guesses):
        sol, info = damped_newton(guess)
        if sol is None:
            continue
        eq = analyse_solution(sol, max_distance=max_distance)
        if eq is None or eq.force_norm > 1e-6:
            continue
        if eq.signature not in results:
            results[eq.signature] = eq
        else:
            if eq.energy < results[eq.signature].energy:
                results[eq.signature] = eq

    return list(results.values())


def format_positions(positions: np.ndarray) -> str:
    rows = []
    for idx, (x, y, z) in enumerate(positions):
        rows.append(f"P{idx}: ({x:+.9f}, {y:+.9f}, {z:+.9f})")
    return "\n".join(rows)


def classify_eigenvalues(
    eigvals: np.ndarray, tol: float = 1e-6
) -> tuple[int, int, int]:
    negative = int(np.sum(eigvals < -tol))
    zero = int(np.sum(np.abs(eigvals) <= tol))
    positive = eigvals.size - negative - zero
    return negative, zero, positive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=int, default=200, help="random initial guesses"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=5.0,
        help="discard solutions whose largest pair distance exceeds this",
    )
    args = parser.parse_args()

    equilibria = multi_start(args.samples, args.seed, args.max_distance)
    if not equilibria:
        print("No equilibria located")
        return

    equilibria.sort(key=lambda eq: eq.energy)
    print(
        f"Found {len(equilibria)} distinct equilibria (out of {args.samples} random seeds)"
    )
    for idx, eq in enumerate(equilibria, start=1):
        neg, zero, pos = classify_eigenvalues(eq.eigenvalues)
        print("-" * 72)
        print(f"Solution {idx}")
        print(f"  Energy        : {eq.energy:.12f}")
        print(f"  Force norm    : {eq.force_norm:.3e}")
        print(f"  Signature     : {eq.signature}")
        name = KNOWN_CONFIGS.get(eq.signature)
        if name:
            print(f"  Identified as : {name}")
        print(f"  Index / zero / pos : {neg} / {zero} / {pos}")
        print("  Positions:")
        print("    " + format_positions(eq.positions).replace("\n", "\n    "))


if __name__ == "__main__":
    main()
