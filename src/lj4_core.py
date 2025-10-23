"""Shared utilities for the 4-body Lennard-Jones simulations.

This module factors out the geometry catalogue, force computations, and
integration helpers so that both the CLI integrator and the interactive
visualiser can rely on the same implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class EquilibriumSpec:
    """Geometry and plotting metadata for a 4-body equilibrium."""

    key: str
    label: str
    positions: np.ndarray
    edges: tuple[tuple[int, int], ...]
    trace_index: int


@dataclass(frozen=True)
class SimulationResult:
    spec: EquilibriumSpec
    omega2: float
    mode_shape: np.ndarray
    masses: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    kinetic: np.ndarray | None
    potential: np.ndarray | None
    total: np.ndarray | None
    energy_initial: float | None
    energy_final: float | None


def _regular_tetrahedron() -> np.ndarray:
    edge = 2.0 ** (1.0 / 6.0)
    base = np.array(
        [
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        ],
        dtype=float,
    )
    scale = edge / (2.0 * math.sqrt(2.0))
    return base * scale


def _square_planar() -> np.ndarray:
    side = 1.1126198392
    half = 0.5 * side
    return np.array(
        [
            (-half, -half, 0.0),
            (half, -half, 0.0),
            (half, half, 0.0),
            (-half, half, 0.0),
        ],
        dtype=float,
    )


def _rhombus_planar() -> np.ndarray:
    positions = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.1202309526, 0.0, 0.0),
            (-0.5555370480, 0.9727774540, 0.0),
            (0.5646939046, 0.9727774540, 0.0),
        ],
        dtype=float,
    )
    positions -= positions.mean(axis=0, keepdims=True)
    return positions


def _triangle_plus_center() -> np.ndarray:
    r_star = (2.0 * (1.0 + 1.0 / (3.0**6)) / (1.0 + 1.0 / (3.0**3))) ** (1.0 / 6.0)
    angles = np.deg2rad([0.0, 120.0, 240.0])
    ring = np.stack(
        [r_star * np.cos(angles), r_star * np.sin(angles), np.zeros_like(angles)],
        axis=1,
    )
    center = np.zeros((1, 3), dtype=float)
    coords = np.vstack([ring, center])
    coords -= coords.mean(axis=0, keepdims=True)
    return coords


def _isosceles_with_interior() -> np.ndarray:
    base = 1.1230225004
    apex_y = 2.0885491154
    interior_y = 0.9695654105
    coords = np.array(
        [
            (-0.5 * base, 0.0, 0.0),
            (0.5 * base, 0.0, 0.0),
            (0.0, apex_y, 0.0),
            (0.0, interior_y, 0.0),
        ],
        dtype=float,
    )
    coords -= coords.mean(axis=0, keepdims=True)
    return coords


def _build_equilibria() -> dict[str, EquilibriumSpec]:
    return {
        "tetrahedron": EquilibriumSpec(
            key="tetrahedron",
            label="tetrahedron (T_d)",
            positions=_regular_tetrahedron(),
            edges=tuple(combinations(range(4), 2)),
            trace_index=0,
        ),
        "rhombus": EquilibriumSpec(
            key="rhombus",
            label="rhombus (θ≈60.27°)",
            positions=_rhombus_planar(),
            edges=((0, 1), (1, 2), (2, 3), (3, 0)),
            trace_index=0,
        ),
        "square": EquilibriumSpec(
            key="square",
            label="square (D_4h)",
            positions=_square_planar(),
            edges=((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)),
            trace_index=0,
        ),
        "triangle_center": EquilibriumSpec(
            key="triangle_center",
            label="triangle + center (C_3v)",
            positions=_triangle_plus_center(),
            edges=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
            trace_index=3,
        ),
        "isosceles_interior": EquilibriumSpec(
            key="isosceles_interior",
            label="isosceles triangle + interior (C_s)",
            positions=_isosceles_with_interior(),
            edges=((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
            trace_index=3,
        ),
    }


EQUILIBRIA = _build_equilibria()


def available_configs() -> tuple[str, ...]:
    """Return the available configuration keys in sorted order."""

    return tuple(sorted(EQUILIBRIA.keys()))


def lj_force_pair_3d(r_vec: np.ndarray) -> np.ndarray:
    r2 = float(r_vec @ r_vec)
    if r2 < 1e-16:
        return np.zeros(3)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    factor = 24.0 * (2.0 * inv_r12 * inv_r2 - inv_r6 * inv_r2)
    return factor * r_vec


def lj_total_forces_3d(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    F = np.zeros_like(X)
    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = X[j] - X[i]
            fij = lj_force_pair_3d(rij)
            F[i] -= fij
            F[j] += fij
    return F


def lj_total_potential(X: np.ndarray) -> float:
    energy = 0.0
    n = X.shape[0]
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = np.linalg.norm(X[j] - X[i])
            inv_r6 = (1.0 / r) ** 6
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy


def kinetic_energy(V: np.ndarray, masses: np.ndarray) -> float:
    return 0.5 * np.sum(masses[:, None] * (V**2))


def total_energy(X: np.ndarray, V: np.ndarray, masses: np.ndarray) -> float:
    return kinetic_energy(V, masses) + lj_total_potential(X)


cbrt2 = 2.0 ** (1.0 / 3.0)
w1 = 1.0 / (2.0 - cbrt2)
w0 = -cbrt2 / (2.0 - cbrt2)
YS = (w1, w0, w1)


def step_yoshida4(
    X: np.ndarray, V: np.ndarray, dt: float, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inv_m = 1.0 / masses[:, None]
    for w in YS:
        F = lj_total_forces_3d(X)
        V = V + 0.5 * (w * dt) * (F * inv_m)
        X = X + (w * dt) * V
        F = lj_total_forces_3d(X)
        V = V + 0.5 * (w * dt) * (F * inv_m)
    return X, V


def pair_hessian(r_vec: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(r_vec))
    if r < 1e-12:
        raise ValueError("Particles overlap; Hessian undefined.")
    e = r_vec / r
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    v1 = 4.0 * (-12.0 * inv_r12 * inv_r + 6.0 * inv_r6 * inv_r)
    v2 = 4.0 * (156.0 * inv_r12 * inv_r2 - 42.0 * inv_r6 * inv_r2)
    outer = np.outer(e, e)
    return (v2 - v1 * inv_r) * outer + (v1 * inv_r) * np.eye(3)


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


def recenter(points: np.ndarray, masses: np.ndarray) -> np.ndarray:
    com = np.average(points, axis=0, weights=masses)
    return points - com


def first_stable_mode(
    positions: np.ndarray, masses: np.ndarray, tol: float = 1e-8
) -> tuple[np.ndarray, float]:
    H = build_hessian(positions)
    weights = np.repeat(np.sqrt(masses), 3)
    Hmw = H / weights[:, None]
    Hmw = Hmw / weights[None, :]
    eigvals, eigvecs = np.linalg.eigh(Hmw)
    for idx, lam in enumerate(eigvals):
        if lam > tol:
            vec = eigvecs[:, idx] / weights
            coords = vec.reshape(-1, 3)
            coords = recenter(coords, masses)
            norm = np.linalg.norm(coords)
            if norm > 0.0:
                coords /= norm
            return coords, float(lam)
    raise RuntimeError("No stable mode located above tolerance")


def prepare_equilibrium(
    config: str, center_mass: float
) -> tuple[EquilibriumSpec, np.ndarray, np.ndarray]:
    if config not in EQUILIBRIA:
        raise KeyError(f"Unknown configuration '{config}'")
    spec = EQUILIBRIA[config]
    base = np.array(spec.positions, copy=True)
    masses = np.ones(base.shape[0], dtype=float)
    if config == "triangle_center":
        masses[:3] = 1.0
        masses[3] = center_mass
    base = recenter(base, masses)
    return spec, base, masses


def simulate_trajectory(
    config: str,
    mode_displacement: float,
    mode_velocity: float,
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
    record_energies: bool = False,
) -> SimulationResult:
    if save_stride < 1:
        raise ValueError("save_stride must be >= 1")

    spec, equilibrium, masses = prepare_equilibrium(config, center_mass)
    mode_shape, omega2 = first_stable_mode(equilibrium, masses)

    X0 = equilibrium + mode_displacement * mode_shape
    V0 = mode_velocity * mode_shape
    X0 = recenter(X0, masses)
    v_com = np.sum(V0 * masses[:, None], axis=0) / masses.sum()
    V0 -= v_com

    nsteps = int(total_time / dt)
    snaps: list[np.ndarray] = [X0.copy()]
    times: list[float] = [0.0]
    kin_vals: list[float] | None = [kinetic_energy(V0, masses)] if record_energies else None
    pot_vals: list[float] | None = [lj_total_potential(X0)] if record_energies else None
    tot_vals: list[float] | None = (
        [kin_vals[0] + pot_vals[0]] if record_energies and kin_vals and pot_vals else None
    )

    Xc = X0.copy()
    Vc = V0.copy()
    for step in range(nsteps):
        Xc, Vc = step_yoshida4(Xc, Vc, dt, masses)
        if (step + 1) % save_stride == 0:
            snaps.append(Xc.copy())
            times.append((step + 1) * dt)
            if record_energies and kin_vals is not None and pot_vals is not None:
                kin = kinetic_energy(Vc, masses)
                pot = lj_total_potential(Xc)
                kin_vals.append(kin)
                pot_vals.append(pot)
                if tot_vals is not None:
                    tot_vals.append(kin + pot)

    if record_energies and kin_vals is not None and pot_vals is not None and tot_vals is not None:
        kinetic = np.array(kin_vals)
        potential = np.array(pot_vals)
        total = np.array(tot_vals)
        energy_initial = float(total[0])
        energy_final = float(total[-1])
    else:
        kinetic = None
        potential = None
        total = None
        energy_initial = None
        energy_final = None

    return SimulationResult(
        spec=spec,
        omega2=omega2,
        mode_shape=mode_shape,
        masses=masses,
        times=np.array(times),
        positions=np.array(snaps),
        kinetic=kinetic,
        potential=potential,
        total=total,
        energy_initial=energy_initial,
        energy_final=energy_final,
    )


__all__ = [
    "EquilibriumSpec",
    "SimulationResult",
    "EQUILIBRIA",
    "available_configs",
    "lj_force_pair_3d",
    "lj_total_forces_3d",
    "lj_total_potential",
    "kinetic_energy",
    "total_energy",
    "step_yoshida4",
    "pair_hessian",
    "build_hessian",
    "recenter",
    "first_stable_mode",
    "prepare_equilibrium",
    "simulate_trajectory",
]
