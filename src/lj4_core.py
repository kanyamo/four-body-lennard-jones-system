"""Shared utilities for the 4-body Lennard-Jones simulations.

This module factors out the geometry catalogue, force computations, and
integration helpers so that both the CLI integrator and the interactive
visualiser can rely on the same implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

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
    initial_velocity: np.ndarray
    masses: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    kinetic: np.ndarray | None
    potential: np.ndarray | None
    total: np.ndarray | None
    energy_initial: float | None
    energy_final: float | None
    mode_indices: tuple[int, ...]
    mode_eigenvalues: tuple[float, ...]
    displacement_coeffs: tuple[float, ...]
    velocity_coeffs: tuple[float, ...]
    mode_shapes: tuple[np.ndarray, ...]
    modal_basis: "ModalBasis"
    modal_coordinates: np.ndarray


@dataclass(frozen=True)
class ModalBasis:
    eigenvalues: np.ndarray
    vectors: np.ndarray
    classifications: tuple[str, ...]
    labels: tuple[str, ...]


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
        "linear_chain": EquilibriumSpec(
            key="linear_chain",
            label="linear chain (D_∞h)",
            positions=np.array(
                [
                    (-1.0842293861, 0.0, 0.0),
                    (-0.3550510257, 0.0, 0.0),
                    (0.3550510257, 0.0, 0.0),
                    (1.0842293861, 0.0, 0.0),
                ],
                dtype=float,
            ),
            edges=((0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)),
            trace_index=0,
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


def _mass_repetition(masses: np.ndarray) -> np.ndarray:
    return np.repeat(masses, 3)


def _mass_inner(vec_a: np.ndarray, vec_b: np.ndarray, mass_diag: np.ndarray) -> float:
    return float(np.dot(vec_a * mass_diag, vec_b))


def _normalize_mass(vec: np.ndarray, mass_diag: np.ndarray, tol: float) -> np.ndarray | None:
    norm_sq = _mass_inner(vec, vec, mass_diag)
    if norm_sq <= tol:
        return None
    inv_norm = 1.0 / math.sqrt(norm_sq)
    return vec * inv_norm


def _translation_seeds(n_particles: int) -> list[np.ndarray]:
    seeds: list[np.ndarray] = []
    for axis in range(3):
        vec = np.zeros((n_particles, 3), dtype=float)
        vec[:, axis] = 1.0
        seeds.append(vec.reshape(-1))
    return seeds


def _rotation_seeds(positions: np.ndarray) -> list[np.ndarray]:
    seeds: list[np.ndarray] = []
    for axis in range(3):
        vec = np.zeros_like(positions)
        for i, (x, y, z) in enumerate(positions):
            if axis == 0:  # rotation about x
                vec[i] = (0.0, -z, y)
            elif axis == 1:  # rotation about y
                vec[i] = (z, 0.0, -x)
            else:  # rotation about z
                vec[i] = (-y, x, 0.0)
        seeds.append(vec.reshape(-1))
    return seeds


def compute_modal_basis(
    positions: np.ndarray,
    masses: np.ndarray,
    zero_tol: float = 1e-8,
) -> ModalBasis:
    if np.any(masses <= 0):
        raise ValueError("Masses must be positive for modal analysis.")

    mass_diag = _mass_repetition(masses)
    weights = np.sqrt(mass_diag)
    H = build_hessian(positions)
    Hmw = H / weights[:, None]
    Hmw = Hmw / weights[None, :]
    eigvals, eigvecs = np.linalg.eigh(Hmw)
    coord_modes = eigvecs / weights[:, None]

    basis_vectors: list[np.ndarray] = []
    eigenvalues: list[float] = []
    classifications: list[str] = []
    labels: list[str] = []
    total_dim = positions.size
    zero_indices: list[int] = []

    scale = max(1.0, float(np.max(np.abs(eigvals))))
    classification_tol = zero_tol * scale

    for idx, lam in enumerate(eigvals):
        vec = coord_modes[:, idx]
        if abs(lam) <= classification_tol:
            zero_indices.append(idx)
            continue
        normalized = _normalize_mass(vec, mass_diag, tol=1e-18)
        if normalized is None:
            continue
        basis_vectors.append(normalized)
        eigenvalues.append(float(lam))
        if lam > 0.0:
            classifications.append("stable")
            labels.append(f"stable_{idx}")
        else:
            classifications.append("unstable")
            labels.append(f"unstable_{idx}")

    expected_total = total_dim
    seeds = _translation_seeds(len(masses))
    seeds.extend(_rotation_seeds(positions))

    def add_seed(seed_vec: np.ndarray, label: str) -> bool:
        if len(basis_vectors) >= expected_total:
            return False
        candidate = seed_vec.astype(float, copy=True)
        for existing in basis_vectors:
            candidate -= _mass_inner(existing, candidate, mass_diag) * existing
        normalized = _normalize_mass(candidate, mass_diag, tol=1e-18)
        if normalized is None:
            return False
        basis_vectors.append(normalized)
        eigenvalues.append(0.0)
        classifications.append("zero")
        labels.append(label)
        return True

    zero_labels = [
        "translation_x",
        "translation_y",
        "translation_z",
        "rotation_x",
        "rotation_y",
        "rotation_z",
    ]

    for seed, label in zip(seeds, zero_labels):
        add_seed(seed, label)

    if len(basis_vectors) < expected_total:
        for idx in zero_indices:
            if len(basis_vectors) >= expected_total:
                break
            vec = coord_modes[:, idx]
            added = add_seed(vec, f"zero_{idx}")
            if not added:
                continue

    if len(basis_vectors) != expected_total:
        raise RuntimeError("Failed to construct a complete modal basis.")

    vectors = np.column_stack(basis_vectors)
    eigenvalues_array = np.array(eigenvalues, dtype=float)
    return ModalBasis(
        eigenvalues=eigenvalues_array,
        vectors=vectors,
        classifications=tuple(classifications),
        labels=tuple(labels),
    )


def recenter(points: np.ndarray, masses: np.ndarray) -> np.ndarray:
    com = np.average(points, axis=0, weights=masses)
    return points - com


def stable_modes(
    positions: np.ndarray, masses: np.ndarray, tol: float = 1e-8
) -> list[tuple[np.ndarray, float]]:
    H = build_hessian(positions)
    weights = np.repeat(np.sqrt(masses), 3)
    Hmw = H / weights[:, None]
    Hmw = Hmw / weights[None, :]
    eigvals, eigvecs = np.linalg.eigh(Hmw)
    modes: list[tuple[np.ndarray, float]] = []
    for idx, lam in enumerate(eigvals):
        if lam > tol:
            vec = eigvecs[:, idx] / weights
            coords = vec.reshape(-1, 3)
            coords = recenter(coords, masses)
            norm = np.linalg.norm(coords)
            if norm > 0.0:
                coords /= norm
            modes.append((coords, float(lam)))
    if not modes:
        raise RuntimeError("No stable mode located above tolerance")
    return modes


def first_stable_mode(
    positions: np.ndarray, masses: np.ndarray, tol: float = 1e-8
) -> tuple[np.ndarray, float]:
    modes = stable_modes(positions, masses, tol)
    return modes[0]


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
    mode_indices: Sequence[int] | None,
    mode_displacements: Sequence[float],
    mode_velocities: Sequence[float],
    dt: float,
    total_time: float,
    save_stride: int,
    center_mass: float,
    record_energies: bool = False,
    random_kick_energy: float = 0.0,
    random_seed: int | None = 12345,
) -> SimulationResult:
    if save_stride < 1:
        raise ValueError("save_stride must be >= 1")

    spec, equilibrium, masses = prepare_equilibrium(config, center_mass)
    all_modes = stable_modes(equilibrium, masses)
    modal_basis = compute_modal_basis(equilibrium, masses)
    mass_diag = _mass_repetition(masses)
    equilibrium_flat = equilibrium.reshape(-1)

    selected_indices = tuple(mode_indices) if mode_indices is not None else (0,)
    if not selected_indices:
        raise ValueError("At least one mode index must be specified")
    if max(selected_indices) >= len(all_modes) or min(selected_indices) < 0:
        raise IndexError(
            "Mode index out of range; ensure the configuration has that many stable modes"
        )

    def _expand(values: Sequence[float], count: int, name: str) -> list[float]:
        vals = list(values)
        if not vals:
            raise ValueError(f"{name} must contain at least one value")
        if len(vals) == 1 and count > 1:
            vals = vals * count
        if len(vals) != count:
            raise ValueError(
                f"{name} must provide either one value or as many values as modes "
                f"({count}), got {len(vals)}"
            )
        return vals

    disp_coeffs = tuple(
        _expand(mode_displacements, len(selected_indices), "mode_displacements")
    )
    vel_coeffs = tuple(
        _expand(mode_velocities, len(selected_indices), "mode_velocities")
    )

    mode_vectors: list[np.ndarray] = []
    mode_eigs: list[float] = []
    combined_disp = np.zeros_like(equilibrium)
    combined_vel = np.zeros_like(equilibrium)
    for idx, disp, vel in zip(selected_indices, disp_coeffs, vel_coeffs):
        vec, lam = all_modes[idx]
        mode_vectors.append(vec)
        mode_eigs.append(lam)
        combined_disp += disp * vec
        combined_vel += vel * vec

    if random_kick_energy > 0.0:
        rng = np.random.default_rng(random_seed)
        noise = rng.standard_normal(size=combined_vel.shape)
        momentum = np.sum(noise * masses[:, None], axis=0) / masses.sum()
        noise -= momentum  # remove net momentum before scaling
        k_noise = kinetic_energy(noise, masses)
        if k_noise > 0.0:
            scale = np.sqrt(random_kick_energy / k_noise)
            noise *= scale
            combined_vel += noise

    omega2 = mode_eigs[0]

    X0 = equilibrium + combined_disp
    V0 = combined_vel
    X0 = recenter(X0, masses)
    v_com = np.sum(V0 * masses[:, None], axis=0) / masses.sum()
    V0 -= v_com
    initial_displacement = X0 - equilibrium
    initial_velocity = V0.copy()

    nsteps = int(total_time / dt)
    snaps: list[np.ndarray] = [X0.copy()]
    times: list[float] = [0.0]
    kin_vals: list[float] | None = (
        [kinetic_energy(V0, masses)] if record_energies else None
    )
    pot_vals: list[float] | None = [lj_total_potential(X0)] if record_energies else None
    tot_vals: list[float] | None = (
        [kin_vals[0] + pot_vals[0]]
        if record_energies and kin_vals and pot_vals
        else None
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

    if (
        record_energies
        and kin_vals is not None
        and pot_vals is not None
        and tot_vals is not None
    ):
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

    total_dim = equilibrium_flat.size
    modal_coordinates = np.zeros((len(snaps), total_dim))
    basis_T = modal_basis.vectors.T
    for idx, snapshot in enumerate(snaps):
        delta_flat = snapshot.reshape(-1) - equilibrium_flat
        projected = basis_T @ (mass_diag * delta_flat)
        modal_coordinates[idx] = projected

    return SimulationResult(
        spec=spec,
        omega2=omega2,
        mode_shape=initial_displacement,
        initial_velocity=initial_velocity,
        masses=masses,
        times=np.array(times),
        positions=np.array(snaps),
        kinetic=kinetic,
        potential=potential,
        total=total,
        energy_initial=energy_initial,
        energy_final=energy_final,
        mode_indices=selected_indices,
        mode_eigenvalues=tuple(mode_eigs),
        displacement_coeffs=disp_coeffs,
        velocity_coeffs=vel_coeffs,
        mode_shapes=tuple(mode_vectors),
        modal_basis=modal_basis,
        modal_coordinates=modal_coordinates,
    )


__all__ = [
    "EquilibriumSpec",
    "SimulationResult",
    "ModalBasis",
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
    "stable_modes",
    "prepare_equilibrium",
    "simulate_trajectory",
    "compute_modal_basis",
]
