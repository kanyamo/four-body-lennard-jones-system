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
    initial_displacement: np.ndarray
    initial_velocity: np.ndarray
    masses: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    dt: float
    total_time: float
    save_stride: int
    center_mass: float
    modal_kick_energy: float
    mode_indices: tuple[int, ...]
    mode_eigenvalues: tuple[float, ...]
    displacement_coeffs: tuple[float, ...]
    velocity_coeffs: tuple[float, ...]
    mode_shapes: tuple[np.ndarray, ...]
    dihedral_edges: tuple[tuple[int, int], ...]


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


DIHEDRAL_EDGES: tuple[tuple[int, int], ...] = tuple(combinations(range(4), 2))
_DIHEDRAL_COMPLEMENTS: dict[tuple[int, int], tuple[int, int]] = {
    edge: tuple(sorted(set(range(4)) - set(edge))) for edge in DIHEDRAL_EDGES
}


def dihedral_angle_for_edge(
    positions: np.ndarray, edge: tuple[int, int], eps: float = 1e-12
) -> float:
    """Compute the dihedral angle (radians) around the given edge.

    The angle is defined between the planes formed by the two triangles that share
    the edge. A perfectly planar configuration yields an angle close to pi.
    """

    i, j = edge
    if positions.shape[0] <= max(i, j):
        raise ValueError("Edge index out of range for provided positions.")
    k, l = _DIHEDRAL_COMPLEMENTS[edge]  # noqa: E741
    pi, pj, pk, pl = positions[i], positions[j], positions[k], positions[l]

    edge_vec = pj - pi
    n1 = np.cross(edge_vec, pk - pi)
    n2 = np.cross(edge_vec, pl - pi)
    n1_norm = float(np.linalg.norm(n1))
    n2_norm = float(np.linalg.norm(n2))
    if n1_norm < eps or n2_norm < eps:
        return math.nan
    cos_theta = float(np.dot(n1, n2) / (n1_norm * n2_norm))
    cos_theta = min(1.0, max(-1.0, cos_theta))
    raw_angle = math.acos(cos_theta)
    # Planar configurations should map to pi regardless of normal orientation.
    return math.pi - min(raw_angle, math.pi - raw_angle)


def dihedral_angles(positions: np.ndarray) -> np.ndarray:
    """Return dihedral angles (radians) for all 6 edges."""

    angles = np.empty(len(DIHEDRAL_EDGES), dtype=float)
    for idx, edge in enumerate(DIHEDRAL_EDGES):
        angles[idx] = dihedral_angle_for_edge(positions, edge)
    return angles


def plot_dihedral_series(
    path,
    times: np.ndarray,
    dihedral_angles: np.ndarray,
    edges: Sequence[tuple[int, int]] | None = None,
    quantity: str = "gap",
) -> None:
    """Plot dihedral evolution over time.

    quantity: "gap" plots (180° - angle) to highlight脱平面量, "angle" plots the
    absolute dihedral angle in degrees.
    """

    import matplotlib.pyplot as plt

    edges = tuple(edges) if edges is not None else DIHEDRAL_EDGES
    if dihedral_angles.shape[1] != len(edges):
        raise ValueError("Mismatch between dihedral data and edge list.")

    if quantity not in {"gap", "angle"}:
        raise ValueError("quantity must be 'gap' or 'angle'")

    plt.figure(figsize=(6, 3.5))
    if quantity == "gap":
        values = np.degrees(np.pi - dihedral_angles)
        ylabel = "dihedral gap (deg from 180)"
        title = "Planarity gap vs time"
    else:
        values = np.degrees(dihedral_angles)
        ylabel = "dihedral angle (deg)"
        title = "Dihedral angles vs time"

    for idx, edge in enumerate(edges):
        plt.plot(times, values[:, idx], label=f"{edge[0]}-{edge[1]}")
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.title(title)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


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


def _normalize_mass(
    vec: np.ndarray, mass_diag: np.ndarray, tol: float
) -> np.ndarray | None:
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


def _rigid_align(
    points: np.ndarray, reference: np.ndarray, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch alignment: remove translation and best-fit rotation."""

    centered_p = recenter(points, masses)
    centered_ref = recenter(reference, masses)
    H = centered_p.T @ centered_ref
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = centered_p @ R
    return aligned, R


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
    modal_kick_energy: float = 0.0,
) -> SimulationResult:
    if save_stride < 1:
        raise ValueError("save_stride must be >= 1")

    spec, equilibrium, masses = prepare_equilibrium(config, center_mass)
    all_modes = stable_modes(equilibrium, masses)
    modal_basis = compute_modal_basis(equilibrium, masses)

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

    if modal_kick_energy > 0.0:

        def select_modal_index(target: str) -> int | None:
            for idx, cls in enumerate(modal_basis.classifications):
                if cls == target:
                    return idx
            return None

        modal_index = select_modal_index("unstable")
        if modal_index is None:
            modal_index = select_modal_index("stable")
        if modal_index is None:
            modal_index = select_modal_index("zero")

        if modal_index is not None:
            base_vector = modal_basis.vectors[:, modal_index]
            direction = base_vector.reshape(combined_vel.shape)
            amplitude = math.sqrt(2.0 * modal_kick_energy)
            combined_vel += amplitude * direction

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
    velocities_series: list[np.ndarray] = [V0.copy()]

    Xc = X0.copy()
    Vc = V0.copy()
    for step in range(nsteps):
        Xc, Vc = step_yoshida4(Xc, Vc, dt, masses)
        if (step + 1) % save_stride == 0:
            snaps.append(Xc.copy())
            times.append((step + 1) * dt)
            velocities_series.append(Vc.copy())

    velocities_arr = np.array(velocities_series)

    return SimulationResult(
        spec=spec,
        omega2=omega2,
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
        masses=masses,
        times=np.array(times),
        positions=np.array(snaps),
        velocities=velocities_arr,
        dt=dt,
        total_time=total_time,
        save_stride=save_stride,
        center_mass=center_mass,
        modal_kick_energy=modal_kick_energy,
        mode_indices=selected_indices,
        mode_eigenvalues=tuple(mode_eigs),
        displacement_coeffs=disp_coeffs,
        velocity_coeffs=vel_coeffs,
        mode_shapes=tuple(mode_vectors),
        dihedral_edges=DIHEDRAL_EDGES,
    )


def compute_energy_series(result: SimulationResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kinetic = np.array([kinetic_energy(v, result.masses) for v in result.velocities])
    potential = np.array([lj_total_potential(x) for x in result.positions])
    total = kinetic + potential
    return kinetic, potential, total


def compute_modal_projections(
    result: SimulationResult,
    modal_basis: ModalBasis | None = None,
    align: bool = True,
) -> tuple[np.ndarray, np.ndarray, ModalBasis]:
    basis = modal_basis or compute_modal_basis(result.spec.positions, result.masses)
    mass_diag = _mass_repetition(result.masses)
    basis_T = basis.vectors.T
    eq_flat = result.spec.positions.reshape(-1)

    coords = np.zeros((len(result.times), eq_flat.size))
    velocities = np.zeros_like(coords)
    for idx, (pos, vel) in enumerate(zip(result.positions, result.velocities)):
        if align:
            aligned_pos, R = _rigid_align(pos, result.spec.positions, result.masses)
            aligned_vel = vel @ R
        else:
            aligned_pos = recenter(pos, result.masses)
            aligned_vel = vel
        delta = aligned_pos.reshape(-1) - eq_flat
        v_flat = aligned_vel.reshape(-1)
        coords[idx] = basis_T @ (mass_diag * delta)
        velocities[idx] = basis_T @ (mass_diag * v_flat)
    return coords, velocities, basis


def compute_modal_energies(
    result: SimulationResult,
    modal_basis: ModalBasis | None = None,
    align: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ModalBasis]:
    coords, velocities, basis = compute_modal_projections(
        result, modal_basis=modal_basis, align=align
    )
    # modal kinetic = 0.5 * v^2 (mass-normalized basisなので質量は1扱い)
    kinetic = 0.5 * velocities**2
    # modal potential = 0.5 * lambda * q^2 （固有値はω^2）
    eigvals = basis.eigenvalues
    potential = 0.5 * coords * (coords * eigvals[None, :])
    total = kinetic + potential
    return kinetic, potential, total, basis


def compute_dihedral_series(
    result: SimulationResult,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.array([dihedral_angles(pos) for pos in result.positions])
    gaps = math.pi - angles
    return angles, gaps


__all__ = [
    "EquilibriumSpec",
    "SimulationResult",
    "ModalBasis",
    "EQUILIBRIA",
    "DIHEDRAL_EDGES",
    "available_configs",
    "dihedral_angle_for_edge",
    "dihedral_angles",
    "plot_dihedral_series",
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
    "compute_modal_projections",
    "compute_dihedral_series",
    "compute_energy_series",
]
