"""Utilities for a 4-body mass-spring system in a planar rhombus-like layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils import recenter

# Particle ordering: R(0), T(1), L(2), B(3)
BASE_POSITIONS = np.array(
    [
        (math.sqrt(3) / 2.0, 0.0, 0.0),  # R
        (0.0, 0.5, 0.0),  # T
        (-math.sqrt(3) / 2.0, 0.0, 0.0),  # L
        (0.0, -0.5, 0.0),  # B
    ],
    dtype=float,
)

# Springs: RT, TL, LB, BR, TB (natural length = 1, k = 1)
SPRING_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (1, 3),
)
SPRING_K = 1.0
SPRING_REST = 1.0


@dataclass(frozen=True)
class EquilibriumSpec:
    key: str
    label: str
    positions: np.ndarray
    edges: tuple[tuple[int, int], ...]
    trace_index: int


@dataclass(frozen=True)
class SimulationResult:
    spec: EquilibriumSpec
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    dt: float
    total_time: float
    save_stride: int
    mode_indices: tuple[int, ...]
    mode_eigenvalues: tuple[float, ...]
    displacement_coeffs: tuple[float, ...]
    velocity_coeffs: tuple[float, ...]
    mode_shapes: tuple[np.ndarray, ...]
    initial_displacement: np.ndarray
    initial_velocity: np.ndarray


def spring_force_pair(
    r_vec: np.ndarray, k: float = SPRING_K, r0: float = SPRING_REST
) -> np.ndarray:
    r = float(np.linalg.norm(r_vec))
    if r < 1e-12:
        return np.zeros(3)
    return -k * (r - r0) * (r_vec / r)


def spring_pair_hessian(
    r_vec: np.ndarray, k: float = SPRING_K, r0: float = SPRING_REST
) -> np.ndarray:
    r = float(np.linalg.norm(r_vec))
    if r < 1e-12:
        raise ValueError("Particles overlap; Hessian undefined.")
    u = r_vec / r
    Wp = k * (r - r0)
    Wpp = k
    outer = np.outer(u, u)
    return (Wpp - Wp / r) * outer + (Wp / r) * np.eye(3)


def total_forces(coords: np.ndarray) -> np.ndarray:
    F = np.zeros_like(coords)
    for i, j in SPRING_EDGES:
        rij = coords[j] - coords[i]
        fij = spring_force_pair(rij)
        F[i] -= fij
        F[j] += fij
    return F


def total_potential(coords: np.ndarray) -> float:
    energy = 0.0
    for i, j in SPRING_EDGES:
        r = float(np.linalg.norm(coords[j] - coords[i]))
        dr = r - SPRING_REST
        energy += 0.5 * SPRING_K * dr * dr
    return energy


def kinetic_energy(V: np.ndarray, masses: np.ndarray) -> float:
    return 0.5 * np.sum(masses[:, None] * (V**2))


def build_hessian(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    H = np.zeros((3 * n, 3 * n), dtype=float)
    for i, j in SPRING_EDGES:
        rij = coords[j] - coords[i]
        kij = spring_pair_hessian(rij)
        sl_i = slice(3 * i, 3 * i + 3)
        sl_j = slice(3 * j, 3 * j + 3)
        H[sl_i, sl_i] += kij
        H[sl_j, sl_j] += kij
        H[sl_i, sl_j] -= kij
        H[sl_j, sl_i] -= kij
    return H


def mass_weighted_hessian(H: np.ndarray, masses: np.ndarray) -> np.ndarray:
    weights = np.repeat(np.sqrt(masses), 3)
    Hmw = H / weights[:, None]
    Hmw = Hmw / weights[None, :]
    return Hmw


def compute_modal_basis(coords: np.ndarray, masses: np.ndarray, zero_tol: float = 1e-8):
    H = build_hessian(coords)
    Hmw = mass_weighted_hessian(H, masses)
    eigvals, eigvecs = np.linalg.eigh(Hmw)
    weights = np.repeat(np.sqrt(masses), 3)
    coord_modes = eigvecs / weights[:, None]
    classifications = []
    for lam in eigvals:
        if abs(lam) <= zero_tol:
            classifications.append("zero")
        elif lam > 0:
            classifications.append("stable")
        else:
            classifications.append("unstable")
    return eigvals, coord_modes, tuple(classifications)


def stable_modes(coords: np.ndarray, masses: np.ndarray, tol: float = 1e-8):
    eigvals, modes, classes = compute_modal_basis(coords, masses, zero_tol=tol)
    stable: list[tuple[np.ndarray, float]] = []
    for lam, cls, vec in zip(eigvals, classes, modes.T):
        if cls == "stable":
            v = vec.reshape(-1, 3)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            stable.append((v, float(lam)))
    if not stable:
        raise RuntimeError("No stable modes found.")
    return stable


def first_stable_mode(coords: np.ndarray, masses: np.ndarray, tol: float = 1e-8):
    return stable_modes(coords, masses, tol)[0]


def step_yoshida4(
    X: np.ndarray, V: np.ndarray, dt: float, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)
    YS = (w1, w0, w1)

    inv_m = 1.0 / masses[:, None]
    for w in YS:
        F = total_forces(X)
        V = V + 0.5 * (w * dt) * (F * inv_m)
        X = X + (w * dt) * V
        F = total_forces(X)
        V = V + 0.5 * (w * dt) * (F * inv_m)
    return X, V


def simulate_trajectory(
    mode_indices: Sequence[int] | None,
    mode_displacements: Sequence[float],
    mode_velocities: Sequence[float],
    dt: float,
    total_time: float,
    save_stride: int,
    kick_velocity: np.ndarray | None = None,
) -> SimulationResult:
    if save_stride < 1:
        raise ValueError("save_stride must be >= 1")

    spec = EquilibriumSpec(
        key="spring_rhombus",
        label="spring rhombus",
        positions=BASE_POSITIONS.copy(),
        edges=SPRING_EDGES,
        trace_index=0,
    )
    masses = np.ones(4, dtype=float)
    equilibrium = spec.positions
    all_modes = stable_modes(equilibrium, masses)

    selected_indices = tuple(mode_indices) if mode_indices is not None else (0,)
    if not selected_indices:
        raise ValueError("At least one mode index must be specified")
    if max(selected_indices) >= len(all_modes) or min(selected_indices) < 0:
        raise IndexError("Mode index out of range")

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

    if kick_velocity is not None:
        if kick_velocity.shape != combined_vel.shape:
            raise ValueError("kick_velocity must have shape (4,3)")
        combined_vel = combined_vel + kick_velocity

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
        times=np.array(times),
        positions=np.array(snaps),
        velocities=velocities_arr,
        masses=masses,
        dt=dt,
        total_time=total_time,
        save_stride=save_stride,
        mode_indices=selected_indices,
        mode_eigenvalues=tuple(mode_eigs),
        displacement_coeffs=disp_coeffs,
        velocity_coeffs=vel_coeffs,
        mode_shapes=tuple(mode_vectors),
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
    )


__all__ = [
    "EquilibriumSpec",
    "SimulationResult",
    "BASE_POSITIONS",
    "SPRING_EDGES",
    "recenter",
    "total_forces",
    "total_potential",
    "kinetic_energy",
    "build_hessian",
    "mass_weighted_hessian",
    "compute_modal_basis",
    "stable_modes",
    "first_stable_mode",
    "simulate_trajectory",
]
