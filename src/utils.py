"""Shared geometry utilities used by both LJ and spring systems."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np

# All 6 edges for a 4-particle system
DIHEDRAL_EDGES: tuple[tuple[int, int], ...] = tuple(combinations(range(4), 2))
_DIHEDRAL_COMPLEMENTS: dict[tuple[int, int], tuple[int, int]] = {
    edge: tuple(sorted(set(range(4)) - set(edge))) for edge in DIHEDRAL_EDGES
}


def recenter(points: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Translate coordinates so that the center of mass is at the origin."""

    com = np.average(points, axis=0, weights=masses)
    return points - com


def dihedral_angle_for_edge(
    positions: np.ndarray, edge: tuple[int, int], eps: float = 1e-12
) -> float:
    """Compute the dihedral angle (radians) around the given edge."""

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


def dihedral_angles(
    positions: np.ndarray, edges: Sequence[tuple[int, int]] | None = None
) -> np.ndarray:
    """Return dihedral angles (radians) for the given edge list."""

    edges = tuple(edges) if edges is not None else DIHEDRAL_EDGES
    angles = np.empty(len(edges), dtype=float)
    for idx, edge in enumerate(edges):
        angles[idx] = dihedral_angle_for_edge(positions, edge)
    return angles


__all__ = ["recenter", "dihedral_angle_for_edge", "dihedral_angles", "DIHEDRAL_EDGES"]
