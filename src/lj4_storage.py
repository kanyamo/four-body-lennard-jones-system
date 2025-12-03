"""シミュレーション結果の保存・読み出しユーティリティ。

計算済みの `SimulationResult` をメタデータ(JSON)と数値配列(NPZ)に分けて保存し、
後から再計算なしでロードして可視化や解析に利用できるようにする。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lj4_core import EquilibriumSpec, ModalBasis, SimulationResult

BUNDLE_VERSION = 1
METADATA_FILENAME = "metadata.json"
ARRAYS_FILENAME = "series.npz"


@dataclass(frozen=True)
class LoadedSimulation:
    """保存済みバンドルを読み出した結果。"""

    result: SimulationResult
    metadata: dict[str, Any]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stack_mode_shapes(
    mode_shapes: tuple[np.ndarray, ...], particle_count: int
) -> np.ndarray:
    if not mode_shapes:
        return np.zeros((0, particle_count, 3), dtype=float)
    return np.stack(mode_shapes, axis=0)


def bundle_from_result(
    result: SimulationResult, run_parameters: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """SimulationResultを保存用メタデータと配列に分離する。"""

    meta = {
        "version": BUNDLE_VERSION,
        "created_at": _timestamp(),
        "config": {
            "key": result.spec.key,
            "label": result.spec.label,
            "trace_index": result.spec.trace_index,
            "edges": [list(edge) for edge in result.spec.edges],
        },
        "run_parameters": dict(run_parameters),
        "omega2": float(result.omega2),
        "mode_selection": {
            "indices": list(result.mode_indices),
            "eigenvalues": list(result.mode_eigenvalues),
            "displacement_coeffs": list(result.displacement_coeffs),
            "velocity_coeffs": list(result.velocity_coeffs),
        },
        "energies": {
            "stored": result.kinetic is not None
            and result.potential is not None
            and result.total is not None,
            "initial": result.energy_initial,
            "final": result.energy_final,
        },
        "modal_basis": {
            "classifications": list(result.modal_basis.classifications),
            "labels": list(result.modal_basis.labels),
        },
        "dihedral_edges": [list(edge) for edge in result.dihedral_edges],
    }

    arrays: dict[str, np.ndarray] = {
        "equilibrium_positions": result.spec.positions,
        "masses": result.masses,
        "times": result.times,
        "positions": result.positions,
        "mode_shape": result.mode_shape,
        "initial_velocity": result.initial_velocity,
        "mode_shapes": _stack_mode_shapes(result.mode_shapes, result.positions.shape[1]),
        "modal_basis_eigenvalues": result.modal_basis.eigenvalues,
        "modal_basis_vectors": result.modal_basis.vectors,
        "modal_coordinates": result.modal_coordinates,
        "dihedral_angles": result.dihedral_angles,
        "dihedral_planarity_gap": result.dihedral_planarity_gap,
    }
    if result.kinetic is not None:
        arrays["kinetic"] = result.kinetic
    if result.potential is not None:
        arrays["potential"] = result.potential
    if result.total is not None:
        arrays["total"] = result.total
    return meta, arrays


def save_simulation_bundle(
    bundle_dir: Path, result: SimulationResult, run_parameters: dict[str, Any]
) -> dict[str, Any]:
    """metadata.json + series.npz の形でバンドルを保存する。"""

    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)
    meta, arrays = bundle_from_result(result, run_parameters)
    (bundle_path / METADATA_FILENAME).write_text(json.dumps(meta, indent=2))
    np.savez_compressed(bundle_path / ARRAYS_FILENAME, **arrays)
    return meta


def load_simulation_bundle(bundle_dir: Path) -> LoadedSimulation:
    """保存済みバンドルを読み込み、SimulationResultとして復元する。"""

    bundle_path = Path(bundle_dir)
    metadata_path = bundle_path / METADATA_FILENAME
    arrays_path = bundle_path / ARRAYS_FILENAME
    meta = json.loads(metadata_path.read_text())

    with np.load(arrays_path) as data:
        arrays = {key: data[key] for key in data.files}

    config_meta = meta["config"]
    spec = EquilibriumSpec(
        key=config_meta["key"],
        label=config_meta["label"],
        positions=arrays["equilibrium_positions"],
        edges=tuple(tuple(edge) for edge in config_meta["edges"]),
        trace_index=int(config_meta["trace_index"]),
    )
    modal_basis = ModalBasis(
        eigenvalues=arrays["modal_basis_eigenvalues"],
        vectors=arrays["modal_basis_vectors"],
        classifications=tuple(meta["modal_basis"]["classifications"]),
        labels=tuple(meta["modal_basis"]["labels"]),
    )
    mode_shapes_arr = arrays.get(
        "mode_shapes",
        np.zeros((0, arrays["positions"].shape[1], 3), dtype=float),
    )
    mode_shapes: tuple[np.ndarray, ...] = tuple(
        mode_shapes_arr[idx] for idx in range(mode_shapes_arr.shape[0])
    )

    def _optional_array(name: str) -> np.ndarray | None:
        return arrays[name] if name in arrays else None

    energies_meta = meta.get("energies", {})
    result = SimulationResult(
        spec=spec,
        omega2=float(meta["omega2"]),
        mode_shape=arrays["mode_shape"],
        initial_velocity=arrays["initial_velocity"],
        masses=arrays["masses"],
        times=arrays["times"],
        positions=arrays["positions"],
        kinetic=_optional_array("kinetic"),
        potential=_optional_array("potential"),
        total=_optional_array("total"),
        energy_initial=energies_meta.get("initial"),
        energy_final=energies_meta.get("final"),
        mode_indices=tuple(meta["mode_selection"]["indices"]),
        mode_eigenvalues=tuple(meta["mode_selection"]["eigenvalues"]),
        displacement_coeffs=tuple(meta["mode_selection"]["displacement_coeffs"]),
        velocity_coeffs=tuple(meta["mode_selection"]["velocity_coeffs"]),
        mode_shapes=mode_shapes,
        modal_basis=modal_basis,
        modal_coordinates=arrays["modal_coordinates"],
        dihedral_edges=tuple(tuple(edge) for edge in meta["dihedral_edges"]),
        dihedral_angles=arrays["dihedral_angles"],
        dihedral_planarity_gap=arrays["dihedral_planarity_gap"],
    )
    return LoadedSimulation(result=result, metadata=meta)
