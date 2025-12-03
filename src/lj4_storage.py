"""シミュレーション結果の保存・読み出しユーティリティ。

計算済みの `SimulationResult` をメタデータ(JSON)と数値配列(NPZ)に分けて保存し、
後から再計算なしでロードして可視化や解析に利用できるようにする。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lj4_core import EquilibriumSpec, SimulationResult

BUNDLE_VERSION = 1
METADATA_FILENAME = "metadata.json"
ARRAYS_FILENAME = "series.npz"
CACHE_DEFAULT_DIRNAME = "results/cache"


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
        "dihedral_edges": [list(edge) for edge in result.dihedral_edges],
        "dt": result.dt,
        "total_time": result.total_time,
        "save_stride": result.save_stride,
        "center_mass": result.center_mass,
        "modal_kick_energy": result.modal_kick_energy,
    }

    arrays: dict[str, np.ndarray] = {
        "equilibrium_positions": result.spec.positions,
        "masses": result.masses,
        "times": result.times,
        "positions": result.positions,
        "velocities": result.velocities,
        "initial_displacement": result.initial_displacement,
        "initial_velocity": result.initial_velocity,
        "mode_shapes": _stack_mode_shapes(
            result.mode_shapes, result.positions.shape[1]
        ),
    }
    return meta, arrays


def _normalize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value


def canonicalize_run_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """順序と型を正規化したパラメータ辞書を作る（ハッシュの安定化用）。"""

    normalized: dict[str, Any] = {}
    for key in sorted(params.keys()):
        normalized[key] = _normalize_value(params[key])
    return normalized


def compute_bundle_dir(
    cache_root: Path, config: str, run_parameters: dict[str, Any]
) -> tuple[Path, str]:
    """パラメータからキャッシュ用バンドルディレクトリを決定する。

    Returns: (bundle_dir, cache_key_hex)
    """

    normalized = canonicalize_run_parameters(run_parameters)
    payload = {
        "config": config,
        "parameters": normalized,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    bundle_dir = Path(cache_root) / config / key
    return bundle_dir, key


def bundle_exists(bundle_dir: Path) -> bool:
    return (bundle_dir / METADATA_FILENAME).exists() and (bundle_dir / ARRAYS_FILENAME).exists()


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
    mode_shapes_arr = arrays.get(
        "mode_shapes",
        np.zeros((0, arrays["positions"].shape[1], 3), dtype=float),
    )
    mode_shapes: tuple[np.ndarray, ...] = tuple(
        mode_shapes_arr[idx] for idx in range(mode_shapes_arr.shape[0])
    )

    result = SimulationResult(
        spec=spec,
        omega2=float(meta["omega2"]),
        initial_displacement=arrays["initial_displacement"],
        initial_velocity=arrays["initial_velocity"],
        masses=arrays["masses"],
        times=arrays["times"],
        positions=arrays["positions"],
        velocities=arrays["velocities"],
        dt=float(meta.get("dt", 0.0)),
        total_time=float(meta.get("total_time", arrays["times"][-1] if len(arrays["times"]) else 0.0)),
        save_stride=int(meta.get("save_stride", 1)),
        center_mass=float(meta.get("center_mass", 1.0)),
        modal_kick_energy=float(meta.get("modal_kick_energy", 0.0)),
        mode_indices=tuple(meta["mode_selection"]["indices"]),
        mode_eigenvalues=tuple(meta["mode_selection"]["eigenvalues"]),
        displacement_coeffs=tuple(meta["mode_selection"]["displacement_coeffs"]),
        velocity_coeffs=tuple(meta["mode_selection"]["velocity_coeffs"]),
        mode_shapes=mode_shapes,
        dihedral_edges=tuple(tuple(edge) for edge in meta["dihedral_edges"]),
    )
    return LoadedSimulation(result=result, metadata=meta)
