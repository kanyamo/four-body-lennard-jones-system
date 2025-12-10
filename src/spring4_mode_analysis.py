#!/usr/bin/env python3
"""Normal-mode analysis for the 4-body spring rhombus system."""

from __future__ import annotations

import argparse
import json
import numpy as np

from spring4_core import (
    BASE_POSITIONS,
    SPRING_EDGES,
    EquilibriumSpec,
    build_hessian,
    compute_modal_basis,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--digits", type=int, default=6)
    ap.add_argument("--zero-tol", type=float, default=1e-8)
    ap.add_argument(
        "--print-vectors",
        choices=["none", "zero", "stable", "unstable", "all"],
        default="none",
    )
    ap.add_argument("--save-json", type=str, default=None)
    return ap.parse_args()


def select_indices(classifications: tuple[str, ...], target: str) -> list[int]:
    if target == "all":
        return list(range(len(classifications)))
    return [i for i, cls in enumerate(classifications) if cls == target]


def main() -> None:
    args = parse_args()
    spec = EquilibriumSpec(
        key="spring_rhombus",
        label="spring rhombus",
        positions=BASE_POSITIONS,
        edges=SPRING_EDGES,
        trace_index=0,
    )
    masses = np.ones(4, dtype=float)

    H = build_hessian(spec.positions)
    lam_h, _ = np.linalg.eigh(H)

    eigvals, coord_modes, classes = compute_modal_basis(
        spec.positions, masses, zero_tol=args.zero_tol
    )
    lam_dyn = eigvals
    vec_dyn = coord_modes

    print("Spring 4-body mode analysis (rhombus)")
    print(f"  zero tolerance = {args.zero_tol:.1e}")
    print()
    print("Hessian eigenvalues λ:")
    for val in lam_h:
        print(f"  {val:.{args.digits}f}")
    print()
    print("Mass-weighted eigenvalues ω² and classifications:")
    for idx, (lam, cls) in enumerate(zip(lam_dyn, classes)):
        if lam > args.zero_tol:
            omega_str = f"{math.sqrt(lam):.{args.digits}f}"
        elif lam < -args.zero_tol:
            omega_str = f"i{math.sqrt(-lam):.{args.digits}f}"
        else:
            omega_str = "0"
        print(
            f"  idx={idx:2d}  ω²={lam:.{args.digits}f}  ω={omega_str:<12}  class={cls}"
        )

    if args.print_vectors != "none":
        targets = select_indices(classes, args.print_vectors)
        if targets:
            print()
            print(f"Eigenvectors ({args.print_vectors}):")
        for idx in targets:
            vec = vec_dyn[:, idx]
            reshaped = vec.reshape(-1, 3)
            print(f"mode {idx} (ω²={lam_dyn[idx]:.{args.digits}f})")
            for pid, comps in enumerate(reshaped):
                print(f"  P{pid}: ({comps[0]:+.6f}, {comps[1]:+.6f}, {comps[2]:+.6f})")
            print()

    if args.save_json:
        data = {
            "configuration": spec.label,
            "positions": spec.positions.tolist(),
            "edges": spec.edges,
            "hessian_eigenvalues": lam_h.tolist(),
            "modes": [
                {
                    "index": i,
                    "omega2": float(lam),
                    "classification": cls,
                    "vector": vec_dyn[:, i].reshape(-1, 3).tolist(),
                }
                for i, (lam, cls) in enumerate(zip(lam_dyn, classes))
            ],
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    import math

    main()
