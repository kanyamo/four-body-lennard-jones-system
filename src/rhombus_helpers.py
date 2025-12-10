from __future__ import annotations

import math

import numpy as np


def solve_rhombus_angle_and_side(
    repulsive_exp: int, attractive_exp: int
) -> tuple[float, float]:
    """Solve for (theta, a) of the planar rhombus under (p,q)-LJ.

    Returns:
        theta (radians), edge length a.
    """

    p = float(repulsive_exp)
    q = float(attractive_exp)
    if p <= q or p <= 0 or q <= 0:
        raise ValueError("Require repulsive_exp p > attractive_exp q > 0.")

    def F(t: float) -> float:
        c2 = t
        s2 = 1.0 - t
        Nc = (2.0 ** (p + 2)) * (c2 ** ((p + 2) / 2.0)) + 1.0
        Dc = (2.0 ** (p + 2)) * (c2 ** ((p + 2) / 2.0)) + (2.0 ** (p - q)) * (
            c2 ** ((p - q) / 2.0)
        )
        Ns = (2.0 ** (p + 2)) * (s2 ** ((p + 2) / 2.0)) + 1.0
        Ds = (2.0 ** (p + 2)) * (s2 ** ((p + 2) / 2.0)) + (2.0 ** (p - q)) * (
            s2 ** ((p - q) / 2.0)
        )
        return Nc * Ds - Ns * Dc

    def find_root(interval: tuple[float, float]) -> float:
        a, b = interval
        fa, fb = F(a), F(b)
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            raise RuntimeError("Root not bracketed")
        for _ in range(80):
            mid = 0.5 * (a + b)
            fm = F(mid)
            if abs(fm) < 1e-14 or (b - a) < 1e-12:
                return mid
            if fa * fm < 0.0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        return 0.5 * (a + b)

    # t = 0.5 is the square; look for non-square roots on both sides
    lower = find_root((1e-6, 0.49))
    upper = find_root((0.51, 1.0 - 1e-6))
    t_candidates = [lower, upper]
    # pick the root closer to the (12,6) acute solution (t≈0.748) as the acute angle
    t_acute = min(t_candidates, key=lambda t: abs(t - 0.75))

    c = math.sqrt(t_acute)
    k = (p * (2.0 ** (p + 2) * c ** (p + 2) + 1.0)) / (
        q * (2.0 ** (p + 2) * c ** (p + 2) + (2.0 ** (p - q)) * c ** (p - q))
    )
    a = k ** (1.0 / (p - q))
    theta = 2.0 * math.acos(c)
    return theta, a
