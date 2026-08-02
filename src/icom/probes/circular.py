"""Circular-manifold characterization (Phase G8): does a cyclic structure yield a
CIRCLE? For cyclic stimuli, `ranks` (1..N by cyclic position) is the ring index, so
cyclic distance = min(|Δrank|, N−|Δrank|).

  - circle_fit      : algebraic (Kåsa) circle fit to a 2-D projection; small
                      rmse_norm ⇒ the points lie on a circle.
  - circular_rsa    : RSA of activation distance vs CYCLIC distance and vs LINEAR
                      distance; a ring tracks cyclic ≫ linear.
  - angular_decode  : does the ANGLE of the 2-D projection recover cyclic position
                      (Fisher–Lee circular correlation)?
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def circle_fit(P):
    """Kåsa algebraic circle fit to 2-D points. Returns {cx, cy, r, rmse_norm}
    with rmse_norm = RMS radial residual / r (0 = perfect circle)."""
    P = np.asarray(P, float)[:, :2]
    x, y = P[:, 0], P[:, 1]
    sol, *_ = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]),
                              x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    rad = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((rad - r) ** 2)))
    return {"cx": float(cx), "cy": float(cy), "r": r,
            "rmse_norm": rmse / (r + 1e-12)}


def circular_rsa(recs, layer, interior_only=False):
    """Mean within-stimulus Spearman of activation distance vs CYCLIC distance and
    vs LINEAR distance (ranks used as ring positions). Returns
    {cyclic_rsa, linear_rsa, n}; a ring gives cyclic_rsa ≫ linear_rsa."""
    from icom.probes.loading import interior_mask
    cyc, lin = [], []
    for r in recs:
        ranks, N = r["ranks"], r["N"]
        m = interior_mask(ranks, N) if interior_only else np.ones(len(ranks), bool)
        m = m & np.isfinite(r["X"][:, layer, :]).all(axis=1)   # drop fp16-overflow rows
        if m.sum() < 4:
            continue
        X = r["X"][m, layer, :].astype(np.float64)
        rk = ranks[m]
        actd = pdist(X)
        dif = np.abs(rk[:, None] - rk[None, :])
        iu = np.triu_indices(len(rk), 1)
        lind = dif[iu]
        cycd = np.minimum(dif, N - dif)[iu]
        if np.std(actd) > 0 and np.std(cycd) > 0:
            cyc.append(spearmanr(actd, cycd)[0])
        if np.std(actd) > 0 and np.std(lind) > 0:
            lin.append(spearmanr(actd, lind)[0])
    return {"cyclic_rsa": float(np.mean(cyc)) if cyc else float("nan"),
            "linear_rsa": float(np.mean(lin)) if lin else float("nan"),
            "n": len(cyc)}


def angular_decode(P, ranks, N):
    """Does the ANGLE of the 2-D projection track cyclic position? If the
    projection is a rank-ordered circle then the projected angle equals the true
    ring angle up to a constant offset (and possibly a reflection), so `ang − true`
    (or `ang + true`) is nearly constant ⇒ its resultant length R ≈ 1. Random ⇒ 0.
    (Robust to the full-ring case where a Fisher–Lee circular mean is undefined.)"""
    P = np.asarray(P, float)[:, :2]
    P = (P - P.mean(0)) / (P.std(0) + 1e-12)               # equalize axis variance (de-ellipse)
    ang = np.arctan2(P[:, 1], P[:, 0])
    true = 2 * np.pi * (np.asarray(ranks) - 1) / N
    R = lambda x: abs(np.mean(np.exp(1j * x)))
    return float(max(R(ang - true), R(ang + true)))       # constant offset/reflection ⇒ ~1
