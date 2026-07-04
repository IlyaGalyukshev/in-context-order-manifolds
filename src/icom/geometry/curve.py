"""Coordinate extraction from N pooled points.

v1 is deliberately conservative: the coordinate is the projection onto PC1 of
the centered cloud. A principal-curve refinement can only add expressive
power (and overfitting risk) — if PC1 already shows a content-order signal,
the signal exists; the curve upgrade is a Stage-2 improvement, not a pilot
requirement. For n_items < 12 even PC1 rank statistics are fragile — callers
should treat such fits as diagnostics only.
"""

from __future__ import annotations

import numpy as np


def fit_coordinate(points: np.ndarray) -> tuple[np.ndarray, dict]:
    """points [N, D] float → (coordinate [N], diagnostics)."""
    X = points.astype(np.float64)
    X = X - X.mean(axis=0)
    # PCA via SVD; PC1 projection is the coordinate
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coord = X @ Vt[0]
    var = S**2
    diag = {
        "pc1_var_ratio": float(var[0] / var.sum()),
        "pc2_var_ratio": float(var[1] / var.sum()) if len(var) > 1 else 0.0,
    }
    return coord, diag
