"""PCA + principal curve → arc-length coordinate.

Points: [n_items × D] for one (stimulus, layer, pooling). Center, project to
top-k PCs, fit a principal curve (spline), parameterize by arc length. For
n_items < 16 a curve fit is overfitting — refuse and fall back to PC1
projection with a warning flag in the output.
"""

from __future__ import annotations


def fit_coordinate(points, max_pcs: int = 5):
    """Return (coordinate per item, fit diagnostics)."""
    raise NotImplementedError
