"""Representational geometry: RSA (metric), TwoNN (intrinsic dim), projections.

- rsa_rank: does activation DISTANCE track |Δrank| (a metric manifold), not just
  linear-separable order? Per stimulus, interior-only, then averaged.
- twonn / intrinsic_dim: the manifold's intrinsic dimension (~1 for a single
  order, ~2 for a grid / partial order, high for no manifold).
- project: low-dim embedding for the 2D/3D manifold pictures.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from icom.probes.loading import interior_mask


def rsa_rank(recs, layer, interior_only=True, min_pts=4):
    """Mean within-stimulus Spearman( activation pairwise distance, |Δrank| ).
    interior-only by default. Returns (mean_rho, n_stimuli_used)."""
    rhos = []
    for r in recs:
        ranks, N = r["ranks"], r["N"]
        mask = interior_mask(ranks, N) if interior_only else np.ones(len(ranks), bool)
        if mask.sum() < min_pts:
            continue
        X = r["X"][mask, layer, :].astype(np.float64)
        rk = ranks[mask].astype(np.float64)
        da = pdist(X)                                   # activation distances
        dr = pdist(rk.reshape(-1, 1))                   # |Δrank|
        if np.std(da) == 0 or np.std(dr) == 0:
            continue
        rho = spearmanr(da, dr)[0]
        if not np.isnan(rho):
            rhos.append(rho)
    return (float(np.mean(rhos)), len(rhos)) if rhos else (float("nan"), 0)


def twonn(X, discard_fraction=0.1):
    """Facco et al. (2017) TwoNN intrinsic-dimension estimator.
    d = slope (through origin) of -log(1-F(mu)) vs log(mu), mu = r2/r1 the ratio
    of the 2nd- to 1st-nearest-neighbour distances. Robust to curvature."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 4:
        return float("nan")
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(X).query(X, k=3)                  # col0=self(0), col1=r1, col2=r2
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = (r1 > 0) & (r2 > r1)
    mu = np.sort(r2[ok] / r1[ok])
    m = len(mu)
    if m < 5:
        return float("nan")
    # always drop >=1 from the top so the empirical CDF never hits F=1 (which
    # sends -log(1-F) -> +inf); keep in [4, m-1].
    keep = min(max(int(m * (1 - discard_fraction)), 4), m - 1)
    mu = mu[:keep]
    F = (np.arange(1, keep + 1)) / m                     # empirical CDF at kept mu
    x = np.log(mu)
    y = -np.log(1.0 - F)
    denom = float(np.sum(x * x))
    # near-degenerate (mu≈1 everywhere, e.g. a perfectly uniform grid) => x≈0;
    # the slope is unidentifiable -> nan rather than a blown-up estimate.
    if denom < 1e-8:
        return float("nan")
    return float(np.sum(x * y) / denom)


def intrinsic_dim(recs, layer, interior_only=False, pool=False):
    """TwoNN per stimulus (median across stimuli) OR on the pooled cloud.
    Per-stimulus N is small (<=16) -> the median over many stimuli is the stable
    estimate; `pool=True` estimates the union cloud's dim instead."""
    if pool:
        Xs = []
        for r in recs:
            mask = interior_mask(r["ranks"], r["N"]) if interior_only else np.ones(len(r["ranks"]), bool)
            Xs.append(r["X"][mask, layer, :])
        X = np.concatenate(Xs) if Xs else np.empty((0, 1))
        return twonn(X)
    ids = []
    for r in recs:
        mask = interior_mask(r["ranks"], r["N"]) if interior_only else np.ones(len(r["ranks"]), bool)
        if mask.sum() >= 5:
            d = twonn(r["X"][mask, layer, :])
            if not np.isnan(d):
                ids.append(d)
    return float(np.median(ids)) if ids else float("nan")


def project(X, method="pca", dim=2, seed=0, n_neighbors=10):
    """Low-dim embedding for visualization. Linear (pca) or nonlinear
    (isomap/spectral via sklearn.manifold — no extra dep; umap if installed)."""
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    dim = min(dim, Xs.shape[1])
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(dim, random_state=seed).fit_transform(Xs)
    if method == "isomap":
        from sklearn.manifold import Isomap
        nn = min(n_neighbors, len(Xs) - 1)
        return Isomap(n_neighbors=max(nn, 2), n_components=dim).fit_transform(Xs)
    if method == "spectral":
        from sklearn.manifold import SpectralEmbedding
        nn = min(n_neighbors, len(Xs) - 1)
        return SpectralEmbedding(n_components=dim, random_state=seed,
                                 n_neighbors=max(nn, 2)).fit_transform(Xs)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=dim, random_state=seed,
                         n_neighbors=min(n_neighbors, len(Xs) - 1)).fit_transform(Xs)
    raise ValueError(f"unknown method {method}")
