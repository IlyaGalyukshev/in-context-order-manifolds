"""Linear interior-only rank probe + permutation null + per-rank MAE.

The PRIMARY geometry metric: cross-stimulus GroupKFold ridge decode of latent
rank, restricted to interior entities (identical role/frequency), scored by
out-of-fold Spearman against a within-stimulus label-permutation null.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def reduce(X, pca=64, seed=0):
    """Standardize then PCA to <= pca dims (guards n_features/n_samples)."""
    Xs = StandardScaler().fit_transform(X)
    k = min(pca, X.shape[0] - 1, X.shape[1])
    if k < 1:
        return Xs
    return PCA(k, random_state=seed).fit_transform(Xs)


def cv_predict(Xr, y, g, alpha=10.0, k=5):
    """Out-of-fold ridge predictions under GroupKFold (groups = stimuli)."""
    k = min(k, len(np.unique(g)))
    if k < 2:
        return None
    oob = np.full(len(y), np.nan)
    for tr, te in GroupKFold(k).split(Xr, y, g):
        oob[te] = Ridge(alpha=alpha).fit(Xr[tr], y[tr]).predict(Xr[te])
    return oob


def cv_spearman(Xr, y, g, alpha=10.0, k=5):
    oob = cv_predict(Xr, y, g, alpha, k)
    if oob is None:
        return float("nan")
    rho = spearmanr(oob, y)[0]
    return abs(rho) if not np.isnan(rho) else float("nan")


def probe_with_null(X, y, g, n_perm=100, pca=64, seed=0, alpha=10.0):
    """Return (real_spearman, null95, p). Null = within-group label permutation
    (position ⟂ order under the shuffle condition), Xr reused so only the CV
    refit is repeated. p = (1 + #{null>=real}) / (1 + n_perm)."""
    if len(np.unique(np.round(y, 6))) < 3 or len(np.unique(g)) < 2:
        return float("nan"), float("nan"), float("nan")
    Xr = reduce(X, pca, seed)
    real = cv_spearman(Xr, y, g, alpha)
    if np.isnan(real):                                   # degenerate probe -> no p
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = y.copy()
        for gg in np.unique(g):
            idx = np.where(g == gg)[0]
            yp[idx] = y[idx][rng.permutation(len(idx))]
        null[i] = cv_spearman(Xr, yp, g, alpha)
    fin = null[np.isfinite(null)]                        # p over FINITE nulls only
    if len(fin) == 0:
        return float(real), float("nan"), float("nan")
    p = (1 + int((fin >= real).sum())) / (1 + len(fin))
    return float(real), float(np.percentile(fin, 95)), float(p)


def per_rank_mae(X, ranks, g, N, pca=64, seed=0, alpha=10.0):
    """MAE of the OOB rank prediction per rank -> which ranks localize (the
    profile that separated v1's endpoint-only signal from real interior code).
    Uses normalized ranks; returns {rank: mae} over the ranks present."""
    y = (np.asarray(ranks) - 1) / (N - 1)
    Xr = reduce(X, pca, seed)
    oob = cv_predict(Xr, y, g, alpha)
    if oob is None:
        return {}
    out = {}
    for r in np.unique(ranks):
        m = ranks == r
        out[int(r)] = float(np.nanmean(np.abs(oob[m] - y[m])))
    return out
