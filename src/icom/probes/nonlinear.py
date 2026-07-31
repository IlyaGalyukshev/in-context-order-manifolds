"""Nonlinear (MLP) rank probe — the curvature signature.

A curved manifold that a linear probe reads only partially should be recovered
better by a small MLP. The gap (nonlinear − linear Spearman) is evidence the
order lives on a curved low-dim manifold rather than a linear rank axis.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

from icom.probes.linear import reduce


def mlp_cv_spearman(X, y, g, pca=64, hidden=(128, 64), k=5, seed=0, max_iter=1500):
    """GroupKFold OOB Spearman of an MLP rank decode on PCA-reduced features."""
    k = min(k, len(np.unique(g)))
    if k < 2:
        return float("nan")
    Xr = reduce(X, pca, seed)
    oob = np.full(len(y), np.nan)
    for tr, te in GroupKFold(k).split(Xr, y, g):
        net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter,
                           random_state=seed, early_stopping=False, alpha=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # silence non-convergence
            net.fit(Xr[tr], y[tr])
        oob[te] = net.predict(Xr[te])
    rho = spearmanr(oob, y)[0]
    return abs(rho) if not np.isnan(rho) else float("nan")
