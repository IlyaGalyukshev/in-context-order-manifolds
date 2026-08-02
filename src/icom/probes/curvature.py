"""Nonlinear-shape characterization (Phase G4): is the order manifold CURVED?

The literature says ordinal/magnitude codes are usually nonlinear (helix, circle,
log, place-cell), so "what shape" is co-primary with "does it decode". Four
complementary, cheap signals over the interior activations at one layer:
  - curvature_profile : linear vs nonlinear(MLP) vs geodesic(Isomap) rank decode;
                        a gain of nonlinear/geodesic over linear ⇒ curvature.
  - principal_curve_curvature : how much better a smooth 1D curve (per-dim
                        polynomial in rank) reconstructs X than the best straight
                        line (PC1). 0 = straight, →1 = strongly curved.
  - curved_irreducible : linear PCA dim vs intrinsic (TwoNN) dim — a straight line
                        has both ≈1; a circle/arc has intrinsic ≈1 but linear ≥2.
  - separability_index : Engels-style test that a 2D feature does NOT factorize
                        into independent 1D marginals (joint/irreducible structure).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from icom.probes.linear import cv_spearman, reduce
from icom.probes.nonlinear import mlp_cv_spearman


def geodesic_cv_spearman(X, y, g, dim=3, n_neighbors=10, seed=0):
    """Isomap (geodesic) embedding then linear GroupKFold rank decode. A large gain
    over the ambient-linear decode indicates a curved but low-dimensional manifold.
    (Isomap is fit unsupervised on all rows, like the PCA in `reduce` — the group
    holdout still prevents label leakage; this is a descriptive gap, not the
    significance test.)"""
    from sklearn.manifold import Isomap
    Xs = StandardScaler().fit_transform(X)
    nn = min(n_neighbors, len(Xs) - 1)
    if nn < 2:
        return float("nan")
    emb = Isomap(n_neighbors=max(nn, 2), n_components=min(dim, Xs.shape[1])).fit_transform(Xs)
    return cv_spearman(emb, y, g)


def curvature_profile(X, y, g):
    """{linear, nonlinear, geodesic, nonlinear_gap, geodesic_gap, curvature_gap}.
    The two gaps are reported SEPARATELY (each = decode − linear); curvature_gap =
    max of the two is a convenience only — headline the individual gaps, each vs a
    label-permutation null (via probe_sweep), not the max (winner's-curse bias)."""
    lin = cv_spearman(reduce(X), y, g)
    nonlin = mlp_cv_spearman(X, y, g)
    geo = geodesic_cv_spearman(X, y, g)
    ng = nonlin - lin if not (np.isnan(nonlin) or np.isnan(lin)) else np.nan
    gg = geo - lin if not (np.isnan(geo) or np.isnan(lin)) else np.nan
    best = np.nanmax([ng, gg]) if not (np.isnan(ng) and np.isnan(gg)) else np.nan
    return {"linear": _r(lin), "nonlinear": _r(nonlin), "geodesic": _r(geo),
            "nonlinear_gap": _r(ng), "geodesic_gap": _r(gg), "curvature_gap": _r(best)}


def principal_curve_curvature(X, y, degree=3):
    """Curvature gain in [0,1): 1 − residual(smooth rank-parameterized curve) /
    residual(best straight line = PC1). 0 = straight, →1 = curved. Descriptive."""
    Xs = StandardScaler().fit_transform(X)
    p = PCA(1).fit(Xs)
    r_lin = float(np.mean((Xs - p.inverse_transform(p.transform(Xs))) ** 2))
    t = (np.asarray(y, float) - np.min(y)) / (np.ptp(y) + 1e-9)
    V = np.vander(t, degree + 1)                        # per-dim polynomial in rank
    coef, *_ = np.linalg.lstsq(V, Xs, rcond=None)
    r_curve = float(np.mean((Xs - V @ coef) ** 2))
    return float(max(0.0, 1.0 - r_curve / (r_lin + 1e-12)))


def curved_irreducible(X, y, pr_thresh=1.5, curve_thresh=0.3):
    """Curved-manifold signature for an OPEN (total-order) manifold: the order
    occupies effectively ≥~1.5 LINEAR dims (participation ratio, robust to the
    noise tail — a #PCs-for-90%-variance count is dominated by noise on high-D
    activations) yet is well described by a smooth 1-D curve parameterized by rank.
    So: a curved 1-D manifold, not a straight line (PR≈1) nor a genuine 2-D blob
    (no rank-curve ⇒ pcv≈0). Rings are handled by `circular.py`, not here.
    Returns {participation_ratio, principal_curve_curvature, curved}."""
    Xs = StandardScaler().fit_transform(X)
    ev = PCA().fit(Xs).explained_variance_
    pr = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))    # effective linear rank
    pcv = principal_curve_curvature(X, y)
    return {"participation_ratio": round(pr, 2), "principal_curve_curvature": round(pcv, 3),
            "curved": bool(pr >= pr_thresh and pcv > curve_thresh)}


def separability_index(P, seed=0, n_perm=20):
    """Engels-style irreducibility for a 2-D feature: compare the real joint
    nearest-neighbour structure to versions with the two coordinates independently
    permuted (destroys the joint, keeps marginals), AVERAGED over `n_perm` draws for
    stability. ~0 ⇒ separable (product of 1-D features); large ⇒ irreducible joint
    structure (e.g. a circle/curve). Note: under-powered for weak non-separability
    (a genuine grid sits near its own null) — compare against the null, don't headline
    a lone threshold."""
    from scipy.spatial import cKDTree
    P = np.asarray(P)[:, :2]
    if len(P) < 4:
        return float("nan")

    def mean_nn(Q):
        dist, _ = cKDTree(Q).query(Q, k=2)
        return float(np.mean(dist[:, 1]))

    real = mean_nn(P)
    vals = []
    for i in range(n_perm):
        rng = np.random.default_rng(seed + i)
        shuf = np.column_stack([P[rng.permutation(len(P)), 0], P[rng.permutation(len(P)), 1]])
        indep = mean_nn(shuf)
        vals.append((indep - real) / (indep + 1e-12))
    return float(np.mean(vals))


def _r(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None
