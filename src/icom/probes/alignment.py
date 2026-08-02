"""Pretrained-axis reuse (Phase G10, CPU part): is the in-context BCS order axis
the SAME direction as a pretrained magnitude/size/space axis?

Both directions live in the same activation space (same model/layer), so they are
directly comparable even though the stimuli are unpaired:
  - contrast_direction : source axis = mean(positive) − mean(negative) activations
                         (CAA/RepE), unit-normalized. For magnitude, positives are
                         "large" and negatives "small" MATCHED number contrasts
                         (avoids the per-digit landmine — the signal is comparative).
  - rank_direction     : BCS order axis = the ridge rank-probe weight direction.
  - direction_cosine   : |cos| between two directions (0 = orthogonal, 1 = aligned).
  - subspace_alignment : mean cosine of principal angles between two column-spans
                         (unpaired subspace comparison).
Causal ablation and cross-domain transfer (train-on-source / test-on-BCS) reuse
`transfer_spearman`; ablation itself is GPU-side.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def contrast_direction(pos_acts, neg_acts):
    """CAA-style direction mean(pos) − mean(neg), unit-normalized."""
    d = np.asarray(pos_acts, float).mean(0) - np.asarray(neg_acts, float).mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def rank_contrast_direction(X, y, frac=0.33):
    """BCS order axis as a CAA-style contrast: mean(top-frac by rank) − mean(bottom
    -frac), raw space. This is the SAME estimator as `contrast_direction`, so their
    cosine is meaningful. PREFER THIS for source-axis alignment (a ridge weight is
    whitened by the activation covariance and is NOT comparable to a raw mean-diff
    — in an anisotropic residual stream the two can disagree even for the same axis)."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    k = max(1, int(len(y) * frac))
    order = np.argsort(y)
    d = X[order[-k:]].mean(0) - X[order[:k]].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def rank_direction(X, y, alpha=10.0):
    """The BCS order axis as the ridge rank-probe WEIGHT (raw space, unit-norm).
    NOTE: ridge is covariance-whitened, so this is NOT directly comparable to a raw
    mean-difference `contrast_direction` — for source-axis alignment use
    `rank_contrast_direction` (same operator on both sides) or whiten both."""
    w = Ridge(alpha=alpha).fit(np.asarray(X, float), y).coef_.ravel()
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def direction_cosine(a, b):
    """|cos| between two direction vectors (sign-invariant)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def subspace_alignment(A, B):
    """Mean cosine of principal angles between the column-spans of A and B
    (each [D, k]). 1 = coincident subspaces, 0 = orthogonal. Unpaired."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))
