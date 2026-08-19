"""Crossnobis (cross-validated Mahalanobis) RDM + whitened-RSA — M2 of the v5 plan.

Given k independent repeat-reads per item (from KV-branch extraction, M1), build an UNBIASED
representational dissimilarity matrix: a cross-validated squared distance on noise-normalized
activations. Two properties the raw cosine/Euclidean RDM lacks, and that the ad-hoc remove-top-k
hack only approximates:
  * E[d_ii] = 0  — a distance between an item and itself is unbiasedly zero (cross-validation
    across independent reads removes the positive bias that noise adds to any non-CV distance);
  * low-variance axes are not swamped by high-variance ones — noise normalization whitens by the
    per-dimension noise scale, so a thin ordinal axis is measured on equal footing.
Refs: Walther et al. 2016; Diedrichsen & Kriegeskorte 2017; Diedrichsen et al. 2021 (whitened RDM).

Noise normalization defaults to UNIVARIATE (diagonal): our regime is D≈4096 with only n·k≈100
reads, where a full multivariate D×D covariance is not estimable — univariate normalization (divide
each dim by its residual SD) is the robust standard there. `multivariate=True` adds Ledoit-Wolf
shrinkage covariance computed in a PCA-reduced space, as a robustness option.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def noise_sd(reads: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-dimension noise SD from within-item residuals across reads.
    reads: [n_items, k_reads, D]. Returns sd: [D] (pooled across items and reads, ddof=1 in k)."""
    reads = np.asarray(reads, dtype=np.float64)
    resid = reads - reads.mean(axis=1, keepdims=True)          # remove each item's mean read
    n, k, D = reads.shape
    var = (resid ** 2).sum(axis=(0, 1)) / max(n * (k - 1), 1)  # ddof=1 per item, pooled
    return np.sqrt(var) + eps


def _reduce_for_cov(reads_z: np.ndarray, max_dim: int):
    """PCA-reduce the (already univariate-whitened) reads for a stable multivariate covariance.
    reads_z: [n, k, D] -> [n, k, d], d = min(max_dim, n*k-1, D)."""
    n, k, D = reads_z.shape
    X = reads_z.reshape(n * k, D)
    Xc = X - X.mean(0)
    d = int(min(max_dim, n * k - 1, D))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:d]                                                  # [d, D]
    return (Xc @ P.T).reshape(n, k, d)


def _lw_precision(resid2d: np.ndarray):
    """Ledoit-Wolf shrinkage precision (Σ⁻¹) from residuals [m, d] toward a diagonal target."""
    m, d = resid2d.shape
    S = (resid2d.T @ resid2d) / max(m - 1, 1)
    mu = np.trace(S) / d
    target = mu * np.eye(d)
    # shrinkage intensity (Ledoit-Wolf, simplified): var of entries / squared off-diag distance
    d2 = ((S - target) ** 2).sum()
    b2 = 0.0
    for row in resid2d:
        b2 += ((np.outer(row, row) - S) ** 2).sum()
    b2 = b2 / (m ** 2)
    lam = float(np.clip((b2 / d2) if d2 > 0 else 1.0, 0.0, 1.0))
    Sigma = lam * target + (1 - lam) * S
    return np.linalg.pinv(Sigma), lam


def crossnobis_rdm(reads: np.ndarray, n_splits: int = 20, multivariate: bool = False,
                   mv_dim: int = 48, seed: int = 0) -> np.ndarray:
    """Unbiased cross-validated RDM from repeat-reads. reads: [n_items, k_reads, D] (k>=2).
    Returns rdm: [n_items, n_items], symmetric, E[diag]≈0 (may be slightly negative from noise).
    Estimator: for many random balanced splits of the k reads into folds A,B,
      d_ij += <mA_i - mA_j, Winv (mB_i - mB_j)> / D_eff ; averaged over splits (cross-validation
    across the independent A/B folds makes it unbiased)."""
    reads = np.asarray(reads, dtype=np.float64)
    n, k, D = reads.shape
    if k < 2:
        raise ValueError("crossnobis needs k>=2 reads per item")
    z = reads / noise_sd(reads)                                # univariate noise normalization
    if multivariate:
        z = _reduce_for_cov(z, mv_dim)
        d = z.shape[2]
        resid = (z - z.mean(axis=1, keepdims=True)).reshape(n * k, d)
        Winv, _ = _lw_precision(resid)
    else:
        Winv = None
    Deff = z.shape[2]
    rng = np.random.default_rng(seed)
    acc = np.zeros((n, n))
    ka = k // 2
    for _ in range(n_splits):
        perm = rng.permutation(k)
        A, B = perm[:ka], perm[ka:2 * ka]                      # disjoint, equal-size folds
        mA = z[:, A, :].mean(axis=1)                           # [n, d]
        mB = z[:, B, :].mean(axis=1)
        dA = mA[:, None, :] - mA[None, :, :]                   # [n, n, d]
        dB = mB[:, None, :] - mB[None, :, :]
        if Winv is not None:
            dB = dB @ Winv.T
        acc += (dA * dB).sum(axis=2) / Deff
    rdm = acc / n_splits
    rdm = 0.5 * (rdm + rdm.T)                                   # symmetrize
    np.fill_diagonal(rdm, 0.0)
    return rdm


# ---- ideal RDMs the crossnobis RDM is compared against ----
def line_rdm(ranks: np.ndarray) -> np.ndarray:
    r = np.asarray(ranks, dtype=np.float64)
    return np.abs(r[:, None] - r[None, :])


def ring_rdm(ranks: np.ndarray, N: int) -> np.ndarray:
    r = np.asarray(ranks, dtype=np.float64)
    dif = np.abs(r[:, None] - r[None, :])
    return np.minimum(dif, N - dif)


def whitened_rsa(rdm: np.ndarray, ideal: np.ndarray, method: str = "spearman") -> float:
    """RSA between a (crossnobis) RDM and an ideal RDM, over the upper triangle."""
    iu = np.triu_indices(rdm.shape[0], 1)
    a, b = rdm[iu], ideal[iu]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(a, b)[0])
    a2, b2 = a - a.mean(), b - b.mean()                        # cosine (whitened-RDM similarity)
    return float((a2 @ b2) / (np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-12))
