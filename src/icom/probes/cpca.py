"""Contrastive PCA (cPCA) — M3 of the v5 plan: isolate a "coherence subspace".

Plain PCA of the REAL (coherent-order) activations returns whatever varies most — and in
the residual stream that is almost always shared, order-irrelevant nuisance (token identity,
card content, positional energy) rather than the thin structure we care about. cPCA (Abid,
Zhang, Zou, Athey 2018) removes exactly that shared nuisance by contrasting against a
BACKGROUND: it returns the top eigenvectors of the CONTRASTIVE covariance

    C = C_fg - alpha * C_bg,

where C_fg is the (mean-centered) covariance of the foreground (real activations) and C_bg
that of the background (the content-matched TWIN — same cards, coherence broken). Directions
with large variance in fg but comparable variance in bg cancel; only the fg-SPECIFIC variance
survives. Setting alpha=0 recovers plain PCA of fg; larger alpha penalizes background variance
more aggressively (see `alpha_spectrum` to pick it).

D >> n handling. In our regime D≈4096 with only n≈100 reads per side, C_fg and C_bg are each
rank-deficient (rank <= n-1) and the full D×D contrastive covariance is neither storable nor
stably diagonalizable. But C = C_fg - alpha*C_bg has all its nonzero eigenvalues INSIDE the
span of the pooled centered rows (its null space is the orthogonal complement, eigenvalue 0
exactly). So we build an orthonormal basis P (d = min(D, n_fg+n_bg-2) rows) for that pooled
span via a thin SVD, diagonalize the small d×d contrastive covariance P C Pᵀ, and lift the
eigenvectors back to D. This is EXACT for the nonzero spectrum (no dimension is discarded that
could have carried contrastive signal), and the lifted component rows are orthonormal by
construction. Refs: Abid et al. 2018 (contrastive PCA); cf. the span trick in crossnobis._reduce_for_cov.
"""

from __future__ import annotations

import numpy as np


def _cov_in_span(X_fg: np.ndarray, X_bg: np.ndarray):
    """Foreground/background covariances expressed in an orthonormal basis of the pooled span.

    Centers each side on ITS OWN mean (a covariance is about spread, not location), stacks the
    two centered blocks, and takes the right singular vectors as an orthonormal basis P [d, D]
    of everything the two clouds span (d = min(D, n_fg+n_bg-2) after dropping the two mean d.o.f.).
    Returns (C_fg_small, C_bg_small, P): the two d×d covariances P C Pᵀ (symmetric, ddof=1) and P.
    """
    Xf = np.asarray(X_fg, dtype=np.float64)
    Xb = np.asarray(X_bg, dtype=np.float64)
    Xf = Xf - Xf.mean(axis=0, keepdims=True)                       # center each cloud on its own mean
    Xb = Xb - Xb.mean(axis=0, keepdims=True)
    n_fg, D = Xf.shape
    n_bg = Xb.shape[0]
    M = np.vstack([Xf, Xb])                                        # [n_fg+n_bg, D], pooled centered rows
    # orthonormal basis of the pooled row space; drop numerically-zero singular directions
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    tol = max(M.shape) * np.finfo(np.float64).eps * (S[0] if S.size else 0.0)
    d = int(min((S > tol).sum(), n_fg + n_bg - 2, D))
    d = max(d, 1)
    P = Vt[:d]                                                     # [d, D], orthonormal rows
    Af = Xf @ P.T                                                  # [n_fg, d] fg coords in the span
    Ab = Xb @ P.T                                                  # [n_bg, d] bg coords in the span
    C_fg = (Af.T @ Af) / max(n_fg - 1, 1)                          # d×d, ddof=1
    C_bg = (Ab.T @ Ab) / max(n_bg - 1, 1)
    C_fg = 0.5 * (C_fg + C_fg.T)                                   # symmetrize away round-off
    C_bg = 0.5 * (C_bg + C_bg.T)
    return C_fg, C_bg, P


def cpca_components(X_fg: np.ndarray, X_bg: np.ndarray, n_comp: int = 6,
                    alpha: float = 1.0) -> np.ndarray:
    """Contrastive PCA components: top eigenvectors of C_fg - alpha*C_bg.

    X_fg: [n_fg, D] foreground (real) activations; X_bg: [n_bg, D] background (twin) activations.
    Returns components: [n_comp, D], rows unit-norm and (up to round-off) mutually orthonormal,
    SORTED by descending contrastive eigenvalue — component 0 is the most fg-specific direction.
    Diagonalization happens in the pooled span (see `_cov_in_span`), so it is stable for D>>n;
    n_comp is clipped to the available span dimension d = min(D, n_fg+n_bg-2)."""
    C_fg, C_bg, P = _cov_in_span(X_fg, X_bg)
    C = C_fg - float(alpha) * C_bg                                 # d×d contrastive covariance
    C = 0.5 * (C + C.T)
    d = C.shape[0]
    n_comp = int(min(n_comp, d))
    evals, evecs = np.linalg.eigh(C)                               # ascending, real (C symmetric)
    order = np.argsort(evals)[::-1][:n_comp]                       # top contrastive eigenvalues
    W = evecs[:, order]                                            # [d, n_comp], orthonormal columns
    comps = W.T @ P                                                # [n_comp, D], lift to full space
    norms = np.linalg.norm(comps, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return comps / norms                                          # unit rows


def project(X: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project X onto cPCA components. X: [n, D] centered on ITS OWN mean, then dotted onto
    each component. components: [n_comp, D]. Returns [n, n_comp] contrastive-subspace scores."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc @ np.asarray(components, dtype=np.float64).T


def alpha_spectrum(X_fg: np.ndarray, X_bg: np.ndarray, alphas) -> np.ndarray:
    """Top contrastive eigenvalue of C_fg - alpha*C_bg for each alpha (helper to pick alpha).

    Returns lam_top: [len(alphas)]. It decreases monotonically in alpha (background variance is
    penalized harder); a useful alpha is typically where lam_top stops tracking the raw fg
    variance and the surviving direction has become fg-SPECIFIC rather than shared-dominant.
    The pooled-span basis is built once and reused across all alphas."""
    C_fg, C_bg, _ = _cov_in_span(X_fg, X_bg)
    out = []
    for a in np.asarray(alphas, dtype=np.float64).ravel():
        C = C_fg - float(a) * C_bg
        C = 0.5 * (C + C.T)
        out.append(float(np.linalg.eigvalsh(C)[-1]))              # largest eigenvalue
    return np.asarray(out, dtype=np.float64)
