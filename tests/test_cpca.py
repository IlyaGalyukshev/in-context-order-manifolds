"""Correctness tests for contrastive PCA (M3) on SYNTHETIC foreground/background data.

These pin the property cPCA has and plain PCA lacks, before the code touches real activations:
  * recovery — when foreground and background SHARE high-variance nuisance but only foreground
    carries an extra signal along a known unit vector u, the top cPCA component finds u while
    PCA of the foreground alone is captured by the nuisance and misses it;
  * null — with foreground and background drawn from the SAME distribution (same nuisance basis,
    independent samples), the top contrastive eigenvalue is small relative to the raw variance
    and no component locks onto a fixed direction;
  * shapes/orthogonality — components are [n_comp, D] with unit, ~orthonormal rows and `project`
    returns [n, n_comp].
"""

import numpy as np

from icom.probes.cpca import cpca_components, project, alpha_spectrum


def _orthonormal(k, D, rng):
    """k orthonormal rows in R^D."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return Q.T[:k]                                                 # [k, D]


def _make_data(n, B, nuis_std, rng, u=None, sig_std=0.0, iso=0.3):
    """Draw n rows: Gaussian nuisance in the FIXED shared basis B [3, D] (per-axis std nuis_std),
    plus optional rank-1 signal along fixed unit u (foreground only), plus isotropic noise.
    B and u are passed in (not regenerated) so fg/bg share the same nuisance subspace."""
    D = B.shape[1]
    X = (rng.standard_normal((n, 3)) * np.asarray(nuis_std)) @ B + iso * rng.standard_normal((n, D))
    if sig_std > 0 and u is not None:
        X = X + (rng.standard_normal((n, 1)) * sig_std) * u[None, :]
    return X


def _signal_axis(B, rng):
    """A unit vector orthogonal to the nuisance basis B (a genuinely new direction)."""
    v = rng.standard_normal(B.shape[1])
    v -= B.T @ (B @ v)
    return v / np.linalg.norm(v)


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def test_cpca_recovers_fg_specific_direction_pca_misses():
    # fg and bg share the SAME high-variance nuisance basis B; fg alone also has a modest signal
    # along a fixed u (variance well below the top nuisance variance, so PCA is dominated by
    # nuisance). cPCA subtracts the shared nuisance and should surface u. Average over a few
    # realizations for a robust threshold.
    D, n = 64, 200
    nuis_std = [3.0, 2.2, 1.6]                                     # nuisance variance ~ 9, 4.8, 2.6
    sig_std = 1.8                                                  # signal variance ~ 3.2 << 9 (PCA still misses it)
    cpca_cos, pca_cos = [], []
    for r in range(5):
        rng = np.random.default_rng(1000 + r)
        B = _orthonormal(3, D, rng)                               # shared nuisance basis
        u = _signal_axis(B, rng)                                  # fixed fg-specific signal axis
        X_fg = _make_data(n, B, nuis_std, rng, u=u, sig_std=sig_std)
        X_bg = _make_data(n, B, nuis_std, rng)                    # background: nuisance only

        comps = cpca_components(X_fg, X_bg, n_comp=6, alpha=1.0)
        cpca_cos.append(_cos(comps[0], u))                        # top contrastive component vs u

        Xc = X_fg - X_fg.mean(0)                                  # plain PCA of fg alone
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        pca_cos.append(_cos(Vt[0], u))                            # PC1 of fg vs u

    m_cpca, m_pca = float(np.mean(cpca_cos)), float(np.mean(pca_cos))
    assert m_cpca > 0.8, (m_cpca, cpca_cos)                       # cPCA finds the fg-specific axis
    assert m_pca < 0.5, (m_pca, pca_cos)                          # PCA is captured by the nuisance
    assert m_cpca - m_pca > 0.3, (m_cpca, m_pca)                  # the load-bearing gap


def test_cpca_null_when_fg_and_bg_same_distribution():
    # fg and bg drawn from the SAME distribution (same nuisance basis B, independent samples, no
    # signal): the contrastive covariance is pure sampling noise. Its top eigenvalue must be small
    # relative to the raw foreground variance, and no component should lock onto a fixed axis u.
    D, n = 64, 250
    nuis_std = [3.0, 2.2, 1.6]
    ratios, aligns = [], []
    for r in range(5):
        rng = np.random.default_rng(2000 + r)
        B = _orthonormal(3, D, rng)
        u = _signal_axis(B, rng)                                  # a fixed direction that carries NO signal here
        X_fg = _make_data(n, B, nuis_std, rng)                   # same generator, independent draws
        X_bg = _make_data(n, B, nuis_std, rng)

        lam_top = alpha_spectrum(X_fg, X_bg, [1.0])[0]           # top contrastive eigenvalue
        var_top = np.linalg.eigvalsh(np.cov(X_fg, rowvar=False))[-1]  # top raw fg variance (~9)
        ratios.append(lam_top / var_top)

        comps = cpca_components(X_fg, X_bg, n_comp=6, alpha=1.0)
        aligns.append(max(_cos(c, u) for c in comps))           # best alignment to the fixed axis

    m_ratio, m_align = float(np.mean(ratios)), float(np.mean(aligns))
    assert m_ratio < 0.35, (m_ratio, ratios)                     # contrastive signal is small vs raw variance
    assert m_align < 0.5, (m_align, aligns)                      # no component locks onto the fixed axis


def test_cpca_shapes_and_orthonormal_rows():
    D, n_comp = 64, 6
    rng = np.random.default_rng(7)
    B = _orthonormal(3, D, rng)
    u = _signal_axis(B, rng)
    X_fg = _make_data(150, B, [3.0, 2.0, 1.5], rng, u=u, sig_std=1.5)
    X_bg = _make_data(150, B, [3.0, 2.0, 1.5], rng)
    comps = cpca_components(X_fg, X_bg, n_comp=n_comp, alpha=1.0)
    assert comps.shape == (n_comp, D)
    assert np.allclose(np.linalg.norm(comps, axis=1), 1.0, atol=1e-6)   # unit rows
    gram = comps @ comps.T                                              # ~orthonormal rows
    assert np.allclose(gram, np.eye(n_comp), atol=1e-6)
    scores = project(X_fg, comps)
    assert scores.shape == (X_fg.shape[0], n_comp)
    # project centers on the data's own mean -> per-component scores have ~zero mean
    assert np.allclose(scores.mean(0), 0.0, atol=1e-9)


def test_cpca_stable_when_D_far_exceeds_n():
    # D >> n: covariances are rank-deficient (rank <= n-1 << D); diagonalizing in the pooled span
    # must stay finite/stable AND still recover the fg-specific direction. With small n the shared
    # nuisance cancels only up to sampling noise, so the signal is made clearly dominant and the
    # cosine is averaged over a few realizations.
    D, n = 1024, 100                                              # D >> n, representative n ~ 100
    nuis_std = [3.0, 2.0, 1.5]
    sig_std = 3.0                                                 # signal variance ~ 9, dominates the residual
    coss = []
    for r in range(3):
        rng = np.random.default_rng(11 + r)
        B = _orthonormal(3, D, rng)
        u = _signal_axis(B, rng)
        comps = cpca_components(_make_data(n, B, nuis_std, rng, u=u, sig_std=sig_std),
                                _make_data(n, B, nuis_std, rng), n_comp=6, alpha=1.0)
        assert comps.shape == (6, D)
        assert np.all(np.isfinite(comps))                        # stable despite D >> n
        coss.append(_cos(comps[0], u))
    assert float(np.mean(coss)) > 0.8, coss                      # recovered despite D >> n
