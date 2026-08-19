"""Smoke/correctness tests for the crossnobis (M2) math on SYNTHETIC repeat-reads.

These pin the two properties the raw RDM lacks, before the code touches real activations:
  * unbiasedness — two items with an IDENTICAL true representation get crossnobis ≈ 0, while a
    naive (non-cross-validated) squared distance is inflated by noise;
  * structure recovery — items laid on a line (resp. ring) give high whitened-RSA to the line
    (resp. ring) ideal RDM, and a structureless "twin" gives ≈ 0.
"""

import numpy as np

from icom.probes.crossnobis import (crossnobis_rdm, line_rdm, ring_rdm, noise_sd,
                                     whitened_rsa)


def _line_reads(n=12, k=8, D=64, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(D); axis /= np.linalg.norm(axis)
    pos = np.arange(n, dtype=float)                       # true 1-D positions == ranks
    mu = pos[:, None] * axis[None, :]                     # [n, D]
    reads = mu[:, None, :] + noise * rng.standard_normal((n, k, D))
    return reads, pos


def test_crossnobis_unbiased_on_identical_items():
    # Unbiasedness is a statement about the EXPECTATION: two items with one true mean and
    # independent noise get E[crossnobis] = 0, whereas the naive (non-CV) squared distance is
    # inflated by noise on every draw. Average over realizations to see the bias vanish.
    D, k, R = 64, 8, 60
    cnb, raw = [], []
    for r in range(R):
        rng = np.random.default_rng(100 + r)
        mu = rng.standard_normal(D)
        reads = np.stack([mu + rng.standard_normal((k, D)),
                          mu + rng.standard_normal((k, D))])       # [2, k, D]
        cnb.append(crossnobis_rdm(reads, n_splits=40, seed=r)[0, 1])
        z = reads / noise_sd(reads)
        raw.append(float(((z[0].mean(0) - z[1].mean(0)) ** 2).sum() / D))
    mcnb, mraw = float(np.mean(cnb)), float(np.mean(raw))
    assert abs(mcnb) < 0.05, mcnb                                   # crossnobis centered on 0 (unbiased)
    assert mraw > 0.2                                              # the naive distance is biased upward
    assert abs(mcnb) < 0.25 * mraw, (mcnb, mraw)                   # and crossnobis removes most of it


def test_crossnobis_line_recovery_and_twin_null():
    reads, pos = _line_reads(n=12, k=8, D=64, noise=1.0, seed=0)
    rdm = crossnobis_rdm(reads, n_splits=50, seed=0)
    rsa_line = whitened_rsa(rdm, line_rdm(pos))
    assert rsa_line > 0.7, rsa_line                              # a line is clearly recovered
    # twin: structureless reads (no position signal) -> RSA ≈ 0
    rng = np.random.default_rng(9)
    twin = rng.standard_normal((12, 8, 64))
    rsa_twin = whitened_rsa(crossnobis_rdm(twin, n_splits=50, seed=0), line_rdm(pos))
    assert abs(rsa_twin) < 0.2, rsa_twin
    assert rsa_line - rsa_twin > 0.5                            # real >> twin, the load-bearing gap


def test_crossnobis_ring_beats_line_on_a_ring():
    # items on a circle: cyclic distance is recovered, linear distance is not.
    n, k, D = 12, 8, 64
    rng = np.random.default_rng(2)
    U = rng.standard_normal((2, D))                            # 2 orthonormal-ish plane axes
    U[1] -= (U[1] @ U[0]) / (U[0] @ U[0]) * U[0]
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    ang = 2 * np.pi * np.arange(n) / n
    mu = np.cos(ang)[:, None] * U[0] + np.sin(ang)[:, None] * U[1]
    reads = mu[:, None, :] + 0.4 * rng.standard_normal((n, k, D))
    rdm = crossnobis_rdm(reads, n_splits=50, seed=0)
    ranks = np.arange(1, n + 1)
    rsa_ring = whitened_rsa(rdm, ring_rdm(ranks, n))
    rsa_line = whitened_rsa(rdm, line_rdm(ranks))
    assert rsa_ring > 0.7, rsa_ring
    assert rsa_ring - rsa_line > 0.2, (rsa_ring, rsa_line)      # ring tracks cyclic ≫ linear


def test_crossnobis_multivariate_runs_and_recovers_line():
    reads, pos = _line_reads(n=12, k=8, D=64, noise=1.0, seed=5)
    rdm = crossnobis_rdm(reads, n_splits=40, multivariate=True, mv_dim=24, seed=0)
    assert np.allclose(rdm, rdm.T) and np.allclose(np.diag(rdm), 0.0)
    assert whitened_rsa(rdm, line_rdm(pos)) > 0.6
