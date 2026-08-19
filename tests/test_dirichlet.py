"""Smoke/correctness tests for the Dirichlet-energy (M5) probe on SYNTHETIC layouts.

Dirichlet energy over the relation graph asks: do the entities a stimulus *relates* (graph edges)
sit CLOSE in activation space? A representation that has folded the relation structure into geometry
has LOW normalized energy (edges are short relative to the average entity pair); a scrambled
placement of entities onto graph nodes has energy ≈ 1. These pins fix that sense before the probe
touches real activations:
  * a local (adjacent-on-a-line) graph reads LOW, random long-range edges read HIGH;
  * shuffling which entity sits at which node — the permutation null — RAISES the energy of a
    locally-smooth graph (denominator is shuffle-invariant, so the ratio moves the honest way);
  * the per-layer profile has the right length and stays finite.
"""

import numpy as np

from icom.probes.dirichlet import dirichlet_energy, dirichlet_profile


def _line(n=20, D=32, noise=0.01, seed=0):
    """n entities on a 1-D line embedded in D dims: X[i] = i·axis (+ tiny isotropic noise).
    Returns (X [n, D], pos [n])."""
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(D); axis /= np.linalg.norm(axis)
    pos = np.arange(n, dtype=float)
    X = pos[:, None] * axis[None, :] + noise * rng.standard_normal((n, D))
    return X, pos


def _random_edges(n, m, seed):
    rng = np.random.default_rng(seed)
    edges = []
    while len(edges) < m:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j:
            edges.append((i, j))
    return edges


def test_low_local_vs_high_longrange():
    # Adjacent edges on a smooth line connect near-neighbours -> LOW normalized energy;
    # rewiring the SAME point cloud to random long-range edges -> HIGH, by a clear margin.
    X, _ = _line(n=20, D=32, seed=0)
    adj = [(i, i + 1) for i in range(len(X) - 1)]
    e_adj = dirichlet_energy(X, adj, normalize=True)
    assert e_adj < 0.3, e_adj                                    # local graph -> low ratio

    e_rand = dirichlet_energy(X, _random_edges(len(X), 3 * len(X), seed=1), normalize=True)
    assert e_rand > 0.5, e_rand                                  # long-range edges ~ average pair
    assert e_rand - e_adj > 0.3, (e_rand, e_adj)                 # the load-bearing gap


def test_perm_null_raises_energy_on_smooth_graph():
    # Permuting which entity sits at which graph node scrambles a locally-smooth graph: adjacent
    # nodes now hold far-apart points, so energy rises. The denominator (mean over all pairs) is
    # permutation-invariant, so the ratio moves purely from the numerator.
    X, _ = _line(n=20, D=32, seed=2)
    adj = [(i, i + 1) for i in range(len(X) - 1)]
    e0 = dirichlet_energy(X, adj, normalize=True)

    rng = np.random.default_rng(3)
    perm = np.array([dirichlet_energy(X[rng.permutation(len(X))], adj, normalize=True)
                     for _ in range(200)])
    assert np.mean(perm) > e0, (float(np.mean(perm)), e0)        # shuffle raises energy on average
    assert np.mean(perm) > 3 * e0                                # clear separation from the smooth graph
    assert np.mean(perm > e0) > 0.9                              # nearly every shuffle is worse


def test_profile_shape_and_finite():
    rng = np.random.default_rng(4)
    N, L, D = 12, 7, 16
    X_layers = rng.standard_normal((N, L, D))                    # [entity, layer, feature]
    edges = [(i, i + 1) for i in range(N - 1)]
    prof = dirichlet_profile(X_layers, edges, normalize=True)
    assert prof.shape == (L,)
    assert np.all(np.isfinite(prof))


def test_guards_and_raw_energy():
    X, _ = _line(n=6, D=8, seed=5)
    assert np.isnan(dirichlet_energy(X, [], normalize=True))           # empty edges -> nan
    assert np.isnan(dirichlet_energy(X[:1], [(0, 0)], normalize=True)) # N<2 -> zero-pair denom -> nan
    # unnormalized energy is the raw Σ ||X[i]-X[j]||²: adjacent unit-spaced edges ≈ (n-1)·1
    adj = [(i, i + 1) for i in range(len(X) - 1)]
    raw = dirichlet_energy(X, adj, normalize=False)
    assert abs(raw - (len(X) - 1)) < 0.5, raw
