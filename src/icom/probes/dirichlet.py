"""Dirichlet energy of activations over the stimulus relation graph — M5 (Park-et-al bridge).

Park et al. read a concept's geometry off how its *comparability graph* embeds: a representation
that has folded the relation structure into geometry places RELATED entities (graph edges) close
together. The Dirichlet energy of a signal on a graph, E = Σ_edges ||X[i] - X[j]||², is exactly
that smoothness functional — small when edges connect nearby points, large when they span the cloud.

We report the SCALE-FREE ratio: divide the raw edge energy by (n_edges · mean squared distance over
ALL entity pairs). That denominator is the energy an "average" edge would carry, so the ratio lives
in ~[0, 1] and is comparable across stimuli / layers / models regardless of the activation norm:
  * ≈ 0  — edges connect near-identical points (relation structure is geometrically smooth);
  * ≈ 1  — edges are no shorter than random pairs (structure is NOT reflected in geometry);
  * > 1  — edges are LONGER than average (an anti-smooth / scrambled placement).
The ratio is permutation-diagnostic: shuffling which entity sits at which graph node leaves the
denominator (a function of the point cloud, not the labelling) fixed while raising the numerator,
so a locally-smooth graph reads well ABOVE its own shuffle null.

Convention: per-entity vectors are indexed [entity, ...] to match the extractor's [N, L+1, D]
records — `dirichlet_energy` takes ONE layer [N, D]; `dirichlet_profile` takes [N, L, D] (entity,
layer, feature) so a caller can hand it `r["X"]` directly and get a per-layer profile of length L.

Refs: Park et al. (relational/comparability geometry); graph Dirichlet energy / Laplacian smoothness.
"""

from __future__ import annotations

import numpy as np


def _mean_pair_sq(X: np.ndarray) -> float:
    """Mean squared Euclidean distance over all unordered entity pairs of X [N, D].
    Returns nan for N < 2 (no pairs)."""
    N = X.shape[0]
    if N < 2:
        return float("nan")
    diffs = X[:, None, :] - X[None, :, :]                 # [N, N, D]
    sq = np.einsum("ijk,ijk->ij", diffs, diffs)           # [N, N] squared distances
    iu = np.triu_indices(N, 1)
    return float(sq[iu].mean())


def dirichlet_energy(X: np.ndarray, edges, normalize: bool = True) -> float:
    """Dirichlet energy of per-entity vectors X [N, D] over the relation graph `edges`.

    edges: iterable of (i, j) row-index pairs (entity indices into X) = the relation-graph edges.
    Raw energy E = Σ_(i,j)∈edges ||X[i] - X[j]||². If `normalize`, divide by
    (n_edges · mean squared distance over ALL entity pairs) → a scale-free ratio in ~[0, 1] where
    a graph whose edges connect nearby points is LOW. Empty edges, N < 2, or a zero/non-finite
    denominator all return nan."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be [N, D]; got shape {X.shape}")
    E = np.asarray(edges, dtype=int)
    if E.size == 0:
        return float("nan")
    E = E.reshape(-1, 2)
    ei, ej = E[:, 0], E[:, 1]
    raw = float(((X[ei] - X[ej]) ** 2).sum())             # Σ_edges ||X[i]-X[j]||²
    if not normalize:
        return raw
    denom = len(E) * _mean_pair_sq(X)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return raw / denom


def dirichlet_profile(X_layers: np.ndarray, edges, normalize: bool = True) -> np.ndarray:
    """Per-layer Dirichlet energy. X_layers is [N, L, D] (entity, layer, feature) — the same
    axis order as the extractor's per-stimulus records — so `dirichlet_profile(r["X"], edges)`
    works directly. Returns a length-L array, energy[layer] = dirichlet_energy(X_layers[:, layer],
    edges, normalize)."""
    X_layers = np.asarray(X_layers, dtype=np.float64)
    if X_layers.ndim != 3:
        raise ValueError(f"X_layers must be [N, L, D]; got shape {X_layers.shape}")
    L = X_layers.shape[1]
    return np.array([dirichlet_energy(X_layers[:, layer, :], edges, normalize)
                     for layer in range(L)])
