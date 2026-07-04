"""Positional-subspace estimation and projection-out (cross-stimulus).

Because entity content is randomized across stimuli, averaging pooled vectors
BY PRESENTATION SLOT across many stimuli cancels content and leaves the
positional component: slot prototypes [n_slots, D]. Their top-k PCs span the
positional subspace; projecting it out of each stimulus's points removes the
trivially-positional geometry. Quality is reported before AND after — the
size of the drop measures how positional the raw manifold was.
"""

from __future__ import annotations

import numpy as np


def slot_prototype_subspace(points_by_stim: list[np.ndarray],
                            slots_by_stim: list[np.ndarray], k: int = 3) -> np.ndarray:
    """points [N,D] float per stimulus → orthonormal basis [k, D]."""
    n_slots = int(max(s.max() for s in slots_by_stim))
    d = points_by_stim[0].shape[1]
    sums = np.zeros((n_slots, d)); counts = np.zeros(n_slots)
    for P, S in zip(points_by_stim, slots_by_stim):
        for p, s in zip(P, S):
            sums[s - 1] += p; counts[s - 1] += 1
    protos = sums / np.maximum(counts, 1)[:, None]
    protos -= protos.mean(axis=0)
    _, _, Vt = np.linalg.svd(protos, full_matrices=False)
    return Vt[:k]


def project_out(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    X = points.astype(np.float64)
    return X - (X @ basis.T) @ basis
