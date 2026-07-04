"""The 2×2 readout — the project's primary measurement.

For a coordinate t over N items with latent ranks r and presentation slots s:
  rho_latent    = spearman(t, r)     — confounded with position in fwd/rev
  rho_position  = spearman(t, s)
  q_content_partial  = spearman(t, r | s)   ← THE dependent variable
  q_position_partial = spearman(t, s | r)
Signs are direction-arbitrary (PC1 has no canonical orientation), so absolute
values are what cross-stimulus aggregation uses; signed values are kept for
within-stimulus diagnostics.

bootstrap_sd resamples items to give a per-stimulus noise scale for the
partial — the reliability guard for Track A's regression.
"""

from __future__ import annotations

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def _pearson(a, b) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def partial_spearman(t, r, s) -> float:
    """spearman(t, r) controlling s, via partial Pearson on ranks."""
    tr, rr, sr = _rank(t), _rank(r), _rank(s)
    r_tr, r_ts, r_rs = _pearson(tr, rr), _pearson(tr, sr), _pearson(rr, sr)
    den = np.sqrt((1 - r_ts**2) * (1 - r_rs**2))
    return float((r_tr - r_ts * r_rs) / den) if den > 1e-9 else 0.0


def quality_readout(t: np.ndarray, ranks: np.ndarray, slots: np.ndarray) -> dict:
    tr, rr, sr = _rank(t), _rank(ranks), _rank(slots)
    return {
        "rho_latent": _pearson(tr, rr),
        "rho_position": _pearson(tr, sr),
        "q_content_partial": partial_spearman(t, ranks, slots),
        "q_position_partial": partial_spearman(t, slots, ranks),
    }


def bootstrap_sd(points: np.ndarray, ranks: np.ndarray, slots: np.ndarray,
                 n_boot: int = 50, seed: int = 0) -> float:
    """SD of q_content_partial under item resampling (fit + readout per draw)."""
    from icom.geometry.curve import fit_coordinate

    rng = np.random.default_rng(seed)
    n = len(ranks)
    vals = []
    for _ in range(n_boot):
        ix = rng.choice(n, size=n, replace=True)
        if len(np.unique(ix)) < 4:
            continue
        t, _ = fit_coordinate(points[ix])
        vals.append(partial_spearman(t, ranks[ix], slots[ix]))
    return float(np.std(vals)) if vals else float("nan")
