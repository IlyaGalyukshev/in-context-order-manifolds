"""Depth dynamics: turn a per-layer profile into the "which layers" answer.

Onset (first significant layer), peak (max decodability), and the longest
contiguous significant band — each reported as a FRACTION of depth so models of
different depth are comparable. Significant = score exceeds the per-layer null95.
"""

from __future__ import annotations

import numpy as np


def _longest_run(mask):
    """(start, end) inclusive of the longest True run, or (None, None)."""
    best_len, best = 0, (None, None)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 > best_len:
                best_len, best = j - i + 1, (i, j)
            i = j + 1
        else:
            i += 1
    return best


def depth_stats(scores, null95):
    """scores, null95: length-(L+1) arrays (layer 0 = embeddings .. L = final).
    Returns onset/peak/band as layer indices and depth fractions."""
    scores = np.asarray(scores, dtype=float)
    null95 = np.asarray(null95, dtype=float)
    L = len(scores) - 1
    frac = (lambda i: float(i) / L) if L > 0 else (lambda i: 0.0)
    sig = scores > null95
    onset = next((i for i in range(len(sig)) if sig[i]), None)
    peak = int(np.nanargmax(scores)) if np.isfinite(scores).any() else None
    a, b = _longest_run(sig)
    return {
        "n_layers": L + 1,
        "onset_layer": onset,
        "onset_frac": None if onset is None else round(frac(onset), 3),
        "peak_layer": peak,
        "peak_frac": None if peak is None else round(frac(peak), 3),
        "peak_score": None if peak is None else round(float(scores[peak]), 3),
        "band_layers": None if a is None else [int(a), int(b)],
        "band_frac": None if a is None else [round(frac(a), 3), round(frac(b), 3)],
        "n_significant": int(np.nansum(sig)),
    }
