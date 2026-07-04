"""Reliability of the per-example quality score.

Track A regresses accuracy on quality per example; if quality is unreliable,
the regression is noise-on-noise and the estimated slope attenuates. Split-half
(odd/even items, Spearman-Brown corrected) and bootstrap-over-items reliability
are computed per (model, layer, pooling, family, condition, N) cell and
reported alongside every regression.
"""

from __future__ import annotations


def split_half_reliability(points, latent_ranks, presentation_slots, n_splits: int = 50) -> float:
    raise NotImplementedError
