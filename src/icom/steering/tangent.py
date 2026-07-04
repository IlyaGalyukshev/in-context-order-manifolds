"""Tangent estimation with cross-stimulus regularization.

A tangent from one prompt's ≤48 points is too noisy to steer with. Stimuli
sharing (model, layer, family, condition, N) are Procrustes-aligned on
rank-matched points; the curve is fit on the pooled cloud and local tangents
are read off the shared fit, then mapped back per stimulus.
"""

from __future__ import annotations


def shared_tangents(pooled_points_by_stimulus, latent_ranks_by_stimulus):
    raise NotImplementedError
