"""Positional-subspace estimation and projection-out.

The positional subspace is estimated from matched control prompts (same card
template, content-free or content-permuted filler) — NOT from the experimental
stimuli themselves. Card-slot mean vectors across many control prompts →
top-k PCs = positional subspace. Experimental points are projected onto its
orthogonal complement before curve fitting; quality is reported before/after
projection so the size of the positional component is itself a result.
"""

from __future__ import annotations


def estimate_position_subspace(control_pooled_acts, k: int = 4):
    raise NotImplementedError


def project_out(points, subspace):
    raise NotImplementedError
