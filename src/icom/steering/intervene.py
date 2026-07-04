"""Steering hooks and the control ladder.

h ← h + α·v at a swept layer (independent of the read-out layer ℓ*), applied
at the target item's card tokens. Directions, in increasing stringency of the
comparison they support:
  along        — local tangent of the shared order-curve fit
  random       — matched-norm random direction (floor; expected inert in 4096-D)
  ortho_sub    — orthogonal to tangent but INSIDE the manifold's top-k PCA
                 subspace (the real control: same subspace, wrong direction)
  foreign      — tangent transplanted from a different stimulus/condition

Headline test is dose–response: sweep α and check the signed prediction that
the item's answered rank drifts monotonically by whole positions, pairwise
answers involving it flip in the predicted direction, and parse-failure rate
stays flat (vs rising under off-manifold pushes).
"""

from __future__ import annotations

DIRECTIONS = ("along", "random", "ortho_sub", "foreign")


def make_steering_hook(direction_vector, alpha: float, token_slice):
    raise NotImplementedError
