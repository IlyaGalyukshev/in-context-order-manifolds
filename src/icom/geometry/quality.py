"""The 2×2 readout — the project's primary measurement.

Given the arc-length coordinate, latent ranks, and presentation slots:
  rho_latent, rho_position, and both partial Spearman correlations.
q_content_partial = rho(coord, latent | slot) is THE dependent variable:
- In SHUFFLE it identifies content-order structure (slot ⟂ rank there).
- In FORWARD/REVERSE raw rho_latent is uninterpretable (|rho(slot, rank)| = 1);
  only cross-condition and position-projected comparisons are meaningful.

Also distinguishes the three Shuffle outcomes explicitly:
  (i) content-order manifold intact  → q_content_partial high
  (ii) positional manifold only      → rho_position high, q_content_partial ≈ 0
  (iii) genuine fragmentation        → both ≈ 0, curve diagnostics poor
The plan's original "fragmented" conflated (ii) and (iii); we do not.
"""

from __future__ import annotations

from icom.generator.schemas import GeometryRow


def quality_readout(coordinate, latent_ranks, presentation_slots) -> dict:
    raise NotImplementedError


def build_geometry_row(stimulus, model: str, layer: int, pooling: str, points) -> GeometryRow:
    raise NotImplementedError
