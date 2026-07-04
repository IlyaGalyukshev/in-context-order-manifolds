"""Question battery: five families, deterministic keys from the latent order.

Generation rules:
- Pairwise questions are NOT sampled uniformly — stratify by rank distance
  d = |rank(A) - rank(B)| with a fixed quota per bin, so accuracy-vs-distance
  curves are estimable (the local-vs-global signature).
- Adjacency/rank questions tag endpoints separately (anchors behave differently).
- Span questions place the queried window at start / middle / end in equal
  proportion, so middle-of-list degradation is measured, not inferred.
- The identical question set is attached to every condition of the same content.
- Output format is aggressively constrained ("answer with only the entity
  name"); each family ships a deterministic parser + fallback regex. Parse
  failures are a separate category, never scored as wrong. Pairwise yes/no is
  additionally scored from logits.
"""

from __future__ import annotations

from icom.generator.schemas import Question


def make_battery(latent_order: list[str], n_distance_bins: int, seed: int) -> list[Question]:
    raise NotImplementedError
