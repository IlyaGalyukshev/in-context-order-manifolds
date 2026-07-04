"""Stimulus construction: 3 families × 4 conditions from one latent order.

Non-negotiables (see README "Design commitments"):
- The three families share identical latent orders and card templates; they
  differ ONLY in how order information is carried (dates / tags / relations).
- Fixed-width surface forms ("Day 03", "Tag 07") so token length is constant
  across items and conditions — no effect may be attributable to length.
- Conditions permute card order only; content and questions are frozen.
- Every card records its presentation slot for the position/content
  decomposition downstream.
- RELATIONAL family: emit only adjacent-pair statements ("the glemb wobbled
  before the drane ..."), so the global order is constructible only by
  transitive closure in-context.
"""

from __future__ import annotations

from icom.generator.schemas import Condition, Stimulus, StimulusFamily


def make_stimulus(
    family: StimulusFamily,
    condition: Condition,
    n_items: int,
    seed: int,
    vocab: list[str],
) -> Stimulus:
    raise NotImplementedError


def make_condition_set(
    family: StimulusFamily, n_items: int, seed: int, vocab: list[str]
) -> dict[Condition, Stimulus]:
    """All four conditions of the SAME content — the unit of counterbalancing."""
    raise NotImplementedError
