"""Generator invariants — the contract the implementation must satisfy.

Skipped until the generator lands; each test encodes a design commitment.
"""

import pytest

pytestmark = pytest.mark.skip(reason="generator not implemented yet (Stage 0)")


def test_conditions_share_content_and_questions():
    """All four conditions of one latent order: same cards (as a set), same
    question battery, same token count; only presentation slots differ."""


def test_fixed_token_length_across_items():
    """Within a stimulus, every card has the same token count (fixed-width markers)."""


def test_relational_family_needs_transitive_closure():
    """RELATIONAL stimuli contain only adjacent-pair statements: no card
    reveals a non-adjacent relation or an absolute coordinate."""


def test_pairwise_distance_stratification():
    """Pairwise questions meet the per-bin quota; bins cover 1 .. N-1."""


def test_presentation_slots_recorded():
    """Every card carries latent_rank AND presentation_slot; in SHUFFLE the
    rank-slot Spearman correlation is ~0 by construction (checked per seed)."""


def test_determinism():
    """Same (config, seed) → byte-identical stimuli and questions."""
