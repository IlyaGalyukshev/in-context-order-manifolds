"""Nonce entity vocabulary.

Requirements:
- Entities must be novel (no pretraining-recoverable order), pronounceable,
  and tokenize into 1–3 tokens on every model in the roster; run
  `check_tokenization` against each tokenizer before generating a dataset.
- Entities are sampled without replacement per stimulus and counterbalanced
  across stimuli so no entity is systematically bound to a rank.
"""

from __future__ import annotations


def build_vocabulary(seed: int, size: int = 500) -> list[str]:
    """Generate nonce entity names (CVC-syllable compounds, e.g. 'glemb', 'snorvic')."""
    raise NotImplementedError


def check_tokenization(vocab: list[str], tokenizer, max_tokens: int = 3) -> list[str]:
    """Return the subset of `vocab` that tokenizes into <= max_tokens pieces."""
    raise NotImplementedError
