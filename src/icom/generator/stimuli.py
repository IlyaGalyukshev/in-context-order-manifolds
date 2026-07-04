"""Stimulus construction: 3 families × conditions from one latent order.

Design invariants (tested in tests/test_generator.py):
- The families share the identical latent order and entity/predicate content;
  they differ ONLY in how order information is carried:
    dated      "Day 07: the glemb polished a metal plate."   (temporal scaffold)
    tagged     "Tag 07: the glemb polished a metal plate."   (numeric, atemporal)
    relational "The glemb polished a metal plate before the drane vented a
                steam pipe."  (adjacent pairs only — global order exists solely
                via in-context transitive closure)
  dated/tagged is a minimal pair: same zero-padded numbers, one token differs.
- Conditions permute card order only; content and questions are frozen, so
  token counts are identical across conditions by construction.
- Every card records latent_rank AND presentation_slot (the position/content
  decomposition downstream depends on both).
- SHUFFLE resamples its permutation until |spearman(slot, rank)| <= 0.25 so
  the identification condition is actually decorrelated at small N.
- POSITION_CONTROL is deliberately not implemented in v1 (pilot runs
  forward/reverse/shuffle); its design is fixed in Stage 2.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from icom.generator.schemas import Card, Condition, Stimulus, StimulusFamily
from icom.utils.seeding import rng_for

MAX_SHUFFLE_ABS_RHO = 0.25


def _marker_values(n: int) -> list[int]:
    """Strictly increasing 2-digit values, evenly spread across 04..96."""
    vals = np.round(np.linspace(4, 96, n)).astype(int)
    for i in range(1, n):  # enforce strict monotonicity at large n
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + 1
    assert vals[-1] <= 99
    return vals.tolist()


def _shuffle_slots(n_cards: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation of card indices with |spearman(order, identity)| <= threshold."""
    ident = np.arange(n_cards)
    for _ in range(2000):
        perm = rng.permutation(n_cards)
        rho = np.corrcoef(ident, perm)[0, 1]  # pearson on ranks == spearman
        if abs(rho) <= MAX_SHUFFLE_ABS_RHO:
            return perm
    raise RuntimeError(f"no acceptable shuffle found for n={n_cards}")


def _content(family: StimulusFamily, n_items: int, seed: int, content_idx: int,
             vocab: list[str], predicates: list[str]):
    """Deterministic content shared by all conditions: entities, predicates, markers."""
    rng = rng_for(seed, "content", family, n_items, content_idx)
    entities = [vocab[i] for i in rng.choice(len(vocab), size=n_items, replace=False)]
    preds = [predicates[i] for i in rng.choice(len(predicates), size=n_items, replace=False)]
    values = _marker_values(n_items)
    key = hashlib.sha256(
        json.dumps([family, n_items, seed, content_idx, entities, preds], sort_keys=True).encode()
    ).hexdigest()[:16]
    return entities, preds, values, key


def _forward_cards(family: StimulusFamily, entities: list[str], preds: list[str],
                   values: list[int]) -> list[Card]:
    """Cards in latent order (slot fields filled later by the condition)."""
    n = len(entities)
    cards = []
    if family is StimulusFamily.RELATIONAL:
        for k in range(n - 1):
            text = (f"The {entities[k]} {preds[k]} before "
                    f"the {entities[k + 1]} {preds[k + 1]}.")
            cards.append(Card(entity=entities[k], entity_b=entities[k + 1],
                              text=text, latent_rank=k + 1, presentation_slot=0))
    else:
        word = "Day" if family is StimulusFamily.DATED else "Tag"
        for k in range(n):
            text = f"{word} {values[k]:02d}: the {entities[k]} {preds[k]}."
            cards.append(Card(entity=entities[k], text=text,
                              latent_rank=k + 1, presentation_slot=0))
    return cards


def _apply_condition(cards: list[Card], condition: Condition,
                     rng: np.random.Generator) -> list[Card]:
    order = np.arange(len(cards))
    if condition is Condition.FORWARD:
        pass
    elif condition is Condition.REVERSE:
        order = order[::-1]
    elif condition is Condition.SHUFFLE:
        order = _shuffle_slots(len(cards), rng)
    else:
        raise NotImplementedError("POSITION_CONTROL is a Stage-2 design decision")
    out = []
    for slot, idx in enumerate(order, start=1):
        c = cards[idx]
        out.append(Card(entity=c.entity, entity_b=c.entity_b, text=c.text,
                        latent_rank=c.latent_rank, presentation_slot=slot))
    return out


def make_condition_set(
    family: StimulusFamily, n_items: int, seed: int, content_idx: int,
    vocab: list[str], predicates: list[str],
    conditions: list[Condition],
) -> tuple[dict[Condition, Stimulus], list[str], str]:
    """All requested conditions of the SAME content. Returns (stimuli, latent_order, key)."""
    entities, preds, values, key = _content(family, n_items, seed, content_idx, vocab, predicates)
    base = _forward_cards(family, entities, preds, values)
    markers = (
        {} if family is StimulusFamily.RELATIONAL
        else {e: f"{'Day' if family is StimulusFamily.DATED else 'Tag'} {v:02d}"
              for e, v in zip(entities, values)}
    )
    out: dict[Condition, Stimulus] = {}
    for cond in conditions:
        rng = rng_for(seed, "cond", family, n_items, content_idx, cond)
        cards = _apply_condition(base, cond, rng)
        prompt = "\n".join(c.text for c in cards)
        out[cond] = Stimulus(
            family=family, condition=cond, n_items=n_items, seed=seed,
            cards=cards, prompt=prompt, content_key=key,
            latent_order=list(entities), markers=markers,
        )
    return out, list(entities), key
