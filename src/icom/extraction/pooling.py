"""Per-entity pooling: locate each entity's tokens in the prompt, pool 4 ways.

Schemes:
  name       — tokens of the entity's name, averaged over ALL its mentions
               (relational entities appear in up to 2 cards)
  marker     — tokens of the card's order marker ("Tag 53"); dated/tagged only
  last_token — last token of the entity's primary card
  card_mean  — mean over all tokens of the primary card
Primary card = the card where the entity plays the `entity` role (earlier in
relational pairs); rank-N relational entities only appear as entity_b, so
their primary card is the one where they are entity_b.

Char spans are found by exact substring/regex search (nonce names are unique
strings), then mapped to token indices via the tokenizer's offset mapping —
robust to chat-template wrapping because we search the final formatted string.
"""

from __future__ import annotations

import re

import numpy as np

POOLING_SCHEMES = ("name", "readout", "marker", "last_token", "card_mean")


def _tokens_in_span(offsets: list[tuple[int, int]], lo: int, hi: int) -> list[int]:
    return [i for i, (s, e) in enumerate(offsets) if s < hi and e > lo and e > s]


def build_spans(prompt: str, stimulus: dict, offsets: list[tuple[int, int]]) -> dict:
    """token indices per entity per scheme: {entity: {scheme: [token_idx, ...]}}"""
    cards = stimulus["cards"]
    markers = stimulus.get("markers") or {}
    spans: dict[str, dict[str, list[int]]] = {}

    card_pos = {}
    for c in cards:
        p = prompt.find(c["text"])
        assert p >= 0, f"card text not found in prompt: {c['text'][:40]}"
        card_pos[c["text"]] = (p, p + len(c["text"]))

    for e in stimulus["latent_order"]:
        primary = next((c for c in cards if c["entity"] == e),
                       None) or next(c for c in cards if c.get("entity_b") == e)
        lo, hi = card_pos[primary["text"]]

        name_toks: list[int] = []
        last_mention: list[int] = []
        # [Tt]he: cards/roster open with "The <entity>" / "the <entity>"
        for m in re.finditer(rf"\b[Tt]he {re.escape(e)}\b", prompt):
            toks = _tokens_in_span(offsets, m.start() + 4, m.end())
            name_toks += toks
            last_mention = toks  # overwritten -> ends as the LAST mention
        card_toks = _tokens_in_span(offsets, lo, hi)

        d: dict[str, list[int]] = {
            "name": sorted(set(name_toks)),        # pooled over all mentions (mention-confounded)
            "readout": last_mention,               # roster token: post-all-cards read locus
            "last_token": [card_toks[-1]],
            "card_mean": card_toks,
        }
        if e in markers:
            mk = markers[e]
            assert prompt[lo:lo + len(mk)] == mk
            d["marker"] = _tokens_in_span(offsets, lo, lo + len(mk))
        spans[e] = d
        assert d["name"], f"no name tokens for {e}"
    return spans


def pool_all(hidden: np.ndarray, spans: dict, latent_order: list[str]) -> dict[str, np.ndarray]:
    """hidden: [L, T, D] → {scheme: [N, L, D] fp16} (marker absent if unavailable)."""
    out = {}
    schemes = set.intersection(*(set(s.keys()) for s in spans.values()))
    for scheme in [s for s in POOLING_SCHEMES if s in schemes]:
        mats = [hidden[:, spans[e][scheme], :].mean(axis=1) for e in latent_order]
        out[scheme] = np.stack(mats).astype(np.float16)  # [N, L, D]
    return out
