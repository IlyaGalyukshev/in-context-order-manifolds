"""Per-card pooling schemes — a genuine free parameter, fixed by ablation.

Schemes: entity-name tokens / order-marker token (date or tag) / last token
of the card / mean over the card's tokens. The ablation compares manifold
quality across schemes; if the marker-token scheme scores suspiciously high
on DATED stimuli, that flags capture of the pretrained date manifold — re-run
on TAGGED/RELATIONAL twins where that scaffold is absent or weaker.
Multi-token names are averaged within the scheme.
"""

from __future__ import annotations

POOLING_SCHEMES = ("name", "marker", "last_token", "card_mean")


def pool_card(hidden_states, card_span, scheme: str):
    raise NotImplementedError
