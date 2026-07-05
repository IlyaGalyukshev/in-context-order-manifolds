"""Question battery: five families, deterministic keys from the latent order.

Generated ONCE per content (latent order) and attached verbatim to every
condition of that content. Wording adapts per stimulus family (a tag order is
not "time"), but within one content the battery is frozen.

Families → geometric predictions (see README):
  reconstruction — global coherence of the whole manifold
  pairwise       — local monotonicity of the coordinate, distance-stratified
  adjacency      — fine-grained local spacing
  rank           — absolute value of the coordinate (two subtypes: position-of-X
                   and entity-at-position-k)
  span           — ordered read-out of a local window at start/middle/end
Answers are constrained ("only the entity name" / "yes or no" / digits) so a
deterministic parser can score them; pairwise is additionally logit-scored.
"""

from __future__ import annotations

import numpy as np

from icom.generator.schemas import Question, QuestionFamily, StimulusFamily
from icom.utils.seeding import rng_for

# Family-specific phrasing of the order relation.
# Wording lessons from the pilot (hard-won, do not soften):
# - pairwise is FORCED CHOICE, not yes/no: on yes/no both pilot models gave a
#   near-constant polar answer (qwen ~88% "yes", olmo ~99% "no") regardless of
#   token budget — the binary channel carried almost no signal.
# - every question ends with a STRICT format clause ("Reply with only ... No
#   explanation.") — the weak "Answer with..." let models narrate, which forced
#   fragile parsing and truncation artifacts.
# - recon disambiguates true order vs text order explicitly, and the battery
#   pairs it with a MENTION-ORDER control twin, separating "can't reconstruct"
#   from "read the question as text order".
_WORDING = {
    StimulusFamily.DATED: {
        "recon": ("The events above are listed in arbitrary order. List all entities in the true "
                  "temporal order of their events, earliest first — this may differ from the "
                  "order in the text."),
        "recon_mention": "List all entities in the order they first appear in the text above, top to bottom.",
        "pair": "Which event happened earlier: the {a} event or the {b} event?",
        "adj": "Which entity comes immediately after the {x} in time?",
        "rank_of": "Counting from the earliest event as position 1, what position is the {x}?",
        "at_rank": "Which entity had the {k}-th earliest event?",
        "span": "List the three entities that come immediately after the {x} in time, earliest first.",
    },
    StimulusFamily.TAGGED: {
        "recon": ("The lines above are listed in arbitrary order. List all entities from the "
                  "lowest tag to the highest tag — this may differ from the order in the text."),
        "recon_mention": "List all entities in the order they first appear in the text above, top to bottom.",
        "pair": "Which has the lower tag: the {a} or the {b}?",
        "adj": "Which entity has the next tag above the {x}?",
        "rank_of": "Counting from the lowest tag as position 1, what position is the {x}?",
        "at_rank": "Which entity has the {k}-th lowest tag?",
        "span": "List the three entities with the next tags above the {x}, lowest first.",
    },
    StimulusFamily.RELATIONAL: {
        "recon": ("The statements above are listed in arbitrary order. List all entities in the "
                  "true order in which they acted, earliest first — this may differ from the "
                  "order in the text."),
        "recon_mention": "List all entities in the order they first appear in the text above, top to bottom.",
        "pair": "Which acted earlier: the {a} or the {b}?",
        "adj": "Which entity acted immediately after the {x}?",
        "rank_of": "Counting from the earliest as position 1, what position is the {x}?",
        "at_rank": "Which entity acted {k}-th from the earliest?",
        "span": "List the three entities that acted immediately after the {x}, earliest first.",
    },
}

FORMAT_SUFFIX = {
    "name": " Reply with only the entity name. No explanation.",
    "choice": " Reply with only one entity name. No explanation.",
    "number": " Reply with only the number. No explanation.",
    "list": " Reply with one entity name per line, nothing else.",
}


def _parse_bins(bins: list) -> list[tuple[int, int | None]]:
    out = []
    for b in bins:
        s = str(b)
        if s.endswith("+"):
            out.append((int(s[:-1]), None))
        elif "-" in s:
            lo, hi = s.split("-")
            out.append((int(lo), int(hi)))
        else:
            out.append((int(s), int(s)))
    return out


def make_battery(
    latent_order: list[str],
    content_key: str,
    family: StimulusFamily,
    seed: int,
    *,
    pairwise_per_bin: int = 3,
    distance_bins: list = (1, "2-3", "4-7", "8+"),
    adjacency_max: int = 10,
    rank_max: int = 10,
) -> list[Question]:
    n = len(latent_order)
    rank_of = {e: i + 1 for i, e in enumerate(latent_order)}
    w = _WORDING[family]
    rng = rng_for(seed, "battery", family, content_key)
    qs: list[Question] = []

    def add(qfam: QuestionFamily, text: str, key, fmt: str, **meta) -> None:
        qs.append(Question(
            stimulus_content_key=content_key, qid=f"{content_key}:{qfam}:{len(qs)}",
            family=qfam, text=text + FORMAT_SUFFIX[fmt], answer_key=key, **meta,
        ))

    # 1. full reconstruction (global) + mention-order control twin
    add(QuestionFamily.RECONSTRUCTION, w["recon"], list(latent_order), "list",
        target_entities=tuple(latent_order))
    add(QuestionFamily.RECONSTRUCTION, w["recon_mention"], "MENTION_ORDER", "list",
        span_location="mention_control", target_entities=tuple(latent_order))

    # 2. pairwise forced-choice, stratified by rank distance. Candidate order
    # is balanced EXACTLY 50/50 (earlier-named-first vs later-named-first) so a
    # trivial "always pick the first-named" strategy scores chance, not 67%.
    # (The previous per-bin j%2 alternation gave 2:1 at 3 questions/bin — a
    # position-baseline confound a data-quality judge caught in the pilot.)
    pw = []
    for lo, hi in _parse_bins(list(distance_bins)):
        hi_eff = min(hi if hi is not None else n - 1, n - 1)
        dists = [d for d in range(lo, hi_eff + 1)]
        if not dists:
            continue
        for _ in range(pairwise_per_bin):
            d = int(rng.choice(dists))
            i = int(rng.integers(1, n - d + 1))          # earlier rank position
            pw.append((latent_order[i - 1], latent_order[i + d - 1], d))
    half = len(pw) // 2
    swap = np.array([False] * (len(pw) - half) + [True] * half)
    rng.shuffle(swap)                                    # exactly half swapped
    for (earlier, later, d), sw in zip(pw, swap):
        a, b = (later, earlier) if sw else (earlier, later)
        add(QuestionFamily.PAIRWISE, w["pair"].format(a=a, b=b), earlier, "choice",
            rank_distance=d, target_entities=(a, b))

    # 3. adjacency / successor (X with a successor; endpoints tagged)
    xs = rng.choice(n - 1, size=min(adjacency_max, n - 1), replace=False)
    for xi in sorted(int(x) for x in xs):
        x = latent_order[xi]
        add(QuestionFamily.ADJACENCY, w["adj"].format(x=x), latent_order[xi + 1], "name",
            is_endpoint=(xi == 0 or xi == n - 2), target_entities=(x,))

    # 4. rank: half "position of X", half "entity at position k"
    half = max(min(rank_max, n) // 2, 1)
    for xi in sorted(int(x) for x in rng.choice(n, size=half, replace=False)):
        x = latent_order[xi]
        add(QuestionFamily.RANK, w["rank_of"].format(x=x), str(xi + 1), "number",
            is_endpoint=(xi in (0, n - 1)), target_entities=(x,))
    for k in sorted(int(k) for k in rng.choice(n, size=half, replace=False)):
        add(QuestionFamily.RANK, w["at_rank"].format(k=k + 1), latent_order[k], "name",
            is_endpoint=(k in (0, n - 1)), target_entities=(latent_order[k],))

    # 5. anchored span: window of 3 after X at start / middle / end
    if n >= 5:
        anchors = {"start": 0, "middle": max(n // 2 - 2, 1), "end": n - 4}
        for loc, xi in anchors.items():
            x = latent_order[xi]
            add(QuestionFamily.SPAN, w["span"].format(x=x),
                latent_order[xi + 1: xi + 4], "list",
                span_location=loc, target_entities=(x,))

    return qs
