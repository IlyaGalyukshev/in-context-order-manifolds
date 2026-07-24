"""Question battery for BCS (v2) stimuli.

Reworded to remove the v1 ambiguities:
- reconstruction speaks to the relation's poles, never "acted".
- pairwise is SWAP-PAIRED (each pair asked in both candidate orders) so the
  first-named bias cancels at the item level; each is flagged interior/endpoint
  and by rank-distance so accuracy can be reported interior-only.
- rank uses "position in the order" only when no number is present (S0/S1).
- adjacency is dropped from order metrics (in BCS the successor is generally
  not stated in a single card, but we still don't treat it as order signal).
"""

from __future__ import annotations

from icom.generator.bcs import RELATIONS
from icom.utils.seeding import rng_for

FMT = {"name": " Reply with only the entity name. No explanation.",
       "choice": " Reply with only one entity name. No explanation.",
       "number": " Reply with only the number. No explanation.",
       "list": " Reply with one entity name per line, nothing else."}


def _bin(d):
    return "d=1" if d == 1 else "d=2-3" if d <= 3 else "d=4-7" if d <= 7 else "d=8+"


def make_battery(stim: dict, *, pairwise_per_bin: int = 4, rank_max: int = 10):
    rel = RELATIONS[stim["relation"]]
    order = stim["latent_order"]
    N = len(order)
    rank = {e: i + 1 for i, e in enumerate(order)}
    interior = lambda e: 3 <= rank[e] <= N - 2
    ck = stim["content_key"]
    rng = rng_for(stim["seed"], "bcsq", stim["relation"], N, ck)
    qs = []

    def add(fam, text, key, fmt, **meta):
        qs.append({"stimulus_content_key": ck, "qid": f"{ck}:{fam}:{len(qs)}",
                   "family": fam, "text": text + FMT[fmt], "answer_key": key, **meta})

    # 1. reconstruction (to the relation poles), + mention-order control twin
    add("reconstruction",
        f"Using only the relations stated above, list all entities from the "
        f"{rel.low_pole} to the {rel.high_pole} (this order may differ "
        f"from the order the lines appear in).",
        list(order), "list", target_entities=tuple(order))
    add("reconstruction",
        "List all entities in the order they first appear in the text above, top to bottom.",
        "MENTION_ORDER", "list", span_location="mention_control", target_entities=tuple(order))

    # 2. pairwise SWAP-PAIRED, stratified by rank distance
    bins = {"d=1": (1, 1), "d=2-3": (2, 3), "d=4-7": (4, 7), "d=8+": (8, N - 1)}
    for b, (lo, hi) in bins.items():
        hi = min(hi, N - 1)
        if lo > hi:
            continue
        for _ in range(pairwise_per_bin):
            d = int(rng.integers(lo, hi + 1))
            i = int(rng.integers(1, N - d + 1))
            a, b_ent = order[i - 1], order[i + d - 1]     # a earlier than b_ent
            both_int = interior(a) and interior(b_ent)
            for first, second in ((a, b_ent), (b_ent, a)):  # SWAP PAIR
                add("pairwise",
                    f"By the relations above, which is {rel.cmp_low}: the "
                    f"{first} or the {second}?",
                    a, "choice", rank_distance=d, both_interior=both_int,
                    target_entities=(first, second))

    # 3. rank (position in the order); poles named, no numbers in S0/S1 prompts
    half = max(min(rank_max, N) // 2, 1)
    for xi in sorted(int(x) for x in rng.choice(N, size=half, replace=False)):
        x = order[xi]
        add("rank",
            f"Counting the {rel.low_pole} as position 1, what is the {x}'s "
            f"position in the order?",
            str(xi + 1), "number", is_endpoint=(xi in (0, N - 1)),
            both_interior=interior(x), target_entities=(x,))
    for k in sorted(int(k) for k in rng.choice(N, size=half, replace=False)):
        add("rank", f"Which entity is at position {k + 1}, counting the "
                    f"{rel.low_pole} as position 1?",
            order[k], "name", is_endpoint=(k in (0, N - 1)),
            both_interior=interior(order[k]), target_entities=(order[k],))
    return qs
