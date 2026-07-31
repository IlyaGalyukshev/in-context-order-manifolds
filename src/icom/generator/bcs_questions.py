"""Question battery for BCS (v2) stimuli.

Reworded to remove the v1 ambiguities:
- reconstruction speaks to the relation's poles, never "acted".
- pairwise is SWAP-PAIRED (each pair asked in both candidate orders) so the
  first-named bias cancels at the item level; each is flagged interior/endpoint
  and by rank-distance so accuracy can be reported interior-only.
- rank uses "position in the order" only when no number is present (S0/S1).

Metric families (added v2.1) — together they test whether BEHAVIOUR reads the
same metric a manifold would encode, all under the position framing (the
`low_pole` as position 1) so they are relation-agnostic (work for S0 too):
- betweenness       — which of three has the MIDDLE rank (ordinal interval).
- successor/predecessor — the immediately-next entity (local order resolution).
- count_between     — how many entities strictly between X and Y (|Δrank|-1,
                      absolute ordinal distance).
- comparative_distance — which of two is closer to a pivot (relative metric).
- extremes          — the overall pole; ENDPOINT-DIAGNOSTIC only (interior_ok
                      is always False), a positive control, never a geometry
                      claim.
Every family carries `interior_ok` (all involved entities in ranks 3..N-2) so
interior-only accuracy is computable, and is swap-/order-randomised so option
position does not leak the key.
"""

from __future__ import annotations

from icom.generator.bcs import RELATIONS
from icom.utils.seeding import rng_for

FMT = {"name": " Reply with only the entity name. No explanation.",
       "choice": " Reply with only one entity name. No explanation.",
       "number": " Reply with only the number. No explanation.",
       "list": " Reply with one entity name per line, nothing else.",
       "_none": ""}  # question already carries its own format instruction


def _bin(d):
    return "d=1" if d == 1 else "d=2-3" if d <= 3 else "d=4-7" if d <= 7 else "d=8+"


def make_battery(stim: dict, *, pairwise_per_bin: int = 4, rank_max: int = 10,
                 betweenness_n: int = 6, succ_pred_n: int = 5,
                 count_between_n: int = 6, comparative_n: int = 6):
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

    def interleave(inter, noni, k):
        """Take up to k, alternating interior-first so interior items are
        represented whenever they exist (mirrors the pairwise discipline)."""
        out = []
        while len(out) < k and (inter or noni):
            take = inter if (inter and (len(out) % 2 == 0 or not noni)) else noni
            out.append(take.pop())
        return out

    # 1. reconstruction (to the relation poles), + mention-order control twin
    add("reconstruction",
        f"Using only the relations stated above, list all entities from the "
        f"{rel.low_pole} to the {rel.high_pole} (this order may differ "
        f"from the order the lines appear in).",
        list(order), "list", target_entities=tuple(order))
    add("reconstruction",
        "List all entities in the order they first appear in the text above, top to bottom.",
        "MENTION_ORDER", "list", span_location="mention_control", target_entities=tuple(order))

    # 2. pairwise SWAP-PAIRED, stratified by rank distance, DISTINCT pairs only
    # (no pseudo-replication), balanced interior/endpoint within each bin.
    bins = {"d=1": (1, 1), "d=2-3": (2, 3), "d=4-7": (4, 7), "d=8+": (8, N - 1)}
    for b, (lo, hi) in bins.items():
        hi = min(hi, N - 1)
        if lo > hi:
            continue
        allpairs = [(order[i - 1], order[i + d - 1], d)
                    for d in range(lo, hi + 1) for i in range(1, N - d + 1)]
        rng.shuffle(allpairs)
        inter = [p for p in allpairs if interior(p[0]) and interior(p[1])]
        noni = [p for p in allpairs if not (interior(p[0]) and interior(p[1]))]
        chosen = []
        while len(chosen) < pairwise_per_bin and (inter or noni):
            take = inter if (inter and (len(chosen) % 2 == 0 or not noni)) else noni
            chosen.append(take.pop())
        for a, b_ent, d in chosen:                       # each pair is DISTINCT
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

    # 4. betweenness — which of three has the MIDDLE rank (ordinal interval).
    triples = [(order[i], order[j], order[k]) for i in range(N)
               for j in range(i + 1, N) for k in range(j + 1, N)]
    rng.shuffle(triples)
    tin = [t for t in triples if all(interior(e) for e in t)]
    tno = [t for t in triples if not all(interior(e) for e in t)]
    for tri in interleave(tin, tno, betweenness_n):
        mid = sorted(tri, key=lambda e: rank[e])[1]
        shown = list(tri); rng.shuffle(shown)
        add("betweenness",
            f"Using only the relations stated, which of these is between the "
            f"other two in the order (counting the {rel.low_pole} as position 1): "
            f"the {shown[0]}, the {shown[1]}, or the {shown[2]}?",
            mid, "choice", interior_ok=all(interior(e) for e in tri),
            span=int(max(rank[e] for e in tri) - min(rank[e] for e in tri)),
            target_entities=tuple(shown))

    # 5. successor / predecessor — the immediately-next entity (local resolution).
    def add_adj(fam, word, cues, ans_of, toward):
        cin = [i for i in cues if interior(order[i]) and interior(ans_of(i))]
        cno = [i for i in cues if not (interior(order[i]) and interior(ans_of(i)))]
        rng.shuffle(cin); rng.shuffle(cno)
        for i in interleave(cin, cno, succ_pred_n):
            x, y = order[i], ans_of(i)
            add(fam,
                f"Using only the relations stated, which entity is immediately "
                f"{word} the {x} in the order (the next position toward the "
                f"{toward}, counting the {rel.low_pole} as position 1)?",
                y, "name", interior_ok=(interior(x) and interior(y)),
                rank_distance=1, target_entities=(x,))
    add_adj("successor", "after", list(range(N - 1)), lambda i: order[i + 1], rel.high_pole)
    add_adj("predecessor", "before", list(range(1, N)), lambda i: order[i - 1], rel.low_pole)

    # 6. count_between — |rank difference| - 1 (absolute ordinal distance).
    cbpairs = [(order[i], order[j]) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(cbpairs)
    cin = [p for p in cbpairs if interior(p[0]) and interior(p[1])]
    cno = [p for p in cbpairs if not (interior(p[0]) and interior(p[1]))]
    for a, b in interleave(cin, cno, count_between_n):
        d = abs(rank[a] - rank[b])
        shown = [a, b]; rng.shuffle(shown)
        add("count_between",
            f"Using only the relations stated, how many entities are strictly "
            f"between the {shown[0]} and the {shown[1]} in the order?",
            str(d - 1), "number", rank_distance=int(d),
            interior_ok=(interior(a) and interior(b)), target_entities=tuple(shown))

    # 7. comparative_distance — which of two is closer to a pivot (no ties).
    comp = [(pi, ai, bi) for pi in range(N) for ai in range(N)
            for bi in range(ai + 1, N)
            if pi not in (ai, bi) and abs(pi - ai) != abs(pi - bi)]
    rng.shuffle(comp)
    cin = [t for t in comp if all(interior(order[k]) for k in t)]
    cno = [t for t in comp if not all(interior(order[k]) for k in t)]
    for pi, ai, bi in interleave(cin, cno, comparative_n):
        x, a, b = order[pi], order[ai], order[bi]
        da, db = abs(pi - ai), abs(pi - bi)
        key = a if da < db else b
        shown = [a, b]; rng.shuffle(shown)
        add("comparative_distance",
            f"Using only the relations stated, which is closer to the {x} in the "
            f"order: the {shown[0]} or the {shown[1]}?",
            key, "choice", interior_ok=all(interior(order[k]) for k in (pi, ai, bi)),
            d_near=int(min(da, db)), d_far=int(max(da, db)),
            target_entities=(x, shown[0], shown[1]))

    # 8. extremes — ENDPOINT-DIAGNOSTIC only (never interior; positive control).
    # Anchored to "in the order" so the S0 poles ("first"/"last") cannot be
    # misread as mention-position rather than rank.
    add("extremes", f"Using only the relations stated, which entity is the "
                    f"{rel.low_pole} of all — the one that comes before every "
                    f"other entity in the order?", order[0], "name",
        is_endpoint=True, interior_ok=False, target_entities=(order[0],))
    add("extremes", f"Using only the relations stated, which entity is the "
                    f"{rel.high_pole} of all — the one that comes after every "
                    f"other entity in the order?", order[-1], "name",
        is_endpoint=True, interior_ok=False, target_entities=(order[-1],))
    return qs


def make_partial_battery(stim: dict, *, per_kind: int = 12):
    """Partial-order battery: within-chain pairwise (comparable) + cross-chain
    INCOMPARABILITY (the sharp test: does the model invent a total order?)."""
    rel = RELATIONS[stim["relation"]]
    ents = stim["latent_order"]; ck = stim["content_key"]
    chain_of = stim["chain_of"]; wrank = stim["within_rank"]
    rng = rng_for(stim["seed"], "bcs_po_q", stim["relation"], ck)
    qs = []

    def add(fam, text, key, fmt, **meta):
        qs.append({"stimulus_content_key": ck, "qid": f"{ck}:{fam}:{len(qs)}",
                   "family": fam, "text": text + FMT[fmt], "answer_key": key, **meta})

    # UNIFIED order-query family: comparable (same-chain, key=earlier) AND
    # incomparable (cross-chain, key='undetermined') under IDENTICAL wording, so
    # a constant 'undetermined' cannot score — the model must actually decide.
    same = [(a, b) for a in ents for b in ents if a < b and chain_of[a] == chain_of[b]]
    diff = [(a, b) for a in ents for b in ents if a < b and chain_of[a] != chain_of[b]]
    rng.shuffle(same); rng.shuffle(diff)

    def order_query(a, b, key, comparable):
        for first, second in ((a, b), (b, a)):           # swap-paired
            add("order_query",
                f"Using only the relations stated, is it determined which is "
                f"{rel.cmp_low}, the {first} or the {second}? Answer with that "
                f"entity's name if determined, otherwise answer 'undetermined'.",
                key, "_none", comparable=comparable, target_entities=(first, second))

    for a, b in same[:per_kind]:                          # comparable -> key = earlier
        order_query(a, b, a if wrank[a] < wrank[b] else b, True)
    for a, b in diff[:per_kind]:                          # incomparable -> 'undetermined'
        order_query(a, b, "undetermined", False)
    return qs


def make_grid_battery(stim: dict, *, per_axis: int = 12):
    """2D-grid battery: per-axis pairwise (each axis separately queryable)."""
    fx, fy = stim["family"].split("|")
    rx, ry = RELATIONS[fx], RELATIONS[fy]
    ents = stim["latent_order"]; ck = stim["content_key"]
    cx, cy = stim["coord_x"], stim["coord_y"]
    rng = rng_for(stim["seed"], "bcs_grid_q", ck)
    qs = []

    def add(text, key, axis, **meta):
        qs.append({"stimulus_content_key": ck, "qid": f"{ck}:pairwise:{len(qs)}",
                   "family": "pairwise", "axis": axis, "text": text + FMT["choice"],
                   "answer_key": key, **meta})

    for axis, coord, rel in (("x", cx, rx), ("y", cy, ry)):
        pairs = [(a, b) for a in ents for b in ents if a < b and coord[a] != coord[b]]
        rng.shuffle(pairs)
        for a, b in pairs[:per_axis]:
            lower = a if coord[a] < coord[b] else b
            for first, second in ((a, b), (b, a)):
                add(f"By the relations above, which is {rel.cmp_low}: the {first} or "
                    f"the {second}?", lower, axis, target_entities=(first, second))
    return qs
