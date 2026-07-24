"""Balanced Comparability Sets (BCS) — v2 stimulus generator.

Fixes the v1 linear-chain confounds proven by the interior-only control
(rank ≡ syntactic role + mention frequency). A stimulus states a REDUNDANT,
DEGREE-REGULAR set of order comparisons with randomized before/after phrasing:

  - degree-regular comparison graph (every entity in exactly d relations)
    => mention count is rank-invariant  (kills the frequency confound)
  - the graph contains the Hamiltonian path => transitive closure is the
    UNIQUE total order (order fully determined, must be integrated)
  - non-adjacent edges present => cannot be solved by local token chaining
  - each edge phrased "a REL b" or "b REL_inv a" with prob 1/2
    => syntactic first-position is 50/50, independent of rank (kills the
       syntactic-role confound)

Any rank signal that survives an interior-only probe on this design is genuine
integrated order representation, not an endpoint/role artifact.

Relation-semantics gradient (the family axis): S0 symbolic, S1 comparative,
S2 grounded-magnitude — to test whether order representation needs meaning.

Output dicts are compatible with extract_activations / analysis: each stimulus
has latent_order, cards (entity/entity_b/text/latent_rank/presentation_slot),
prompt, family, condition, content_key, n_items, plus BCS metadata + gate report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from icom.utils.seeding import rng_for


# ------------------------------------------------------------------ relations
@dataclass(frozen=True)
class Relation:
    name: str
    fwd: str          # "a REL b" asserts rank(a) < rank(b), a named first
    inv: str          # "b REL_inv a" asserts the same, b named first
    low_pole: str     # reconstruction wording: from {low_pole} to {high_pole}
    high_pole: str
    preamble: str = ""  # optional in-context transitivity declaration (S0)


RELATIONS = {
    # S0 — arbitrary symbolic, transitivity declared in-context
    "s0_zib": Relation(
        "s0_zib", "zibs", "is zibbed by", "first", "last",
        preamble="In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z.",
    ),
    # S1 — meaningful transitive comparatives over nonce entities
    "s1_size": Relation("s1_size", "is smaller than", "is larger than", "smallest", "largest"),
    "s1_loud": Relation("s1_loud", "is quieter than", "is louder than", "quietest", "loudest"),
    "s1_heat": Relation("s1_heat", "is cooler than", "is hotter than", "coolest", "hottest"),
}
# S2 (grounded magnitude with a nonce unit) is generated specially — see below.


# ------------------------------------------------------------- comparison graph
def _path_edges(n):
    return {(i, i + 1) for i in range(n - 1)}


def regular_graph_with_path(n: int, d: int, rng: np.random.Generator, max_tries: int = 200):
    """A simple d-regular graph on ranks 0..n-1 that CONTAINS the path
    {(i,i+1)}. Edges returned as (lo,hi) rank pairs (lo<hi => lo is earlier).

    Configuration model over residual stubs (after mandatory path edges) with
    rejection of self/multi/adjacent-duplicate; falls back to a circulant.
    Requires n*d even and 2<=d<=n-1.
    """
    assert 2 <= d <= n - 1 and (n * d) % 2 == 0, f"infeasible d={d} for n={n}"
    path = _path_edges(n)
    deg0 = {i: 0 for i in range(n)}
    for a, b in path:
        deg0[a] += 1; deg0[b] += 1
    for _ in range(max_tries):
        edges = set(path)
        residual = []
        for i in range(n):
            residual += [i] * (d - deg0[i])
        rng.shuffle(residual)
        ok = True
        for k in range(0, len(residual), 2):
            a, b = residual[k], residual[k + 1]
            lo, hi = (a, b) if a < b else (b, a)
            if a == b or (lo, hi) in edges:
                ok = False; break
            edges.add((lo, hi))
        if ok and all(_degree(edges, i) == d for i in range(n)):
            return edges
    return circulant_graph(n, d)


def circulant_graph(n: int, d: int):
    """C_n(1..d/2) on a cycle: exactly d-regular, contains the path, symmetric
    across ranks (only the earlier/later ROLE depends on rank -> the cleanest,
    hardest variant: order recoverable only by global transitive integration)."""
    assert d % 2 == 0
    edges = set()
    for i in range(n):
        for k in range(1, d // 2 + 1):
            j = (i + k) % n
            lo, hi = (i, j) if i < j else (j, i)
            edges.add((lo, hi))
    return edges


def _degree(edges, i):
    return sum(1 for e in edges if i in e)


def eulerian_orientation(edges, n):
    """Orient each undirected edge so every vertex is the TAIL (first-named) of
    exactly deg/2 edges. Exists because the comparison graph is d-regular with
    d even (all degrees even => each component Eulerian). Returns {frozenset:
    tail_vertex}. Decouples 'named first' from 'earlier' => subject-slot
    fraction is exactly 0.5 for every entity, killing the syntactic-role
    confound deterministically (not just in expectation)."""
    adj = {i: [] for i in range(n)}
    eid = {}
    for k, (a, b) in enumerate(edges):
        adj[a].append((b, k)); adj[b].append((a, k))
        eid[k] = (a, b)
    used = [False] * len(edges)
    tail = {}
    ptr = {i: 0 for i in range(n)}
    for start in range(n):
        # walk an Eulerian trail from `start` over its component (Hierholzer)
        stack = [start]
        path = []
        while stack:
            v = stack[-1]
            while ptr[v] < len(adj[v]) and used[adj[v][ptr[v]][1]]:
                ptr[v] += 1
            if ptr[v] == len(adj[v]):
                path.append(stack.pop())
            else:
                w, k = adj[v][ptr[v]]
                used[k] = True
                stack.append(w)
        path.reverse()
        for a, b in zip(path, path[1:]):
            # find the edge id between a,b not yet oriented
            for w, k in adj[a]:
                if w == b and k not in tail:
                    tail[k] = a  # a is the tail => first-named
                    break
    # any unoriented (isolated safety) -> default tail = smaller endpoint
    for k, (a, b) in enumerate(edges):
        tail.setdefault(k, a)
    return tail


# ------------------------------------------------------------------ gates
def gate_report(entities, edges, cards, ranks, slots, d):
    n = len(entities)
    degs = [_degree(edges, i) for i in range(n)]
    subj_frac = _subject_slot_fraction(cards, entities)  # per-entity fraction named-first
    return {
        "degree_regular": all(dg == d for dg in degs),
        "degrees": degs,
        "unique_total_order": _unique_topo(edges, n),
        "has_nonadjacent": any(hi - lo >= 2 for lo, hi in edges),
        # mention count == degree for all -> corr(rank, mentioncount)==0 trivially
        "corr_rank_mentions": float(np.corrcoef(ranks, degs)[0, 1]) if np.std(degs) > 0 else 0.0,
        "corr_rank_subjfrac": float(np.corrcoef(ranks, subj_frac)[0, 1]) if np.std(subj_frac) > 0 else 0.0,
        "corr_rank_slot": float(np.corrcoef(ranks, slots)[0, 1]),
    }


def _unique_topo(edges, n):
    """True iff the DAG (lo->hi) admits exactly one topological order."""
    succ = {i: set() for i in range(n)}
    indeg = {i: 0 for i in range(n)}
    for lo, hi in edges:
        if hi not in succ[lo]:
            succ[lo].add(hi); indeg[hi] += 1
    order = []
    avail = [i for i in range(n) if indeg[i] == 0]
    while avail:
        if len(avail) > 1:
            return False  # a choice point => not unique
        u = avail.pop()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                avail.append(v)
    return len(order) == n and order == list(range(n))


def _subject_slot_fraction(cards, entities):
    """Per entity: fraction of its cards where it is the first-named."""
    first = {e: 0 for e in entities}
    total = {e: 0 for e in entities}
    for c in cards:
        a, b = c["entity"], c["entity_b"]
        total[a] += 1; total[b] += 1
        first[a] += 1  # entity is named first in the card by construction of `entity`
    return np.array([first[e] / max(total[e], 1) for e in entities])


# ------------------------------------------------------------------ rendering
def _card_text(rel: Relation, earlier: str, later: str, flip: bool):
    """Return (text, first_named, second_named). flip decides phrasing."""
    if not flip:
        return f"The {earlier} {rel.fwd} the {later}.", earlier, later
    return f"The {later} {rel.inv} the {earlier}.", later, earlier


def build_stimulus(family: str, n_items: int, seed: int, idx: int,
                   vocab, d: int = 4, balanced: bool = False,
                   condition: str = "shuffle", incoherent: bool = False):
    """One BCS stimulus. family is a RELATIONS key. Returns a dict."""
    rel = RELATIONS[family]
    rng = rng_for(seed, "bcs", family, n_items, idx, d, balanced)
    entities = [vocab[i] for i in rng.choice(len(vocab), size=n_items, replace=False)]

    edges = (circulant_graph(n_items, d) if balanced
             else regular_graph_with_path(n_items, d, rng))
    edge_list = sorted(edges)

    cards = []
    if incoherent:
        # coherence-null twin: reverse a few edges' claimed direction to inject
        # a cycle (no valid total order); random first-named (control only).
        for (lo, hi) in _inject_cycle(edge_list, rng):
            e_lo, e_hi = min(lo, hi), max(lo, hi)
            earlier, later = entities[lo], entities[hi]  # as CLAIMED (may be false)
            flip = bool(rng.integers(2))
            text, first, second = _card_text(rel, earlier, later, flip)
            cards.append({"lo": e_lo, "hi": e_hi, "text": text, "entity": first, "entity_b": second})
    else:
        # deterministic first-named balance via Eulerian orientation:
        # each entity is named-first in exactly d/2 cards => subjfrac == 0.5.
        tail = eulerian_orientation(edge_list, n_items)
        for k, (lo, hi) in enumerate(edge_list):
            earlier, later = entities[lo], entities[hi]
            first_is_earlier = (tail[k] == lo)
            if first_is_earlier:
                text = f"The {earlier} {rel.fwd} the {later}."; first, second = earlier, later
            else:
                text = f"The {later} {rel.inv} the {earlier}."; first, second = later, earlier
            cards.append({"lo": lo, "hi": hi, "text": text, "entity": first, "entity_b": second})

    # presentation order (condition)
    order = np.arange(len(cards))
    if condition == "forward":
        order = _order_by_min_rank(cards)          # cards sorted by earlier rank
    elif condition == "shuffle":
        order = _decorrelated_order(cards, entities, rng)
    cards = [cards[i] for i in order]
    for slot, c in enumerate(cards, 1):
        c["presentation_slot"] = slot

    # per-entity latent rank + slot = MEAN card position of its mentions
    # (matches pooling-over-mentions; naturally ~decorrelated under shuffle).
    rank_of = {e: r + 1 for r, e in enumerate(entities)}
    pos_sum = {e: 0.0 for e in entities}; pos_cnt = {e: 0 for e in entities}
    for si, c in enumerate(cards):
        for e in (c["entity"], c["entity_b"]):
            pos_sum[e] += si; pos_cnt[e] += 1
    meanpos = np.array([pos_sum[e] / max(pos_cnt[e], 1) for e in entities])
    slots = meanpos.argsort().argsort() + 1  # rank-normalize to 1..N
    ranks = np.array([rank_of[e] for e in entities])

    prompt = (rel.preamble + "\n\n" if rel.preamble else "") + "\n".join(c["text"] for c in cards)
    for r, e in enumerate(entities):
        for c in cards:
            if c["entity"] == e or c["entity_b"] == e:
                pass
    # attach per-card latent_rank of the earlier entity (for compat)
    for c in cards:
        c["latent_rank"] = c["lo"] + 1

    report = gate_report(entities, set((c["lo"], c["hi"]) for c in cards), cards,
                         ranks, slots, d) if not incoherent else {"incoherent": True}

    content_key = hashlib.sha256(
        json.dumps([family, n_items, seed, idx, d, balanced, incoherent, entities, edge_list],
                   sort_keys=True).encode()).hexdigest()[:16]
    stim = {
        "family": family, "condition": condition, "n_items": n_items, "seed": seed,
        "relation": rel.name, "degree": d, "balanced": balanced, "incoherent": incoherent,
        "latent_order": list(entities),
        "cards": [{"entity": c["entity"], "entity_b": c["entity_b"], "text": c["text"],
                   "latent_rank": c["latent_rank"], "presentation_slot": c["presentation_slot"]}
                  for c in cards],
        "prompt": prompt, "content_key": content_key,
        "entity_ranks": {e: int(rank_of[e]) for e in entities},
        "entity_slots": {e: int(s) for e, s in zip(entities, slots)},
        "gate": report,
    }
    stim["stimulus_id"] = hashlib.sha256(
        json.dumps([family, condition, n_items, seed, idx, d, balanced, incoherent, prompt],
                   sort_keys=True).encode()).hexdigest()[:16]
    return stim


def _order_by_min_rank(cards):
    return np.argsort([c["lo"] for c in cards], kind="stable")


def _decorrelated_order(cards, entities, rng, thresh=0.15, tries=6000):
    """Shuffle card order until each entity's MEAN mention position ⟂ latent rank."""
    n = len(entities); m = len(cards)
    ranks = np.arange(1, n + 1)
    ent_idx = {e: r for r, e in enumerate(entities)}
    # precompute which entities each card touches
    touch = [(ent_idx[c["entity"]], ent_idx[c["entity_b"]]) for c in cards]
    best, best_rho = None, np.inf
    for _ in range(tries):
        perm = rng.permutation(m)
        psum = np.zeros(n); pcnt = np.zeros(n)
        for si, ci in enumerate(perm):
            a, b = touch[ci]
            psum[a] += si; psum[b] += si; pcnt[a] += 1; pcnt[b] += 1
        meanpos = psum / np.maximum(pcnt, 1)
        rho = abs(np.corrcoef(ranks, meanpos.argsort().argsort() + 1)[0, 1])
        if rho <= thresh:
            return perm
        if rho < best_rho:
            best, best_rho = perm, rho
    return best


def _inject_cycle(edge_list, rng, k=2):
    """Flip the direction of k edges to create at least one cycle (no total order)."""
    el = [list(e) for e in edge_list]
    picks = rng.choice(len(el), size=min(k, len(el)), replace=False)
    for p in picks:
        el[p] = [el[p][1], el[p][0]]  # reversed: now hi<lo claim -> inconsistent
    return [tuple(e) for e in el]
