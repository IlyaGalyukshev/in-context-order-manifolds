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
    low_pole: str     # reconstruction superlative for rank 1 ("smallest", "first")
    high_pole: str    # reconstruction superlative for rank N
    cmp_low: str      # pairwise comparative ("smaller", "earlier in the order")
    preamble: str = ""  # optional in-context transitivity declaration (S0)


RELATIONS = {
    # S0 — arbitrary symbolic, transitivity declared in-context
    "s0_zib": Relation(
        "s0_zib", "zibs", "is zibbed by", "first", "last", "earlier in the zib-order",
        preamble="In this puzzle, 'zibs' is a transitive relation: if X zibs Y and Y zibs Z, then X zibs Z. 'X zibs Y' means X comes before Y in the zib-order.",
    ),
    # S1 — meaningful transitive comparatives over nonce entities
    "s1_size": Relation("s1_size", "is smaller than", "is larger than", "smallest", "largest", "smaller"),
    "s1_loud": Relation("s1_loud", "is quieter than", "is louder than", "quietest", "loudest", "quieter"),
    "s1_heat": Relation("s1_heat", "is cooler than", "is hotter than", "coolest", "hottest", "cooler"),
}
# S2 (grounded magnitude with a nonce unit) is generated specially — see below.


# ------------------------------------------------------------- comparison graph
def _path_edges(n):
    return {(i, i + 1) for i in range(n - 1)}


def regular_graph_with_path(n: int, d: int, rng: np.random.Generator, max_tries: int = 60,
                            prefer: str = "any"):
    """A simple d-regular graph on ranks 0..n-1 CONTAINING the path {(i,i+1)}
    (forces the UNIQUE total order). Padding edges are chosen greedily by
    distance preference:
      prefer='near' (easy) — short padding, local redundancy, chain-solvable;
      prefer='far'  (hard) — long padding, forces reconciling far comparisons;
      prefer='any'  — random padding.
    Requires n*d even and 2<=d<=n-1.
    """
    assert 2 <= d <= n - 1 and (n * d) % 2 == 0, f"infeasible d={d} for n={n}"
    for _ in range(max_tries):
        edges = set(_path_edges(n))
        deg = {i: _degree(edges, i) for i in range(n)}
        stuck = False
        while True:
            deficient = [i for i in range(n) if deg[i] < d]
            if not deficient:
                break
            i = int(rng.choice(deficient))
            cands = [j for j in range(n) if j != i and deg[j] < d
                     and (min(i, j), max(i, j)) not in edges]
            if not cands:
                stuck = True; break
            if prefer == "near":
                cands.sort(key=lambda j: (abs(i - j), rng.random()))
            elif prefer == "far":
                cands.sort(key=lambda j: (-abs(i - j), rng.random()))
            else:
                rng.shuffle(cands)
            j = cands[0]
            edges.add((min(i, j), max(i, j))); deg[i] += 1; deg[j] += 1
        if not stuck and all(deg[i] == d for i in range(n)):
            return edges
    return circulant_graph(n, d)


def feasible_degree(m: int, d: int) -> int:
    """Largest even degree <= min(d, m-1) that admits an Eulerian orientation."""
    de = min(d, m - 1)
    if de % 2:
        de -= 1
    return max(2, de)


def chain_edges(m: int, d: int, rng):
    """Edges for a length-m chain (path + regular padding). m<=2 => single edge."""
    if m <= 1:
        return []
    if m == 2:
        return [(0, 1)]
    return sorted(regular_graph_with_path(m, feasible_degree(m, d), rng))


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
def roster_line(entities, rng, rank_of=None, thresh=0.15, tries=3000):
    """Appended readout tail: every entity mentioned ONCE, after all cards, in a
    rank-decorrelated random order. Gives each entity a clean post-integration
    read position (its roster token) so 'which layer' is not confounded by
    'which mention'. Returns (text_line, roster_order)."""
    n = len(entities)
    order = list(rng.permutation(n))
    if rank_of is not None:
        ranks = np.array([rank_of[e] for e in entities])
        best, bestrho = order, np.inf
        for _ in range(tries):
            perm = list(rng.permutation(n))
            pos_rank = np.array([ranks[p] for p in perm])          # rank at each roster slot
            rho = abs(np.corrcoef(np.arange(n), pos_rank)[0, 1])
            if rho <= thresh:
                best = perm; break
            if rho < bestrho:
                best, bestrho = perm, rho
        order = best
    names = [entities[p] for p in order]
    return "Entities: " + ", ".join(f"the {e}" for e in names) + ".", names


def _card_text(rel: Relation, earlier: str, later: str, flip: bool):
    """Return (text, first_named, second_named). flip decides phrasing."""
    if not flip:
        return f"The {earlier} {rel.fwd} the {later}.", earlier, later
    return f"The {later} {rel.inv} the {earlier}.", later, earlier


def _prefer(difficulty):
    return {"easy": "near", "hard": "far"}.get(difficulty, "any")


def build_stimulus(family: str, n_items: int, seed: int, idx: int,
                   vocab, d: int = 4, balanced: bool = False,
                   condition: str = "shuffle", incoherent: bool = False,
                   difficulty: str = None, readout: bool = True):
    """One BCS stimulus. family is a RELATIONS key. Returns a dict.

    difficulty overrides `balanced`: 'easy' = banded circulant (order recoverable
    by LOCAL chaining), 'hard' = random-regular (LONG edges, needs GLOBAL
    integration). Both are degree-regular / confound-clean.
    readout appends a rank-decorrelated entity roster for a clean read position.
    """
    prefer = _prefer(difficulty)
    rel = RELATIONS[family]
    rng = rng_for(seed, "bcs", family, n_items, idx, d, balanced)
    entities = [vocab[i] for i in rng.choice(len(vocab), size=n_items, replace=False)]

    edges = (circulant_graph(n_items, d) if balanced
             else regular_graph_with_path(n_items, d, rng, prefer=prefer))
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
    readout_order = None
    if readout:
        line, readout_order = roster_line(entities, rng, rank_of=rank_of)
        prompt += "\n\n" + line
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
        "readout_order": readout_order, "gate": report,
    }
    stim["stimulus_id"] = hashlib.sha256(
        json.dumps([family, condition, n_items, seed, idx, d, balanced, incoherent, prompt],
                   sort_keys=True).encode()).hexdigest()[:16]
    return stim


# ---------------------------------------------------------- non-total structures
def build_partial_order(family: str, n_chains: int, chain_len: int, seed: int, idx: int,
                        vocab, d: int = 4, condition: str = "shuffle"):
    """K disjoint total orders (chains) stated with BCS edges WITHIN each chain
    and NONE across => cross-chain pairs are INCOMPARABLE. Tests whether the
    model represents K separate 1D orders (≥2 components) and respects
    incomparability instead of inventing a single total order.

    Per-entity labels: chain_id + within_chain_rank. There is no global rank.
    """
    rel = RELATIONS[family]
    rng = rng_for(seed, "bcs_po", family, n_chains, chain_len, seed, idx)
    N = n_chains * chain_len
    ents = [vocab[i] for i in rng.choice(len(vocab), size=N, replace=False)]
    chains = [ents[c * chain_len:(c + 1) * chain_len] for c in range(n_chains)]

    cards = []
    chain_of = {}; wrank = {}
    for c, chain in enumerate(chains):
        edges = chain_edges(chain_len, d, rng)
        tail = eulerian_orientation(edges, chain_len)
        for k, (lo, hi) in enumerate(edges):
            earlier, later = chain[lo], chain[hi]
            if tail[k] == lo:
                text = f"The {earlier} {rel.fwd} the {later}."; first, second = earlier, later
            else:
                text = f"The {later} {rel.inv} the {earlier}."; first, second = later, earlier
            cards.append({"lo": lo, "hi": hi, "text": text, "entity": first, "entity_b": second})
        for r, e in enumerate(chain):
            chain_of[e] = c; wrank[e] = r + 1

    order = _condition_order(cards, ents, chain_of, wrank, condition, rng)
    cards = [cards[i] for i in order]
    for slot, c in enumerate(cards, 1):
        c["presentation_slot"] = slot; c["latent_rank"] = c["lo"] + 1

    prompt = (rel.preamble + "\n\n" if rel.preamble else "") + "\n".join(c["text"] for c in cards)
    _, readout_order = roster_line(ents, rng)
    prompt += "\n\nEntities: " + ", ".join(f"the {e}" for e in readout_order) + "."
    key = hashlib.sha256(json.dumps(
        ["po", family, n_chains, chain_len, seed, idx, ents], sort_keys=True).encode()).hexdigest()[:16]
    stim = {
        "family": family, "structure": "partial_order", "condition": condition,
        "n_items": N, "n_chains": n_chains, "chain_len": chain_len, "seed": seed,
        "relation": rel.name, "degree": d,
        "latent_order": list(ents),  # arbitrary listing; NOT a global order
        "chain_of": {e: int(chain_of[e]) for e in ents},
        "within_rank": {e: int(wrank[e]) for e in ents},
        "cards": [{"entity": c["entity"], "entity_b": c["entity_b"], "text": c["text"],
                   "latent_rank": c["latent_rank"], "presentation_slot": c["presentation_slot"]}
                  for c in cards],
        "prompt": prompt, "content_key": key,
    }
    stim["stimulus_id"] = hashlib.sha256((key + condition).encode()).hexdigest()[:16]
    return stim


def build_grid2d(family_x: str, family_y: str, N: int, seed: int, idx: int,
                 vocab, d: int = 4, condition: str = "shuffle"):
    """TWO INDEPENDENT GLOBAL total orders over the SAME N entities: an x-order
    (relation family_x, e.g. size) and an independent y-order (family_y, e.g.
    loudness). Each entity has coord (rank_x, rank_y) in 1..N. Both orders are
    fully determined (BCS comparisons over all N per axis). Tests whether a 2D
    manifold forms with x and y separately decodable and disentangled."""
    rx, ry = RELATIONS[family_x], RELATIONS[family_y]
    rng = rng_for(seed, "bcs_2ord", family_x, family_y, N, seed, idx)
    ents = [vocab[i] for i in rng.choice(len(vocab), size=N, replace=False)]
    rank_x = {e: i + 1 for i, e in enumerate(ents)}          # x-order = listing
    yperm = list(rng.permutation(N))                          # independent y-order
    y_chain = [ents[i] for i in yperm]                        # y-rank 1..N along this
    rank_y = {e: r + 1 for r, e in enumerate(y_chain)}

    cards = []
    _emit_chain_cards(ents, rx, N, d, rng, cards)             # x comparisons (all N)
    _emit_chain_cards(y_chain, ry, N, d, rng, cards)          # y comparisons (all N)

    order = list(rng.permutation(len(cards))) if condition == "shuffle" else list(range(len(cards)))
    cards = [cards[i] for i in order]
    for slot, cc in enumerate(cards, 1):
        cc["presentation_slot"] = slot; cc["latent_rank"] = 0

    preamble = " ".join(p for p in (rx.preamble, ry.preamble) if p)
    prompt = (preamble + "\n\n" if preamble else "") + "\n".join(cc["text"] for cc in cards)
    _, readout_order = roster_line(ents, rng)
    prompt += "\n\nEntities: " + ", ".join(f"the {e}" for e in readout_order) + "."
    key = hashlib.sha256(json.dumps(
        ["2ord", family_x, family_y, N, seed, idx, ents], sort_keys=True).encode()).hexdigest()[:16]
    stim = {
        "family": f"{family_x}|{family_y}", "structure": "grid2d", "condition": condition,
        "n_items": N, "seed": seed, "degree": d,
        "latent_order": list(ents),
        "coord_x": {e: rank_x[e] for e in ents}, "coord_y": {e: rank_y[e] for e in ents},
        "cards": [{"entity": cc["entity"], "entity_b": cc["entity_b"], "text": cc["text"],
                   "latent_rank": 0, "presentation_slot": cc["presentation_slot"]} for cc in cards],
        "prompt": prompt, "content_key": key,
    }
    stim["stimulus_id"] = hashlib.sha256((key + condition).encode()).hexdigest()[:16]
    return stim


def _emit_chain_cards(chain, rel, m, d, rng, cards):
    """Append BCS cards ordering `chain` (index r => rank r) by `rel`."""
    edges = chain_edges(m, d, rng)
    tail = eulerian_orientation(edges, m)
    for k, (lo, hi) in enumerate(edges):
        earlier, later = chain[lo], chain[hi]
        if tail.get(k, lo) == lo:
            text = f"The {earlier} {rel.fwd} the {later}."; first, second = earlier, later
        else:
            text = f"The {later} {rel.inv} the {earlier}."; first, second = later, earlier
        cards.append({"lo": lo, "hi": hi, "text": text, "entity": first, "entity_b": second})


def _condition_order(cards, ents, chain_of, wrank, condition, rng):
    m = len(cards)
    if condition == "forward":
        return sorted(range(m), key=lambda i: (chain_of[cards[i]["entity"]], cards[i]["lo"]))
    return list(rng.permutation(m))


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
