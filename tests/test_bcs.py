"""Invariant tests for the v2 Balanced Comparability Sets generator.

Each test pins a confound-control that the v1 linear chain violated. A failure
means the endpoint/role/frequency artifact could return — treat as a blocker.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from icom.generator.bcs import (build_stimulus, circulant_graph, eulerian_orientation,
                                 regular_graph_with_path, _degree, _unique_topo)

SEED = 20260724
POOL = Path(__file__).resolve().parents[1] / "data" / "pools" / "entities_v1.json"
VOCAB = json.load(open(POOL))["names"]
CONFIGS = [(N, d, bal) for N in (7, 9, 12, 16) for d in (4,) for bal in (False, True)]


@pytest.mark.parametrize("N,d,bal", CONFIGS)
def test_gates_hold(N, d, bal):
    """The six generation gates hold for every stimulus in the config."""
    for idx in range(25):
        s = build_stimulus("s1_size", N, SEED, idx, VOCAB, d=d, balanced=bal, condition="shuffle")
        g = s["gate"]
        assert g["degree_regular"], (N, d, bal, idx, g["degrees"])
        assert g["unique_total_order"]
        assert g["has_nonadjacent"]
        assert abs(g["corr_rank_mentions"]) < 1e-9    # frequency ⟂ rank (D2)
        assert abs(g["corr_rank_subjfrac"]) < 1e-9    # syntactic slot ⟂ rank (D1)
        assert abs(g["corr_rank_slot"]) <= 0.15       # presentation position ⟂ rank


@pytest.mark.parametrize("N,d,bal", CONFIGS)
def test_first_named_balanced(N, d, bal):
    """Every entity is named-first in exactly d/2 cards (Eulerian balance)."""
    s = build_stimulus("s1_size", N, SEED, 0, VOCAB, d=d, balanced=bal, condition="shuffle")
    first = Counter(c["entity"] for c in s["cards"])
    assert set(first.values()) == {d // 2}, first


@pytest.mark.parametrize("N,d", [(7, 4), (9, 4), (12, 4), (16, 4)])
def test_regular_graph_is_regular_and_ordered(N, d):
    rng = np.random.default_rng(0)
    edges = regular_graph_with_path(N, d, rng)
    assert all(_degree(edges, i) == d for i in range(N))
    assert _unique_topo(edges, N)                     # contains the path
    assert all(lo < hi for lo, hi in edges)


@pytest.mark.parametrize("N,d", [(8, 4), (10, 6), (16, 4)])
def test_eulerian_orientation_balanced(N, d):
    rng = np.random.default_rng(1)
    edges = sorted(regular_graph_with_path(N, d, rng))
    tail = eulerian_orientation(edges, N)
    outdeg = Counter(tail[k] for k in range(len(edges)))
    assert all(outdeg[i] == d // 2 for i in range(N)), outdeg


def test_coherence_null_has_no_total_order():
    """The null twin injects a cycle: no unique total order recoverable."""
    for idx in range(10):
        z = build_stimulus("s0_zib", 9, SEED, idx, VOCAB, d=4, condition="shuffle", incoherent=True)
        # rebuild the CLAIMED directed edges from the cards
        ent = {e: i for i, e in enumerate(z["latent_order"])}
        claimed = set()
        for c in z["cards"]:
            t = c["text"]
            if " zibs the " in t:
                a, b = t.split("The ")[1].split(" zibs the ")
            else:
                b, a = t.split("The ")[1].split(" is zibbed by the ")
            a = a.strip().rstrip("."); b = b.strip().rstrip(".")
            claimed.add((ent[a], ent[b]))
        assert not _unique_topo(claimed, 9)


def test_determinism():
    a = build_stimulus("s1_size", 12, SEED, 5, VOCAB, d=4, condition="shuffle")
    b = build_stimulus("s1_size", 12, SEED, 5, VOCAB, d=4, condition="shuffle")
    assert a["prompt"] == b["prompt"] and a["content_key"] == b["content_key"]
    c = build_stimulus("s1_size", 12, SEED, 6, VOCAB, d=4, condition="shuffle")
    assert c["content_key"] != a["content_key"]


def test_forward_condition_sorted_by_rank():
    s = build_stimulus("s1_size", 9, SEED, 0, VOCAB, d=4, condition="forward")
    los = [c["latent_rank"] for c in s["cards"]]
    assert los == sorted(los)


def test_circulant_regular_and_contains_path():
    for N, d in [(8, 4), (12, 6), (16, 4)]:
        edges = circulant_graph(N, d)
        assert all(_degree(edges, i) == d for i in range(N))
        assert all((i, i + 1) in edges for i in range(N - 1))


def test_partial_order_no_cross_chain_edges():
    """Cross-chain pairs must be incomparable: zero cards relate different chains."""
    from icom.generator.bcs import build_partial_order
    for idx in range(10):
        s = build_partial_order("s1_size", 2, 5, SEED, idx, VOCAB, d=4, condition="shuffle")
        ci = s["chain_of"]
        assert sum(1 for c in s["cards"] if ci[c["entity"]] != ci[c["entity_b"]]) == 0
        assert set(ci.values()) == {0, 1}
        # each chain independently degree-regular in mentions
        from collections import Counter
        cnt = Counter()
        for c in s["cards"]:
            cnt[c["entity"]] += 1; cnt[c["entity_b"]] += 1
        for chain in (0, 1):
            degs = {cnt[e] for e in ci if ci[e] == chain}
            assert len(degs) == 1, degs  # rank-invariant mention count within chain


def test_grid2d_two_independent_global_orders():
    from icom.generator.bcs import build_grid2d
    g = build_grid2d("s1_size", "s1_loud", 9, SEED, 0, VOCAB, d=4, condition="shuffle")
    assert g["n_items"] == 9
    assert ("smaller than" in g["prompt"] or "larger than" in g["prompt"])
    assert ("louder than" in g["prompt"] or "quieter than" in g["prompt"])
    # both coordinates are GLOBAL total orders 1..N (every cross-pair determined)
    assert {g["coord_x"][e] for e in g["latent_order"]} == set(range(1, 10))
    assert {g["coord_y"][e] for e in g["latent_order"]} == set(range(1, 10))
    # x and y independent: not the same ranking
    assert [g["coord_x"][e] for e in g["latent_order"]] != [g["coord_y"][e] for e in g["latent_order"]]


def test_order_query_family_not_degenerate():
    """The order-query family MIXES comparable (key=entity) and incomparable
    (key='undetermined') under identical wording, so a constant answer can't
    score. Cross-chain => undetermined; same-chain => a determined entity key."""
    from icom.generator.bcs import build_partial_order
    from icom.generator.bcs_questions import make_partial_battery
    s = build_partial_order("s1_size", 2, 5, SEED, 0, VOCAB, d=4, condition="shuffle")
    qs = [q for q in make_partial_battery(s) if q["family"] == "order_query"]
    ci = s["chain_of"]
    und = [q for q in qs if q["answer_key"] == "undetermined"]
    det = [q for q in qs if q["answer_key"] != "undetermined"]
    assert und and det, "must contain BOTH determined and undetermined"
    for q in und:
        a, b = q["target_entities"]; assert ci[a] != ci[b]      # cross-chain
    for q in det:
        a, b = q["target_entities"]; assert ci[a] == ci[b]      # same-chain
        assert q["answer_key"] in (a, b)


def test_coherence_null_always_has_cycle():
    """Every coherence-null twin must admit NO valid total order (has a cycle)."""
    from icom.generator.bcs import build_stimulus, _has_cycle
    import re
    for idx in range(30):
        z = build_stimulus("s1_size", 9, SEED, idx, VOCAB, d=4, condition="shuffle",
                           incoherent=True)
        ent = {e: i for i, e in enumerate(z["latent_order"])}
        directed = []
        for c in z["cards"]:
            t = c["text"]
            if " is smaller than " in t:
                a, b = re.match(r"The (\w+) is smaller than the (\w+)\.", t).groups()
            else:
                b, a = re.match(r"The (\w+) is larger than the (\w+)\.", t).groups()
            directed.append((ent[a], ent[b]))
        assert _has_cycle(directed, 9), f"null idx={idx} has no cycle (coherent!)"


def test_pairwise_pairs_distinct():
    """Total-order pairwise: no unordered pair is asked more than once (beyond
    its swap), i.e. distinct pairs per bin (no pseudo-replication)."""
    from icom.generator.bcs import build_stimulus
    from icom.generator.bcs_questions import make_battery
    s = build_stimulus("s1_size", 12, SEED, 0, VOCAB, d=4, condition="shuffle")
    pw = [q for q in make_battery(s) if q["family"] == "pairwise"]
    unordered = [frozenset(q["target_entities"]) for q in pw]
    from collections import Counter
    c = Counter(unordered)
    assert all(v == 2 for v in c.values()), f"pairs not distinct: {c.most_common(3)}"
