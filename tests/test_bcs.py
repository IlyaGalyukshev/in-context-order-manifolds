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


def test_determinacy_bridges_decouple_q_from_difficulty():
    """E4-fix: at FIXED m, sweeping `bridges` 0..m-1 raises the determined-pair fraction q monotonically
    to 1.0, while the block substrate (entities/ranks) and per-block reading load stay constant — bridges
    only ADD cards, never alter the within-block graphs. Twin (incoherent) still admits a cycle."""
    from icom.generator.bcs import build_determinacy, _has_cycle
    import re
    for idx in range(8):
        qs, orders, ncards = [], [], []
        for b in (0, 1, 2):
            s = build_determinacy("s1_size", 16, SEED, idx, VOCAB, m=3, d=4, difficulty="hard", bridges=b)
            qs.append(s["q_determined"]); orders.append(tuple(s["latent_order"])); ncards.append(len(s["cards"]))
            assert s["determinacy_bridges"] == b
        assert qs[0] < qs[1] < qs[2], f"q not strictly increasing with bridges: {qs} (idx={idx})"
        assert abs(qs[2] - 1.0) < 1e-9, f"m-1 bridges must fully determine the order: q={qs[2]}"
        assert orders[0] == orders[1] == orders[2], "bridge sweep must share the entity/rank substrate"
        # bridges only ADD cross-block cards → within-block reading load is identical across the sweep
        assert ncards[1] == ncards[0] + 1 and ncards[2] == ncards[0] + 2, f"bridges changed card count oddly: {ncards}"
    z = build_determinacy("s1_size", 16, SEED, 0, VOCAB, m=3, d=4, difficulty="hard", incoherent=True, bridges=1)
    ent = {e: i for i, e in enumerate(z["latent_order"])}
    directed = []
    for c in z["cards"]:
        if " is smaller than " in c["text"]:
            a, b = re.match(r"The (\w+) is smaller than the (\w+)\.", c["text"]).groups()
        else:
            b, a = re.match(r"The (\w+) is larger than the (\w+)\.", c["text"]).groups()
        directed.append((ent[a], ent[b]))
    assert _has_cycle(directed, 16), "determinacy twin must contain a cycle"


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


def _order_rank(s):
    order = s["latent_order"]
    return order, {e: i + 1 for i, e in enumerate(order)}


def test_metric_family_keys_correct():
    """Every metric-family key is independently recomputable from latent_order,
    across S0/S1 families and N — a wrong key would silently corrupt accuracy."""
    from icom.generator.bcs_questions import make_battery
    for fam in ("s0_zib", "s1_size", "s1_loud"):
        for N in (7, 9, 12, 16):
            s = build_stimulus(fam, N, SEED, N, VOCAB, d=4, condition="shuffle")
            order, rank = _order_rank(s)
            for q in make_battery(s):
                f, k, te = q["family"], q["answer_key"], q.get("target_entities", ())
                if f == "betweenness":
                    assert k == sorted(te, key=lambda e: rank[e])[1]
                elif f == "successor":
                    assert k == order[rank[te[0]]]              # rank(x)+1
                elif f == "predecessor":
                    assert k == order[rank[te[0]] - 2]          # rank(x)-1
                elif f == "count_between":
                    a, b = te
                    assert k == str(abs(rank[a] - rank[b]) - 1)
                elif f == "comparative_distance":
                    x, a, b = te
                    near = a if abs(rank[a] - rank[x]) < abs(rank[b] - rank[x]) else b
                    assert k == near
                elif f == "extremes":
                    assert k == te[0] and k in (order[0], order[-1])


def test_metric_families_present_and_interior_flagged():
    """All new families are emitted; interior-answerable items exist (so
    interior-only accuracy is computable); extremes are ALWAYS endpoint-only."""
    from icom.generator.bcs_questions import make_battery
    qs = make_battery(build_stimulus("s1_size", 12, SEED, 1, VOCAB, d=4, condition="shuffle"))
    fams = Counter(q["family"] for q in qs)
    for f in ("betweenness", "successor", "predecessor", "count_between",
              "comparative_distance", "extremes"):
        assert fams[f] > 0, f
    for f in ("betweenness", "successor", "predecessor", "count_between", "comparative_distance"):
        assert any(q["family"] == f and q.get("interior_ok") for q in qs), f
    assert all(q.get("interior_ok") is False and q.get("is_endpoint")
               for q in qs if q["family"] == "extremes")


def test_cyclic_gates_and_no_endpoints():
    """The cyclic ring is degree-regular, has non-adjacent edges, forms exactly
    ONE valid cycle (distinct from the invalid-cycle null), is position-symmetric
    (no endpoints: every entity is named-first in ~half its cards), and each
    entity is mentioned exactly d times (frequency ⟂ position)."""
    from icom.generator.bcs import build_cyclic
    for N in (9, 12, 16):
        s = build_cyclic("s1_size", N, SEED, N, VOCAB, d=4, condition="shuffle")
        g = s["gate"]
        assert g["degree_regular"] and g["has_nonadjacent"] and g["unique_cycle"]
        assert abs(g["corr_pos_subjfrac"]) < 1e-9          # first-named ⟂ position (Eulerian)
        assert g["corr_pos_slot_circular"] <= 0.25         # slot ⟂ circular position (shuffle)
        # mention count == degree for all -> frequency is position-invariant
        cnt = Counter()
        for c in s["cards"]:
            cnt[c["entity"]] += 1; cnt[c["entity_b"]] += 1
        assert set(cnt.values()) == {4}, cnt
        # every entity plays BOTH roles somewhere (no subject-only endpoint)
        firsts = {c["entity"] for c in s["cards"]}
        seconds = {c["entity_b"] for c in s["cards"]}
        assert firsts == seconds == set(s["latent_order"])


def test_cyclic_battery_keys_correct():
    """Cyclic successor/predecessor/distance/order keys recompute from cyclic_pos."""
    from icom.generator.bcs import build_cyclic
    from icom.generator.bcs_questions import make_cyclic_battery
    for N in (9, 12, 16):
        s = build_cyclic("s0_zib", N, SEED, N + 1, VOCAB, d=4, condition="shuffle")
        pos = s["cyclic_pos"]; at = {p: e for e, p in pos.items()}
        for q in make_cyclic_battery(s):
            f, k, te = q["family"], q["answer_key"], q["target_entities"]
            if f == "cyclic_successor":
                assert k == at[(pos[te[0]] + 1) % N]
            elif f == "cyclic_predecessor":
                assert k == at[(pos[te[0]] - 1) % N]
            elif f == "cyclic_distance":
                a, b = te; assert k == str((pos[b] - pos[a]) % N)
            elif f == "cyclic_order":
                x, a, b = te
                assert k == (a if (pos[a] - pos[x]) % N < (pos[b] - pos[x]) % N else b)


def test_metric_families_not_degenerate():
    """Keys/positions aren't constant: comparative answer uses both slots,
    betweenness middle appears in every shown position, count_between spans
    multiple values (a constant answer could otherwise score)."""
    from icom.generator.bcs_questions import make_battery
    cmp_pos, btw_pos, cb_vals = Counter(), Counter(), set()
    for idx in range(30):
        s = build_stimulus("s1_size", 12, SEED, idx, VOCAB, d=4, condition="shuffle")
        for q in make_battery(s):
            if q["family"] == "comparative_distance":
                shown = q["target_entities"][1:]
                cmp_pos["first" if q["answer_key"] == shown[0] else "second"] += 1
            elif q["family"] == "betweenness":
                shown = list(q["target_entities"]); btw_pos[shown.index(q["answer_key"])] += 1
            elif q["family"] == "count_between":
                cb_vals.add(int(q["answer_key"]))
    assert min(cmp_pos.values()) / sum(cmp_pos.values()) > 0.3   # both option slots used
    assert set(btw_pos) == {0, 1, 2}                             # middle lands in every slot
    assert len(cb_vals) >= 4                                     # distance values spread


def test_declared_list_shares_order_and_states_it():
    """E2/D2: declared-list stimulus shares entities+order with the derived (D1) cell, states the
    full order explicitly, carries no coherence twin, and stays span-extractable."""
    kw = dict(family="s0_zib", n_items=9, seed=SEED, idx=0, vocab=VOCAB, condition="shuffle")
    d1 = build_stimulus(**kw, declared=None)
    d2 = build_stimulus(**kw, declared="list")
    assert d1["latent_order"] == d2["latent_order"]              # content-matched
    assert d2["stimulus_id"] != d1["stimulus_id"]
    assert d2["declared"] == "list" and d2["incoherent"] is False
    # the order is stated in rank sequence
    order = d2["latent_order"]
    pos = [d2["prompt"].index(f"the {e}") for e in order]        # first mention = list position
    assert pos == sorted(pos)                                    # list is in latent-order sequence
    # span-extractable: each card text present; readout (last mention) is the roster, not the list
    roster = d2["prompt"].rfind("Entities:")
    assert all(d2["prompt"].find(c["text"]) >= 0 for c in d2["cards"])
    assert all(d2["prompt"].rfind(f"the {e}") >= roster for e in order)     # readout = roster
    assert all(d2["prompt"].find(f"the {e}") < roster for e in order)       # card = ordered list
