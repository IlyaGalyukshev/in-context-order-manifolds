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
