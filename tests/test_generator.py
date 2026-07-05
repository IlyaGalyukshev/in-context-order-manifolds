"""Generator invariants — the contract the implementation must satisfy.

Each test encodes a design commitment from the plan critique; breaking one
invalidates downstream analyses, so treat failures as release blockers.
"""

import json

import numpy as np
import pytest

from icom.generator.questions import make_battery
from icom.generator.schemas import Condition, QuestionFamily, StimulusFamily
from icom.generator.stimuli import MAX_SHUFFLE_ABS_RHO, make_condition_set

SEED = 20260704
CONDITIONS = [Condition.FORWARD, Condition.REVERSE, Condition.SHUFFLE]
VOCAB = [f"zz{c1}{c2}orp" for c1 in "bcdfglmnprst" for c2 in "aeiou"]  # 60 fake nonces
# digit-free like the real pool (supervisor bans numbers; the relational
# no-absolute-coordinate invariant depends on this)
PREDICATES = [f"polished a {a}{b}ic plate" for a in "bcdfglmnpr" for b in "aeiou"]


def _set(family, n=16, idx=0, conditions=CONDITIONS):
    return make_condition_set(family, n, SEED, idx, VOCAB, PREDICATES, conditions)


@pytest.mark.parametrize("family", list(StimulusFamily))
def test_conditions_share_content_and_questions(family):
    """All conditions of one latent order: same cards as a set, same total
    length; only presentation slots differ. Questions attach to content_key."""
    if family is StimulusFamily.RELATIONAL:
        pass
    stims, latent, key = _set(family)
    texts = {c: sorted(card.text for card in s.cards) for c, s in stims.items()}
    assert texts[Condition.FORWARD] == texts[Condition.REVERSE] == texts[Condition.SHUFFLE]
    lengths = {len(s.prompt) for s in stims.values()}
    assert len(lengths) == 1, "prompt char length must be identical across conditions"
    assert all(s.content_key == key for s in stims.values())


@pytest.mark.parametrize("family", [StimulusFamily.DATED, StimulusFamily.TAGGED])
def test_fixed_width_markers(family):
    """Markers are fixed-width strings ('Day 07'), so no card's length varies
    with its rank via the marker."""
    stims, _, _ = _set(family)
    s = stims[Condition.FORWARD]
    widths = {len(m) for m in s.markers.values()}
    assert widths == {6}
    for card in s.cards:
        assert card.text.split(":")[0] == s.markers[card.entity]


def test_relational_family_needs_transitive_closure():
    """RELATIONAL stimuli contain only adjacent-pair statements: no card
    reveals a non-adjacent relation or an absolute coordinate (no digits)."""
    stims, latent, _ = _set(StimulusFamily.RELATIONAL)
    rank = {e: i + 1 for i, e in enumerate(latent)}
    s = stims[Condition.SHUFFLE]
    assert len(s.cards) == len(latent) - 1
    for card in s.cards:
        assert card.entity_b is not None
        assert rank[card.entity_b] - rank[card.entity] == 1, "non-adjacent pair leaked"
        assert not any(ch.isdigit() for ch in card.text), "absolute coordinate leaked"
    # every entity mentioned, chain covers all adjacent pairs exactly once
    pairs = {(c.entity, c.entity_b) for c in s.cards}
    assert pairs == {(latent[i], latent[i + 1]) for i in range(len(latent) - 1)}


def test_pairwise_distance_stratification():
    """Forced-choice pairwise: per-bin quotas, key = the earlier entity, and
    the earlier entity is named first in ~half the questions (position balance)."""
    _, latent, key = _set(StimulusFamily.TAGGED)
    qs = make_battery(latent, key, StimulusFamily.TAGGED, SEED,
                      pairwise_per_bin=3, distance_bins=[1, "2-3", "4-7", "8+"])
    pw = [q for q in qs if q.family is QuestionFamily.PAIRWISE]
    assert len(pw) == 12
    bins = {(1, 1): 0, (2, 3): 0, (4, 7): 0, (8, 15): 0}
    for q in pw:
        for (lo, hi) in bins:
            if lo <= q.rank_distance <= hi:
                bins[(lo, hi)] += 1
                break
    assert all(v == 3 for v in bins.values()), bins
    rank = {e: i + 1 for i, e in enumerate(latent)}
    first_named_earlier = 0
    for q in pw:
        a, b = q.target_entities
        assert q.answer_key in (a, b)
        assert q.answer_key == (a if rank[a] < rank[b] else b)
        first_named_earlier += q.answer_key == a
    # EXACT 50/50 position balance: "always pick first-named" must score chance.
    # (A judge caught the old 2:1 alternation as a trivial-baseline confound.)
    assert abs(first_named_earlier - len(pw) / 2) <= 0.5, \
        f"position balance broken: earlier named first {first_named_earlier}/{len(pw)}"


def test_mention_order_control_twin():
    """Recon has a mention-order control twin with the sentinel key — its true
    key is condition-dependent and computed at scoring time."""
    _, latent, key = _set(StimulusFamily.RELATIONAL)
    qs = make_battery(latent, key, StimulusFamily.RELATIONAL, SEED)
    recs = [q for q in qs if q.family is QuestionFamily.RECONSTRUCTION]
    assert len(recs) == 2
    assert any(q.answer_key == "MENTION_ORDER" for q in recs)
    assert any(q.answer_key == latent for q in recs)


def test_presentation_slots_recorded():
    """Every card carries latent_rank AND presentation_slot; in SHUFFLE the
    rank-slot correlation is near zero by construction; in FORWARD/REVERSE
    it is exactly ±1."""
    stims, _, _ = _set(StimulusFamily.TAGGED)
    for cond, expected in [(Condition.FORWARD, 1.0), (Condition.REVERSE, -1.0)]:
        ranks = [c.latent_rank for c in stims[cond].cards]
        slots = [c.presentation_slot for c in stims[cond].cards]
        assert abs(np.corrcoef(ranks, slots)[0, 1] - expected) < 1e-9
    sh = stims[Condition.SHUFFLE]
    rho = np.corrcoef([c.latent_rank for c in sh.cards],
                      [c.presentation_slot for c in sh.cards])[0, 1]
    assert abs(rho) <= MAX_SHUFFLE_ABS_RHO
    for s in stims.values():
        assert sorted(c.presentation_slot for c in s.cards) == list(range(1, len(s.cards) + 1))


def test_determinism():
    """Same (config, seed) → byte-identical stimuli and questions."""
    import dataclasses
    a_stims, a_lat, a_key = _set(StimulusFamily.RELATIONAL, idx=7)
    b_stims, b_lat, b_key = _set(StimulusFamily.RELATIONAL, idx=7)
    dump = lambda stims: json.dumps({c.value: dataclasses.asdict(s) for c, s in stims.items()},
                                    sort_keys=True)
    assert dump(a_stims) == dump(b_stims) and a_lat == b_lat and a_key == b_key
    qa = make_battery(a_lat, a_key, StimulusFamily.RELATIONAL, SEED)
    qb = make_battery(b_lat, b_key, StimulusFamily.RELATIONAL, SEED)
    assert [dataclasses.asdict(q) for q in qa] == [dataclasses.asdict(q) for q in qb]
    # different content idx → different content
    c_stims, c_lat, c_key = _set(StimulusFamily.RELATIONAL, idx=8)
    assert c_key != a_key and c_lat != a_lat


def test_battery_shared_across_conditions_by_construction():
    """The battery depends only on (latent order, content_key, family, seed) —
    generating it per condition would be a bug; this pins the signature."""
    _, latent, key = _set(StimulusFamily.DATED)
    q1 = make_battery(latent, key, StimulusFamily.DATED, SEED)
    q2 = make_battery(latent, key, StimulusFamily.DATED, SEED)
    assert [q.qid for q in q1] == [q.qid for q in q2]
    fams = {q.family for q in q1}
    assert fams == set(QuestionFamily)
