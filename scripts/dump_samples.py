#!/usr/bin/env python
"""Dump curated, human-readable SAMPLES of every stimulus & question type.

Writes docs/samples/sample_stimuli.md (readable: full prompt + a few keyed
questions per family) and docs/samples/samples.jsonl (full stimulus+battery
records). Deterministic; regenerate with:

  python scripts/dump_samples.py --out docs/samples
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from icom.generator.bcs import (build_stimulus, build_partial_order, build_grid2d,
                                build_cyclic)
from icom.generator.bcs_questions import (make_battery, make_partial_battery,
                                          make_grid_battery, make_cyclic_battery)

SEED = 20260724


def q_by_family(qs, per=2):
    by = defaultdict(list)
    for q in qs:
        if len(by[q["family"]]) < per:
            by[q["family"]].append(q)
    return by


def render(title, stim, qs, note="", max_cards=None):
    L = [f"## {title}", ""]
    if note:
        L += [f"*{note}*", ""]
    order = stim["latent_order"]
    if stim.get("structure") == "cyclic":
        L.append("**cyclic order (pos:entity):** " +
                 " → ".join(f"{stim['cyclic_pos'][e]}:{e}" for e in order) + " → (wrap)")
    elif stim.get("structure") == "grid2d":
        L.append("**coords (entity: x,y):** " +
                 ", ".join(f"{e}:{stim['coord_x'][e]},{stim['coord_y'][e]}" for e in order[:8]) + " …")
    elif stim.get("structure") == "partial_order":
        L.append("**chains:** " + "; ".join(
            "chain%d=[%s]" % (c, ",".join(e for e in order if stim["chain_of"][e] == c))
            for c in sorted(set(stim["chain_of"].values()))))
    else:
        L.append("**latent order (rank 1..N):** " + " < ".join(f"{i+1}:{e}" for i, e in enumerate(order)))
    L += ["", "**prompt:**", "```"]
    prompt = stim["prompt"]
    if max_cards:
        lines = prompt.split("\n")
        prompt = "\n".join(lines[:max_cards]) + f"\n… ({len(lines)} lines total)"
    L += [prompt, "```", "", "**questions (family → key):**"]
    for fam, items in q_by_family(qs).items():
        for q in items:
            L.append(f"- `{fam}` — {q['text']}  **→ {q['answer_key']}**")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/samples")
    ap.add_argument("--pool", default="data/pools/entities_v1.json")
    args = ap.parse_args()
    vocab = json.load(open(args.pool))["names"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    blocks, records = [], []

    def add(title, stim, qs, note="", max_cards=None):
        blocks.append(render(title, stim, qs, note, max_cards))
        records.append({"title": title, "stimulus": stim, "questions": qs})

    # total order — semantics, difficulty, condition, new heat family, N=24
    s = build_stimulus("s1_size", 9, SEED, 1, vocab, difficulty="easy", condition="shuffle")
    add("Total order · S1 size · N=9 · easy · shuffle", s, make_battery(s),
        "The primary design. Interior = ranks 3..N−2. Metric families included.")
    s = build_stimulus("s0_zib", 9, SEED, 2, vocab, difficulty="hard", condition="shuffle")
    add("Total order · S0 zib (symbolic) · N=9 · hard · shuffle", s, make_battery(s),
        "S0 = arbitrary transitive relation declared in-context (no magnitude lexicon).")
    s = build_stimulus("s1_size", 9, SEED, 1, vocab, difficulty="easy", condition="forward")
    add("Total order · S1 size · N=9 · easy · FORWARD (ceiling/control)", s, [],
        "Same content as the shuffle twin; forward = cards sorted by rank (position leaks order).")
    s = build_stimulus("s1_heat", 9, SEED, 4, vocab, difficulty="easy", condition="shuffle")
    add("Total order · S1 heat (cooler/hotter) · N=9 · easy · shuffle", s, make_battery(s),
        "Extra S1 relation family (semantics-gradient robustness).")
    s = build_stimulus("s1_size", 24, SEED, 0, vocab, difficulty="hard", condition="shuffle")
    add("Total order · S1 size · N=24 · hard · shuffle (length stress)", s, [],
        "Large-N stress cell (SSM state-bottleneck test).", max_cards=10)

    # coherence null
    z = build_stimulus("s1_size", 9, SEED, 2, vocab, condition="shuffle", incoherent=True)
    add("Coherence-null twin · S1 size · N=9", z, [],
        "An INVALID cycle is injected (no valid total order). Must decode at chance. "
        "Distinct from the cyclic ring below (which is a VALID cycle).", max_cards=12)

    # non-total structures
    s = build_partial_order("s1_size", 2, 0, SEED, 0, vocab, condition="shuffle", chain_lens=[5, 4])
    add("Partial order · 2 chains [5,4] · S1 size · shuffle", s, make_partial_battery(s),
        "Cross-chain pairs are INCOMPARABLE (order_query mixes comparable + 'undetermined').")
    s = build_grid2d("s1_size", "s1_loud", 9, SEED, 0, vocab, condition="shuffle")
    add("2-D grid · size × loud · N=9 · shuffle", s, make_grid_battery(s),
        "Two INDEPENDENT global orders over the same entities; per-axis pairwise.")

    # cyclic ring (nonlinear litmus)
    s = build_cyclic("s1_size", 12, SEED, 3, vocab, condition="shuffle")
    add("Cyclic ring · S1 size · N=12 · shuffle (nonlinear litmus)", s, make_cyclic_battery(s),
        "A VALID single cycle (positions wrap). No endpoints; first-named ⟂ position (Eulerian). "
        "Predicts a CIRCULAR manifold.")
    s = build_cyclic("s0_zib", 9, SEED, 5, vocab, condition="shuffle")
    add("Cyclic ring · S0 zib · N=9 · shuffle", s, make_cyclic_battery(s)[:6])

    (out / "sample_stimuli.md").write_text(
        "# BCS sample stimuli & questions\n\n"
        "*Curated, deterministic examples of every stimulus type and question "
        "family (regenerate: `python scripts/dump_samples.py`). Full records in "
        "`samples.jsonl`.*\n\n" + "\n---\n\n".join(blocks) + "\n")
    with open(out / "samples.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(blocks)} sample blocks -> {out/'sample_stimuli.md'} + samples.jsonl")


if __name__ == "__main__":
    main()
