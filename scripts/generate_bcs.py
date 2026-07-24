#!/usr/bin/env python
"""Generate a v2 BCS dataset (CPU only). Writes stimuli.jsonl, questions.jsonl,
a coherence-null twin set, and a gate summary. Ready for extraction.

  python scripts/generate_bcs.py --out data/bcs_pilot \
      --families s0_zib,s1_size --n-grid 7,9,12 --per-cell 100 --degree 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icom.generator.bcs import build_stimulus, build_partial_order, build_grid2d
from icom.generator.bcs_questions import make_battery, make_partial_battery, make_grid_battery

SEED = 20260724


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/pools/entities_v1.json")
    ap.add_argument("--families", default="s0_zib,s1_size")
    ap.add_argument("--n-grid", default="7,9,12")
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--conditions", default="shuffle,forward")
    ap.add_argument("--balanced", action="store_true", help="circulant (hardest) variant")
    ap.add_argument("--with-null", action="store_true", default=True)
    ap.add_argument("--structures", action="store_true",
                    help="also emit partial-order (2 chains) + 2D-grid stimuli")
    ap.add_argument("--struct-per-cell", type=int, default=100)
    args = ap.parse_args()

    vocab = json.load(open(args.pool))["names"]
    fams = args.families.split(",")
    ngrid = [int(x) for x in args.n_grid.split(",")]
    conds = args.conditions.split(",")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    n_stim = n_q = n_null = 0
    gate_fail = 0
    with open(out / "stimuli.jsonl", "w") as fs, open(out / "questions.jsonl", "w") as fq, \
         open(out / "stimuli_null.jsonl", "w") as fn:
        for fam in fams:
            for N in ngrid:
                for idx in range(args.per_cell):
                    battery_written = False
                    for cond in conds:
                        st = build_stimulus(fam, N, SEED, idx, vocab, d=args.degree,
                                            balanced=args.balanced, condition=cond)
                        g = st["gate"]
                        base_ok = (g["degree_regular"] and g["unique_total_order"]
                                   and abs(g["corr_rank_subjfrac"]) < 1e-9
                                   and abs(g["corr_rank_mentions"]) < 1e-9)
                        # the position-decorrelation gate applies ONLY to shuffle
                        # (forward is position=order by design, not a failure)
                        slot_ok = (cond != "shuffle") or (abs(g["corr_rank_slot"]) <= 0.20)
                        if not (base_ok and slot_ok):
                            gate_fail += 1
                        fs.write(json.dumps(st) + "\n"); n_stim += 1
                        if not battery_written:  # battery shared across conditions
                            for q in make_battery(st):
                                fq.write(json.dumps(q) + "\n"); n_q += 1
                            battery_written = True
                    if args.with_null:
                        z = build_stimulus(fam, N, SEED, idx, vocab, d=args.degree,
                                           balanced=args.balanced, condition="shuffle",
                                           incoherent=True)
                        fn.write(json.dumps(z) + "\n"); n_null += 1

    n_struct = 0
    if args.structures:
        with open(out / "stimuli.jsonl", "a") as fs, open(out / "questions.jsonl", "a") as fq:
            for idx in range(args.struct_per_cell):
                for fam in fams:
                    for m in (4, 5, 6):  # chain length -> partial order 2 x m
                        for cond in conds:
                            st = build_partial_order(fam, 2, m, SEED, idx, vocab,
                                                     d=args.degree, condition=cond)
                            fs.write(json.dumps(st) + "\n"); n_struct += 1
                            if cond == conds[0]:
                                for q in make_partial_battery(st):
                                    fq.write(json.dumps(q) + "\n"); n_q += 1
                # two independent global orders over N entities (2D)
                for Ngrid in (9, 12):
                    for cond in conds:
                        st = build_grid2d("s1_size", "s1_loud", Ngrid, SEED, idx, vocab,
                                          d=args.degree, condition=cond)
                        fs.write(json.dumps(st) + "\n"); n_struct += 1
                        if cond == conds[0]:
                            for q in make_grid_battery(st):
                                fq.write(json.dumps(q) + "\n"); n_q += 1

    meta = {"families": fams, "n_grid": ngrid, "per_cell": args.per_cell, "degree": args.degree,
            "conditions": conds, "balanced": args.balanced, "n_stimuli": n_stim + n_struct,
            "n_total_order": n_stim, "n_structures": n_struct,
            "n_questions": n_q, "n_null": n_null, "gate_failures": gate_fail}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n_stim} stimuli, {n_q} questions, {n_null} null twins -> {out} "
          f"(gate_failures={gate_fail})")


if __name__ == "__main__":
    main()
