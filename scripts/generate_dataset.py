#!/usr/bin/env python
"""Step 1/4: generate stimuli + question batteries from configs/generation.yaml.

Deterministic from (config, seed): reads COMMITTED pools (predicates, entities),
never calls an API. Emits:
  <out>/stimuli.jsonl    one record per (content × condition)
  <out>/questions.jsonl  one record per question (shared across conditions
                         of one content via content_key)
  <out>/meta.json        provenance
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import yaml

from icom.generator.questions import make_battery
from icom.generator.schemas import Condition, StimulusFamily
from icom.generator.stimuli import make_condition_set


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default=None, help="override: comma-separated")
    ap.add_argument("--n-grid", default=None, help="override: comma-separated ints")
    ap.add_argument("--per-cell", type=int, default=None)
    ap.add_argument("--conditions", default=None, help="override: comma-separated")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["seed"]
    families = [StimulusFamily(f) for f in
                (args.families.split(",") if args.families else cfg["families"])]
    n_grid = ([int(x) for x in args.n_grid.split(",")] if args.n_grid else cfg["n_items_grid"])
    per_cell = args.per_cell or cfg["stimuli_per_cell"]
    conditions = [Condition(c) for c in
                  (args.conditions.split(",") if args.conditions else cfg["conditions"])]
    qcfg = cfg["questions"]

    pools_cfg = cfg["pools"]
    predicates = json.load(open(pools_cfg["predicates"]))["predicates"]
    entities_pool = json.load(open(pools_cfg["entities"]))["names"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_stim = n_q = 0
    with open(out / "stimuli.jsonl", "w") as fs, open(out / "questions.jsonl", "w") as fq:
        for family in families:
            for n_items in n_grid:
                for idx in range(per_cell):
                    stims, latent, key = make_condition_set(
                        family, n_items, seed, idx, entities_pool, predicates, conditions)
                    for st in stims.values():
                        rec = dataclasses.asdict(st)
                        rec["stimulus_id"] = st.stimulus_id
                        fs.write(json.dumps(rec) + "\n")
                        n_stim += 1
                    for q in make_battery(
                        latent, key, family, seed,
                        pairwise_per_bin=qcfg["pairwise_per_distance_bin"],
                        distance_bins=qcfg["distance_bins"],
                        adjacency_max=qcfg["adjacency_max"],
                        rank_max=qcfg["rank_max"],
                    ):
                        fq.write(json.dumps(dataclasses.asdict(q)) + "\n")
                        n_q += 1

    meta = {
        "config": cfg, "families": [f.value for f in families], "n_grid": n_grid,
        "per_cell": per_cell, "conditions": [c.value for c in conditions],
        "n_stimuli": n_stim, "n_questions": n_q,
        "pool_meta": json.load(open(pools_cfg["predicates"]))["meta"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n_stim} stimuli, {n_q} questions → {out}")


if __name__ == "__main__":
    main()
