#!/usr/bin/env python
"""Step 4/4: pooled acts → coordinate → 2×2 quality readout → geometry.parquet.

Two variants per (stimulus, layer, scheme):
  raw        — coordinate on the pooled points as-is
  posproj    — after projecting out the slot-prototype positional subspace
               (estimated cross-stimulus within each model×family×condition
               group; leave-one-stimulus-out is unnecessary at k=3 << n_stim)

Prints a summary: per (model, family, condition, scheme) the best layer by
mean |q_content_partial| with both variants. CPU-only; run on the DGX cores.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from icom.geometry.curve import fit_coordinate
from icom.geometry.position import project_out, slot_prototype_subspace
from icom.geometry.quality import bootstrap_sd, quality_readout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True, help="acts root (contains <model>/ dirs)")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--pos-k", type=int, default=3)
    ap.add_argument("--bootstrap-layers", default="best",
                    help="'best' = bootstrap only at each group's best layer; 'none'")
    args = ap.parse_args()

    acts_root = Path(args.acts)
    files = sorted(acts_root.glob("*/*.npz"))
    print(f"{len(files)} activation files")

    # group by (model, family, condition) for the positional subspace
    groups: dict[tuple, list] = defaultdict(list)
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        groups[(meta["model"], meta["family"], meta["condition"])].append((f, meta))

    rows = []
    for (model, family, condition), members in sorted(groups.items()):
        print(f"--- {model} / {family} / {condition} ({len(members)} stimuli)")
        # load all pooled arrays for the group once
        data = []
        for f, meta in members:
            z = np.load(f, allow_pickle=False)
            schemes = [k for k in ("name", "marker", "last_token", "card_mean") if k in z]
            data.append({"id": f.stem, "meta": meta, "ranks": z["ranks"], "slots": z["slots"],
                         **{s: z[s] for s in schemes}})
        schemes = [k for k in ("name", "marker", "last_token", "card_mean") if k in data[0]]
        n_layers = data[0][schemes[0]].shape[1]

        for scheme in schemes:
            for layer in range(n_layers):
                pts = [d[scheme][:, layer, :].astype(np.float64) for d in data]
                slots = [d["slots"] for d in data]
                basis = slot_prototype_subspace(pts, slots, k=args.pos_k)
                for d, P in zip(data, pts):
                    # shuffle row storage order: defense against any index-
                    # correlated artifact (rows are stored in latent order)
                    perm = np.random.default_rng(
                        int(d["id"][:8], 16) ^ layer).permutation(len(d["ranks"]))
                    ranks_p, slots_p = d["ranks"][perm], d["slots"][perm]
                    for variant, X in (("raw", P[perm]), ("posproj", project_out(P, basis)[perm])):
                        t, diag = fit_coordinate(X)
                        q = quality_readout(t, ranks_p, slots_p,
                                            null_seed=int(d["id"][:8], 16) ^ layer)
                        rows.append({
                            "model": model, "family": family, "condition": condition,
                            "scheme": scheme, "layer": layer, "variant": variant,
                            "stimulus_id": d["id"], "content_key": d["meta"]["content_key"],
                            "n_items": d["meta"]["n_items"], **q, **diag,
                        })

    df = pd.DataFrame(rows)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows → {args.out}")

    # summary: best layer per group×scheme×variant by mean |partial|,
    # with the permuted-ranks null alongside (|rho| is bias-inflated at small N —
    # the signal is the GAP over the null, not the raw value). NaN rows
    # (degenerate clouds, undefined partials in fwd/rev) are excluded.
    df["abs_q"] = df["q_content_partial"].abs()
    df["abs_null"] = df["q_content_null"].abs()
    valid = df[df["abs_q"].notna()]
    summ = (valid.groupby(["model", "family", "condition", "scheme", "variant", "layer"])
            [["abs_q", "abs_null"]].mean().reset_index())
    best = summ.loc[summ.groupby(["model", "family", "condition", "scheme", "variant"])
                    ["abs_q"].idxmax()]
    print("\n=== best layer by mean |q_content_partial| (vs permuted-ranks null) ===")
    for _, r in best.iterrows():
        print(f"{r['model']:14s} {r['family']:10s} {r['condition']:8s} {r['scheme']:10s} "
              f"{r['variant']:7s} L{int(r['layer']):>2d}  |q|={r['abs_q']:.3f}  null={r['abs_null']:.3f}")

    # bootstrap reliability at each group's best raw layer (cheap, informative)
    if args.bootstrap_layers == "best":
        print("\n=== bootstrap SD of q_content_partial at best raw layer ===")
        for (model, family, condition), members in sorted(groups.items()):
            sub = best[(best["model"] == model) & (best["family"] == family)
                       & (best["condition"] == condition) & (best["variant"] == "raw")]
            if sub.empty:
                continue
            r = sub.loc[sub["abs_q"].idxmax()]
            scheme, layer = r["scheme"], int(r["layer"])
            f, meta = members[0]
            z = np.load(f, allow_pickle=False)
            sd = bootstrap_sd(z[scheme][:, layer, :].astype(np.float64), z["ranks"], z["slots"])
            print(f"{model:14s} {family:10s} {condition:8s} {scheme:10s} L{layer:>2d} sd≈{sd:.3f}")


if __name__ == "__main__":
    main()
