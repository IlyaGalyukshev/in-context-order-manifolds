#!/usr/bin/env python
"""Clincher for the relational rank signal: is the 0.42 name-token probe a
GLOBAL ordered manifold, or LOCAL neighbour-bleed? Plus real p-values and a
geometry->behaviour link. All on existing pilot activations.

Three tests, per (model, family, condition, scheme, best_layer):
1. Real permutation p with 200 within-stimulus label permutations.
2. Global-vs-local: cross-validated ridge predicts a rank coordinate; measure
   pairwise ordering accuracy of that coordinate RESOLVED BY RANK DISTANCE.
   GLOBAL manifold -> accuracy flat/high across distances (far pairs easy too).
   LOCAL neighbour-bleed -> high at distance 1, decays toward chance at large
   distances. Also report the within-stimulus Spearman distribution.
3. Geometry->behaviour: per stimulus, correlate the within-stimulus rank
   coordinate quality with that stimulus's behavioural reconstruction tau.

Neighbour-IDENTITY memorisation is already excluded by design: entities are
re-randomised per stimulus and CV groups by stimulus, so a probe cannot learn
"neighbour=X -> rank=Y"; the distance-resolved test checks the remaining
local-vs-global question.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# best layers from probe_rank.parquet (name scheme, shuffle)
CELLS = [
    ("qwen3-4b", "relational", "shuffle", "name", 22),
    ("qwen3-4b", "relational", "forward", "name", 20),   # global positive control
    ("qwen3-4b", "tagged", "shuffle", "name", 20),        # strong contrast
    ("olmo3-7b-inst", "relational", "shuffle", "name", 18),
    ("olmo3-7b-inst", "tagged", "shuffle", "name", 15),
]


def oob_predictions(Xr, y, groups, alpha=10.0):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oob = np.full(len(y), np.nan)
    for tr, te in gkf.split(Xr, y, groups):
        oob[te] = Ridge(alpha=alpha).fit(Xr[tr], y[tr]).predict(Xr[te])
    return oob


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--battery-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--pca", type=int, default=64)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    # behavioural reconstruction tau per stimulus (true-order recon:0)
    beh = {}
    for m in ("qwen3-4b", "olmo3-7b-inst"):
        for l in open(Path(args.battery_dir) / f"battery_{m}.jsonl"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r["q_family"] == "reconstruction" and ":reconstruction:1" not in r["qid"]:
                if r.get("tau") is not None and not (isinstance(r["tau"], float) and np.isnan(r["tau"])):
                    beh[(m, r["stimulus_id"])] = r["tau"]

    rows, dist_rows, link_rows = [], [], []
    for model, family, condition, scheme, layer in CELLS:
        files = []
        for f in sorted((Path(args.acts) / model).glob("*.npz")):
            meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
            if meta["family"] == family and meta["condition"] == condition:
                files.append(f)
        Xs, ranks, slots, groups, sids = [], [], [], [], []
        for gi, f in enumerate(files):
            z = np.load(f, allow_pickle=False)
            Xs.append(z[scheme][:, layer, :].astype(np.float32))
            ranks.append(z["ranks"]); slots.append(z["slots"])
            groups.append(np.full(len(z["ranks"]), gi)); sids.append((gi, f.stem))
        X = np.concatenate(Xs); ranks = np.concatenate(ranks)
        groups = np.concatenate(groups)
        y = (ranks - ranks.min()) / (ranks.max() - ranks.min())
        Xr = PCA(n_components=min(args.pca, X.shape[0] - 1),
                 random_state=0).fit_transform(StandardScaler().fit_transform(X))

        oob = oob_predictions(Xr, y, groups)
        real = abs(spearmanr(oob, y)[0])
        # 200-perm real p
        null = []
        for _ in range(args.n_perm):
            yp = y.copy()
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                yp[idx] = y[idx][rng.permutation(len(idx))]
            null.append(abs(spearmanr(oob_predictions(Xr, yp, groups), yp)[0]))
        null = np.array(null)
        p = (1 + int((null >= real).sum())) / (1 + args.n_perm)

        # distance-resolved pairwise ordering accuracy + within-stimulus spearman
        bins = {"d=1": [], "d=2-3": [], "d=4-7": [], "d=8+": []}
        within = []
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            pr, rk = oob[idx], ranks[idx]
            within.append(spearmanr(pr, rk)[0])
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    d = abs(rk[a] - rk[b])
                    correct = np.sign(pr[a] - pr[b]) == np.sign(rk[a] - rk[b])
                    key = "d=1" if d == 1 else "d=2-3" if d <= 3 else "d=4-7" if d <= 7 else "d=8+"
                    bins[key].append(correct)
            # geometry->behaviour link
            sid = dict(sids)[g]
            if (model, sid) in beh:
                link_rows.append({"model": model, "family": family, "condition": condition,
                                  "within_spearman": abs(spearmanr(pr, rk)[0]),
                                  "behav_tau": beh[(model, sid)]})
        rows.append({"model": model, "family": family, "condition": condition, "scheme": scheme,
                     "layer": layer, "probe": round(real, 3), "p_value": round(p, 4),
                     "null_mean": round(float(null.mean()), 3),
                     "within_mean": round(float(np.nanmean(within)), 3)})
        dr = {"model": model, "family": family, "condition": condition}
        for k, v in bins.items():
            dr[k] = round(float(np.mean(v)), 3) if v else None
        dist_rows.append(dr)
        print(f"{model:13s} {family:10s} {condition:8s} probe={real:.2f} p={p:.4f} "
              f"within={np.nanmean(within):+.2f} | dist "
              + " ".join(f"{k}:{dr[k]}" for k in ("d=1", "d=2-3", "d=4-7", "d=8+")), flush=True)

    out = Path(args.out)
    pd.DataFrame(rows).to_parquet(out)
    pd.DataFrame(dist_rows).to_csv(str(out).replace(".parquet", "_distance.csv"), index=False)
    link = pd.DataFrame(link_rows)
    link.to_csv(str(out).replace(".parquet", "_link.csv"), index=False)
    print("\n=== geometry->behaviour link (within-cell Spearman of probe-quality vs recon tau) ===")
    for (m, fam, cond), g in link.groupby(["model", "family", "condition"]):
        if len(g) >= 10 and g["within_spearman"].std() > 0 and g["behav_tau"].std() > 0:
            rho, pv = spearmanr(g["within_spearman"], g["behav_tau"])
            print(f"{m:13s} {fam:10s} {cond:8s} rho={rho:+.2f} p={pv:.3f} n={len(g)}")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
