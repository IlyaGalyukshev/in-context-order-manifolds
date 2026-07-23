#!/usr/bin/env python
"""Harden Beat 3: is the relational in-context order manifold recoverable by a
NONLINEAR probe at least as well as by the linear one (closing the "linear-in-PCs
could be missing structure" caveat), and does the GLOBAL signature survive?

Per (model, relational, shuffle, name, peak layer): CV OOB rank-Spearman for
  linear ridge | kernel ridge (RBF) | small MLP,
plus distance-resolved pairwise ordering accuracy for the strongest probe and a
100-permutation null. If nonlinear >= linear and far-pair accuracy stays high,
the flagship manifold is real and not a linear-probe artifact in either
direction. Tagged shuffle included as the strong-signal reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PEAK = {("qwen3-4b", "relational"): 22, ("qwen3-4b", "tagged"): 20,
        ("olmo3-7b-inst", "relational"): 18, ("olmo3-7b-inst", "tagged"): 15}


def make_probe(kind):
    if kind == "linear":
        return Ridge(alpha=10.0)
    if kind == "kernel":
        return KernelRidge(alpha=1.0, kernel="rbf", gamma=1.0 / 64)
    return MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=400, random_state=0)


def cv_oob(Xr, y, groups, kind):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oob = np.full(len(y), np.nan)
    for tr, te in gkf.split(Xr, y, groups):
        oob[te] = make_probe(kind).fit(Xr[tr], y[tr]).predict(Xr[te])
    return oob


def load(acts, model, family, condition, layer):
    Xs, ranks, groups = [], [], []
    for gi, f in enumerate(sorted((Path(acts) / model).glob("*.npz"))):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["family"] == family and meta["condition"] == condition:
            Xs.append(z["name"][:, layer, :].astype(np.float32))
            ranks.append(z["ranks"]); groups.append(np.full(len(z["ranks"]), gi))
    X = np.concatenate(Xs); ranks = np.concatenate(ranks); groups = np.concatenate(groups)
    return X, ranks, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="qwen3-4b,olmo3-7b-inst")
    ap.add_argument("--n-perm", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rows, dist_rows = [], []
    for model in args.models.split(","):
        for family in ("relational", "tagged"):
            layer = PEAK[(model, family)]
            X, ranks, groups = load(args.acts, model, family, "shuffle", layer)
            y = (ranks - ranks.min()) / (ranks.max() - ranks.min())
            Xr = PCA(n_components=64, random_state=0).fit_transform(StandardScaler().fit_transform(X))
            res = {}
            for kind in ("linear", "kernel", "mlp"):
                res[kind] = abs(spearmanr(cv_oob(Xr, y, groups, kind), y)[0])
            best = max(res, key=res.get)
            # distance-resolved accuracy for the best probe
            oob = cv_oob(Xr, y, groups, best)
            bins = {"d=1": [], "d=2-3": [], "d=4-7": [], "d=8+": []}
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        d = abs(ranks[idx[a]] - ranks[idx[b]])
                        ok = np.sign(oob[idx[a]] - oob[idx[b]]) == np.sign(ranks[idx[a]] - ranks[idx[b]])
                        k = "d=1" if d == 1 else "d=2-3" if d <= 3 else "d=4-7" if d <= 7 else "d=8+"
                        bins[k].append(ok)
            # perm null for best
            null = []
            for _ in range(args.n_perm):
                yp = y.copy()
                for g in np.unique(groups):
                    idx = np.where(groups == g)[0]
                    yp[idx] = y[idx][rng.permutation(len(idx))]
                op = cv_oob(Xr, yp, groups, best)
                ok = ~np.isnan(op)
                null.append(abs(spearmanr(op[ok], yp[ok])[0]))
            null = np.array(null)
            p = (1 + int((null >= res[best]).sum())) / (1 + args.n_perm)
            rows.append({"model": model, "family": family, "layer": layer,
                         "linear": round(res["linear"], 3), "kernel": round(res["kernel"], 3),
                         "mlp": round(res["mlp"], 3), "best": best, "best_val": round(res[best], 3),
                         "null_p95": round(float(np.percentile(null, 95)), 3), "p_value": round(p, 4)})
            dr = {"model": model, "family": family, "probe": best}
            for k, v in bins.items():
                dr[k] = round(float(np.mean(v)), 3)
            dist_rows.append(dr)
            r = rows[-1]
            print(f"{model:13s} {family:10s} L{layer} | linear={r['linear']:.2f} kernel={r['kernel']:.2f} "
                  f"mlp={r['mlp']:.2f} best={best}={r['best_val']:.2f} null95={r['null_p95']:.2f} p={r['p_value']:.3f}"
                  f" | dist " + " ".join(f"{k}:{dr[k]}" for k in ("d=1", "d=2-3", "d=4-7", "d=8+")), flush=True)

    pd.DataFrame(rows).to_parquet(args.out)
    pd.DataFrame(dist_rows).to_csv(str(args.out).replace(".parquet", "_distance.csv"), index=False)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
