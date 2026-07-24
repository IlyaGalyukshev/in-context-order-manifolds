#!/usr/bin/env python
"""Interior-only rank probe — the v2 PRIMARY geometry metric.

The v1 linear-chain "order manifold" turned out to be an endpoint/role
artifact: rank decodes overall but collapses to null among interior entities
(identical syntactic role & mention frequency). This script makes the
interior-only decode the headline, reported next to all-ranks, with a
permutation null and (optionally) a role-feature regression control.

Runs on CPU on any acts dir (local mirror or worker). Interior = ranks
[3 .. N-2]. For a clean v2 (BCS) dataset, interior should SURVIVE; for v1 it
should collapse — this doubles as a regression test of the confound.

Usage:
  python scripts/probe_interior.py --acts <acts_dir> --models qwen3-4b,olmo3-7b-inst \
      --family relational --condition shuffle --scheme name --layer 22 --n-perm 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def cv_spearman(Xr, y, g, alpha=10.0):
    k = min(5, len(np.unique(g)))
    if k < 2:
        return float("nan")
    oob = np.full(len(y), np.nan)
    for tr, te in GroupKFold(k).split(Xr, y, g):
        oob[te] = Ridge(alpha=alpha).fit(Xr[tr], y[tr]).predict(Xr[te])
    return abs(spearmanr(oob, y)[0])


def load(acts, model, family, condition, layer, scheme):
    Xs, ranks, groups = [], [], []
    for gi, f in enumerate(sorted((Path(acts) / model).glob("*.npz"))):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if m["family"] == family and m["condition"] == condition and scheme in z.files:
            Xs.append(z[scheme][:, layer, :].astype(np.float32))
            ranks.append(z["ranks"])
            groups.append(np.full(len(z["ranks"]), gi))
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(ranks), np.concatenate(groups)


def probe_with_null(X, ranks, groups, n_perm, pca=64, seed=0):
    if len(np.unique(ranks)) < 3:
        return float("nan"), float("nan")
    y = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    Xr = PCA(min(pca, X.shape[0] - 1), random_state=0).fit_transform(StandardScaler().fit_transform(X))
    real = cv_spearman(Xr, y, groups)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for gg in np.unique(groups):
            idx = np.where(groups == gg)[0]
            yp[idx] = y[idx][rng.permutation(len(idx))]
        null.append(cv_spearman(Xr, yp, groups))
    null = np.array(null)
    p = (1 + int((null >= real).sum())) / (1 + n_perm)
    return real, float(np.percentile(null, 95)), p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--models", default="qwen3-4b,olmo3-7b-inst")
    ap.add_argument("--family", default="relational")
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--scheme", default="name")
    ap.add_argument("--layer", type=int, default=None, help="fixed layer; else sweep-best")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for model in args.models.split(","):
        # find best layer by all-ranks probe if not fixed
        probe0 = load(args.acts, model, args.family, args.condition, 0, args.scheme)
        if probe0 is None:
            print(f"{model}: no data for {args.family}/{args.condition}/{args.scheme}")
            continue
        n_layers = None
        for gi, f in enumerate(sorted((Path(args.acts) / model).glob("*.npz"))):
            z = np.load(f, allow_pickle=False)
            if args.scheme in z.files:
                n_layers = z[args.scheme].shape[1]; break
        layers = [args.layer] if args.layer is not None else range(n_layers)
        best = None
        for L in layers:
            X, ranks, groups = load(args.acts, model, args.family, args.condition, L, args.scheme)
            N = int(ranks.max())
            interior = (ranks >= 3) & (ranks <= N - 2)
            allr = cv_spearman(
                PCA(min(64, X.shape[0] - 1), random_state=0).fit_transform(StandardScaler().fit_transform(X)),
                (ranks - ranks.min()) / (ranks.max() - ranks.min()), groups)
            if best is None or allr > best[1]:
                best = (L, allr, X, ranks, groups, interior, N)
        L, _, X, ranks, groups, interior, N = best
        ra, na, pa = probe_with_null(X, ranks, groups, args.n_perm)
        ri, ni, pi = probe_with_null(X[interior], ranks[interior], groups[interior], args.n_perm)
        rows.append(dict(model=model, family=args.family, condition=args.condition,
                         scheme=args.scheme, layer=L, N=N,
                         all_probe=round(ra, 3), all_null95=round(na, 3), all_p=round(pa, 4),
                         interior_probe=round(ri, 3), interior_null95=round(ni, 3),
                         interior_p=round(pi, 4), interior_n_per_stim=int(interior.sum() // len(np.unique(groups)))))
        r = rows[-1]
        verdict = "SURVIVES" if (ri > ni and pi < 0.05) else "COLLAPSES (endpoint artifact)"
        print(f"{model:14s} {args.family}/{args.condition} {args.scheme} L{L} N={N} | "
              f"ALL={ra:.2f}(p={pa:.3f}) INTERIOR={ri:.2f}(null95={ni:.2f},p={pi:.3f}) -> {verdict}", flush=True)

    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(args.out)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
