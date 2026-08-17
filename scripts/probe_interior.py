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


def scheme_key(z, scheme):
    """Resolve the pooling-scheme array key: canonical 'name'/'readout'/... (extract_activations)
    or the legacy 'loc_<scheme>' layout. Returns None if absent."""
    if scheme in z.files:
        return scheme
    if ("loc_" + scheme) in z.files:
        return "loc_" + scheme
    return None


def load(acts, model, family, condition, layer, scheme, is_null=False, difficulty="all"):
    """is_null selects the coherence-null twins; difficulty in {all,easy,hard} filters by the
    stored difficulty label. Backward-compatible with older acts whose meta lacks is_null and
    whose scheme arrays use the 'loc_' prefix."""
    Xs, ranks, groups = [], [], []
    for gi, f in enumerate(sorted((Path(acts) / model).glob("*.npz"))):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        key = scheme_key(z, scheme)
        if m["family"] != family or m["condition"] != condition or key is None:
            continue
        if bool(m.get("is_null", False)) != is_null:
            continue
        if difficulty != "all" and m.get("difficulty") not in (difficulty, None):
            continue
        Xs.append(z[key][:, layer, :].astype(np.float32))
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
    ap.add_argument("--difficulty", default="all", choices=["all", "easy", "hard"],
                    help="filter stimuli by difficulty label (from meta)")
    ap.add_argument("--coherence", action="store_true",
                    help="also decode the coherence-null twins (is_null) and report the real-twin increment")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def ld(model, layer, is_null=False):
        return load(args.acts, model, args.family, args.condition, layer, args.scheme,
                    is_null=is_null, difficulty=args.difficulty)

    rows = []
    for model in args.models.split(","):
        probe0 = ld(model, 0)
        if probe0 is None:
            print(f"{model}: no data for {args.family}/{args.condition}/{args.scheme}/{args.difficulty}")
            continue
        n_layers = None
        for gi, f in enumerate(sorted((Path(args.acts) / model).glob("*.npz"))):
            z = np.load(f, allow_pickle=False)
            k = scheme_key(z, args.scheme)
            if k is not None:
                n_layers = z[k].shape[1]; break
        layers = [args.layer] if args.layer is not None else range(n_layers)
        best = None
        for L in layers:
            X, ranks, groups = ld(model, L)
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
        # coherence increment (real interior - twin interior at the SAME layer): the load-bearing
        # BCS metric — a coherent order should decode ABOVE its incoherent-cycle twin (local chaining).
        coh = twin_i = None
        if args.coherence:
            T = ld(model, L, is_null=True)
            if T is not None:
                Xt, rt, gt = T; it = (rt >= 3) & (rt <= int(rt.max()) - 2)
                twin_i, _, _ = probe_with_null(Xt[it], rt[it], gt[it], args.n_perm)
                coh = ri - twin_i
        rows.append(dict(model=model, family=args.family, condition=args.condition,
                         scheme=args.scheme, difficulty=args.difficulty, layer=L, N=N,
                         all_probe=round(ra, 3), all_null95=round(na, 3), all_p=round(pa, 4),
                         interior_probe=round(ri, 3), interior_null95=round(ni, 3), interior_p=round(pi, 4),
                         twin_interior=round(twin_i, 3) if twin_i is not None else None,
                         coherence_increment=round(coh, 3) if coh is not None else None,
                         interior_n_per_stim=int(interior.sum() // len(np.unique(groups)))))
        verdict = "SURVIVES" if (ri > ni and pi < 0.05) else "COLLAPSES (endpoint artifact)"
        cohs = f" COH(real-twin)={coh:+.3f}" if coh is not None else ""
        print(f"{model:14s} {args.family}/{args.condition} {args.scheme} diff={args.difficulty} L{L} N={N} | "
              f"ALL={ra:.2f}(p={pa:.3f}) INTERIOR={ri:.2f}(null95={ni:.2f},p={pi:.3f}){cohs} -> {verdict}", flush=True)

    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(args.out)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
