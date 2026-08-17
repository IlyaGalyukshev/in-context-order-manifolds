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


def oof_pred(X, ranks, groups, pca=64):
    """OOF ridge predictions per entity (for bootstrapping the Spearman over stimuli)."""
    y = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    Xr = PCA(min(pca, X.shape[0] - 1), random_state=0).fit_transform(StandardScaler().fit_transform(X))
    o = np.full(len(y), np.nan)
    k = min(5, len(np.unique(groups)))
    for tr, te in GroupKFold(k).split(Xr, y, groups):
        o[te] = Ridge(alpha=10.).fit(Xr[tr], y[tr]).predict(Xr[te])
    return o, y


def boot_coherence(pr, yr, gr, pt, yt, gt, B, seed=0):
    """Bootstrap CI on the coherence increment (real interior |Spearman| − twin interior |Spearman|),
    resampling stimuli independently in each pool (twins are separate stimuli)."""
    rng = np.random.default_rng(seed)
    ur = {u: np.where(gr == u)[0] for u in np.unique(gr)}
    ut = {u: np.where(gt == u)[0] for u in np.unique(gt)}
    kr, kt = list(ur), list(ut)

    def sp(p, y, idx):
        r = spearmanr(p[idx], y[idx])[0]
        return abs(r) if r == r else 0.0
    vals = []
    for _ in range(B):
        ri = np.concatenate([ur[kr[j]] for j in rng.integers(0, len(kr), len(kr))])
        ti = np.concatenate([ut[kt[j]] for j in rng.integers(0, len(kt), len(kt))])
        vals.append(sp(pr, yr, ri) - sp(pt, yt, ti))
    lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    return [round(float(lo), 3), round(float(hi), 3)], bool(lo > 0)


def load_all(acts, model, family, condition, scheme, is_null=False, difficulty="all", n_items=None):
    """Every layer at once: X [Σn, L, D], ranks, groups — so per-layer decodes + held-out layer
    selection reuse one load instead of re-reading npz per layer. n_items filters length (for
    the signal-vs-N curve) so mixed-N acts are not pooled."""
    Xs, ranks, groups = [], [], []
    for gi, f in enumerate(sorted((Path(acts) / model).glob("*.npz"))):
        z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"]))
        key = scheme_key(z, scheme)
        if m["family"] != family or m["condition"] != condition or key is None:
            continue
        if bool(m.get("is_null", False)) != is_null:
            continue
        if n_items is not None and int(m.get("n_items", 0)) != int(n_items):
            continue
        if difficulty != "all" and m.get("difficulty") not in (difficulty, None):
            continue
        Xs.append(z[key].astype(np.float32)); ranks.append(z["ranks"]); groups.append(np.full(len(z["ranks"]), gi))
    if not Xs:
        return None
    return np.concatenate(Xs, 0), np.concatenate(ranks), np.concatenate(groups)


def _reduce(X, pca=64):
    return PCA(min(pca, X.shape[0] - 1), random_state=0).fit_transform(StandardScaler().fit_transform(X))


def _ynorm(ranks):
    return (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-9)


def boot_spearman_ci(pred, y, groups, B, seed=0):
    """Bootstrap CI on |Spearman| by resampling stimuli (groups) over fixed OOF predictions."""
    rng = np.random.default_rng(seed)
    idx = {u: np.where(groups == u)[0] for u in np.unique(groups)}; ks = list(idx)
    vals = []
    for _ in range(B):
        sel = np.concatenate([idx[ks[j]] for j in rng.integers(0, len(ks), len(ks))])
        r = spearmanr(pred[sel], y[sel])[0]; vals.append(abs(r) if r == r else 0.0)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


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
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap CI (over stimuli) on the interior decode and, with --coherence, the increment (e.g. 2000)")
    ap.add_argument("--seed", type=int, default=0, help="seed for held-out layer split + bootstrap")
    ap.add_argument("--n-items", type=int, default=None, help="filter to one length N (for the signal-vs-N curve)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def ld(model, layer, is_null=False):
        return load(args.acts, model, args.family, args.condition, layer, args.scheme,
                    is_null=is_null, difficulty=args.difficulty)

    rows = []
    for model in args.models.split(","):
        A = load_all(args.acts, model, args.family, args.condition, args.scheme, difficulty=args.difficulty, n_items=args.n_items)
        if A is None:
            print(f"{model}: no data for {args.family}/{args.condition}/{args.scheme}/{args.difficulty}"); continue
        Xall, ranks, groups = A; n_layers = Xall.shape[1]; N = int(ranks.max())
        interior = (ranks >= 3) & (ranks <= N - 2)
        ug = np.unique(groups); n_stim = len(ug)
        Xi, yi, gi_ = Xall[interior], _ynorm(ranks[interior]), groups[interior]

        def dec_at(L, gsub=None):
            m = np.ones(len(yi), bool) if gsub is None else np.isin(gi_, gsub)
            if m.sum() < 5 or len(np.unique(yi[m])) < 3:
                return float("nan")
            return cv_spearman(_reduce(Xi[m, L, :]), yi[m], gi_[m])

        layers = [args.layer] if args.layer is not None else list(range(n_layers))
        full = np.array([dec_at(L) for L in layers])
        L_arg = layers[int(np.nanargmax(full))]; ri_arg = float(np.nanmax(full))   # OPTIMISTIC (argmax on all data)

        # HELD-OUT layer selection (leak-free MAIN number): choose the layer on half the stimuli,
        # decode the OTHER half at it; swap; average. Removes the layer-selection leak that argmax has.
        if args.layer is None and n_stim >= 8:
            perm = np.random.default_rng(args.seed).permutation(ug)
            gA, gB = perm[:n_stim // 2], perm[n_stim // 2:]
            LA = layers[int(np.nanargmax([dec_at(L, gA) for L in layers]))]
            LB = layers[int(np.nanargmax([dec_at(L, gB) for L in layers]))]
            ri_ho = float(np.nanmean([dec_at(LA, gB), dec_at(LB, gA)]))
        else:
            ri_ho = ri_arg

        # characterisation at the argmax layer: perm null + bootstrap CI + n
        ri, ni, pi = probe_with_null(Xi[:, L_arg, :], ranks[interior], gi_, args.n_perm)
        ra, na, pa = probe_with_null(Xall[:, L_arg, :], ranks, groups, args.n_perm)
        ri_ci = None
        if args.bootstrap:
            pr, yr = oof_pred(Xi[:, L_arg, :], ranks[interior], gi_)
            ri_ci = boot_spearman_ci(pr, yr, gi_, args.bootstrap, args.seed)

        coh = twin_i = None; coh_ci = None; coh_sig = None
        if args.coherence:
            T = load_all(args.acts, model, args.family, args.condition, args.scheme, is_null=True, difficulty=args.difficulty, n_items=args.n_items)
            if T is not None:
                Xt, rt, gt = T; it = (rt >= 3) & (rt <= int(rt.max()) - 2)
                twin_i, _, _ = probe_with_null(Xt[it][:, L_arg, :], rt[it], gt[it], args.n_perm)
                coh = ri - twin_i
                if args.bootstrap:
                    prc, yrc = oof_pred(Xi[:, L_arg, :], ranks[interior], gi_)
                    pt, yt = oof_pred(Xt[it][:, L_arg, :], rt[it], gt[it])
                    coh_ci, coh_sig = boot_coherence(prc, yrc, gi_, pt, yt, gt[it], args.bootstrap)

        rows.append(dict(model=model, family=args.family, condition=args.condition, scheme=args.scheme,
                         difficulty=args.difficulty, layer_argmax=L_arg, N=N, n_stim=n_stim,
                         interior_heldout=round(ri_ho, 3), interior_argmax=round(ri_arg, 3),
                         interior_ci=ri_ci, interior_null95=round(ni, 3), interior_p=round(pi, 4),
                         all_probe=round(ra, 3), all_p=round(pa, 4),
                         twin_interior=round(twin_i, 3) if twin_i is not None else None,
                         coherence_increment=round(coh, 3) if coh is not None else None,
                         coherence_ci=coh_ci, coherence_sig=coh_sig))
        verdict = "SURVIVES" if (ri > ni and pi < 0.05) else "COLLAPSES"
        cis = f"CI{ri_ci}" if ri_ci else ""
        cohs = (f" COH={coh:+.3f}{('CI'+str(coh_ci)+('SIG' if coh_sig else 'ns')) if coh_ci else ''}"
                if coh is not None else "")
        print(f"{model:14s} {args.family}/{args.condition} {args.scheme} diff={args.difficulty} | n={n_stim} "
              f"INTERIOR heldout={ri_ho:.3f} argmax={ri_arg:.3f}@L{L_arg} {cis} (null95={ni:.2f},p={pi:.3f}){cohs} -> {verdict}",
              flush=True)

    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(args.out)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
