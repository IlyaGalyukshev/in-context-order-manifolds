#!/usr/bin/env python
"""M3 — cPCA coherence-subspace probe: does the RANK axis live in the real-vs-twin contrastive
subspace? Contrastive PCA (real foreground, twin background) returns the directions that vary in the
COHERENT order but not in the incoherent-cycle twin. We then OOF-decode rank INSIDE that subspace
(GroupKFold over stimuli) and compare to a plain-PCA subspace of the same dimension. cPCA ≫ PCA ⇒
the coherence-specific variance is where the ordinal axis lives — a principled replacement for the
ad-hoc remove-top-k. Uses the stored k-means (mean_<scheme>) from extract_repeat --store.

  python scripts/probe_cpca.py --acts results/e1_20260820/acts --model qwen3-4b \
      --family s0_zib --n-items 12 --scheme card_mean --n-comp 6 --json out/cpca.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from icom.probes import cpca_components, cv_spearman, interior_mask
from icom.probes.cpca import project as cpca_project


def _load_means(acts, model, family, condition, scheme, is_null, n_items):
    recs = []
    for f in sorted((Path(acts) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if m["family"] != family or m["condition"] != condition:
            continue
        if bool(m.get("is_null", False)) != is_null or int(m["n_items"]) != int(n_items):
            continue
        key = f"mean_{scheme}" if f"mean_{scheme}" in z.files else (scheme if scheme in z.files else None)
        if key is None:
            continue
        X = z[key].astype(np.float64)
        if X.ndim == 4:
            X = X.mean(axis=1)
        recs.append({"X": X, "ranks": z["ranks"].astype(int), "N": int(m["n_items"])})
    return recs


def _stack(recs, layer):
    """Interior entities pooled at `layer`: (X [n,D], y [n] norm rank, g [n] stimulus id)."""
    Xs, ys, gs = [], [], []
    for gi, r in enumerate(recs):
        mask = interior_mask(r["ranks"], r["N"]) & np.isfinite(r["X"][:, layer, :]).all(axis=1)
        if mask.sum() < 2:
            continue
        Xs.append(r["X"][mask, layer, :])
        ys.append((r["ranks"][mask] - 1) / (r["N"] - 1))
        gs.append(np.full(int(mask.sum()), gi))
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def run(acts, model, family, condition, scheme, n_items, n_comp, alpha, n_boot, seed):
    real = _load_means(acts, model, family, condition, scheme, False, n_items)
    twin = _load_means(acts, model, family, condition, scheme, True, n_items)
    if not real or not twin:
        return None
    L = real[0]["X"].shape[1]

    def cell(layer):
        sr, st = _stack(real, layer), _stack(twin, layer)
        if sr is None or st is None:
            return None
        Xr, y, g = sr
        comps = cpca_components(Xr, st[0], n_comp=n_comp, alpha=alpha)
        cpca_dec = cv_spearman(cpca_project(Xr, comps), y, g)          # rank decode in coherence subspace
        pca_dec = cv_spearman(PCA(min(n_comp, Xr.shape[1]), random_state=0).fit_transform(
            Xr - Xr.mean(0)), y, g)                                    # matched-dim plain PCA
        return float(cpca_dec), float(pca_dec)

    prof = [cell(l) for l in range(L)]
    cp = np.array([c[0] if c else np.nan for c in prof])
    pc = np.array([c[1] if c else np.nan for c in prof])
    if not np.isfinite(cp).any():
        return None
    peak = int(np.nanargmax(cp))
    # bootstrap CI over stimuli on (cPCA - PCA) at peak
    rng = np.random.default_rng(seed); idx = np.arange(len(real)); diff = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        rb = [real[i] for i in b]
        sr = _stack(rb, peak)
        if sr is None:
            continue
        st = _stack(twin, peak)
        comps = cpca_components(sr[0], st[0], n_comp=n_comp, alpha=alpha)
        cd = cv_spearman(cpca_project(sr[0], comps), sr[1], sr[2])
        pd = cv_spearman(PCA(min(n_comp, sr[0].shape[1]), random_state=0).fit_transform(
            sr[0] - sr[0].mean(0)), sr[1], sr[2])
        diff.append(float(cd - pd))
    diff = np.array([x for x in diff if x == x])
    ci = [_r(np.percentile(diff, 2.5)), _r(np.percentile(diff, 97.5))] if len(diff) > 2 else [None, None]
    return dict(model=model, family=family, n_items=n_items, scheme=scheme, n_comp=n_comp,
                n_real=len(real), n_twin=len(twin), peak_layer=peak,
                cpca_decode=_r(cp[peak]), pca_decode=_r(pc[peak]),
                cpca_minus_pca=_r(cp[peak] - pc[peak]), diff_ci=ci,
                cpca_beats_pca=bool(ci[0] is not None and ci[0] > 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--families", default="s0_zib"); ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--scheme", default="card_mean"); ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--n-comp", type=int, default=6); ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = []
    for family in args.families.split(","):
        r = run(args.acts, args.model, family, args.condition, args.scheme, args.n_items,
                args.n_comp, args.alpha, args.n_boot, args.seed)
        if r is None:
            print(f"{args.model} {family} N{args.n_items}: no data", flush=True); continue
        results.append(r)
        print(f"{r['model']:12s} {family} N{r['n_items']} {r['scheme']} | peak L{r['peak_layer']} "
              f"cPCA_decode={r['cpca_decode']} PCA_decode={r['pca_decode']} "
              f"cPCA-PCA={r['cpca_minus_pca']}{r['diff_ci']} {'SIG' if r['cpca_beats_pca'] else 'ns'}", flush=True)
    if args.json and results:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
