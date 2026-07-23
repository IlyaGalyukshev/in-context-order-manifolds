#!/usr/bin/env python
"""Option A: is latent RANK decodable from pooled activations — with the RIGHT
tool, not PC1?

PC1 (a straight line) can miss a curved manifold; a supervised, cross-validated
probe cannot (if rank is recoverable at all, the probe finds it). This is the
make-or-break re-analysis of the pilot geometry null, on EXISTING activations.

Method, per (model, family, condition, scheme, layer):
  - pool all items across stimuli in the cell -> X [n_items, D]
  - reduce with PCA(64) (unsupervised, label-blind) once per cell/layer
  - GroupKFold-5 ridge regression predicting normalized rank, groups = stimulus
    (train and test NEVER share a stimulus). Metric = Spearman(oob pred, rank).
  - positive control: same probe predicting the presentation SLOT.
  - baseline: the old PC1 coordinate's Spearman with rank.
Significance: permute rank labels WITHIN each stimulus (keeps items + slot
structure, breaks activation<->rank), redo the whole layer sweep, take max over
layers. Null = distribution of that max over N_PERM permutations; the real
statistic is max-over-layers of the true probe. p accounts for layer selection.

Decisive reads:
  - slot control high everywhere  => probe works.
  - tagged 'marker' rank high      => probe detects a known signal.
  - relational 'name' rank >> null => an assembled-order manifold PC1 missed.
  - relational 'name' rank ~ null  => a REAL null (not a method artifact).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SCHEMES = ("name", "marker", "last_token", "card_mean")


def cv_spearman(Xr: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float = 10.0) -> float:
    """Out-of-fold Spearman of a ridge probe, grouped by stimulus."""
    n_groups = len(np.unique(groups))
    k = min(5, n_groups)
    if k < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=k)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(Xr, y, groups):
        m = Ridge(alpha=alpha).fit(Xr[tr], y[tr])
        oof[te] = m.predict(Xr[te])
    rho, _ = spearmanr(oof, y)
    return float(rho)


def load_cell(files):
    """Return per-cell arrays keyed by scheme: {scheme: X[n,L,D]}, ranks, slots, groups."""
    per_scheme = defaultdict(list)
    ranks, slots, groups = [], [], []
    for gi, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        schemes = [s for s in SCHEMES if s in z.files]
        for s in schemes:
            per_scheme[s].append(z[s].astype(np.float32))  # [N, L, D]
        n = len(z["ranks"])
        ranks.append(z["ranks"]); slots.append(z["slots"]); groups.append(np.full(n, gi))
    out = {s: np.concatenate(v, axis=0) for s, v in per_scheme.items()}  # [n_tot, L, D]
    return out, np.concatenate(ranks), np.concatenate(slots), np.concatenate(groups)


def permute_within_stimulus(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    yp = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        yp[idx] = y[idx][rng.permutation(len(idx))]
    return yp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="qwen3-4b,olmo3-7b-inst")
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for model in args.models.split(","):
        files = defaultdict(list)
        for f in sorted((Path(args.acts) / model).glob("*.npz")):
            meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
            files[(meta["family"], meta["condition"])].append(f)

        for (family, condition), fs in sorted(files.items()):
            X_by_scheme, y_rank, y_slot, groups = load_cell(fs)
            yr = (y_rank - y_rank.min()) / (y_rank.max() - y_rank.min())
            ys = (y_slot - y_slot.min()) / (y_slot.max() - y_slot.min())
            for scheme, X in X_by_scheme.items():
                L = X.shape[1]
                # precompute PCA-reduced X per layer (label-blind)
                Xr_layers = []
                for layer in range(L):
                    Xl = StandardScaler().fit_transform(X[:, layer, :])
                    k = min(args.pca, Xl.shape[1], Xl.shape[0] - 1)
                    Xr_layers.append(PCA(n_components=k, random_state=0).fit_transform(Xl))
                # real: rank + slot probe per layer; PC1 baseline per layer
                real_rank = np.array([cv_spearman(Xr_layers[l], yr, groups) for l in range(L)])
                real_slot = np.array([cv_spearman(Xr_layers[l], ys, groups) for l in range(L)])
                pc1 = []
                for l in range(L):
                    c = Xr_layers[l][:, 0]
                    # per-group PC1 sign is arbitrary; use |spearman| pooled
                    pc1.append(abs(spearmanr(c, y_rank)[0]))
                pc1 = np.array(pc1)

                best_l = int(np.nanargmax(np.abs(real_rank)))
                real_max = float(np.nanmax(np.abs(real_rank)))
                # permutation null: max over layers of |spearman| under within-stimulus label perm
                null_max = []
                for _ in range(args.n_perm):
                    yp = permute_within_stimulus(yr, groups, rng)
                    vals = [abs(cv_spearman(Xr_layers[l], yp, groups)) for l in range(L)]
                    null_max.append(np.nanmax(vals))
                null_max = np.array(null_max)
                p = (1 + int((null_max >= real_max).sum())) / (1 + args.n_perm)

                rows.append(dict(
                    model=model, family=family, condition=condition, scheme=scheme,
                    best_layer=best_l, probe_rank=round(real_max, 3),
                    null_p95=round(float(np.percentile(null_max, 95)), 3),
                    null_mean=round(float(null_max.mean()), 3), p_value=round(p, 4),
                    slot_ctrl=round(float(np.nanmax(np.abs(real_slot))), 3),
                    pc1_rank=round(float(np.nanmax(pc1)), 3),
                ))
                r = rows[-1]
                print(f"{model:13s} {family:10s} {condition:8s} {scheme:10s} "
                      f"probe={r['probe_rank']:.2f} null95={r['null_p95']:.2f} p={r['p_value']:.3f} "
                      f"| slot={r['slot_ctrl']:.2f} pc1={r['pc1_rank']:.2f} L{r['best_layer']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_parquet(out)
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
