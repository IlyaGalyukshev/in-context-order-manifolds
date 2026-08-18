#!/usr/bin/env python
"""Form litmus: does the emergent geometry take the STRUCTURE's own form, or collapse to a line?

Two structures, each judged by NUMBERS + bootstrap CI vs a within-stimulus permutation null
(never "it looks like a ring"):

  grid2d : both coordinate axes decodable (decode_x, decode_y > null) AND near-orthogonal
           (|cos(axis_x, axis_y)| ~ 0) AND non-cross-decoding (an x-axis projection does NOT
           predict the y-coordinate) => a FACTORED 2-D grid. Collinear axes / high cross-decode
           => the two axes are ENTANGLED (a line wearing a grid's labels).
  cyclic : activation distance tracks CYCLIC distance >> LINEAR distance (ring_gap = cyclic_rsa
           - linear_rsa > null) AND the 2-D projection angle recovers ring position
           (angular_decode > null) => a RING; else an arc / line.

Per-layer sweep -> peak (max-over-layers permutation null = family-wise correction for the layer
search); bootstrap CI over stimuli on the peak-layer metrics. Efficient: the expensive per-layer
reductions (PCA embedding / 2-D projections / pairwise distances) are INVARIANT to the label
permutation, so they are computed ONCE per layer and the null/bootstrap only re-score the cheap
decodes. Same acts / conventions as the other probes.

  python scripts/probe_form.py --acts acts --model gemma-4-12b-it --family s0_zib \
      --structure grid2d --condition shuffle --scheme readout --n-boot 1000 --n-perm 200 \
      --json out/form_grid.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from icom.probes import cv_spearman, interior_mask, load_records, n_layers, project, reduce


# ----------------------------------------------------------------------------- grid2d
def _fit_axis(X, c):
    """Unit rank-axis in raw D-space for target coordinate c (StandardScaler+Ridge, as steer_rank)."""
    sc = StandardScaler().fit(X)
    rg = Ridge(alpha=10.0).fit(sc.transform(X), c)
    w = rg.coef_ / sc.scale_
    return w / (np.linalg.norm(w) + 1e-9)


def _grid_layer_data(recs, layer):
    """Interior entities pooled across stimuli at `layer`, with the PCA embedding precomputed once
    (invariant to label permutation). Returns (X[n,D], Xr[n,k], cx[n], cy[n], g[n]) or None."""
    Xs, cxs, cys, gs = [], [], [], []
    for gi, r in enumerate(recs):
        if "coord_x" not in r or "coord_y" not in r:
            continue
        m = interior_mask(r["ranks"], r["N"]) & np.isfinite(r["X"][:, layer, :]).all(axis=1)
        if m.sum() < 4:
            continue
        cx, cy = r["coord_x"][m], r["coord_y"][m]
        if np.std(cx) < 1e-9 or np.std(cy) < 1e-9:
            continue
        Xs.append(r["X"][m, layer, :])
        cxs.append((cx - cx.mean()) / (cx.std() + 1e-9))
        cys.append((cy - cy.mean()) / (cy.std() + 1e-9))
        gs.append(np.full(int(m.sum()), gi))
    if not Xs:
        return None
    X = np.concatenate(Xs)
    return X, reduce(X), np.concatenate(cxs), np.concatenate(cys), np.concatenate(gs)


def _grid_decode(Xr, cx, cy, g):
    return float(cv_spearman(Xr, cx, g)), float(cv_spearman(Xr, cy, g))


def _grid_full_metrics(X, Xr, cx, cy, g):
    dx, dy = _grid_decode(Xr, cx, cy, g)
    wx, wy = _fit_axis(X, cx), _fit_axis(X, cy)
    px, py = X @ wx, X @ wy
    cxx, cyy = [], []
    for gg in np.unique(g):                                # cross-decode WITHIN stimulus, averaged
        idx = g == gg
        if idx.sum() >= 4 and np.std(px[idx]) > 0 and np.std(cy[idx]) > 0:
            cxx.append(abs(spearmanr(px[idx], cy[idx])[0]))
        if idx.sum() >= 4 and np.std(py[idx]) > 0 and np.std(cx[idx]) > 0:
            cyy.append(abs(spearmanr(py[idx], cx[idx])[0]))
    cross = float(np.nanmean(cxx + cyy)) if (cxx or cyy) else float("nan")
    return dict(decode_x=dx, decode_y=dy, axis_cos=float(abs(wx @ wy)), cross=cross)


def _perm_within(cx, cy, g, rng):
    cxp, cyp = cx.copy(), cy.copy()
    for gg in np.unique(g):                                # joint (cx,cy) permutation within stimulus
        idx = np.where(g == gg)[0]
        p = rng.permutation(len(idx))
        cxp[idx], cyp[idx] = cx[idx][p], cy[idx][p]
    return cxp, cyp


# ----------------------------------------------------------------------------- cyclic
def _cyclic_layer_data(recs, layer):
    """Per-stimulus interior data at `layer` with the pairwise distances + 2-D projection
    precomputed (both invariant to ring-position permutation). Returns a list of
    {actd, P, ranks, N} or None."""
    out = []
    for r in recs:
        m = interior_mask(r["ranks"], r["N"]) & np.isfinite(r["X"][:, layer, :]).all(axis=1)
        if m.sum() < 4:
            continue
        X = r["X"][m, layer, :].astype(np.float64)
        out.append({"actd": pdist(X), "P": project(X, dim=2),
                    "ranks": r["ranks"][m].astype(int), "N": int(r["N"])})
    return out or None


def _R(x):
    return abs(np.mean(np.exp(1j * x)))


def _cyclic_metrics(data, ranks_list=None):
    """ring_gap = cyclic_rsa - linear_rsa and angular decode, averaged over stimuli. `ranks_list`
    overrides each stimulus's ring positions (for the permutation null); actd/P are reused."""
    cyc, lin, ang = [], [], []
    for i, d in enumerate(data):
        rk = d["ranks"] if ranks_list is None else ranks_list[i]
        N = d["N"]
        dif = np.abs(rk[:, None] - rk[None, :])
        iu = np.triu_indices(len(rk), 1)
        lind, cycd = dif[iu], np.minimum(dif, N - dif)[iu]
        if np.std(d["actd"]) > 0 and np.std(cycd) > 0:
            cyc.append(spearmanr(d["actd"], cycd)[0])
        if np.std(d["actd"]) > 0 and np.std(lind) > 0:
            lin.append(spearmanr(d["actd"], lind)[0])
        P = (d["P"] - d["P"].mean(0)) / (d["P"].std(0) + 1e-12)
        theta = np.arctan2(P[:, 1], P[:, 0]); true = 2 * np.pi * (rk - 1) / N
        ang.append(float(max(_R(theta - true), _R(theta + true))))
    cr = float(np.mean(cyc)) if cyc else float("nan")
    lr = float(np.mean(lin)) if lin else float("nan")
    return dict(cyclic_rsa=cr, linear_rsa=lr, ring_gap=cr - lr,
                angular=float(np.nanmean(ang)) if ang else float("nan"), n=len(cyc))


# ----------------------------------------------------------------------------- driver
def _boot_ci(vals):
    a = np.array([v for v in vals if v == v])
    if len(a) < 2:
        return None, None
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def run_cell(acts, model, family, condition, scheme, structure, n_boot, n_perm, seed):
    recs = load_records(acts, model, family, condition, scheme, structure=structure, with_extras=True)
    if not recs:
        return None
    L = n_layers(recs)
    rng = np.random.default_rng(seed)

    # ---- precompute the expensive, permutation-invariant per-layer data ONCE ----
    if structure == "grid2d":
        per = [_grid_layer_data(recs, l) for l in range(L)]
        def score_layer(d, cxp=None, cyp=None):
            if d is None:
                return np.nan
            X, Xr, cx, cy, g = d
            dx, dy = _grid_decode(Xr, cx if cxp is None else cxp, cy if cyp is None else cyp, g)
            return 0.5 * (dx + dy)
        obs = [(_grid_full_metrics(*d) if d else None) for d in per]
    elif structure == "cyclic":
        per = [_cyclic_layer_data(recs, l) for l in range(L)]
        def score_layer(d, ranks_list=None):
            if d is None:
                return np.nan
            return _cyclic_metrics(d, ranks_list)["ring_gap"]
        obs = [(_cyclic_metrics(d) if d else None) for d in per]
    else:
        raise SystemExit(f"--structure must be grid2d|cyclic (got {structure})")

    scores = np.array([score_layer(per[l]) for l in range(L)])
    if not np.isfinite(scores).any():
        return None
    peak = int(np.nanargmax(scores))
    peak_m = obs[peak]

    # ---- max-over-layers permutation null (cheap: only re-scores decode on precomputed data) ----
    null_max, null_peak = [], []
    for _ in range(n_perm):
        ns = np.full(L, np.nan)
        for l in range(L):
            d = per[l]
            if d is None:
                continue
            if structure == "grid2d":
                _, _, cx, cy, g = d
                cxp, cyp = _perm_within(cx, cy, g, rng)
                ns[l] = score_layer(d, cxp, cyp)
            else:
                rl = [rng.permutation(s["ranks"]) for s in d]
                ns[l] = score_layer(d, rl)
        null_max.append(np.nanmax(ns) if np.isfinite(ns).any() else np.nan)
        null_peak.append(ns[peak])
    fmax = np.array([v for v in null_max if v == v])
    p_fwer = (1 + int((fmax >= scores[peak]).sum())) / (1 + len(fmax)) if len(fmax) else None

    # ---- bootstrap CI over stimuli at the peak layer. Resample the PRECOMPUTED peak-layer data
    # (rows/stimuli) with replacement -> no per-draw re-PCA / re-projection; only the cheap decode
    # (grid) or the per-stimulus RSA (cyclic) is recomputed. Groups get fresh ids per draw. ----
    boot = {k: [] for k in peak_m}
    if structure == "grid2d":
        X, Xr, cx, cy, g = per[peak]
        groups = np.unique(g)
        for _ in range(n_boot):
            parts, newg = [], []
            for ng, gg in enumerate(rng.choice(groups, size=len(groups), replace=True)):
                idx = np.where(g == gg)[0]
                parts.append(idx); newg.append(np.full(len(idx), ng))
            ii = np.concatenate(parts); g2 = np.concatenate(newg)
            mb = _grid_full_metrics(X[ii], Xr[ii], cx[ii], cy[ii], g2)
            for k in boot:
                boot[k].append(mb[k])
    else:
        data = per[peak]
        for _ in range(n_boot):
            rb = [data[i] for i in rng.choice(len(data), size=len(data), replace=True)]
            mb = _cyclic_metrics(rb)
            for k in boot:
                boot[k].append(mb[k])
    ci = {k: _boot_ci(v) for k, v in boot.items()}

    return dict(model=model, family=family, condition=condition, scheme=scheme, structure=structure,
                n_stimuli=len(recs), peak_layer=peak, peak_frac=round(peak / max(1, L - 1), 3),
                score_peak=_r(scores[peak]), score_null95=_r(float(np.nanpercentile(null_peak, 95))),
                p_fwer=_r(p_fwer, 4),
                metrics={k: _r(v) for k, v in peak_m.items()},
                ci95={k: [_r(lo), _r(hi)] for k, (lo, hi) in ci.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--model", default="gemma-4-12b-it")
    ap.add_argument("--families", default="s0_zib")
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--scheme", default="readout")
    ap.add_argument("--structure", default="grid2d", help="grid2d|cyclic")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = []
    for family in args.families.split(","):
        try:
            r = run_cell(args.acts, args.model, family, args.condition, args.scheme,
                         args.structure, args.n_boot, args.n_perm, args.seed)
        except Exception as e:
            print(f"{args.model} {family}/{args.structure}: ERROR {e}", flush=True)
            continue
        if r is None:
            print(f"{args.model} {family}/{args.structure}: no data", flush=True)
            continue
        results.append(r)
        m = r["metrics"]
        extra = (f"decode_x={m['decode_x']} decode_y={m['decode_y']} axis_cos={m['axis_cos']} "
                 f"cross={m['cross']}" if args.structure == "grid2d"
                 else f"cyclic_rsa={m['cyclic_rsa']} linear_rsa={m['linear_rsa']} angular={m['angular']}")
        print(f"{r['model']:14s} {family}/{args.structure} {args.scheme} | peak L{r['peak_layer']}"
              f"({r['peak_frac']}) score={r['score_peak']}(null95={r['score_null95']},"
              f"p_fwer={r['p_fwer']}) | {extra} | ci95={r['ci95']}", flush=True)

    if args.json and results:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
