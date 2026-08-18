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
search); bootstrap CI over stimuli on the peak-layer metrics. One parameterized tool, same acts /
conventions as the other probes.

  python scripts/probe_form.py --acts acts --model gemma-4-12b-it --family s0_zib \
      --structure grid2d --condition shuffle --scheme readout --n-boot 1000 --n-perm 200 \
      --json out/form_grid.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from icom.probes import (circular_rsa, angular_decode, cv_spearman, interior_mask,
                         load_records, n_layers, project, reduce)


# ----------------------------------------------------------------------------- grid2d
def _fit_axis(X, c):
    """Unit rank-axis in raw D-space for target coordinate c (StandardScaler+Ridge, as steer_rank)."""
    sc = StandardScaler().fit(X)
    rg = Ridge(alpha=10.0).fit(sc.transform(X), c)
    w = rg.coef_ / sc.scale_
    return w / (np.linalg.norm(w) + 1e-9)


def _stack_grid(recs, layer):
    """Interior entities pooled across stimuli at `layer`: X[n,D], cx[n], cy[n] (each coord
    z-scored per stimulus so mixed content pools comparably), g[n] stimulus id."""
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
    return (np.concatenate(Xs), np.concatenate(cxs), np.concatenate(cys), np.concatenate(gs))


def _grid_metrics(S):
    X, cx, cy, g = S
    Xr = reduce(X)
    dx, dy = cv_spearman(Xr, cx, g), cv_spearman(Xr, cy, g)
    wx, wy = _fit_axis(X, cx), _fit_axis(X, cy)
    px, py = X @ wx, X @ wy
    # cross-decode computed WITHIN stimulus then averaged (factored grid => ~0)
    cxx, cyy = [], []
    for gg in np.unique(g):
        idx = g == gg
        if idx.sum() >= 4 and np.std(px[idx]) > 0 and np.std(cy[idx]) > 0:
            cxx.append(abs(spearmanr(px[idx], cy[idx])[0]))
        if idx.sum() >= 4 and np.std(py[idx]) > 0 and np.std(cx[idx]) > 0:
            cyy.append(abs(spearmanr(py[idx], cx[idx])[0]))
    cross = float(np.nanmean(cxx + cyy)) if (cxx or cyy) else float("nan")
    return dict(decode_x=float(dx), decode_y=float(dy), axis_cos=float(abs(wx @ wy)), cross=cross)


def _grid_score(mrec):  # headline scalar the layer search maximizes: both axes decode
    return 0.5 * (mrec["decode_x"] + mrec["decode_y"])


def _grid_null_once(S, rng):
    X, cx, cy, g = S
    cxp, cyp = cx.copy(), cy.copy()
    for gg in np.unique(g):                                # joint (cx,cy) permutation within stimulus
        idx = np.where(g == gg)[0]
        p = rng.permutation(len(idx))
        cxp[idx], cyp[idx] = cx[idx][p], cy[idx][p]
    return _grid_metrics((X, cxp, cyp, g))


# ----------------------------------------------------------------------------- cyclic
def _cyclic_metrics(recs, layer):
    rr = circular_rsa(recs, layer, interior_only=True)
    # angular decode on the 2-D PCA projection, per stimulus, averaged
    ang = []
    for r in recs:
        m = interior_mask(r["ranks"], r["N"]) & np.isfinite(r["X"][:, layer, :]).all(axis=1)
        if m.sum() < 4:
            continue
        P = project(r["X"][m, layer, :], dim=2)
        ang.append(angular_decode(P, r["ranks"][m], r["N"]))
    return dict(cyclic_rsa=rr["cyclic_rsa"], linear_rsa=rr["linear_rsa"],
                ring_gap=float(rr["cyclic_rsa"] - rr["linear_rsa"]),
                angular=float(np.nanmean(ang)) if ang else float("nan"), n=rr["n"])


def _cyclic_score(mrec):
    return mrec["ring_gap"]


def _cyclic_null_once(recs, layer, rng):
    shuffled = []
    for r in recs:
        rr = dict(r)
        rr["ranks"] = rng.permutation(r["ranks"])          # ring position <-> entity broken
        shuffled.append(rr)
    return _cyclic_metrics(shuffled, layer)


# ----------------------------------------------------------------------------- driver
def _boot_ci(vals):
    a = np.array([v for v in vals if v == v])
    if len(a) < 2:
        return None, None
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def run_cell(acts, model, family, condition, scheme, structure, n_boot, n_perm, seed):
    recs = load_records(acts, model, family, condition, scheme, structure=structure, with_extras=True)
    if not recs:
        return None
    L = n_layers(recs)
    rng = np.random.default_rng(seed)

    if structure == "grid2d":
        per = [_grid_metrics(S) if (S := _stack_grid(recs, l)) else None for l in range(L)]
        score = _grid_score
    elif structure == "cyclic":
        per = [_cyclic_metrics(recs, l) for l in range(L)]
        score = _cyclic_score
    else:
        raise SystemExit(f"--structure must be grid2d|cyclic (got {structure})")

    scores = np.array([score(m) if m else np.nan for m in per])
    if not np.isfinite(scores).any():
        return None
    peak = int(np.nanargmax(scores))
    peak_m = per[peak]

    # max-over-layers permutation null (family-wise correction for the layer search)
    null_max, null_peak = [], []
    for _ in range(n_perm):
        if structure == "grid2d":
            ncols = [_grid_null_once(S, rng) if (S := _stack_grid(recs, l)) else None for l in range(L)]
        else:
            ncols = [_cyclic_null_once(recs, l, rng) for l in range(L)]
        ns = np.array([score(m) if m else np.nan for m in ncols])
        null_max.append(np.nanmax(ns) if np.isfinite(ns).any() else np.nan)
        null_peak.append(ns[peak])
    fmax = np.array([v for v in null_max if v == v])
    p_fwer = (1 + int((fmax >= scores[peak]).sum())) / (1 + len(fmax)) if len(fmax) else None

    # bootstrap CI over stimuli at the peak layer (resample whole records)
    idx0 = np.arange(len(recs))
    boot = {k: [] for k in peak_m}
    for _ in range(n_boot):
        bi = rng.choice(idx0, size=len(idx0), replace=True)
        rb = [recs[i] for i in bi]
        if structure == "grid2d":
            S = _stack_grid(rb, peak)
            mb = _grid_metrics(S) if S else None
        else:
            mb = _cyclic_metrics(rb, peak)
        if mb:
            for k in boot:
                boot[k].append(mb[k])
    ci = {k: _boot_ci(v) for k, v in boot.items()}

    def _r(x, nd=3):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)

    out = dict(model=model, family=family, condition=condition, scheme=scheme, structure=structure,
               n_stimuli=len(recs), peak_layer=peak, peak_frac=round(peak / max(1, L - 1), 3),
               score_peak=_r(scores[peak]), score_null95=_r(float(np.nanpercentile(null_peak, 95))),
               p_fwer=_r(p_fwer, 4),
               metrics={k: _r(v) for k, v in peak_m.items()},
               ci95={k: [_r(lo), _r(hi)] for k, (lo, hi) in ci.items()})
    return out


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
