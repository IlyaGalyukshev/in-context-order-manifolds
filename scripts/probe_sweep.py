#!/usr/bin/env python
"""Phase-C exhaustive offline probe sweep over stored BCS activations (CPU).

For each (model x family x condition x N x scheme) it sweeps ALL layers and runs
the probe catalog, then summarizes per cell:
  per-layer  : linear-all, linear-interior (+perm null95/p), RSA-interior,
               intrinsic-dim-interior
  per-cell   : depth onset/peak/band, nonlinear@peak (+curvature gap),
               per-rank MAE@peak, transfer (cross-condition / cross-N),
               coherence-null gate@peak.
Writes a long CSV (per-layer) and a summary JSON; optional Markdown.

  python scripts/probe_sweep.py --acts acts --models qwen3-4b \
      --families s1_size --conditions shuffle --Ns 9,12 --scheme readout \
      --transfer-condition forward --null-acts acts_null \
      --n-perm 100 --csv out/sweep_layers.csv --json out/sweep_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icom.probes import (depth_stats, intrinsic_dim, load_records, mlp_cv_spearman,
                         n_layers, per_rank_mae, probe_with_null, reduce, rsa_rank,
                         stack_all_layers, stack_layer, transfer_spearman)
from icom.probes.linear import cv_spearman


def _permute_within(y, g, rng):
    yp = y.copy()
    for gg in np.unique(g):
        idx = np.where(g == gg)[0]
        yp[idx] = y[idx][rng.permutation(len(idx))]
    return yp


def sweep_cell(acts, model, family, condition, N, scheme, n_perm, seed,
               transfer_condition=None, null_acts=None):
    recs = load_records(acts, model, family, condition, scheme, N=N)
    if not recs:
        return None, []
    L = n_layers(recs)
    Sint = stack_all_layers(recs, interior_only=True)
    Sall = stack_all_layers(recs, interior_only=False)
    if Sint is None:                                  # no interior entities at all
        return None, []
    Xint, yint, gint = Sint
    Xri = [reduce(X) for X in Xint]                   # PCA per layer (unsupervised)
    lin_int = [cv_spearman(Xr, yint, gint) for Xr in Xri]
    if Sall is not None:
        Xra = [reduce(X) for X in Sall[0]]
        lin_all = [cv_spearman(Xr, Sall[1], Sall[2]) for Xr in Xra]
    else:
        lin_all = [float("nan")] * L

    # ---- MAX-OVER-LAYERS permutation null (one within-stimulus permutation
    # evaluated at EVERY layer -> the family-wise correction for the layer search
    # that a naive per-layer argmax would otherwise inflate).
    rng = np.random.default_rng(seed)
    per_layer_null = [[] for _ in range(L)]
    null_max = []
    for _ in range(n_perm):
        yp = _permute_within(yint, gint, rng)
        sp = [cv_spearman(Xr, yp, gint) for Xr in Xri]
        for layer in range(L):
            per_layer_null[layer].append(sp[layer])
        null_max.append(np.nanmax(sp) if np.isfinite(sp).any() else np.nan)
    null95 = [float(np.nanpercentile(per_layer_null[l], 95))
              if np.isfinite(per_layer_null[l]).any() else float("nan") for l in range(L)]

    def _pl_p(layer):                                 # per-layer (uncorrected) p
        fin = np.array([v for v in per_layer_null[layer] if np.isfinite(v)])
        r = lin_int[layer]
        return None if (np.isnan(r) or not len(fin)) else (1 + int((fin >= r).sum())) / (1 + len(fin))

    long_rows = []
    for layer in range(L):
        rsa, rsa_n = rsa_rank(recs, layer, interior_only=True)
        idi = intrinsic_dim(recs, layer, interior_only=True)
        long_rows.append(dict(model=model, family=family, condition=condition, N=N,
                              scheme=scheme, layer=layer, lin_all=_r(lin_all[layer]),
                              lin_interior=_r(lin_int[layer]), interior_null95=_r(null95[layer]),
                              interior_p_layer=_r(_pl_p(layer), 4), rsa_interior=_r(rsa),
                              id_interior=_r(idi), rsa_n=rsa_n))

    if not np.isfinite(lin_int).any():
        return None, long_rows
    peak = int(np.nanargmax(lin_int))
    peak_score = lin_int[peak]
    fin_max = np.array([v for v in null_max if np.isfinite(v)])
    p_fwer = ((1 + int((fin_max >= peak_score).sum())) / (1 + len(fin_max))) if len(fin_max) else None
    depth = depth_stats(lin_int, null95)              # band uses per-layer null95

    nonlin = mlp_cv_spearman(Xint[peak], yint, gint)
    s_all_pk = stack_layer(recs, peak, interior_only=False)
    prm = (per_rank_mae(s_all_pk[0], np.rint(s_all_pk[1] * (N - 1) + 1).astype(int),
                        s_all_pk[2], N) if s_all_pk else {})

    transfer = {}
    if transfer_condition and transfer_condition != condition:
        recs_b = load_records(acts, model, family, transfer_condition, scheme, N=N)
        b = stack_layer(recs_b, peak, interior_only=True) if recs_b else None
        if b:
            transfer[f"to_{transfer_condition}"] = _r(transfer_spearman(Xint[peak], yint, b[0], b[1]))

    coh_null = None
    if null_acts:
        recs_z = load_records(null_acts, model, family, condition, scheme, N=N)
        z = stack_layer(recs_z, peak, interior_only=True) if recs_z else None
        if z:
            rz, nz, pz = probe_with_null(*z, n_perm=n_perm, seed=seed)
            coh_null = dict(interior=_r(rz), null95=_r(nz), p=_r(pz, 4))

    # FWER-corrected verdict: peak must beat the max-over-layers null.
    survives = p_fwer is not None and p_fwer < 0.05
    summary = dict(model=model, family=family, condition=condition, N=N, scheme=scheme,
                   n_stimuli=len(recs), **{f"depth_{k}": v for k, v in depth.items()},
                   interior_peak=_r(peak_score), interior_peak_null95=_r(null95[peak]),
                   interior_peak_p_layer=_r(_pl_p(peak), 4), interior_peak_p_fwer=_r(p_fwer, 4),
                   nonlinear_peak=_r(nonlin), curvature_gap=_r(nonlin - peak_score),
                   per_rank_mae={int(k): round(v, 3) for k, v in prm.items()},
                   transfer=transfer, coherence_null=coh_null,
                   verdict="MANIFOLD" if survives else "no-interior-signal")
    return summary, long_rows


def _r(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--models", default="qwen3-4b")
    ap.add_argument("--families", default="s1_size")
    ap.add_argument("--conditions", default="shuffle")
    ap.add_argument("--Ns", default="9,12")
    ap.add_argument("--scheme", default="readout")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transfer-condition", default=None)
    ap.add_argument("--null-acts", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    summaries, long_all = [], []
    for model in args.models.split(","):
        for family in args.families.split(","):
            for condition in args.conditions.split(","):
                for N in [int(x) for x in args.Ns.split(",")]:
                    try:
                        summ, rows = sweep_cell(args.acts, model, family, condition, N,
                                                args.scheme, args.n_perm, args.seed,
                                                args.transfer_condition, args.null_acts)
                    except Exception as e:                 # one bad cell must not kill the sweep
                        print(f"{model} {family}/{condition} N={N} {args.scheme}: ERROR {e}", flush=True)
                        continue
                    long_all += rows
                    if summ is None:
                        print(f"{model} {family}/{condition} N={N} {args.scheme}: no interior data", flush=True)
                        continue
                    summaries.append(summ)
                    d = summ
                    print(f"{model:12s} {family}/{condition} N={N} {args.scheme} | "
                          f"peak L{d['depth_peak_layer']}({d['depth_peak_frac']}) "
                          f"int={d['interior_peak']}(null95={d['interior_peak_null95']},"
                          f"p_fwer={d['interior_peak_p_fwer']}) nonlin={d['nonlinear_peak']} "
                          f"band={d['depth_band_frac']} -> {d['verdict']}", flush=True)

    if args.csv and long_all:
        import pandas as pd
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(long_all).to_csv(args.csv, index=False)
        print(f"wrote per-layer -> {args.csv}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summaries, indent=2))
        print(f"wrote summary -> {args.json}")
    if args.md and summaries:
        Path(args.md).write_text(_md(summaries))
        print(f"wrote md -> {args.md}")


def _md(summaries):
    L = ["# Probe sweep summary", "",
         "*p_fwer = max-over-layers permutation p (family-wise correction for the "
         "layer search); MANIFOLD requires p_fwer < 0.05.*", "",
         "| model | family/cond | N | scheme | peak (frac) | interior | null95 | p_fwer | nonlin | band | verdict |",
         "|---|---|--:|---|---|--:|--:|--:|--:|---|---|"]
    for d in summaries:
        L.append(f"| {d['model']} | {d['family']}/{d['condition']} | {d['N']} | {d['scheme']} | "
                 f"L{d['depth_peak_layer']} ({d['depth_peak_frac']}) | {d['interior_peak']} | "
                 f"{d['interior_peak_null95']} | {d['interior_peak_p_fwer']} | {d['nonlinear_peak']} | "
                 f"{d['depth_band_frac']} | {d['verdict']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
