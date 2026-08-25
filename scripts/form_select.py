#!/usr/bin/env python
"""0.5 — Form model-selection (X→Y). Per stimulus, pick the best-fitting geometric TEMPLATE
(line / ring / 2-block / grid) for the crossnobis RDM by whitened-RSA, with NO prior. The distribution
of winners shows WHAT FORM the model actually built: a line-order stimulus whose RDM best matches the
LINE template (over ring / 2-block / grid) is Y=line ASSEMBLED from X — not an X→X surface reflection.
This is the direct rebuttal to "the geometry just mirrors the input structure".

  python scripts/form_select.py --acts results/e1hard_20260820/acts --model qwen3-4b \
      --family s0_zib --n-items 12 --scheme card_mean --templates line,ring,2block --json out/form.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icom.probes import interior_mask
from icom.probes.crossnobis import crossnobis_rdm, line_rdm, ring_rdm, whitened_rsa


def two_block_rdm(ranks: np.ndarray) -> np.ndarray:
    """2-cluster template: entities split by median rank into two blocks; within-block dist 0, cross 1."""
    n = len(ranks); order = np.argsort(ranks); half = n // 2
    grp = np.zeros(n, int); grp[order[half:]] = 1
    return (grp[:, None] != grp[None, :]).astype(float)


def grid_rdm(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    P = np.stack([cx, cy], 1).astype(float)
    return np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--families", default="s0_zib"); ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--scheme", default="card_mean"); ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--structure", default=None, help="filter meta.structure (total_order/cyclic/grid2d)")
    ap.add_argument("--layer", type=int, default=None); ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--templates", default="line,ring,2block")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    tmpl = [t for t in args.templates.split(",") if t]
    results = []
    for family in args.families.split(","):
        winners = {t: 0 for t in tmpl}; rsa_sum = {t: [] for t in tmpl}; nstim = 0
        for f in sorted((Path(args.acts) / args.model).glob("*.npz")):
            z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"]))
            if m.get("family") != family or m.get("condition") != args.condition:
                continue
            if bool(m.get("is_null", False)):
                continue
            if args.structure and m.get("structure", "total_order") != args.structure:
                continue
            if args.n_items and int(m["n_items"]) != args.n_items:
                continue
            ranks = z["ranks"].astype(int); N = int(m["n_items"])
            mask = interior_mask(ranks, N)
            if f"rdm_{args.scheme}" in z.files:
                Lp = z[f"rdm_{args.scheme}"].shape[2]; layer = args.layer if args.layer is not None else Lp // 2
                R = z[f"rdm_{args.scheme}"][:, :, layer].astype(np.float64)
            elif args.scheme in z.files:
                arr = z[args.scheme]; Lp = arr.shape[2]; layer = args.layer if args.layer is not None else Lp // 2
                R = crossnobis_rdm(arr[:, :, layer, :].astype(np.float64), n_splits=args.n_splits, seed=0)
            else:
                continue
            fin = np.isfinite(R).all(1) & mask; idx = np.where(fin)[0]
            if len(idx) < 4:
                continue
            R = R[np.ix_(idx, idx)]; rk = ranks[idx]
            cands = {}
            if "line" in tmpl:
                cands["line"] = line_rdm(rk)
            if "ring" in tmpl:
                cp = z["cyclic_pos"][idx].astype(int) if "cyclic_pos" in z.files else rk
                cands["ring"] = ring_rdm(cp, N)
            if "2block" in tmpl:
                cands["2block"] = two_block_rdm(rk)
            if "grid" in tmpl and "coord_x" in z.files:
                cands["grid"] = grid_rdm(z["coord_x"][idx], z["coord_y"][idx])
            scores = {t: whitened_rsa(R, c) for t, c in cands.items()}
            scores = {t: s for t, s in scores.items() if s == s}
            if not scores:
                continue
            best = max(scores, key=scores.get); winners[best] = winners.get(best, 0) + 1
            for t, s in scores.items():
                rsa_sum[t].append(s)
            nstim += 1
        res = {"model": args.model, "family": family, "scheme": args.scheme, "n_stim": nstim,
               "winner_frac": {t: round(winners.get(t, 0) / max(nstim, 1), 3) for t in tmpl},
               "mean_rsa": {t: (round(float(np.mean(rsa_sum[t])), 3) if rsa_sum[t] else None) for t in tmpl}}
        results.append(res)
        print(f"{args.model} {family}/{args.scheme}: n={nstim} winners={res['winner_frac']} "
              f"mean_rsa={res['mean_rsa']}", flush=True)
    if args.json and results:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
