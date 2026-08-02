#!/usr/bin/env python
"""Phase G4/G8 — shape characterization of the order manifold over stored acts (CPU).

For each (model × family × condition × N × scheme) at a chosen layer:
  - total/partial/grid  (G4): linear vs nonlinear vs geodesic decode, principal-
    curve curvature, curved-irreducible (linear vs intrinsic dim), Engels
    separability of the top-2 PCs — is the manifold CURVED?
  - cyclic (G8): circle-fit rmse, cyclic-vs-linear RSA, angular decode — is it a RING?

Layer: --layer, else the interior-decode peak found by a quick per-layer sweep.

  python scripts/probe_geometry.py --acts acts --models qwen3-4b \
      --families s1_size --conditions shuffle --Ns 9,12 --scheme readout \
      --structure total_order --json out/geometry.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icom.probes import (angular_decode, circle_fit, circular_rsa, curvature_profile,
                         curved_irreducible, cv_spearman, load_records, n_layers,
                         principal_curve_curvature, project, reduce, separability_index,
                         stack_layer)


def peak_layer(recs):
    L = n_layers(recs)
    best, bl = 0, -np.inf
    for layer in range(L):
        s = stack_layer(recs, layer, interior_only=True)
        v = cv_spearman(reduce(s[0]), s[1], s[2]) if s else np.nan
        if np.isfinite(v) and v > bl:
            bl, best = v, layer
    return best


def geometry_cell(acts, model, family, condition, N, scheme, structure, layer):
    recs = load_records(acts, model, family, condition, scheme, N=N, structure=structure)
    if not recs:
        return None
    lyr = layer if layer is not None else peak_layer(recs)
    out = {"model": model, "family": family, "condition": condition, "N": N,
           "scheme": scheme, "structure": structure, "layer": lyr, "n_stimuli": len(recs)}
    if structure == "cyclic":
        # pool all entities (no endpoints); rank == cyclic position. Drop
        # fp16-overflow rows before projecting (mirror stack_layer).
        Xs, ranks = [], []
        for r in recs:
            fin = np.isfinite(r["X"][:, lyr, :]).all(axis=1)
            Xs.append(r["X"][fin, lyr, :]); ranks.append(r["ranks"][fin])
        X = np.concatenate(Xs); rk = np.concatenate(ranks)
        P2 = project(X, "pca", 2)
        rsa = circular_rsa(recs, lyr)
        ang = round(angular_decode(P2, rk, int(np.max(rk))), 3)
        out["circle"] = circle_fit(P2)                 # necessary-not-sufficient (a line fits a circle)
        out["circular_rsa"] = rsa
        out["angular_decode"] = ang
        # ring verdict combines the ROBUST signals (circle_fit alone can't reject a line/arc)
        out["ring"] = bool(np.isfinite(rsa["cyclic_rsa"]) and np.isfinite(rsa["linear_rsa"])
                           and rsa["cyclic_rsa"] - rsa["linear_rsa"] > 0.1 and ang > 0.8)
    else:
        s = stack_layer(recs, lyr, interior_only=True)
        if not s:
            return out
        X, y, g = s
        out["curvature"] = curvature_profile(X, y, g)
        out["principal_curve_curvature"] = round(principal_curve_curvature(X, y), 3)
        out["curved_irreducible"] = curved_irreducible(X, y)
        out["separability_top2"] = round(separability_index(project(X, "pca", 2)), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--models", default="qwen3-4b")
    ap.add_argument("--families", default="s1_size")
    ap.add_argument("--conditions", default="shuffle")
    ap.add_argument("--Ns", default="9,12")
    ap.add_argument("--scheme", default="readout")
    ap.add_argument("--structure", default="total_order")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for model in args.models.split(","):
        for family in args.families.split(","):
            for condition in args.conditions.split(","):
                for N in [int(x) for x in args.Ns.split(",")]:
                    try:
                        r = geometry_cell(args.acts, model, family, condition, N,
                                          args.scheme, args.structure, args.layer)
                    except Exception as e:
                        print(f"{model} {family}/{condition} N={N}: ERROR {e}", flush=True)
                        continue
                    if r is None:
                        continue
                    rows.append(r)
                    if args.structure == "cyclic":
                        print(f"{model:12s} {family}/{condition} N={N} L{r['layer']} | "
                              f"cyclic_rsa={r['circular_rsa']['cyclic_rsa']:.2f} "
                              f"(lin={r['circular_rsa']['linear_rsa']:.2f}) "
                              f"angular={r['angular_decode']} "
                              f"-> {'RING' if r['ring'] else 'not-ring'}", flush=True)
                    else:
                        c = r.get("curvature", {})
                        print(f"{model:12s} {family}/{condition} N={N} L{r['layer']} | "
                              f"lin={c.get('linear')} nonlin={c.get('nonlinear')} "
                              f"geo={c.get('geodesic')} gap={c.get('curvature_gap')} "
                              f"pcurve={r.get('principal_curve_curvature')} "
                              f"curved={r.get('curved_irreducible', {}).get('curved')}", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
