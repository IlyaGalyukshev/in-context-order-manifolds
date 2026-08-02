#!/usr/bin/env python
"""Phase G10 (CPU part) — does the in-context BCS order axis reuse a pretrained
magnitude/size/space axis?

Consumes stored activations (extraction is GPU-side, done separately):
  - BCS acts: the standard per-stimulus npz (acts/<model>/*.npz), for a
    (family × condition × N) cell.
  - source-axis acts: <src>/<model>/<axis>.npz with arrays `pos` and `neg`
    ([n_pairs × (L+1) × D]) = activations of the positive/negative prompts from
    scripts/build_source_axes.py (magnitude / space / size).

At a chosen layer it computes the BCS rank direction (ridge weight on interior
entities) and each source direction (mean(pos) − mean(neg) = CAA), then reports
|cos| and principal-angle subspace alignment. Causal ablation is GPU-side.

  python scripts/align_axes.py --acts acts --src src_axes --model qwen3-4b \
      --family s0_zib --condition shuffle --N 12 --scheme readout --layer 18 \
      --json out/align.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icom.probes import (contrast_direction, direction_cosine, load_records,
                         n_layers, rank_contrast_direction, stack_layer, subspace_alignment)


def bcs_rank_dir(acts, model, family, condition, N, scheme, layer, structure="total_order"):
    recs = load_records(acts, model, family, condition, scheme, N=N, structure=structure)
    if not recs:
        return None, 0
    s = stack_layer(recs, layer, interior_only=True)
    if not s:
        return None, n_layers(recs)
    X, y, _ = s
    # CAA-style contrast (same estimator as the source axis) — comparable cosine.
    return rank_contrast_direction(X, y), n_layers(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--src", required=True, help="source-axis acts dir (<src>/<model>/<axis>.npz)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="s0_zib")
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--scheme", default="readout")
    ap.add_argument("--structure", default="total_order")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--axes", default="magnitude,space,size")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rank_dir, n_lyr = bcs_rank_dir(args.acts, args.model, args.family, args.condition,
                                   args.N, args.scheme, args.layer, args.structure)
    if rank_dir is None:
        raise SystemExit("no BCS interior activations for that cell")

    out = {"model": args.model, "family": args.family, "condition": args.condition,
           "N": args.N, "layer": args.layer, "axes": {}}
    for axis in args.axes.split(","):
        p = Path(args.src) / args.model / f"{axis}.npz"
        if not p.exists():
            out["axes"][axis] = {"error": "no source acts"}; continue
        z = np.load(p, allow_pickle=False)
        if "pos" not in z.files or "neg" not in z.files:
            out["axes"][axis] = {"error": "npz missing pos/neg"}; continue
        # guard: same layer indexing as BCS, and a decodable layer index
        if z["pos"].shape[1] != n_lyr or z["neg"].shape[1] != n_lyr:
            out["axes"][axis] = {"error": f"layer mismatch: source L={z['pos'].shape[1]} vs BCS L={n_lyr}"}
            continue
        if not len(z["pos"]) or not len(z["neg"]) or args.layer >= z["pos"].shape[1]:
            out["axes"][axis] = {"error": "empty source or layer out of range"}; continue
        pos = z["pos"][:, args.layer, :]; neg = z["neg"][:, args.layer, :]
        src_dir = contrast_direction(pos, neg)
        out["axes"][axis] = {
            "cosine": round(direction_cosine(rank_dir, src_dir), 3),
            "subspace": round(subspace_alignment(rank_dir[:, None], src_dir[:, None]), 3),
            "n_pairs": int(len(pos)),
        }
        print(f"{args.model} {args.family} L{args.layer} · {axis}: "
              f"|cos|(order,{axis})={out['axes'][axis]['cosine']}", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
