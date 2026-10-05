#!/usr/bin/env python
"""Real − twin increment at EVERY layer, per locus — the input-layer (layer 0) diagnostic.

A locus whose increment is already present at layer 0 (embeddings, no context) reflects stimulus
structure (e.g. card-level pooling mixes in the partner's tokens), not computation; a clean locus is
zero at layer 0 and rises with depth. Bootstrap CI over stimuli per layer.

  python scripts/layer_profile.py --acts <acts> --model <tag> --families s1_size,s0_quomp,s0_zib \
      --schemes card_mean,readout --n-items 12 --json out/profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_crossnobis as pc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--families", default="s0_zib"); ap.add_argument("--schemes", default="card_mean,readout")
    ap.add_argument("--condition", default="shuffle"); ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); out = []
    for fam in args.families.split(","):
        for sc in args.schemes.split(","):
            R = pc.load_repeat(args.acts, args.model, fam, args.condition, sc, is_null=False, n_items=args.n_items)
            T = pc.load_repeat(args.acts, args.model, fam, args.condition, sc, is_null=True, n_items=args.n_items)
            if not R or not T:
                continue
            L = R[0]["RDM"].shape[2] if R[0]["mode"] == "rdm" else R[0]["X"].shape[2]
            SR = np.array([[pc._stim_rsa(r, l, "line", 20, 0) for l in range(L)] for r in R])
            ST = np.array([[pc._stim_rsa(t, l, "line", 20, 0) for l in range(L)] for t in T])
            inc = np.nanmean(SR, 0) - np.nanmean(ST, 0)
            bs = np.array([np.nanmean(SR[rng.integers(0, len(SR), len(SR))], 0) - np.nanmean(ST[rng.integers(0, len(ST), len(ST))], 0)
                           for _ in range(args.n_boot)])
            lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
            out.append(dict(model=args.model, family=fam, scheme=sc, L=int(L), n_real=len(R), n_twin=len(T),
                            increment=[float(x) for x in inc], ci_lo=[float(x) for x in lo], ci_hi=[float(x) for x in hi]))
            pk = int(np.nanargmax(inc))
            print(f"{args.model} {fam:9s} {sc:9s} layer0={inc[0]:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}]  peak L{pk}/{L - 1} "
                  f"({pk / (L - 1):.0%}) {inc[pk]:+.3f}", flush=True)
    json.dump(out, open(args.json, "w"))


if __name__ == "__main__":
    main()
