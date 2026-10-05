#!/usr/bin/env python
"""Per-stimulus whitened-RSA at a CROSS-FITTED layer — the input for paired cross-model statistics.

Each real stimulus is scored at the layer that maximises mean RSA on the OTHER half of real stimuli
(halves fixed by a hash of content_key, hence identical across models); twins are scored the same
way. No stimulus is ever evaluated at a layer chosen on itself, and per-stimulus values let
compare_models.py resample the SAME stimuli jointly across models (paired CIs, TOST, scale slopes).

Also stores the same per-stimulus RSA at layer 0 (embeddings, no context) as a control: a locus whose
increment already exists at layer 0 reflects stimulus structure, not computation.

Writes one small JSON per (model, family):
  {model, family, scheme, condition, n_items, L, layers, real:{key:rsa}, twin:{key:rsa}, real_l0, twin_l0}

  python scripts/per_stim_rsa.py --acts <acts_dir> --model <tag> \
      --families s0_zib,s0_quomp,s1_size,s1_loud,s1_heat --scheme card_mean --n-items 12 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_crossnobis as pc  # noqa: E402


def half(key) -> int:
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 2


def keyed(recs):
    """content_key, de-duplicated deterministically (files are read in sorted order for every model)."""
    seen, out = {}, []
    for r in recs:
        k = str(r.get("content_key"))
        seen[k] = seen.get(k, 0) + 1
        out.append(k if seen[k] == 1 else f"{k}#{seen[k]}")
    return out


def profile(recs, L, ideal, n_splits, seed):
    return np.array([[pc._stim_rsa(r, l, ideal, n_splits, seed) for l in range(L)] for r in recs], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--families", default="s0_zib")
    ap.add_argument("--scheme", default="card_mean")
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--difficulty", default=None, choices=["easy", "hard"])
    ap.add_argument("--ideal", default="line", choices=["line", "ring"])
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for fam in args.families.split(","):
        kw = dict(n_items=args.n_items, difficulty=args.difficulty)
        real = pc.load_repeat(args.acts, args.model, fam, args.condition, args.scheme, is_null=False, **kw)
        twin = pc.load_repeat(args.acts, args.model, fam, args.condition, args.scheme, is_null=True, **kw)
        if not real:
            print(f"{args.model} {fam}: no real stimuli", flush=True)
            continue
        L = real[0]["RDM"].shape[2] if real[0]["mode"] == "rdm" else real[0]["X"].shape[2]
        kr, kt = keyed(real), keyed(twin)
        Sr = profile(real, L, args.ideal, args.n_splits, args.seed)
        St = profile(twin, L, args.ideal, args.n_splits, args.seed) if twin else np.zeros((0, L))
        hr = np.array([half(k) for k in kr]); ht = np.array([half(k) for k in kt])
        layers = {}
        for h in (0, 1):
            prof = np.nanmean(Sr[hr == h], axis=0) if (hr == h).any() else np.nanmean(Sr, axis=0)
            layers[h] = int(np.nanargmax(prof))
        real_v = {k: float(Sr[i, layers[1 - hr[i]]]) for i, k in enumerate(kr)}
        twin_v = {k: float(St[j, layers[1 - ht[j]]]) for j, k in enumerate(kt)}
        # embedding-layer control: the same per-stimulus RSA at layer 0 (no contextual processing yet)
        real_l0 = {k: float(Sr[i, 0]) for i, k in enumerate(kr)}
        twin_l0 = {k: float(St[j, 0]) for j, k in enumerate(kt)}
        rec = dict(model=args.model, family=fam, scheme=args.scheme, condition=args.condition,
                   n_items=args.n_items, difficulty=args.difficulty, L=int(L),
                   layers={str(h): l for h, l in layers.items()}, real=real_v, twin=twin_v,
                   real_l0=real_l0, twin_l0=twin_l0)
        fn = Path(args.out) / f"{args.model}__{fam}__{args.scheme}__N{args.n_items}.json"
        json.dump(rec, open(fn, "w"))
        rv = np.array([v for v in real_v.values() if v == v]); tv = np.array([v for v in twin_v.values() if v == v])
        print(f"{args.model:28s} {fam:9s} L{layers[0]}/{layers[1]} of {L}  real={rv.mean():.3f} (n={len(rv)})  "
              f"twin={tv.mean() if len(tv) else float('nan'):.3f} (n={len(tv)})  incr={rv.mean() - (tv.mean() if len(tv) else np.nan):+.3f}",
              flush=True)
        r0 = np.array([v for v in real_l0.values() if v == v]); t0 = np.array([v for v in twin_l0.values() if v == v])
        print(f"{'':28s} {'':9s} embedding-layer control L0: incr={r0.mean() - (t0.mean() if len(t0) else np.nan):+.3f}", flush=True)


if __name__ == "__main__":
    main()
