#!/usr/bin/env python
"""M4 — per-stimulus Procrustes 'spaghetti' (the honest replacement for centroid manifolds).

Centroid plots average away the per-prompt variance and manufacture a smooth curve (the N40 caveat).
This instead keeps every stimulus: PCA-embed each stimulus's interior k-mean reps at the peak layer,
orient by rank, generalized-Procrustes-align all stimuli into ONE frame (matching by rank label),
and emit every per-stimulus curve + the median tube with a CI band — for real vs twin. You see both
the FORM and the SPREAD. Emits JSON for the artifact viewer.

  python scripts/viz_spaghetti.py --acts results/e1hard_20260820/acts --model qwen3-4b \
      --family s0_zib --n-items 12 --scheme card_mean --layer 18 --out out/spaghetti.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from icom.probes import interior_mask


def _load(acts, model, family, condition, scheme, n_items, is_null):
    recs = []
    for f in sorted((Path(acts) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if m["family"] != family or m["condition"] != condition or int(m["n_items"]) != int(n_items):
            continue
        if bool(m.get("is_null", False)) != is_null:
            continue
        key = f"mean_{scheme}" if f"mean_{scheme}" in z.files else (scheme if scheme in z.files else None)
        if key is None:
            continue
        X = z[key].astype(np.float64)
        if X.ndim == 4:
            X = X.mean(axis=1)
        recs.append({"X": X, "ranks": z["ranks"].astype(int), "N": int(m["n_items"])})
    return recs


def _shape(rec, layer, dim=3):
    """Interior reps at `layer` → [n_int, dim] PCA, rows ordered by rank, oriented so PC1 increases
    with rank (sign fix), centered. Returns (coords[n_int,dim], ranks[n_int]) or None."""
    mask = interior_mask(rec["ranks"], rec["N"]) & np.isfinite(rec["X"][:, layer, :]).all(axis=1)
    if mask.sum() < 4:
        return None
    X = rec["X"][mask, layer, :]; ranks = rec["ranks"][mask]
    order = np.argsort(ranks); X, ranks = X[order], ranks[order]
    P = PCA(min(dim, X.shape[0] - 1), random_state=0).fit_transform(X - X.mean(0))
    if P.shape[1] < dim:                                        # pad to dim
        P = np.column_stack([P, np.zeros((P.shape[0], dim - P.shape[1]))])
    if np.corrcoef(P[:, 0], ranks)[0, 1] < 0:                  # orient PC1 ↑ with rank
        P[:, 0] *= -1
    return P - P.mean(0), ranks


def _procrustes(A, ref):
    """Rotate+scale A to best match ref (both [n,dim], centered). Returns aligned A."""
    U, _, Vt = np.linalg.svd(ref.T @ A)
    R = (U @ Vt).T                                             # rotation A→ref
    Ar = A @ R
    s = (Ar * ref).sum() / (Ar * Ar).sum() if (Ar * Ar).sum() > 0 else 1.0
    return Ar * s


def _align_set(shapes):
    """Generalized Procrustes: align all [n,dim] shapes (shared rank order) to their mean (2 passes)."""
    if not shapes:
        return []
    cur = [s.copy() for s in shapes]
    for _ in range(2):
        ref = np.mean(cur, axis=0)
        ref = ref - ref.mean(0)
        cur = [_procrustes(s, ref) for s in cur]
    return cur


def _tube(aligned):
    """median + 2.5/97.5 band per rank-position across stimuli. aligned: list of [n,dim]."""
    A = np.stack(aligned)                                      # [S, n, dim]
    return {"median": np.median(A, axis=0).tolist(),
            "lo": np.percentile(A, 2.5, axis=0).tolist(),
            "hi": np.percentile(A, 97.5, axis=0).tolist()}


def build(acts, model, family, condition, scheme, n_items, layer, dim, max_curves, seed):
    out = {}
    for tag, is_null in (("real", False), ("twin", True)):
        recs = _load(acts, model, family, condition, scheme, n_items, is_null)
        shapes, ranks_ref = [], None
        for r in recs:
            sh = _shape(r, layer, dim)
            if sh is None or len(sh[1]) != (n_items - 4):      # keep only full interior sets, comparable
                continue
            shapes.append(sh[0]); ranks_ref = sh[1]
        if not shapes:
            out[tag] = None; continue
        aligned = _align_set(shapes)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(aligned))[:max_curves]
        out[tag] = {"n_stim": len(aligned), "ranks": ranks_ref.tolist(),
                    "curves": [aligned[i].tolist() for i in idx], "tube": _tube(aligned)}
    return {"model": model, "family": family, "n_items": n_items, "scheme": scheme,
            "layer": layer, "dim": dim, **out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="s0_zib"); ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--scheme", default="card_mean"); ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--max-curves", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = build(args.acts, args.model, args.family, args.condition, args.scheme, args.n_items,
                args.layer, args.dim, args.max_curves, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res))
    for tag in ("real", "twin"):
        d = res.get(tag)
        print(f"{args.model} {args.family} N{args.n_items} L{args.layer} {tag}: "
              f"{'n_stim=' + str(d['n_stim']) + ' curves=' + str(len(d['curves'])) if d else 'no data'}", flush=True)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
