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


def _template(card):
    return card["text"].replace(card["entity_b"], "<B>").replace(card["entity"], "<A>")


def card_graphs(stimuli_path, null_path=None):
    """content_key -> (edges as rank-index pairs, set of on-cycle ranks or None). For twins the reversed claim is
    the card whose phrasing template (learned from real stimuli, where every claim follows the latent order) points
    against the latent order; the cycle it closes runs along the Hamiltonian path between the two ranks."""
    reals = [json.loads(l) for l in open(stimuli_path)]
    sign = {}                                                   # template -> True if <A> precedes <B>
    for st in reals:
        er = st.get("entity_ranks") or {}
        for c in st.get("cards", []):
            if c.get("entity") in er and c.get("entity_b") in er:
                sign[_template(c)] = er[c["entity"]] < er[c["entity_b"]]
    out, cyc_by_set = {}, {}
    for st in ([json.loads(l) for l in open(null_path)] if null_path else []):
        er = st.get("entity_ranks") or {}
        rev = [c for c in st.get("cards", []) if c.get("entity") in er and _template(c) in sign
               and sign[_template(c)] != (er[c["entity"]] < er[c["entity_b"]])]
        cyc = None
        if len(rev) == 1:
            lo, hi = sorted((er[rev[0]["entity"]], er[rev[0]["entity_b"]]))
            cyc = set(range(lo, hi + 1))
        out[st.get("content_key")] = ([(er[c["entity"]] - 1, er[c["entity_b"]] - 1) for c in st.get("cards", [])
                                       if c.get("entity") in er and c.get("entity_b") in er], cyc)
        cyc_by_set[frozenset(er)] = cyc
    for st in reals:
        er = st.get("entity_ranks")
        if er:
            out[st.get("content_key")] = ([(er[c["entity"]] - 1, er[c["entity_b"]] - 1) for c in st.get("cards", [])
                                           if c.get("entity") in er and c.get("entity_b") in er],
                                          cyc_by_set.get(frozenset(er)))
    return out


def subset_mask(ranks, N, sub, edges, cyc):
    """pair subsets over interior entities: onehop/multihop (stated graph); oncycle = both entities on the twin's
    cycle (their order is undefined in the twin), offcycle = at least one entity off it (their order is still
    implied in the twin). The same pairs are used for the real stimulus and its twin."""
    if sub in ("onehop", "multihop"):
        return pc._pair_mask(ranks, N, sub, edges=edges)
    if cyc is None:
        return None
    on = np.array([r in cyc for r in ranks])
    both = on[:, None] & on[None, :]
    return both if sub == "oncycle" else ~both


def keyed(recs):
    """content_key, de-duplicated deterministically (files are read in sorted order for every model)."""
    seen, out = {}, []
    for r in recs:
        k = str(r.get("content_key"))
        seen[k] = seen.get(k, 0) + 1
        out.append(k if seen[k] == 1 else f"{k}#{seen[k]}")
    return out


def ext_rsa(rec, layer, order):
    """whitened RSA of the interior crossnobis RDM against an external entity order (rdm store)."""
    if rec["mode"] != "rdm":
        return float("nan")
    R = rec["RDM"][:, :, layer]
    mask = pc.interior_mask(rec["ranks"], rec["N"]) & np.isfinite(R).all(axis=1)
    idx = np.where(mask)[0]
    ents = [rec["entities"][i] for i in idx]
    if len(idx) < 4 or any(e not in order for e in ents):
        return float("nan")
    return float(pc.whitened_rsa(R[np.ix_(idx, idx)], pc.line_rdm(np.array([order[e] for e in ents]))))


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
    ap.add_argument("--pair-subsets", default=None,
                    help="e.g. onehop,multihop: also score each real stimulus on a PAIR SUBSET of the stated-relation "
                         "graph (needs --stimuli) at the same cross-fitted layer, and test subsets within stimuli")
    ap.add_argument("--stimuli", default=None, help="stimuli.jsonl (real) for --pair-subsets edge graphs")
    ap.add_argument("--stimuli-null", default=None,
                    help="stimuli_null.jsonl: also score twins on the same pair subsets, and enable oncycle/offcycle")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--entity-order", default=None,
                    help="comma list of entity names in an EXTERNAL order (e.g. calendar months): also score each "
                         "stimulus against that order at the same cross-fitted layer (familiar-token control)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    graphs = card_graphs(args.stimuli, args.stimuli_null) if (args.pair_subsets and args.stimuli) else {}
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
        ext = {}
        if args.entity_order:
            order = {e: i for i, e in enumerate(args.entity_order.split(","))}
            ext = dict(real_ext={k: ext_rsa(r, layers[1 - hr[i]], order) for i, (k, r) in enumerate(zip(kr, real))},
                       twin_ext={k: ext_rsa(t, layers[1 - ht[j]], order) for j, (k, t) in enumerate(zip(kt, twin))},
                       real_ext_l0={k: ext_rsa(r, 0, order) for k, r in zip(kr, real)})
        subsets, twin_subsets = {}, {}
        if args.pair_subsets and graphs:
            def score(recs, keys, hs):
                vals = {}
                for i, (k, r) in enumerate(zip(keys, recs)):
                    g = graphs.get(r.get("content_key"))
                    ir = pc._interior_rdm(r, layers[1 - hs[i]], args.n_splits, args.seed)
                    if g is None or ir is None:
                        continue
                    rdm, ranks, N = ir
                    mask = subset_mask(ranks, N, sub, g[0], g[1])
                    if mask is None or mask.sum() < 2 * 2:
                        continue
                    vals[k] = float(pc.whitened_rsa(rdm, pc.line_rdm(ranks), mask=mask))
                return vals
            for sub in args.pair_subsets.split(","):
                subsets[sub] = score(real, kr, hr)
                if args.stimuli_null and twin:
                    twin_subsets[sub] = score(twin, kt, ht)
        rec = dict(model=args.model, family=fam, scheme=args.scheme, condition=args.condition,
                   n_items=args.n_items, difficulty=args.difficulty, L=int(L),
                   layers={str(h): l for h, l in layers.items()}, real=real_v, twin=twin_v,
                   real_l0=real_l0, twin_l0=twin_l0, real_subsets=subsets, twin_subsets=twin_subsets, **ext)
        fn = Path(args.out) / f"{args.model}__{fam}__{args.scheme}__N{args.n_items}.json"
        json.dump(rec, open(fn, "w"))
        rv = np.array([v for v in real_v.values() if v == v]); tv = np.array([v for v in twin_v.values() if v == v])
        print(f"{args.model:28s} {fam:9s} L{layers[0]}/{layers[1]} of {L}  real={rv.mean():.3f} (n={len(rv)})  "
              f"twin={tv.mean() if len(tv) else float('nan'):.3f} (n={len(tv)})  incr={rv.mean() - (tv.mean() if len(tv) else np.nan):+.3f}",
              flush=True)
        r0 = np.array([v for v in real_l0.values() if v == v]); t0 = np.array([v for v in twin_l0.values() if v == v])
        if len(subsets) == 2:
            a, b = list(subsets)
            ks = [k for k in subsets[a] if k in subsets[b] and subsets[a][k] == subsets[a][k] and subsets[b][k] == subsets[b][k]]
            d = np.array([subsets[a][k] - subsets[b][k] for k in ks])
            rng = np.random.default_rng(args.seed)
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)]) if len(d) else np.array([np.nan])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"{'':28s} {'':9s} within-real {a}={np.mean([subsets[a][k] for k in ks]):.3f} {b}={np.mean([subsets[b][k] for k in ks]):.3f} "
                  f"diff={d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] n={len(d)}", flush=True)
        if ext:
            mv = lambda d: np.nanmean(list(d.values())) if d else float("nan")
            print(f"{'':28s} {'':9s} EXTERNAL order: real={mv(ext['real_ext']):.3f} twin={mv(ext['twin_ext']):.3f} "
                  f"layer0={mv(ext['real_ext_l0']):.3f}  (in-context order above)", flush=True)
        print(f"{'':28s} {'':9s} embedding-layer control L0: incr={r0.mean() - (t0.mean() if len(t0) else np.nan):+.3f}", flush=True)


if __name__ == "__main__":
    main()
