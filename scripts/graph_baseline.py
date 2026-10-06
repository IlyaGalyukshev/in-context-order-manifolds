#!/usr/bin/env python
"""No-integration baselines for the real - twin order increment, scored with the model's RSA metric.

A coherence-null twin keeps the real stimulus's undirected card graph and reverses the claimed
direction of one non-adjacent edge, so any undirected co-occurrence code has real - twin = 0 exactly.
The baselines below use the claimed directions:
  hop h  : entity score = (# entities claimed later within h steps) - (# claimed earlier within h steps)
           h = 1 is the local tally over stated cards; h = inf is the transitive closure (full integration)
Each baseline RDM |s_i - s_j| over interior entities is scored against the ideal line with the same
whitened_rsa used for hidden states; we report real, twin and the paired real - twin increment
(bootstrap over stimuli), per family and pooled.

  python scripts/graph_baseline.py --stimuli <core>/stimuli.jsonl --stimuli-null <core>/stimuli_null.jsonl \
      --families s0_zib,s0_quomp,s1_size,s1_loud,s1_heat --n-items 12 --hops 1,2,3,inf --json out.json \
      [--dumps <per_stim_rsa dir> --models <tag,tag>]

Because the baselines are noise-free and hidden-state RSA is attenuated by noise, the scale-free
comparison is the relative drop 1 - twin/real (noise attenuates real and twin alike); with --dumps the
same quantity is computed for each model from per_stim_rsa.py outputs (bootstrap over stimuli).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from icom.generator.bcs import RELATIONS  # noqa: E402
from icom.probes.crossnobis import line_rdm, whitened_rsa  # noqa: E402


def claimed_edges(st):
    """(earlier, later) entity-index pairs as CLAIMED by the cards (true for real, one reversed in twin)."""
    rel = RELATIONS[st["relation"]]
    idx = {e: i for i, e in enumerate(st["latent_order"])}
    out = []
    for c in st["cards"]:
        a, b, t = c["entity"], c["entity_b"], c["text"]
        if f" {rel.inv} " in t:
            out.append((idx[b], idx[a]))           # "The b REL_inv the a": a earlier... first-named is later
        else:
            out.append((idx[a], idx[b]))           # "The a REL the b": a earlier
    return out


def scores(edges, n, h):
    """later-minus-earlier reach within h steps (h=None: transitive closure)."""
    A = np.zeros((n, n), dtype=bool)
    for a, b in edges:
        A[a, b] = True
    reach, frontier = A.copy(), A.copy()
    steps = 1
    while h is None or steps < h:
        nxt = (frontier.astype(int) @ A.astype(int)) > 0
        new = nxt & ~reach
        if not new.any():
            break
        reach |= new; frontier = new; steps += 1
    return reach.sum(1).astype(float) - reach.sum(0).astype(float)


def rsa(st, s):
    N = int(st["n_items"]); ranks = np.arange(1, N + 1)
    inter = np.where((ranks >= 3) & (ranks <= N - 2))[0]
    R = np.abs(s[inter, None] - s[None, inter])
    return whitened_rsa(R, line_rdm(ranks[inter]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stimuli", required=True); ap.add_argument("--stimuli-null", required=True)
    ap.add_argument("--families", default="s0_zib,s0_quomp,s1_size,s1_loud,s1_heat")
    ap.add_argument("--n-items", type=int, default=12); ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--hops", default="1,2,3,inf"); ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json", default=None)
    ap.add_argument("--dumps", default=None); ap.add_argument("--models", default=None)
    ap.add_argument("--scheme", default="readout")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    hops = [None if h == "inf" else int(h) for h in a.hops.split(",")]
    keep = lambda s: s["family"] in a.families.split(",") and int(s["n_items"]) == a.n_items and s.get("condition") == a.condition
    twins = {}
    for l in open(a.stimuli_null):
        t = json.loads(l)
        if keep(t):
            twins.setdefault((t["family"], tuple(t["latent_order"])), t)
    pairs = [(s, twins.get((s["family"], tuple(s["latent_order"])))) for s in map(json.loads, open(a.stimuli)) if keep(s)]
    pairs = [(s, t) for s, t in pairs if t is not None]
    res = {"n_pairs": len(pairs), "rows": []}
    for h in hops:
        hl = "inf" if h is None else str(h)
        per = {}
        for s, t in pairs:
            n = int(s["n_items"])
            r = rsa(s, scores(claimed_edges(s), n, h)); w = rsa(t, scores(claimed_edges(t), n, h))
            per.setdefault(s["family"], []).append((r, w))
        allv = [v for vs in per.values() for v in vs]
        for fam, vs in list(per.items()) + [("pooled", allv)]:
            v = np.array([x for x in vs if np.isfinite(x[0]) and np.isfinite(x[1])])
            d = v[:, 0] - v[:, 1]
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n_boot)]
            row = dict(hop=hl, family=fam, n=int(len(d)), real=float(v[:, 0].mean()), twin=float(v[:, 1].mean()),
                       increment=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            row["rel_drop"] = float(1 - row["twin"] / row["real"]) if row["real"] else float("nan")
            res["rows"].append(row)
            print(f"hop={hl:>3s} {fam:9s} n={row['n']:3d} real={row['real']:.3f} twin={row['twin']:.3f} "
                  f"incr={row['increment']:+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]", flush=True)
    if a.dumps and a.models:
        res["models"] = []
        for m in a.models.split(","):
            R, T = {}, {}
            for fam in a.families.split(","):
                f = os.path.join(a.dumps, f"{m}__{fam}__{a.scheme}__N{a.n_items}.json")
                if os.path.exists(f):
                    d = json.load(open(f))
                    R[fam] = np.array([v for v in d["real"].values() if np.isfinite(v)])
                    T[fam] = np.array([v for v in d["twin"].values() if np.isfinite(v)])
            if not R:
                continue
            drop = lambda RR, TT: 1 - np.mean([t.mean() for t in TT.values()]) / np.mean([r.mean() for r in RR.values()])
            bs = []
            for _ in range(a.n_boot):
                bs.append(drop({k: v[rng.integers(0, len(v), len(v))] for k, v in R.items()},
                               {k: v[rng.integers(0, len(v), len(v))] for k, v in T.items()}))
            row = dict(model=m, families=sorted(R), real=float(np.mean([r.mean() for r in R.values()])),
                       twin=float(np.mean([t.mean() for t in T.values()])), rel_drop=float(drop(R, T)),
                       ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            res["models"].append(row)
            print(f"model {m:28s} real={row['real']:.3f} twin={row['twin']:.3f} rel_drop={row['rel_drop']:.3f} "
                  f"[{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]", flush=True)
    for r in res["rows"]:
        if r["family"] == "pooled":
            print(f"baseline hop={r['hop']:>3s} rel_drop={r['rel_drop']:.3f}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
