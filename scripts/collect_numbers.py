#!/usr/bin/env python
"""Collect per-model behaviour and steering summaries into one JSON for the paper's number macros.

  --battery TAG=path/to/battery_*.jsonl   per-family mean score (same definition as summarize.py GATE)
  --steer   TAG=path/to/steer_rank.log     along-axis slope vs matched-norm off-axis null, as printed by
                                           steer_rank.py ('<family> <scheme> L<layer> | along_slope=...')

  python scripts/collect_numbers.py --battery gemma12b=.../battery.jsonl --steer gemma12b_n64=.../g9.log \
      --out results_figdata/v11/collected.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

STEER_RE = re.compile(r"^(\S+)\s+(\S+)\s+L(\d+)\s+\|\s+along_slope=([+-]?[\d.]+)\s+offaxis_null=([+-]?[\d.]+)"
                      r"±([\d.]+)\s+\(n=(\d+)\)\s+p\(\|off\|>=\|along\|\)=([\d.]+)")
PERM_RE = re.compile(r"^(\S+)\s+\S+\s+L\d+\s+\|\s+along_slope=[+-]?[\d.]+\s+permaxis_null=([+-]?[\d.]+)"
                     r"±([\d.]+)\s+\(n=(\d+)\)\s+p_perm\(\|perm\|>=\|along\|\)=([\d.]+)")


def gate_lp(path, Q):
    """pairwise accuracy from the log-prob margin toward the first-named candidate (as summarize.py GATE_LP)."""
    ok = n = 0
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("q_family") == "pairwise" and r.get("logit_margin") is not None and r.get("qid") in Q:
            q = Q[r["qid"]]
            ok += (r["logit_margin"] > 0) == (q["answer_key"] == q["target_entities"][0]); n += 1
    return {"acc": ok / n, "n": n} if n else None


def distance_effect(path, Q, stated):
    """pairwise accuracy (generated answer) by rank distance, split into stated (one card) and inferred pairs."""
    acc = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        q = Q.get(r.get("qid"))
        if q is None or r.get("score") is None or r.get("rank_distance") is None or r["score"] != r["score"]:
            continue
        kind = "stated" if frozenset(q["target_entities"]) in stated.get(r["stimulus_id"], set()) else "inferred"
        a = acc.setdefault(kind, {}).setdefault(int(r["rank_distance"]), [0.0, 0])
        a[0] += float(r["score"]); a[1] += 1
    return {k: {d: {"acc": v[0] / v[1], "n": v[1]} for d, v in sorted(dd.items())} for k, dd in acc.items()}


def _graph(st):
    """rank of each entity, undirected hop distances, and the one-step tally (claimed earlier minus later)."""
    from collections import deque
    order = st["latent_order"]; idx = {e: i for i, e in enumerate(order)}; n = len(order)
    adj = {i: set() for i in range(n)}; tally = [0] * n
    for c in st.get("cards", []):
        a, b = idx[c["entity"]], idx[c["entity_b"]]
        adj[a].add(b); adj[b].add(a)
        lo, hi = (a, b) if a < b else (b, a)                     # real stimuli: claims match the latent order
        tally[lo] += 1; tally[hi] -= 1
    hop = {}
    for s0 in range(n):
        d = {s0: 0}; q = deque([s0])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in d:
                    d[v] = d[u] + 1; q.append(v)
        hop[s0] = d
    return idx, hop, tally


def distance_model(path, Q, stims, n_boot=500, seed=0):
    """Symbolic distance effect with controls: logistic regression of pairwise correctness on rank distance,
    hop distance and |one-step tally difference|, inferred pairs between interior entities only; cluster
    bootstrap over stimuli for the rank-distance coefficient (standardized predictors)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    G = {sid: _graph(st) for sid, st in stims.items()}
    X, y, g = [], [], []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        q = Q.get(r.get("qid")); sid = r.get("stimulus_id")
        if q is None or sid not in G or r.get("score") is None or r["score"] != r["score"]:
            continue
        idx, hop, tally = G[sid]; n = len(idx)
        a, b = idx.get(q["target_entities"][0]), idx.get(q["target_entities"][1])
        if a is None or b is None or not (2 <= a <= n - 3 and 2 <= b <= n - 3):
            continue
        h = hop[a].get(b)
        if h is None or h < 2:                                   # inferred pairs only
            continue
        X.append([abs(a - b), h, abs(tally[a] - tally[b])]); y.append(float(r["score"]) > 0.5); g.append(sid)
    if len(set(y)) < 2:
        return None
    X = np.array(X, float); y = np.array(y); g = np.array(g)
    Z = (X - X.mean(0)) / X.std(0)
    fit = lambda Zs, ys: LogisticRegression(C=1e4, max_iter=2000).fit(Zs, ys).coef_[0]
    beta = fit(Z, y)
    rng = np.random.default_rng(seed); ug = np.unique(g); bs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug)); m = np.concatenate([np.where(g == u)[0] for u in pick])
        if len(set(y[m])) == 2:
            bs.append(fit(Z[m], y[m])[0])
    lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (None, None)
    return {"beta_rank": float(beta[0]), "beta_rank_ci": [lo, hi], "beta_hop": float(beta[1]),
            "beta_tally": float(beta[2]), "n_pairs": int(len(y)), "n_stim": int(len(ug))}


def gate(path):
    acc = {}
    for line in open(path):
        try:
            j = json.loads(line)
        except Exception:
            continue
        s, q = j.get("score"), j.get("q_family")
        if s is not None and s == s:
            a = acc.setdefault(q, [0.0, 0]); a[0] += float(s); a[1] += 1
    return {q: {"acc": v[0] / v[1], "n": v[1]} for q, v in sorted(acc.items()) if v[1]}


def steer(path):
    out = {}
    for line in open(path, errors="replace"):
        m = STEER_RE.match(line.strip())
        if m:
            fam, scheme, L, al, nm, ns, n, p = m.groups()
            out[fam] = {"scheme": scheme, "layer": int(L), "along_slope": float(al), "null_mean": float(nm),
                        "null_sd": float(ns), "n_offaxis": int(n), "p": float(p)}
        m = PERM_RE.match(line.strip())
        if m:                                                  # permuted-rank control axes (steer_rank --control-axes perm:K)
            fam, pm, ps, n, p = m.groups()
            out.setdefault(fam, {}).update({"perm_mean": float(pm), "perm_sd": float(ps), "n_perm": int(n), "p_perm": float(p)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battery", nargs="*", default=[])
    ap.add_argument("--steer", nargs="*", default=[])
    ap.add_argument("--questions", default=None)
    ap.add_argument("--stimuli", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--update", action="store_true", help="merge the given tags into an existing --out instead of rewriting it")
    a = ap.parse_args()
    res = {"gate": {}, "steer": {}}
    if a.update and os.path.exists(a.out):
        res = json.load(open(a.out))
    Q = {}
    if a.questions:
        for line in open(a.questions):
            q = json.loads(line)
            if q.get("family") == "pairwise" and q.get("target_entities"):
                Q[q["qid"]] = q
    stated = {}; stims = {}
    if a.stimuli:
        for line in open(a.stimuli):
            st = json.loads(line)
            stated[st["stimulus_id"]] = {frozenset((c["entity"], c["entity_b"])) for c in st.get("cards", [])}
            stims[st["stimulus_id"]] = st
    for kv in a.battery:
        tag, path = kv.split("=", 1); res["gate"][tag] = gate(path)
        if Q and stated:
            res["gate"][tag]["distance"] = distance_effect(path, Q, stated)
            dm = distance_model(path, Q, stims)
            if dm:
                res["gate"][tag]["distance_model"] = dm
                print(f"sde   {tag:12s} beta_rank={dm['beta_rank']:+.2f} {dm['beta_rank_ci']} hop={dm['beta_hop']:+.2f} "
                      f"tally={dm['beta_tally']:+.2f} n={dm['n_pairs']}", flush=True)
        if Q:
            lp = gate_lp(path, Q)
            if lp:
                res["gate"][tag]["pairwise_lp"] = lp
    for kv in a.steer:
        tag, path = kv.split("=", 1); res["steer"][tag] = steer(path)
    json.dump(res, open(a.out, "w"), indent=1)
    for tag, g in res["gate"].items():
        print(f"gate  {tag:12s} pairwise={g.get('pairwise', {}).get('acc', float('nan')):.3f} "
              f"reconstruction={g.get('reconstruction', {}).get('acc', float('nan')):.3f} "
              f"lp={g.get('pairwise_lp', {}).get('acc', float('nan')):.3f}")
    for tag, s in res["steer"].items():
        for fam, r in s.items():
            print(f"steer {tag:12s} {fam:8s} L{r['layer']} slope={r['along_slope']:+.3f} p={r['p']:.3f}"
                  + (f" p_perm={r['p_perm']:.3f}" if "p_perm" in r else ""))


if __name__ == "__main__":
    main()
