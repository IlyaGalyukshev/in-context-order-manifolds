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
import re

STEER_RE = re.compile(r"^(\S+)\s+(\S+)\s+L(\d+)\s+\|\s+along_slope=([+-]?[\d.]+)\s+offaxis_null=([+-]?[\d.]+)"
                      r"±([\d.]+)\s+\(n=(\d+)\)\s+p\(\|off\|>=\|along\|\)=([\d.]+)")


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
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battery", nargs="*", default=[])
    ap.add_argument("--steer", nargs="*", default=[])
    ap.add_argument("--questions", default=None)
    ap.add_argument("--stimuli", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"gate": {}, "steer": {}}
    Q = {}
    if a.questions:
        for line in open(a.questions):
            q = json.loads(line)
            if q.get("family") == "pairwise" and q.get("target_entities"):
                Q[q["qid"]] = q
    stated = {}
    if a.stimuli:
        for line in open(a.stimuli):
            st = json.loads(line)
            stated[st["stimulus_id"]] = {frozenset((c["entity"], c["entity_b"])) for c in st.get("cards", [])}
    for kv in a.battery:
        tag, path = kv.split("=", 1); res["gate"][tag] = gate(path)
        if Q and stated:
            res["gate"][tag]["distance"] = distance_effect(path, Q, stated)
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
            print(f"steer {tag:12s} {fam:8s} L{r['layer']} slope={r['along_slope']:+.3f} p={r['p']:.3f}")


if __name__ == "__main__":
    main()
