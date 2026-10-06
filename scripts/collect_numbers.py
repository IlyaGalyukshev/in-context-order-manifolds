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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"gate": {}, "steer": {}}
    for kv in a.battery:
        tag, path = kv.split("=", 1); res["gate"][tag] = gate(path)
    for kv in a.steer:
        tag, path = kv.split("=", 1); res["steer"][tag] = steer(path)
    json.dump(res, open(a.out, "w"), indent=1)
    for tag, g in res["gate"].items():
        print(f"gate  {tag:12s} pairwise={g.get('pairwise', {}).get('acc', float('nan')):.3f} "
              f"reconstruction={g.get('reconstruction', {}).get('acc', float('nan')):.3f}")
    for tag, s in res["steer"].items():
        for fam, r in s.items():
            print(f"steer {tag:12s} {fam:8s} L{r['layer']} slope={r['along_slope']:+.3f} p={r['p']:.3f}")


if __name__ == "__main__":
    main()
