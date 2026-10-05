#!/usr/bin/env python
"""Paired cross-model statistics on per_stim_rsa.py dumps (same stimuli resampled jointly).

For every family and every model: the cross-fitted coherence increment (real − twin) with a
bootstrap CI and a one-sided bootstrap p, then Benjamini–Hochberg q over all (model, family) tests.
Against a reference model: the paired difference of increments with a 95% CI and a TOST
equivalence test at ±margin (90% CI inside the margin). With --params: the slope of the increment
on log10(parameters) with a bootstrap CI (per family and pooled over families), plus slope TOST.

  python scripts/compare_models.py --dumps <dir> --models A,B,C [--labels ...] [--params 2,4,12,31]
      [--ref A] [--margin 0.05] [--slope-margin 0.03] [--families ...] [--n-boot 5000] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def bh(pvals):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k); q[i] = prev
    return q


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if len(a) > 1 else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="dir with per_stim_rsa.py JSONs")
    ap.add_argument("--models", required=True, help="comma list of model tags (as in the dumps)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--params", default=None, help="comma list of parameter counts (B) for the scale slope")
    ap.add_argument("--ref", default=None, help="reference model tag for paired differences / TOST")
    ap.add_argument("--families", default=None)
    ap.add_argument("--scheme", default="card_mean")
    ap.add_argument("--margin", type=float, default=0.05, help="TOST equivalence margin on increment differences")
    ap.add_argument("--slope-margin", type=float, default=0.03, help="TOST margin on slope per log10(params)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    models = args.models.split(","); labels = (args.labels or args.models).split(",")
    lab = dict(zip(models, labels))
    params = [float(x) for x in args.params.split(",")] if args.params else None
    D = {}
    for f in glob.glob(os.path.join(args.dumps, "*.json")):
        d = json.load(open(f))
        if d["model"] in models and d["scheme"] == args.scheme:
            D[(d["model"], d["family"])] = d
    fams = args.families.split(",") if args.families else sorted({f for (_, f) in D})
    rng = np.random.default_rng(args.seed)
    out = {"per_model": [], "pairs": [], "slope": []}
    boot_pool = {m: [] for m in models}                      # per family boot increments, pooled later

    print(f"=== COMPARE [{args.scheme}] models={','.join(labels)} families={','.join(fams)} B={args.n_boot} ===")
    for fam in fams:
        ms = [m for m in models if (m, fam) in D]
        if not ms:
            continue
        rk = sorted(set.intersection(*[{k for k, v in D[(m, fam)]["real"].items() if v == v} for m in ms]))
        tk = sorted(set.intersection(*[{k for k, v in D[(m, fam)]["twin"].items() if v == v} for m in ms]))
        if not rk or not tk:
            print(f"{fam}: no common stimuli"); continue
        R = np.array([[D[(m, fam)]["real"][k] for k in rk] for m in ms])      # [M, nr]
        T = np.array([[D[(m, fam)]["twin"][k] for k in tk] for m in ms])      # [M, nt]
        inc = R.mean(1) - T.mean(1)
        B = np.empty((args.n_boot, len(ms)))
        for b in range(args.n_boot):
            ir = rng.integers(0, R.shape[1], R.shape[1]); it = rng.integers(0, T.shape[1], T.shape[1])
            B[b] = R[:, ir].mean(1) - T[:, it].mean(1)
        for j, m in enumerate(ms):
            p = (1 + int((B[:, j] <= 0).sum())) / (1 + args.n_boot)
            lo, hi = ci(B[:, j])
            out["per_model"].append(dict(model=m, label=lab[m], family=fam, n_real=len(rk), n_twin=len(tk),
                                         increment=float(inc[j]), ci=[lo, hi], p=p))
            boot_pool[m].append(B[:, j])
        if args.ref and args.ref in ms:
            r = ms.index(args.ref)
            for j, m in enumerate(ms):
                if j == r:
                    continue
                diff = B[:, j] - B[:, r]
                lo, hi = ci(diff); l90, h90 = ci(diff, 5, 95)
                out["pairs"].append(dict(family=fam, a=lab[m], b=lab[args.ref], diff=float(inc[j] - inc[r]),
                                         ci=[lo, hi], ci90=[l90, h90],
                                         equivalent=bool(l90 > -args.margin and h90 < args.margin),
                                         differs=bool(lo > 0 or hi < 0)))
        if params:
            x = np.log10([params[models.index(m)] for m in ms])
            if len(ms) >= 3:
                sl = np.polyfit(x, inc, 1)[0]; bs = np.array([np.polyfit(x, B[b], 1)[0] for b in range(args.n_boot)])
                lo, hi = ci(bs); l90, h90 = ci(bs, 5, 95)
                out["slope"].append(dict(family=fam, slope=float(sl), ci=[lo, hi], ci90=[l90, h90],
                                         flat_equivalent=bool(l90 > -args.slope_margin and h90 < args.slope_margin)))

    # BH-FDR over all per-model tests
    if out["per_model"]:
        q = bh([r["p"] for r in out["per_model"]])
        for r, qq in zip(out["per_model"], q):
            r["q_bh"] = float(qq)
    # pooled-over-families increment per model and pooled slope (families weighted equally)
    pooled = {}
    for m in models:
        if boot_pool[m] and len(boot_pool[m]) == len(fams):
            pb = np.mean(np.stack(boot_pool[m], 1), 1)
            pt = np.mean([r["increment"] for r in out["per_model"] if r["model"] == m])
            pooled[m] = pb
            lo, hi = ci(pb)
            out.setdefault("pooled", []).append(dict(model=m, label=lab[m], increment=float(pt), ci=[lo, hi]))
    if params and len(pooled) >= 3:
        ms = [m for m in models if m in pooled]
        x = np.log10([params[models.index(m)] for m in ms]); P = np.stack([pooled[m] for m in ms], 1)
        pt = np.array([next(r["increment"] for r in out["pooled"] if r["model"] == m) for m in ms])
        bs = np.array([np.polyfit(x, P[b], 1)[0] for b in range(P.shape[0])])
        lo, hi = ci(bs); l90, h90 = ci(bs, 5, 95)
        out["slope_pooled"] = dict(slope=float(np.polyfit(x, pt, 1)[0]), ci=[lo, hi], ci90=[l90, h90],
                                   flat_equivalent=bool(l90 > -args.slope_margin and h90 < args.slope_margin))

    for r in out["per_model"]:
        print("INC   %-10s %-14s incr=%+.3f [%+.3f,%+.3f] p=%.4f q=%.4f n=%d/%d %s" % (
            r["family"], r["label"], r["increment"], r["ci"][0], r["ci"][1], r["p"], r["q_bh"],
            r["n_real"], r["n_twin"], "SIG" if r["q_bh"] < 0.05 and r["ci"][0] > 0 else "ns"))
    for r in out.get("pooled", []):
        print("POOL  %-14s incr=%+.3f [%+.3f,%+.3f]" % (r["label"], r["increment"], r["ci"][0], r["ci"][1]))
    for r in out["pairs"]:
        print("PAIR  %-10s %s − %s = %+.3f [%+.3f,%+.3f] 90%%[%+.3f,%+.3f] %s%s" % (
            r["family"], r["a"], r["b"], r["diff"], r["ci"][0], r["ci"][1], r["ci90"][0], r["ci90"][1],
            "EQUIV(±%.2f) " % args.margin if r["equivalent"] else "", "DIFFERS" if r["differs"] else ""))
    for r in out["slope"]:
        print("SLOPE %-10s %+.4f/log10 [%+.4f,%+.4f] 90%%[%+.4f,%+.4f] %s" % (
            r["family"], r["slope"], r["ci"][0], r["ci"][1], r["ci90"][0], r["ci90"][1],
            "FLAT-EQUIV(±%.2f)" % args.slope_margin if r["flat_equivalent"] else ""))
    if "slope_pooled" in out:
        r = out["slope_pooled"]
        print("SLOPE pooled     %+.4f/log10 [%+.4f,%+.4f] 90%%[%+.4f,%+.4f] %s" % (
            r["slope"], r["ci"][0], r["ci"][1], r["ci90"][0], r["ci90"][1],
            "FLAT-EQUIV(±%.2f)" % args.slope_margin if r["flat_equivalent"] else ""))
    print("=== END COMPARE ===")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
