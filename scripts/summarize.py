#!/usr/bin/env python
"""Print a compact RESULT SUMMARY (numbers only) for a model's OUT dir.

The mlc jobs run in a closed contour with ~10 GB and no easy way to drag GB of
results out — so instead of downloading files, each job prints its findings here
and we read them from `mlc job logs`. Reads the small JSON result files +
e9b_patch parquet; steering slopes are already printed inline by steer_rank.py.

  python scripts/summarize.py <OUT_DIR> [questions.jsonl]
With questions.jsonl, also prints GATE_LP: pairwise accuracy from the forced-choice log-prob margin
(graded; informative where generation is at floor in small models).
"""
from __future__ import annotations

import glob
import json
import os
import sys


def jrows(f):
    try:
        d = json.load(open(f))
        return d if isinstance(d, list) else [d]
    except Exception:
        return []


def main() -> None:
    D = sys.argv[1].rstrip("/")
    name = os.path.basename(D)
    out = [f"=== RESULT SUMMARY [{name}] ==="]

    # RSA manifold by family x locus
    for f in sorted(glob.glob(D + "/rsa_*_N12.json")):
        for r in jrows(f):
            inc = r.get("increment")
            if isinstance(inc, (int, float)):
                out.append("RSA   %-9s %-9s inc=%+.3f (real=%.3f twin=%.3f) L%s %s"
                           % (r.get("family", ""), r.get("scheme", ""), inc,
                              r.get("rsa_real", float("nan")), r.get("rsa_twin", float("nan")),
                              r.get("peak_layer", "?"), "SIG" if r.get("sig_vs_twin") else "ns"))

    # one-number-per-file families (N-curve, difficulty, condition, declared, assembly, dynamics, strata)
    for tag in ("ncurve_card_mean", "ncurve_readout", "diff", "cond", "e2", "e7q", "e8", "r8", "decay"):
        for f in sorted(glob.glob(D + "/" + tag + "*.json")):
            for r in jrows(f):
                inc = r.get("increment")
                if isinstance(inc, (int, float)):
                    out.append("%-24s inc=%+.3f %s"
                               % (os.path.basename(f)[:-5], inc, "SIG" if r.get("sig_vs_twin") else "ns"))

    # form (structure litmus): winner fractions
    for f in sorted(glob.glob(D + "/form_*.json")):
        for r in jrows(f):
            wf = r.get("winner_frac")
            if wf:
                out.append("FORM  %-16s %s" % (os.path.basename(f)[5:-5],
                           " ".join("%s=%.2f" % (k, v) for k, v in wf.items())))

    # P0.2 coupling (geometry -> correctness)
    for f in sorted(glob.glob(D + "/coupling/*.json")):
        for r in jrows(f):
            b = r.get("beta_margin")
            if isinstance(b, (int, float)):
                ci = r.get("beta_ci", ["", ""])
                out.append("COUP  %-9s beta=%+.3f [%s,%s] %s (corr=%.3f cv_rho=%.3f)"
                           % (r.get("family", ""), b, ci[0], ci[1],
                              "SIG" if r.get("sig") else "ns",
                              r.get("corr_margin_correct", float("nan")), r.get("cv_rho", float("nan"))))

    # E9b causal entity-patch: directional shift toward donor rank
    for f in sorted(glob.glob(D + "/e9b_patch_*.parquet")):
        try:
            import numpy as np
            import pandas as pd
            d = pd.read_parquet(f)
            piv = d.pivot_table(index=["stim", "true_rank_A", "true_rank_B", "true_rank_C"],
                                columns="cond", values="answered", aggfunc="first").reset_index()

            def toward(a0, a1, rr, ra):
                if pd.isna(a0) or pd.isna(a1) or rr == ra:
                    return np.nan
                return 1.0 if np.sign(a1 - a0) == np.sign(rr - ra) else 0.0
            tB = pd.Series([toward(r.baseline, getattr(r, "patchB", np.nan), r.true_rank_B, r.true_rank_A)
                            for r in piv.itertuples()]).dropna()
            tC = pd.Series([toward(r.baseline, getattr(r, "patchC", np.nan), r.true_rank_C, r.true_rank_A)
                            for r in piv.itertuples()]).dropna()
            sch = d["scheme"].iloc[0] if "scheme" in d else os.path.basename(f)
            out.append("E9b   %-9s towardB=%.3f towardC=%.3f (n=%d)"
                       % (sch, tB.mean() if len(tB) else float("nan"),
                          tC.mean() if len(tC) else float("nan"), len(piv)))
        except Exception as e:
            out.append("E9b   %s (summary skipped: %s)" % (os.path.basename(f), str(e)[:50]))

    # behavioural gate (pairwise / reconstruction are the integration metrics that license geometry)
    for f in glob.glob(D + "/battery/*.jsonl"):
        acc = {}
        for l in open(f):
            try:
                j = json.loads(l)
            except Exception:
                continue
            s = j.get("score"); q = j.get("q_family")
            if s is not None and s == s:
                a = acc.setdefault(q, [0.0, 0]); a[0] += float(s); a[1] += 1
        if acc:
            out.append("GATE  " + "  ".join("%s=%.3f" % (k, v[0] / v[1])
                       for k, v in sorted(acc.items()) if v[1]))
        break

    if len(sys.argv) > 2:
        Q = {}
        for l in open(sys.argv[2]):
            try:
                q = json.loads(l)
            except Exception:
                continue
            if q.get("family") == "pairwise" and q.get("target_entities"):
                Q[q["qid"]] = q
        for f in glob.glob(D + "/battery/*.jsonl"):
            ok = n = 0
            for l in open(f):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r.get("q_family") != "pairwise" or r.get("logit_margin") is None or r.get("qid") not in Q:
                    continue
                q = Q[r["qid"]]
                ok += (r["logit_margin"] > 0) == (q["answer_key"] == q["target_entities"][0]); n += 1
            if n:
                out.append("GATE_LP pairwise log-prob accuracy=%.3f (n=%d)" % (ok / n, n))
            break
    out.append("=== END SUMMARY ===")
    print("\n".join(out), flush=True)


if __name__ == "__main__":
    main()
