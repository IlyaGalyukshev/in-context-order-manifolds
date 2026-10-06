#!/usr/bin/env python
"""Write every number the paper cites as a LaTeX macro, read from the result files under --figdata.

The paper never types a result by hand: sections use \\N<name> macros defined here, so a number in
the text is always the number in the result file. Values from the closed mlc contour are read from
v11/mlc_31b.json (transcribed from the job logs, see its _source field).

  python scripts/make_numbers.py --figdata results_figdata --out paper/numbers.sty [--dump]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

DIG = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
       "7": "seven", "8": "eight", "9": "nine"}


def mname(*parts):
    s = "".join(str(p)[:1].upper() + str(p)[1:] for p in parts)
    s = "".join(DIG.get(c, c) for c in s)
    return "N" + re.sub(r"[^A-Za-z]", "", s)


def f3(x, sign=False):
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("-", "$-$")


def f2(x, sign=False):
    s = f"{x:+.2f}" if sign else f"{x:.2f}"
    return s.replace("-", "$-$")


def ci(c, d=3):
    f = f3 if d == 3 else f2
    return f"[{f(c[0])}, {f(c[1])}]"


def pv(p):
    return "0.0002" if p < 0.0005 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def load(figdata, rel):
    p = os.path.join(figdata, rel)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figdata", default="results_figdata")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    F = a.figdata
    M = {}

    def put(name, val):
        M[name] = val

    # ---- input-layer diagnostic (Gemma-4-31B depth profile) ----
    prof = load(F, "v11/profile_31B.json") or []
    card = [r for r in prof if r["scheme"] == "card_mean"]
    ro = [r for r in prof if r["scheme"] == "readout"]
    if card:
        best = max(card, key=lambda r: r["increment"][0])
        put(mname("Card", "LzeroMax"), f3(best["increment"][0], True))
        put(mname("Card", "LzeroMaxFam"), {"s0_quomp": "quomp", "s0_zib": "zib", "s1_size": "size"}.get(best["family"], best["family"]))
        put(mname("Card", "LzeroMaxCI"), ci([best["ci_lo"][0], best["ci_hi"][0]]))
    if ro:
        fr = [int(np.argmax(r["increment"])) / (r["L"] - 1) for r in ro if max(r["increment"]) > 0.05]
        put(mname("Ro", "PeakDepthLo"), f"{int(round(100 * min(fr)))}")
        put(mname("Ro", "PeakDepthHi"), f"{int(round(100 * max(fr)))}")

    # ---- pooling artefact in a model at chance (Qwen3-0.6B, same probe for both loci) ----
    q6 = {}
    for f in glob.glob(os.path.join(F, "v11", "qwen06", "rsa_*_N12.json")):
        r = json.load(open(f)); r = r[0] if isinstance(r, list) else r
        q6[os.path.basename(f)] = r
    if q6:
        cm = [r for k, r in q6.items() if "card_mean" in k]; rr = [r for k, r in q6.items() if "readout" in k]
        put(mname("QsixCardSig"), str(sum(bool(r.get("sig_vs_twin")) for r in cm))); put(mname("QsixCardN"), str(len(cm)))
        put(mname("QsixRoSig"), str(sum(bool(r.get("sig_vs_twin")) for r in rr)))
        sc = [r["increment"] for r in cm if r.get("sig_vs_twin")]
        if sc:
            put(mname("QsixCardLo"), f3(min(sc), True)); put(mname("QsixCardHi"), f3(max(sc), True))

    # ---- no-integration baseline: relative real->twin drop, baseline vs models ----
    gb = [load(F, "v11/graph_baseline_gemma.json"), load(F, "v11/graph_baseline_qwen.json")]
    if gb[0]:
        for r in gb[0]["rows"]:
            if r["family"] == "pooled":
                put(mname("GbDrop", "hop", r["hop"]), f2(r["rel_drop"])); put(mname("GbIncr", "hop", r["hop"]), f3(r["increment"], True))
                put(mname("GbReal", "hop", r["hop"]), f3(r["real"])); put(mname("GbTwin", "hop", r["hop"]), f3(r["twin"]))
        lab = {"google_gemma-4-E2B-it": "EtwoB", "google_gemma-4-E4B-it": "EfourB", "google_gemma-4-12B-it": "Gtwelve",
               "Qwen_Qwen3-4B": "Qfour", "Qwen_Qwen3-8B": "Qeight"}
        above = 0
        for d in gb:
            for m in (d or {}).get("models", []):
                if m["model"] in lab:
                    put(mname("GbModel", lab[m["model"]]), f2(m["rel_drop"])); put(mname("GbModelCI", lab[m["model"]]), ci(m["ci"], 2))
                ab = {"google_gemma-4-E2B-it": "EtwoB", "google_gemma-4-E4B-it": "EfourB", "google_gemma-4-12B-it": "Gtwelve",
                      "Qwen_Qwen3-0.6B": "Qzerosix", "Qwen_Qwen3-1.7B": "Qonesev", "Qwen_Qwen3-4B": "Qfour", "Qwen_Qwen3-8B": "Qeight"}
                if m["model"] in ab:
                    put(mname("AbsReal", ab[m["model"]]), f3(m["real"])); put(mname("AbsTwin", ab[m["model"]]), f3(m["twin"]))
                if m["model"].startswith("google") and m["ci"][0] > next(r["rel_drop"] for r in gb[0]["rows"] if r["family"] == "pooled" and r["hop"] == "1"):
                    above += 1
        put(mname("Gb", "GemmaAbove"), str(above))

    # ---- thin axis (contrastive vs plain PCA rank decoding at the roster token) ----
    for f in sorted(glob.glob(os.path.join(F, "cpu_probes_20260824", "cpca_*_readout.json"))):
        r = json.load(open(f))[0]
        tag = {"qwen3-4b": "Qfour", "olmo3-7b-inst": "Olmo", "gemma-4-12b-it": "Gtwelve"}[r["model"]]
        put(mname("Cpca", tag), f2(r["cpca_decode"])); put(mname("Pca", tag), f2(r["pca_decode"]))
        put(mname("CpcaDiffCI", tag), ci(r["diff_ci"], 2))

    # ---- ladders (Gemma-4, Qwen3), layer-0 control ----
    G = load(F, "v11/cmp_ladder_ro.json"); G0 = load(F, "v11/cmp_ladder_ro_L0.json"); Q = load(F, "v11/cmp_qwen_ro.json")
    for d in (G, Q):
        for r in d["pooled"]:
            put(mname("Pooled", r["label"]), f3(r["increment"], r["increment"] < 0))
            put(mname("PooledCI", r["label"]), ci(r["ci"]))
    sig = lambda rows: sum(1 for r in rows if r["q_bh"] < 0.05 and r["ci"][0] > 0)
    put(mname("Gemma", "SigCells"), str(sig(G["per_model"]))); put(mname("Gemma", "Cells"), str(len(G["per_model"])))
    put(mname("Qwen", "SigCells"), str(sig(Q["per_model"])))
    put(mname("Qwen", "Slope"), f3(Q["slope_pooled"]["slope"], True)); put(mname("Qwen", "SlopeCI"), ci(Q["slope_pooled"]["ci"]))
    if G0:
        put(mname("Gemma", "LzeroMaxAbs"), f3(max(abs(r["increment"]) for r in G0["pooled"])))

    # ---- behaviour ----
    C = load(F, "v11/collected.json") or {"gate": {}, "steer": {}}
    mlc = load(F, "v11/mlc_31b.json") or {}
    gate = {k: v.get("pairwise", {}).get("acc") for k, v in C["gate"].items()}
    rec = {k: v.get("reconstruction", {}).get("acc") for k, v in C["gate"].items()}
    if mlc.get("gate"):
        gate["31B"] = mlc["gate"]["pairwise"]; rec["31B"] = mlc["gate"]["reconstruction"]
    for k, v in gate.items():
        if v is not None:
            put(mname("Pair", k), f2(v))
    for k in ("E2B", "E4B"):
        if rec.get(k) is not None:
            put(mname("Recon", k), f2(rec[k]))

    # ---- stated vs inferred ----
    for tag, rel in (("Gemma", "v11/sub_ladder.json"), ("Qwen", "v11/sub_qwen.json")):
        s = load(F, rel)
        if s:
            p = s["pooled"]
            put(mname("Sub", tag), f3(p["diff"], True)); put(mname("SubCI", tag), ci(p["ci"]))
            put(mname("SubP", tag), pv(p["p"])); put(mname("SubPos", tag), f"{p['n_pos']}"); put(mname("SubN", tag), f"{p['n']}")

    # ---- months ----
    mo = load(F, "v11/cmp_months.json")
    lab = {"gemma12b": "Gtwelve", "qwen4b": "Qfour", "olmo7b": "Olmo"}
    nsig = 0
    for r in mo["pooled"]:
        put(mname("Month", lab[r["label"]]), f3(r["increment"], True)); put(mname("MonthCI", lab[r["label"]]), ci(r["ci"]))
        nsig += r["ci"][0] > 0
    put(mname("Month", "SigBoot"), str(nsig)); put(mname("Month", "NBoot"), str(len(mo["pooled"])))
    for r in mlc.get("months", []):
        fam = r["family"].split("_")[1]
        put(mname("MonthThirtyone", fam), f3(r["increment"], True))
        put(mname("CalReal", fam), f3(r["external_real"])); put(mname("CalTwin", fam), f3(r["external_twin"]))
        put(mname("CalLzero", fam), f3(r["external_layer0"]))

    # ---- question read-out ----
    ql = {"Gthirtyone": mlc.get("query_local")}
    for tag, f in (("Gtwelve", "gemma-4-12b"), ("Olmo", "olmo3-7b"), ("Qfour", "qwen3-4b")):
        ql[tag] = load(F, f"v11/ql/{f}_summary.json")
    nql = 0
    for tag, q in ql.items():
        if not q:
            continue
        put(mname("QlOrd", tag), f3(q["order_real"]["auc"])); put(mname("QlNon", tag), f3(q["nonorder_real"]["auc"]))
        put(mname("QlTwin", tag), f3(q["order_twin"]["auc"]))
        put(mname("QlDiff", tag), f3(q["order_minus_nonorder"]["diff"], True)); put(mname("QlDiffCI", tag), ci(q["order_minus_nonorder"]["ci"]))
        nql += q["order_minus_nonorder"]["ci"][0] > 0
    put(mname("Ql", "Sig"), str(nql)); put(mname("Ql", "N"), str(sum(1 for q in ql.values() if q)))

    # ---- transplant (31B) ----
    for r in mlc.get("transplant", []):
        if r["layer"] != 24:
            continue
        fam = r["family"].split("_")[1]
        put(mname("TpB", fam), f2(r["towardB"])); put(mname("TpC", fam), f2(r["towardC"]))
        put(mname("TpD", fam), f2(r["delta"], True)); put(mname("TpDCI", fam), ci(r["ci"], 2))
        put(mname("TpTwinD", fam), f2(r["delta_real_minus_twin"], True)); put(mname("TpTwinDCI", fam), ci(r["ci_real_minus_twin"], 2))
        put(mname("TpN", fam), str(r["n"]))

    # ---- steering ----
    st = mlc.get("steer", {})
    for fam in ("s0_zib", "s1_size"):
        if fam in st:
            put(mname("SteerThirtyone", fam.split("_")[1]), f3(st[fam]["along_slope"], True))
            put(mname("SteerPThirtyone", fam.split("_")[1]), f"{st[fam]['p']:.2f}")
    if "s0_zib" in st and "answered_along" in st["s0_zib"]:
        put(mname("Dose", "Lo"), f"{st['s0_zib']['answered_along'][0]:.1f}"); put(mname("Dose", "Hi"), f"{st['s0_zib']['answered_along'][-1]:.1f}")
    for fam in ("s0_zib", "s1_size"):
        if fam in st and st[fam].get("offaxis_sd"):
            z = (st[fam]["along_slope"] - st[fam]["offaxis_mean"]) / st[fam]["offaxis_sd"]
            put(mname("SteerZThirtyone", fam.split("_")[1]), f"{z:.0f}")
    for tag, s_ in C["steer"].items():
        for fam, r in s_.items():
            if r["null_sd"] > 0:
                put(mname("SteerZ", tag.replace("_", ""), fam.split("_")[1]), f"{(r['along_slope'] - r['null_mean']) / r['null_sd']:.1f}".replace("-", "$-$"))
    put(mname("Steer", "NOff"), str(st.get("n_offaxis", 49))); put(mname("Steer", "PFloor"), f"{1 / (st.get('n_offaxis', 49) + 1):.2f}")
    for tag, s in C["steer"].items():
        for fam, r in s.items():
            t = tag.replace("_", "")
            put(mname("Steer", t, fam.split("_")[1]), f3(r["along_slope"], True))
            put(mname("SteerP", t, fam.split("_")[1]), f"{r['p']:.2f}")
            put(mname("SteerL", t), str(r["layer"]))

    # ---- coupling (robust model) ----
    cm = {"Gthirtyone": mlc.get("coupling", [])}
    for tag, m in (("Gtwelve", "google_gemma-4-12B-it"), ("Olmo", "allenai_Olmo-3-7B-Instruct"), ("Qeight", "Qwen_Qwen3-8B"),
                   ("Qfour", "Qwen_Qwen3-4B"), ("Qonesev", "Qwen_Qwen3-1.7B"), ("Qzerosix", "Qwen_Qwen3-0.6B")):
        cm[tag] = [r for f in sorted(glob.glob(os.path.join(F, "v11", "coupling", m, "*.json"))) for r in json.load(open(f))]
    qmax = 0
    for tag, rows in cm.items():
        rob = [r for r in rows if "beta_robust" in r]
        sigr = [r for r in rob if r["beta_robust_ci"][0] > 0]
        put(mname("CoupRob", tag), str(len(sigr))); put(mname("CoupPlain", tag), str(sum(r["beta_ci"][0] > 0 for r in rows)))
        put(mname("CoupFam", tag), str(len(rows)))
        if sigr:
            put(mname("CoupRobLo", tag), f2(min(r["beta_robust"] for r in sigr))); put(mname("CoupRobHi", tag), f2(max(r["beta_robust"] for r in sigr)))
        if tag.startswith("Q"):
            qmax = max(qmax, len(sigr))
    put(mname("Coup", "QwenMax"), str(qmax))

    # ---- confound audit ----
    t = open(os.path.join(F, "v11", "audit_core.txt")).read() if os.path.exists(os.path.join(F, "v11", "audit_core.txt")) else ""
    m = re.search(r"PAIRWISE ANSWER BIAS \(n=(\d+);.*?first-named-in-question ([\d.]+) \| gold==mentioned-earlier-in-text ([\d.]+)", t)
    for feat, nm in (("mention_count", "Mention"), ("subj_frac", "Subj"), ("mean_pos", "Pos")):
        mm = re.search(rf"{feat}\s+([\d.]+)", t)
        if mm:
            put(mname("Audit", nm), f"{float(mm.group(1)):.2f}")
    if m:
        put(mname("Audit", "N"), f"{int(m.group(1)):,}".replace(",", "{,}"))
        put(mname("Audit", "First"), f"{100 * float(m.group(2)):.1f}"); put(mname("Audit", "Earlier"), f"{100 * float(m.group(3)):.1f}")

    lines = ["% AUTO-GENERATED by scripts/make_numbers.py from results_figdata - do not edit by hand.",
             "\\ProvidesPackage{numbers}"]
    for k in sorted(M):
        lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
    if a.dump:
        print("\n".join(lines))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {len(M)} macros -> {a.out}")


if __name__ == "__main__":
    main()
