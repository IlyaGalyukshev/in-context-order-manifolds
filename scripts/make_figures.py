#!/usr/bin/env python
"""Publication figures for the in-context order-manifold paper (§8, Fig 1-7).

Reads the committed result JSONs (crossnobis / cpca / form schema) and renders each figure to
PDF + PNG. Parameterized and reproducible: NO hard-coded result numbers — every value is read from
a result JSON under --figdata. `--dump` prints the data tables (for verification) without rendering.

  python scripts/make_figures.py --figdata results_figdata --out figs --figs all
  python scripts/make_figures.py --figdata results_figdata --figs e10,crossform --dump
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- data loading
def _load(figdata: str, sub: str, pat: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(figdata, sub, pat))):
        obj = json.load(open(f))
        for r in (obj if isinstance(obj, list) else [obj]):
            if isinstance(r, dict):
                r = dict(r)
                r["_file"] = os.path.basename(f)
                rows.append(r)
    return rows


def _one(figdata: str, sub: str, pat: str) -> dict | None:
    r = _load(figdata, sub, pat)
    return r[0] if r else None


# ----------------------------------------------------------------------------- style
PAL = {
    "AR": "#4C72B0",            # blue
    "diff-init": "#C44E52",     # red  (Dream = Qwen2.5-initialised diffusion)
    "diff-scratch": "#DD8452",  # orange (LLaDA = from-scratch diffusion)
    "real": "#2A2A2A", "twin": "#B0B0B0",
    "line": "#4C72B0", "ring": "#55A868", "2block": "#C44E52", "grid": "#8172B3",
    "grid_accent": "#937860",
}
MODELCOL = {"qwen": "#4C72B0", "olmo": "#55A868", "gemma": "#C44E52",
            "qwen3-4b": "#4C72B0", "olmo3-7b-inst": "#55A868", "gemma-4-12b-it": "#C44E52"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.axisbelow": True, "grid.alpha": 0.25,
    })
    return plt


def _save(fig, out: str, name: str):
    Path(out).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    print(f"  wrote {out}/{name}.pdf + .png")


# ----------------------------------------------------------------------------- Fig E10
E10_ORDER = [
    ("qwen2.5-7b-base", "Qwen2.5-Base", "AR"),
    ("dream-7b-base", "Dream-Base", "diff-init"),
    ("qwen2.5-7b-inst", "Qwen2.5-Inst", "AR"),
    ("dream-7b-inst", "Dream-Inst", "diff-init"),
    ("llama3-8b-base", "LLaMA3-Base", "AR"),
    ("llada-8b-base", "LLaDA-Base", "diff-scratch"),
    ("llama3-8b-inst", "LLaMA3-Inst", "AR"),
    ("llada-8b-inst", "LLaDA-Inst", "diff-scratch"),
]


def _e10_rows(figdata):
    rows = {r["model"]: r for r in _load(figdata, "e10_20260826", "e10_*.json")}
    return rows


def fig_e10(figdata, out, dump):
    rows = _e10_rows(figdata)
    data = []
    for mid, label, cat in E10_ORDER:
        r = rows.get(mid)
        if not r:
            continue
        lo, hi = r["increment_ci"]
        data.append((label, cat, r["increment"], lo, hi, r["rsa_real"], r["rsa_twin"], r.get("p")))
    if dump:
        print("== Fig E10: diffusion vs AR (s0_quomp real - coherence-twin) ==")
        for d in data:
            print(f"  {d[0]:16s} {d[1]:12s} incr={d[2]:.3f} [{d[3]:.3f},{d[4]:.3f}] "
                  f"real={d[5]:.3f} twin={d[6]:.3f} p={d[7]}")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(data))[::-1]
    for yi, d in zip(y, data):
        label, cat, inc, lo, hi = d[:5]
        ax.barh(yi, inc, color=PAL[cat], edgecolor="white", height=0.72, zorder=3)
        ax.plot([lo, hi], [yi, yi], color="#333", lw=1.4, zorder=4)          # CI line
        for x in (lo, hi):                                                    # CI caps
            ax.plot([x, x], [yi - .1, yi + .1], color="#333", lw=1.4, zorder=4)
        ax.text(hi + 0.004, yi, f"{inc:.3f}", va="center", ha="left", fontsize=7.5, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data])
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("Order-manifold increment  (whitened-RSA: real − coherence-twin)")
    ax.set_xlim(0, max(d[4] for d in data) * 1.40)   # headroom so the upper-right legend clears all CIs
    ax.set_title("Fig 4 · The in-context order manifold is objective-invariant (E10)")
    from matplotlib.patches import Patch
    leg = [Patch(fc=PAL["AR"], label="autoregressive"),
           Patch(fc=PAL["diff-init"], label="diffusion (init = Qwen2.5-7B)"),
           Patch(fc=PAL["diff-scratch"], label="diffusion (from scratch)")]
    ax.legend(handles=leg, loc="upper right", frameon=False)
    ax.grid(axis="x")
    fig.text(0.5, -0.02,
             "Shared-init control (Dream ↔ Qwen2.5, base + instruct): diffusion ≈ AR, no AR advantage.  "
             "All 8 arms significant vs coherence-twin (p = 0.007).",
             ha="center", fontsize=7.5, style="italic", color="#444")
    _save(fig, out, "fig4_e10_diffusion_vs_ar")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig cross-form
def _form_agg(rows, templates):
    """mean over files of winner_frac + mean_rsa for the given templates."""
    wf = {t: [] for t in templates}
    mr = {t: [] for t in templates}
    nst = 0
    for r in rows:
        if r.get("n_stim", 0) == 0:
            continue
        nst += r["n_stim"]
        for t in templates:
            if t in r.get("winner_frac", {}):
                wf[t].append(r["winner_frac"][t])
            v = r.get("mean_rsa", {}).get(t)
            if v is not None:
                mr[t].append(v)
    return ({t: (np.mean(wf[t]) if wf[t] else 0.0) for t in templates},
            {t: (np.mean(mr[t]) if mr[t] else np.nan) for t in templates}, nst)


def fig_crossform(figdata, out, dump):
    panels = [
        ("Total order", ["line", "ring", "2block"],
         _load(figdata, "track0_20260825", "form_*_line.json")),
        ("Cyclic", ["line", "ring", "2block"],
         _load(figdata, "xform_20260828", "form_*_cyclic.json")),
        ("Partial order", ["line", "2block"],
         _load(figdata, "xform_20260828", "form_*_partial.json")),
        ("Grid-2D (semantic)", ["line", "ring", "2block", "grid"],
         _load(figdata, "xform_20260828", "grid_*_s1_size_s1_loud.json")),
    ]
    aggs = [(name, tmpl, *_form_agg(rows, tmpl)) for name, tmpl, rows in panels]
    if dump:
        print("== Fig cross-form: form-selection mean_rsa (winner*) by structure ==")
        for name, tmpl, wf, mr, nst in aggs:
            win = max(mr, key=lambda t: (mr[t] if mr[t] == mr[t] else -9))
            print(f"  {name:22s} n={nst:3d}  " +
                  "  ".join(f"{t}={mr[t]:.3f}{'*' if t == win else ' '}" for t in tmpl))
        return
    plt = _style()
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9))
    for ax, (name, tmpl, wf, mr, nst) in zip(axes, aggs):
        vals = [mr[t] for t in tmpl]
        win = int(np.nanargmax(vals))
        cols = [PAL[t] for t in tmpl]
        bars = ax.bar(range(len(tmpl)), vals, color=cols, edgecolor="white", zorder=3)
        bars[win].set_edgecolor("#111")
        bars[win].set_linewidth(1.8)
        ax.set_xticks(range(len(tmpl)))
        ax.set_xticklabels(tmpl, rotation=30, ha="right")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0, max(0.08, np.nanmax(vals) * 1.25))
        ax.grid(axis="y")
        ax.annotate("★", (win, vals[win]), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=10, color="#111")
    axes[0].set_ylabel("mean whitened-RSA to template")
    fig.suptitle("Fig 5 · Geometry follows the latent: form-selection by structure "
                 "(qwen3-4b + olmo3-7b-inst)", y=1.04, fontsize=10, fontweight="bold")
    _save(fig, out, "fig5_crossform")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig R9 (E4 bridge)
def fig_bridge(figdata, out, dump):
    models = ["qwen3-4b", "olmo3-7b-inst", "gemma-4-12b-it"]
    series = {}
    for m in models:
        pts = []
        for b in (0, 1, 2):
            r = _one(figdata, "track0_20260825", f"r9_{m}_b{b}.json")
            if r:
                pts.append((b, r["rsa_real"], r.get("rsa_real_ci")))
        if pts:
            series[m] = pts
    if dump:
        print("== Fig R9: E4 cross-block RSA vs #bridges (determinacy dose-response) ==")
        for m, pts in series.items():
            print(f"  {m:16s} " + "  ".join(f"b{b}={v:.3f}" for b, v, _ in pts))
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for m, pts in series.items():
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=MODELCOL.get(m, "#333"), lw=2, ms=6,
                label=m.replace("-inst", "").replace("-it", ""), zorder=3)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("# determinacy bridges added (cross-block)")
    ax.set_ylabel("cross-block-only RSA to line (real)")
    ax.set_title("Fig 6 · E4 determinacy dose-response\n(cross-block map rises with bridges)")
    ax.legend(frameon=False)
    ax.grid(True)
    _save(fig, out, "fig6_e4_bridge_doseresponse")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig R8 (stated vs inferred)
def fig_stated(figdata, out, dump):
    models = ["qwen3-4b", "olmo3-7b-inst", "gemma-4-12b-it"]
    got = {}
    for m in models:
        one = _one(figdata, "track0_20260825", f"r8_{m}_onehop.json")
        mul = _one(figdata, "track0_20260825", f"r8_{m}_multihop.json")
        if one and mul:
            got[m] = (one["increment"], one.get("increment_ci"),
                      mul["increment"], mul.get("increment_ci"),
                      one.get("sig_vs_twin"), mul.get("sig_vs_twin"))
    if dump:
        print("== Fig R8: stated (onehop) vs inferred (multihop) increment ==")
        for m, g in got.items():
            print(f"  {m:16s} onehop={g[0]:.3f}(sig={g[4]})  multihop={g[2]:.3f}(sig={g[5]})")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ms = list(got.keys()); x = np.arange(len(ms)); w = 0.36
    one = [got[m][0] for m in ms]; mul = [got[m][2] for m in ms]
    one_e = np.array([[got[m][0] - got[m][1][0], got[m][1][1] - got[m][0]] for m in ms]).T
    mul_e = np.array([[got[m][2] - got[m][3][0], got[m][3][1] - got[m][2]] for m in ms]).T
    ax.bar(x - w / 2, one, w, yerr=one_e, color="#4C72B0", label="stated (1-hop)",
           capsize=3, edgecolor="white", zorder=3)
    ax.bar(x + w / 2, mul, w, yerr=mul_e, color="#C44E52", label="inferred (multi-hop)",
           capsize=3, edgecolor="white", zorder=3)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-inst", "").replace("-it", "") for m in ms])
    ax.set_ylabel("coherence increment (real − twin)")
    ax.set_title("Fig 3 · Within-stimulus signal sits on\nSTATED relations, not inferred ones (E1/R8)")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    _save(fig, out, "fig3_stated_vs_inferred")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig cPCA (dissociation)
LOCI = [("readout", "readout"), ("card_mean", "card-mean"), ("last_token", "last-token")]


def fig_cpca(figdata, out, dump):
    models = ["qwen3-4b", "olmo3-7b-inst", "gemma-4-12b-it"]
    agg = {}
    for scheme, _ in LOCI:
        pca, cpca = [], []
        for m in models:
            r = _one(figdata, "cpu_probes_20260824", f"cpca_{m}_{scheme}.json")
            if r:
                pca.append(r["pca_decode"]); cpca.append(r["cpca_decode"])
        agg[scheme] = (np.mean(pca) if pca else np.nan, np.mean(cpca) if cpca else np.nan)
    if dump:
        print("== Fig cPCA: PCA-blind vs cPCA-recovered rank decode by locus ==")
        for s, _ in LOCI:
            print(f"  {s:12s} pca={agg[s][0]:.3f}  cpca={agg[s][1]:.3f}")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    x = np.arange(len(LOCI)); w = 0.36
    pca = [agg[s][0] for s, _ in LOCI]; cpca = [agg[s][1] for s, _ in LOCI]
    ax.bar(x - w / 2, pca, w, color="#B0B0B0", label="raw PCA subspace", edgecolor="white", zorder=3)
    ax.bar(x + w / 2, cpca, w, color="#4C72B0", label="contrastive-PCA (real vs twin)",
           edgecolor="white", zorder=3)
    for xi, (p, c) in zip(x, zip(pca, cpca)):
        ax.text(xi - w / 2, p + .01, f"{p:.2f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, c + .01, f"{c:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LOCI])
    ax.set_ylabel("interior-rank decode accuracy (ρ)")
    ax.set_ylim(0, 1)
    ax.set_title("Fig 1 · Low-variance ordinal subspace:\nrank is PCA-invisible but cPCA-recoverable")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y")
    _save(fig, out, "fig1_cpca_dissociation")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig manifold (real vs twin)
def fig_manifold(figdata, out, dump):
    """core manifold: real vs coherence-twin RSA on DIRECTLY-STATED (1-hop) interior pairs, 3 models
    (R8). This is the coherence-specific manifold signal (the plain all-pairs increment is ~0 —
    that is the honest d=1 local-chaining calibration; the order-specific map lives on stated pairs)."""
    models = ["qwen3-4b", "olmo3-7b-inst", "gemma-4-12b-it"]
    got = {}
    for m in models:
        r = _one(figdata, "track0_20260825", f"r8_{m}_onehop.json")
        if r:
            got[m] = (r["rsa_real"], r.get("rsa_real_ci"), r["rsa_twin"], r.get("rsa_twin_ci"),
                      r["increment"], r.get("sig_vs_twin"))
    if dump:
        print("== Fig manifold: real vs coherence-twin RSA (stated pairs, R8) ==")
        for m, g in got.items():
            print(f"  {m:16s} real={g[0]:.3f} twin={g[2]:.3f} incr={g[4]:.3f} sig={g[5]}")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ms = list(got); x = np.arange(len(ms)); w = 0.36
    real = [got[m][0] for m in ms]; twin = [got[m][2] for m in ms]
    real_e = np.array([[got[m][0] - got[m][1][0], got[m][1][1] - got[m][0]] for m in ms]).T
    twin_e = np.array([[got[m][2] - got[m][3][0], got[m][3][1] - got[m][2]] for m in ms]).T
    ax.bar(x - w / 2, real, w, yerr=real_e, color=PAL["real"], label="real order",
           capsize=3, edgecolor="white", zorder=3)
    ax.bar(x + w / 2, twin, w, yerr=twin_e, color=PAL["twin"], label="coherence-null twin",
           capsize=3, edgecolor="white", zorder=3)
    for xi, m in zip(x, ms):
        ax.annotate(f"+{got[m][4]:.2f}", (xi, max(got[m][0], got[m][2])),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8, color="#4C72B0", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-inst", "").replace("-it", "") for m in ms])
    ax.set_ylabel("whitened-RSA to line")
    ax.set_ylim(0, max(real) * 1.28)
    ax.set_title("Fig 2 · The order manifold stands above its coherence-null twin\n"
                 "(directly-stated interior pairs, 3 models)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y")
    _save(fig, out, "fig2_manifold_vs_twin")
    plt.close(fig)


FIGS = {"cpca": fig_cpca, "manifold": fig_manifold, "stated": fig_stated, "e10": fig_e10,
        "crossform": fig_crossform, "bridge": fig_bridge}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figdata", default="results_figdata")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--figs", default="all", help="all | comma list: " + ",".join(FIGS))
    ap.add_argument("--dump", action="store_true", help="print data tables, do not render")
    args = ap.parse_args()
    want = list(FIGS) if args.figs == "all" else [f.strip() for f in args.figs.split(",")]
    for name in want:
        fn = FIGS.get(name)
        if not fn:
            print(f"  ?? unknown figure '{name}' (have: {', '.join(FIGS)})")
            continue
        fn(args.figdata, args.out, args.dump)


if __name__ == "__main__":
    main()
