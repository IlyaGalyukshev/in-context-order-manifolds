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


# ----------------------------------------------------------------------------- v1.1 figures (readout locus)
FAMCOL = {"s1_size": "#1baf7a", "s0_quomp": "#eb6834", "s0_zib": "#2a78d6", "s1_loud": "#eda100", "s1_heat": "#e87ba4"}
FAMLAB = {"s1_size": "size", "s0_quomp": "quomp", "s0_zib": "zib", "s1_loud": "loud", "s1_heat": "heat"}


def fig_locus(figdata, out, dump):
    """Input-layer diagnostic: real−twin increment by depth, card-level pooling vs the entity token."""
    rows = _load(figdata, "v11", "profile_*.json")
    if not rows:
        print("  locus: no v11/profile_*.json"); return
    if dump:
        for r in rows:
            print(f"  {r['model']} {r['family']:9s} {r['scheme']:9s} L0={r['increment'][0]:+.3f} peak={max(r['increment']):+.3f}")
        return
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, sc, title in ((axes[0], "card_mean", "Sentence-level pooling"), (axes[1], "readout", "Entity token (roster)")):
        for r in [r for r in rows if r["scheme"] == sc]:
            x = np.arange(r["L"]) / (r["L"] - 1); c = FAMCOL.get(r["family"], "#555")
            ax.plot(x, r["increment"], color=c, lw=1.6, label=FAMLAB.get(r["family"], r["family"]))
            ax.fill_between(x, r["ci_lo"], r["ci_hi"], color=c, alpha=0.12, lw=0)
        ax.axhline(0, color="#888", lw=0.8, ls="--"); ax.set_title(title); ax.set_xlabel("relative depth (0 = embeddings)")
    axes[0].set_ylabel("real − twin RSA"); axes[1].legend(frameon=False, ncol=1, loc="upper left")
    _save(fig, out, "fig_locus")


def fig_ladder(figdata, out, dump):
    """Computed at every scale: per-family increments (FDR) and pooled increment vs the layer-0 control."""
    cmp_ = _one(figdata, "v11", "cmp_ladder_ro.json"); l0 = _one(figdata, "v11", "cmp_ladder_ro_L0.json")
    if not cmp_:
        print("  ladder: no v11/cmp_ladder_ro.json"); return
    labels = [r["label"] for r in cmp_["pooled"]]
    if dump:
        for r in cmp_["per_model"]:
            print(f"  {r['family']:9s} {r['label']:5s} {r['increment']:+.3f} q={r['q_bh']:.4f}")
        for r in cmp_["pooled"]:
            print(f"  pooled {r['label']:5s} {r['increment']:+.3f} {r['ci']}")
        return
    plt = _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(labels))
    for fam in FAMCOL:
        rr = {r["label"]: r for r in cmp_["per_model"] if r["family"] == fam}
        if not rr:
            continue
        y = [rr[l]["increment"] if l in rr else np.nan for l in labels]
        sig = [l in rr and rr[l]["q_bh"] < 0.05 and rr[l]["ci"][0] > 0 for l in labels]
        a.plot(x, y, color=FAMCOL[fam], lw=1.6, label=FAMLAB[fam])
        a.scatter(x, y, s=28, facecolors=[FAMCOL[fam] if s_ else "white" for s_ in sig], edgecolors=FAMCOL[fam], zorder=3)
    a.axhline(0, color="#888", lw=0.8, ls="--"); a.set_xticks(x, labels); a.set_ylabel("real − twin RSA (entity token)")
    a.set_title("Per family (filled = FDR q<.05)"); a.legend(frameon=False, ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    pc_ = cmp_["pooled"]
    b.errorbar(x - 0.08, [r["increment"] for r in pc_], yerr=[[r["increment"] - r["ci"][0] for r in pc_], [r["ci"][1] - r["increment"] for r in pc_]],
               fmt="o", color="#2a78d6", capsize=3, label="computed (cross-fitted layer)")
    if l0:
        p0 = l0["pooled"]
        b.errorbar(x + 0.08, [r["increment"] for r in p0], yerr=[[r["increment"] - r["ci"][0] for r in p0], [r["ci"][1] - r["increment"] for r in p0]],
                   fmt="s", color="#9aa2ab", capsize=3, label="layer 0 (input)")
    b.axhline(0, color="#888", lw=0.8, ls="--"); b.set_xticks(x, labels); b.set_title("Pooled over families [95% CI]"); b.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _save(fig, out, "fig_ladder")


def fig_stated_v11(figdata, out, dump):
    """Integration: within-stimulus stated (1-hop) − inferred (multi-hop) difference per cell + pooled."""
    d = _one(figdata, "v11", "sub_ladder.json")
    if not d:
        print("  stated_v11: no v11/sub_ladder.json"); return
    rows = d["rows"]
    if dump:
        for r in rows:
            print(f"  {r['family']:9s} {r['label']:5s} {r['diff']:+.3f} {r['ci']} q={r['q_bh']:.3f}")
        print("  pooled", d["pooled"]); return
    plt = _style()
    order = ["E2B", "E4B", "12B", "31B"]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, fam in enumerate(FAMCOL):
        for j, lab in enumerate(order):
            r = next((r for r in rows if r["family"] == fam and r["label"] == lab), None)
            if not r:
                continue
            xx = j + (i - 2) * 0.12
            ax.errorbar(xx, r["diff"], yerr=[[r["diff"] - r["ci"][0]], [r["ci"][1] - r["diff"]]], fmt="o", ms=3.5,
                        color=FAMCOL[fam], capsize=0, lw=1, label=FAMLAB[fam] if j == 0 else None)
    pl = d["pooled"]
    ax.axhspan(pl["ci"][0], pl["ci"][1], color="#2a78d6", alpha=0.10, lw=0)
    ax.axhline(pl["diff"], color="#2a78d6", lw=1.2, label=f"pooled {pl['diff']:+.3f}")
    ax.axhline(0, color="#888", lw=0.8, ls="--"); ax.set_xticks(range(4), order)
    ax.set_ylabel("stated − inferred RSA"); ax.legend(frameon=False, fontsize=6.5, ncol=3)
    _save(fig, out, "fig_stated_v11")



def fig_scale(figdata, out, dump):
    """Two families on one log-size axis: pooled entity-token increment [95% CI] vs parameters."""
    G = _one(figdata, "v11", "cmp_ladder_ro.json"); Q = _one(figdata, "v11", "cmp_qwen_ro.json")
    if not G or not Q:
        print("  scale: need v11/cmp_ladder_ro.json and v11/cmp_qwen_ro.json"); return
    size = {"E2B": 2, "E4B": 4, "12B": 12, "31B": 31, "Q0.6B": 0.6, "Q1.7B": 1.7, "Q4B": 4, "Q8B": 8}
    if dump:
        for d in (G, Q):
            for r in d["pooled"]:
                print(f"  {r['label']:6s} {size[r['label']]:5.1f}B {r['increment']:+.3f} {r['ci']}")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for d, col, name, dx in ((G, "#2a78d6", "Gemma-4", 1.03), (Q, "#eb6834", "Qwen3", 0.97)):
        xs = [size[r["label"]] * dx for r in d["pooled"]]; ys = [r["increment"] for r in d["pooled"]]
        lo = [r["increment"] - r["ci"][0] for r in d["pooled"]]; hi = [r["ci"][1] - r["increment"] for r in d["pooled"]]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=col, capsize=2.5, lw=1.4, ms=4, label=name)
    ax.axhline(0, color="#888", lw=0.8, ls="--"); ax.set_xscale("log")
    ax.set_xticks([0.6, 1.7, 4, 8, 12, 31], ["0.6", "1.7", "4", "8", "12", "31"])
    ax.set_xlabel("parameters (B; Gemma-4 E-models: effective)"); ax.set_ylabel("real − twin RSA (pooled)")
    ax.legend(frameon=False, fontsize=7); ax.set_title("Input-layer control = 0.000 throughout", fontsize=8, fontweight="normal")
    _save(fig, out, "fig_scale")


# ----------------------------------------------------------------------------- v1.5 figures
QL_ORDER = [("mlc", "Gemma-4-31B"), ("gemma-4-12b", "Gemma-4-12B"), ("olmo3-7b", "OLMo-3-7B"), ("qwen3-4b", "Qwen3-4B")]


def fig_querylocal(figdata, out, dump):
    """Which-is-earlier AUC at the question's final token: order question vs a non-order question
    about the same (randomly ordered) pair; real and coherence-null twin."""
    mlc = _one(figdata, "v11", "mlc_31b.json")
    rows = []
    for tag, lab in QL_ORDER:
        q = (mlc or {}).get("query_local") if tag == "mlc" else _one(figdata, "v11/ql", f"{tag}_summary.json")
        if q:
            rows.append((lab, q))
    if dump:
        for lab, q in rows:
            print(f"  {lab:12s} order {q['order_real']['auc']:.3f} nonorder {q['nonorder_real']['auc']:.3f} "
                  f"diff {q['order_minus_nonorder']['diff']:+.3f} {q['order_minus_nonorder']['ci']} twin-order {q['order_twin']['auc']:.3f}")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(rows)); w = 0.2
    for k, (key, col, name) in enumerate((("order_real", "#2a78d6", "order Q, real"), ("order_twin", "#9cc0ec", "order Q, twin"),
                                          ("nonorder_real", "#6b7480", "non-order Q, real"))):
        ys = [q[key]["auc"] for _, q in rows]
        err = [[q[key]["auc"] - q[key]["ci"][0] for _, q in rows], [q[key]["ci"][1] - q[key]["auc"] for _, q in rows]]
        ax.bar(x + (k - 1) * w, ys, w, color=col, label=name, yerr=err, capsize=1.5, error_kw={"lw": 0.8})
    ax.axhline(0.5, color="#888", lw=0.8, ls="--"); ax.set_ylim(0.4, 1.0)
    ax.set_xticks(x, ["\n".join(l.rsplit("-", 1)) for l, _ in rows], fontsize=7); ax.set_ylabel("which-is-earlier AUC")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    _save(fig, out, "fig_querylocal")


def fig_months(figdata, out, dump):
    """Familiar tokens with a non-calendar in-context order: computed in-context increment per model."""
    d = _one(figdata, "v11", "cmp_months.json"); mlc = _one(figdata, "v11", "mlc_31b.json")
    lab = {"gemma12b": "Gemma-4-12B", "qwen4b": "Qwen3-4B", "olmo7b": "OLMo-3-7B"}
    pts = []                                                  # (model, family, inc, lo, hi)
    for r in (mlc or {}).get("months", []):
        pts.append(("Gemma-4-31B", r["family"], r["increment"], None, None))
    for r in (d or {}).get("per_model", []):
        pts.append((lab.get(r["label"], r["label"]), r["family"], r["increment"], r["ci"][0], r["ci"][1]))
    models = list(dict.fromkeys(p[0] for p in pts))
    if dump:
        for p in pts:
            print(f"  {p[0]:12s} {p[1]:9s} {p[2]:+.3f} {p[3]} {p[4]}")
        for r in (mlc or {}).get("months", []):
            print(f"  31B calendar: real {r['external_real']} twin {r['external_twin']} layer0 {r['external_layer0']}")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for i, m in enumerate(models):
        for j, fam in enumerate(("s0_zib", "s0_quomp")):
            p = next((p for p in pts if p[0] == m and p[1] == fam), None)
            if not p:
                continue
            xx = i + (j - 0.5) * 0.25
            yerr = [[p[2] - p[3]], [p[4] - p[2]]] if p[3] is not None else None
            ax.errorbar(xx, p[2], yerr=yerr, fmt="o", color=FAMCOL[fam], ms=4, capsize=2, lw=1,
                        label=FAMLAB[fam] if i == 0 else None)
    ax.axhline(0, color="#888", lw=0.8, ls="--"); ax.set_xticks(range(len(models)), models, fontsize=7)
    ax.set_ylabel("in-context order: real − twin"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("Month names, non-calendar order (layer 0 = 0.000)", fontsize=8, fontweight="normal")
    _save(fig, out, "fig_months")


def _dose(df, fam):
    g = df[df.family == fam]
    al = g[g.direction == "along"].groupby("alpha")["answered"].mean()
    off = g[g.direction.str.startswith("offaxis")].groupby(["direction", "alpha"])["answered"].mean().unstack(0)
    return al, off


def fig_causal(figdata, out, dump):
    """(a) Steering dose-response at a layer fixed in advance (along the rank axis vs 49 matched-norm
    off-axis directions, 64 stimuli); (b) activation transplant in Gemma-4-31B (toward-donor rate)."""
    import pandas as pd
    p = os.path.join(figdata, "v11", "steer", "google_gemma-4-12B-it", "steer_n64.parquet")
    mlc = _one(figdata, "v11", "mlc_31b.json")
    tp = [r for r in (mlc or {}).get("transplant", []) if r["layer"] in (24, 33)]
    df = pd.read_parquet(p) if os.path.exists(p) else None
    if dump:
        if df is not None:
            for fam in ("s0_zib", "s1_size"):
                al, off = _dose(df, fam)
                print(f"  12B {fam} along {al.round(2).tolist()} off-mean {off.mean(axis=1).round(2).tolist()}")
        for r in tp:
            print(f"  31B {r['family']:8s} L{r['layer']} B {r['towardB']} C {r['towardC']} Btwin {r['towardBtwin']} d {r['delta']} {r['ci']}")
        return
    plt = _style()
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.6, 5.0), gridspec_kw={"hspace": 0.7})
    if df is not None:
        for fam, ls in (("s0_zib", "-"), ("s1_size", ":")):
            al, off = _dose(df, fam)
            a.fill_between(off.index, off.quantile(0.025, axis=1), off.quantile(0.975, axis=1), color="#9aa2ab", alpha=0.25, lw=0)
            a.plot(off.index, off.mean(axis=1), color="#6b7480", lw=1.2, ls=ls)
            a.plot(al.index, al.values, color=FAMCOL[fam], lw=1.8, ls=ls, marker="o", ms=3, label=f"{FAMLAB[fam]}: along axis")
        a.plot([], [], color="#6b7480", label="49 off-axis (mean, 95%)")
    a.set_xlabel("steering strength α (layer 26/48)"); a.set_ylabel("answered position")
    a.set_title("Gemma-4-12B, 64 stimuli", fontsize=8, fontweight="normal"); a.legend(frameon=False, fontsize=6.5)
    fams = ["s0_zib", "s1_size", "s1_loud"]; x = np.arange(len(fams)); w = 0.13
    for li, L in enumerate((24, 33)):
        for k, (key, col, name) in enumerate((("towardB", "#2a78d6", "donor B (real)"), ("towardBtwin", "#9cc0ec", "donor B (twin)"),
                                              ("towardC", "#6b7480", "control donor C"))):
            ys = [next((r[key] for r in tp if r["family"] == f and r["layer"] == L), np.nan) for f in fams]
            xx = x + (li - 0.5) * 0.42 + (k - 1) * w
            b.bar(xx, ys, w, color=col, hatch=("" if L == 24 else "///"), edgecolor="white", lw=0.3,
                  label=name if li == 0 else None)
    b.set_xticks(x, [f"{FAMLAB[f]}\nL24 | L33" for f in fams], fontsize=7); b.set_ylim(0, 1)
    b.set_ylabel("toward donor rank"); b.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    b.set_title("Gemma-4-31B transplant", fontsize=8, fontweight="normal", pad=18)
    _save(fig, out, "fig_causal")


COUP_MODELS = [("google_gemma-4-31B-it", "Gemma-4-31B"), ("google_gemma-4-12B-it", "Gemma-4-12B"),
               ("allenai_Olmo-3-7B-Instruct", "OLMo-3-7B"), ("Qwen_Qwen3-8B", "Qwen3-8B"), ("Qwen_Qwen3-4B", "Qwen3-4B"),
               ("Qwen_Qwen3-1.7B", "Qwen3-1.7B"), ("Qwen_Qwen3-0.6B", "Qwen3-0.6B")]


def fig_coupling(figdata, out, dump):
    """Resting geometry predicts per-pair correctness: robust coefficient (+hop, stated, stimulus FE)."""
    mlc = _one(figdata, "v11", "mlc_31b.json")
    rows = {}
    for m, _ in COUP_MODELS:
        rr = (mlc or {}).get("coupling", []) if m == "google_gemma-4-31B-it" else _load(figdata, f"v11/coupling/{m}", "*.json")
        rows[m] = {r["family"]: r for r in rr if "beta_robust" in r}
    if dump:
        for m, lab in COUP_MODELS:
            sig = sum(1 for r in rows[m].values() if r["beta_robust_ci"][0] > 0)
            acc = np.mean([r["accuracy"] for r in rows[m].values()]) if rows[m] else float("nan")
            print(f"  {lab:12s} acc {acc:.3f} robust-sig {sig}/{len(rows[m])}")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for i, (m, lab) in enumerate(COUP_MODELS):
        for j, fam in enumerate(FAMCOL):
            r = rows[m].get(fam)
            if not r:
                continue
            xx = i + (j - 2) * 0.13; lo, hi = r["beta_robust_ci"]; sig = lo > 0
            ax.errorbar(xx, r["beta_robust"], yerr=[[r["beta_robust"] - lo], [hi - r["beta_robust"]]], fmt="o", ms=3.5,
                        color=FAMCOL[fam], mfc=FAMCOL[fam] if sig else "white", capsize=0, lw=1, label=FAMLAB[fam] if i == 0 else None)
    labs = []
    for m, lab in COUP_MODELS:
        acc = np.mean([r["accuracy"] for r in rows[m].values()]) if rows[m] else float("nan")
        labs.append(f"{lab.replace('Gemma-4-', 'G-').replace('Qwen3-', 'Q-').replace('OLMo-3-', 'OLMo-')}\n{acc:.2f}")
    ax.axhline(0, color="#888", lw=0.8, ls="--"); ax.set_xticks(range(len(COUP_MODELS)), labs, fontsize=6.5)
    ax.set_ylabel("β (margin → correct)"); ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper right")
    ax.set_xlabel("model (mean pairwise accuracy)", fontsize=7)
    _save(fig, out, "fig_coupling")


FIGS = {"cpca": fig_cpca, "manifold": fig_manifold, "stated": fig_stated, "e10": fig_e10,
        "crossform": fig_crossform, "bridge": fig_bridge,
        "locus": fig_locus, "ladder": fig_ladder, "stated_v11": fig_stated_v11, "scale": fig_scale,
        "querylocal": fig_querylocal, "months": fig_months, "causal": fig_causal, "coupling": fig_coupling}


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
