#!/usr/bin/env python
"""Phase-D deliverable: 2D/3D pictures of the order manifold.

Pools interior (or all) entities across stimuli at a layer, projects (PCA +
nonlinear embeddings via sklearn.manifold; UMAP if installed) and colors by
normalized latent rank. Emits a per-layer contact sheet (the manifold forming
and dissolving with depth) with the coherence-null control on an adjacent row,
plus an optional interactive 3D HTML.

HONESTY RULE: a projection illustrates, it never decides. The coherence-null
panel (should be a blob, not an arc) ships in the same figure; the quantitative
claim is probe_sweep's interior-only decode vs the nulls.

  python scripts/visualize_manifold.py --acts acts --model qwen3-4b \
      --family s1_size --condition shuffle --scheme readout \
      --layers 0,8,16,24 --method pca --interior-only \
      --out fig/manifold.png --null-acts acts_null --html fig/manifold3d.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from icom.probes.geometry import project
from icom.probes.loading import interior_mask, load_records, n_layers


def collect(acts, model, family, condition, scheme, layer, interior_only=True):
    """Pool entities across stimuli at one layer -> (X [Σn, D], normrank [Σn])."""
    recs = load_records(acts, model, family, condition, scheme)
    Xs, cs = [], []
    for r in recs:
        ranks, N = r["ranks"], r["N"]
        mask = interior_mask(ranks, N) if interior_only else np.ones(len(ranks), bool)
        if mask.sum() < 1:
            continue
        Xs.append(r["X"][mask, layer, :])
        cs.append((ranks[mask] - 1) / (N - 1))       # normalized rank in [0,1]
    if not Xs:
        return None, None
    return np.concatenate(Xs).astype(np.float64), np.concatenate(cs)


def rank_axis_alignment(coords, normrank):
    """Best |Spearman| between any projected axis and rank — a non-decisive
    sanity number for the picture (high on an arc, ~0 on a blob). Uses the best
    axis, not just PC1, since rank need not fall on the max-variance direction.
    The quantitative claim lives in probe_sweep, not here."""
    from scipy.stats import spearmanr
    if coords is None or len(coords) < 4:
        return float("nan")
    rhos = [abs(spearmanr(coords[:, j], normrank)[0]) for j in range(coords.shape[1])]
    rhos = [r for r in rhos if not np.isnan(r)]
    return float(max(rhos)) if rhos else float("nan")


def contact_sheet(entries, model, family, condition, scheme, layers, method,
                  interior_only, out_png, seed=0):
    """entries: list of (row_label, acts_dir). Rows x len(layers) grid of 2D
    projections colored by normalized rank. The coherence-null row (if provided)
    sits directly beneath the order row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow, ncol = len(entries), len(layers)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    Lref = None
    sm = None
    for ri, (label, acts) in enumerate(entries):
        recs = load_records(acts, model, family, condition, scheme)
        Lref = n_layers(recs) if recs else Lref
        for ci, layer in enumerate(layers):
            ax = axes[ri][ci]
            X, c = collect(acts, model, family, condition, scheme, layer, interior_only)
            if X is None or len(X) < 4:
                ax.set_axis_off(); ax.set_title(f"{label} L{layer}\n(no data)", fontsize=8)
                continue
            P = project(X, method=method, dim=2, seed=seed)
            sc = ax.scatter(P[:, 0], P[:, 1], c=c, cmap="viridis", s=10, alpha=0.7,
                            vmin=0, vmax=1, linewidths=0)
            sm = sc
            frac = round(layer / (Lref - 1), 2) if Lref and Lref > 1 else "?"
            align = rank_axis_alignment(P, c)
            ax.set_title(f"{label} · L{layer} ({frac})\nrank-axis|ρ|={align:.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{model} · {family}/{condition} · {scheme} · {method} · "
                 f"{'interior' if interior_only else 'all'} entities (color = rank)", fontsize=10)
    if sm is not None:
        cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
        cb.set_label("normalized rank (low→high)", fontsize=8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def interactive3d(acts, model, family, condition, scheme, layer, method,
                  interior_only, out_html, seed=0):
    """Rotatable 3D projection colored by rank (plotly, self-contained HTML)."""
    import plotly.graph_objects as go
    X, c = collect(acts, model, family, condition, scheme, layer, interior_only)
    if X is None or len(X) < 4:
        raise SystemExit("no data for 3D projection")
    P = project(X, method=method, dim=3, seed=seed)
    if P.shape[1] < 3:
        P = np.pad(P, ((0, 0), (0, 3 - P.shape[1])))
    fig = go.Figure(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
        marker=dict(size=4, color=c, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="rank")),
    ))
    fig.update_layout(title=f"{model} {family}/{condition} {scheme} L{layer} ({method})")
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="s1_size")
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--scheme", default="readout")
    ap.add_argument("--layers", default="0,8,16,24")
    ap.add_argument("--method", default="pca", choices=["pca", "isomap", "spectral", "umap"])
    ap.add_argument("--interior-only", action="store_true", default=True)
    ap.add_argument("--all-entities", dest="interior_only", action="store_false")
    ap.add_argument("--null-acts", default=None, help="coherence-null acts for the control row")
    ap.add_argument("--out", required=True, help="contact-sheet PNG")
    ap.add_argument("--html", default=None, help="optional interactive 3D HTML")
    ap.add_argument("--html-layer", type=int, default=None)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    entries = [("order", args.acts)]
    if args.null_acts:
        entries.append(("coherence-null", args.null_acts))
    png = contact_sheet(entries, args.model, args.family, args.condition, args.scheme,
                        layers, args.method, args.interior_only, args.out)
    print(f"wrote contact sheet -> {png}")
    if args.html:
        L = args.html_layer if args.html_layer is not None else layers[-1]
        html = interactive3d(args.acts, args.model, args.family, args.condition,
                             args.scheme, L, args.method, args.interior_only, args.html)
        print(f"wrote 3D -> {html}")


if __name__ == "__main__":
    main()
