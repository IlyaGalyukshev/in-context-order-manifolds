#!/usr/bin/env python
"""Contrastive-bottleneck emergent-shape probe (the flagship "does X -> Y, Y may != X" test).

A 2-layer MLP encoder f: R^D -> R^3 is trained so entities close in the STRUCTURE'S OWN metric
are close in a 3-D latent (KL(P||Q): P from the structure graph-distance, Q from latent cosine).
The loss never names line/ring/plane -> form-neutral. Read OUT-OF-FOLD on held-out stimuli, then
measure per-stimulus descriptors and aggregate over stimuli (the unit is one stimulus, 16 pts —
NOT a cross-stimulus pool, which would blur into a 2-D sphere blob):
  betti1 = max H1 persistence / diameter (ripser)   -> loop (ring) vs none (line/arc)
  PRdim  = participation ratio (sum l)^2 / sum l^2   -> ~1 (line) vs ~2 (ring/plane)
  curv   = mean turning angle along structure order  -> 0 line, const ring
NULL (per structure) = permute the structure coordinates within each stimulus; the shape MUST
collapse (paired real-null CI) or the method is an artifact.

The metric is derived from the cards: the stored ranks / cyclic_pos / coord_x,y ARE the generator's
solution to the relation graph (verified == a from-scratch transitive-closure / offset-propagation
reconstruction at Spearman 1.0), so total_order -> |dr|, cyclic -> wrap-around ring distance,
grid2d -> L1 over (x,y). CPU-friendly; needs torch + ripser (pip install .[shape]).

Usage:
  python scripts/probe_shape.py --acts <dir> --model gemma-4-12b-it \
      --structure cyclic --family s0_zib --scheme readout --difficulty all
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def gdist(coords, metric, K):
    if metric == "ring":
        d = np.abs(coords[:, None, 0] - coords[None, :, 0]); return np.minimum(d, K - d)
    if metric == "grid":
        return np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)
    return np.abs(coords[:, None, 0] - coords[None, :, 0])  # line


def load(acts, model, family, structure, scheme, n_items, difficulty, is_null):
    metric = {"total_order": "line", "cyclic": "ring", "grid2d": "grid"}[structure]
    out = []
    for f in sorted((Path(acts) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"]))
        if m.get("family") != family or m.get("structure", "total_order") != structure:
            continue
        if scheme not in z.files or (n_items and m.get("n_items") != n_items):
            continue
        if bool(m.get("is_null", False)) != is_null:
            continue
        if difficulty != "all" and m.get("difficulty") not in (difficulty, None):
            continue
        if metric == "ring":
            gc = z["cyclic_pos"].astype(float)[:, None]; gp = gc[:, 0]
        elif metric == "grid":
            gc = np.stack([z["coord_x"], z["coord_y"]], 1).astype(float); gp = gc[:, 0]
        else:
            gc = z["ranks"].astype(float)[:, None]; gp = gc[:, 0]
        out.append({"x": z[scheme].astype(np.float32), "gc": gc, "gp": gp})
    return out, metric


def train_encode(train, test, layer, metric, K, epochs, sigma=2.0, tau=0.3):
    import torch, torch.nn as nn
    D = train[0]["x"].shape[2]
    sc = StandardScaler().fit(np.concatenate([d["x"][:, layer, :] for d in train]))
    pools, Ps = [], []
    for d in train:
        Dg = gdist(d["gc"], metric, K); P = np.exp(-Dg ** 2 / (2 * sigma ** 2)); np.fill_diagonal(P, 0)
        P = P / (P.sum(1, keepdims=True) + 1e-9)
        pools.append(torch.tensor(sc.transform(d["x"][:, layer, :]), dtype=torch.float32))
        Ps.append(torch.tensor(P, dtype=torch.float32))

    class Enc(nn.Module):
        def __init__(s):
            super().__init__(); s.net = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 3))
        def forward(s, x):
            z = s.net(x); return z / (z.norm(dim=1, keepdim=True) + 1e-8)
    enc = Enc(); opt = torch.optim.Adam(enc.parameters(), lr=2e-3, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for x, P in zip(pools, Ps):
            z = enc(x); sim = (z @ z.T) / tau; sim.fill_diagonal_(-1e9)
            loss = loss - (P * torch.log_softmax(sim, 1)).sum(1).mean()
        (loss / len(pools)).backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        return [enc(torch.tensor(sc.transform(d["x"][:, layer, :]), dtype=torch.float32)).numpy() for d in test]


def desc_stim(Z, gp, rng):
    from ripser import ripser
    Z = Z + 1e-6 * rng.standard_normal(Z.shape)
    h1 = ripser(Z, maxdim=1)["dgms"][1]
    diam = np.linalg.norm(Z.max(0) - Z.min(0)) + 1e-9
    b1 = 0.0 if len(h1) == 0 else float((h1[:, 1] - h1[:, 0]).max() / diam)
    C = Z - Z.mean(0); lam = np.linalg.eigvalsh(C.T @ C); lam = lam[lam > 1e-9]
    pr = float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12)) if len(lam) else float("nan")
    Po = Z[np.argsort(gp)]; dd = np.diff(Po, 0); dd = dd / (np.linalg.norm(dd, axis=1, keepdims=True) + 1e-9)
    cv = float(np.mean(np.arccos(np.clip((dd[:-1] * dd[1:]).sum(1), -1, 1)))) if len(Po) >= 3 else float("nan")
    return b1, pr, cv


def pick_layer(data, scheme_layers):
    # cheap: choose the layer maximizing OOF ridge decode of gp (structure position)
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    y = np.concatenate([d["gp"] for d in data]); g = np.concatenate([[i] * len(d["gp"]) for i, d in enumerate(data)])
    best = (0, -9)
    for l in range(scheme_layers):
        Z = StandardScaler().fit_transform(np.concatenate([d["x"][:, l, :] for d in data]))
        P = PCA(min(64, Z.shape[0] - 1), random_state=0).fit_transform(Z)
        o = np.full(len(y), np.nan)
        for tr, te in GroupKFold(4).split(P, y, g):
            o[te] = Ridge(alpha=10.).fit(P[tr], y[tr]).predict(P[te])
        s = abs(spearmanr(o, y)[0] or 0)
        if s > best[1]:
            best = (l, s)
    return best[0]


def oof_embed(data, layer, metric, K, epochs, nfold=4):
    g = np.arange(len(data)); Z = [None] * len(data)
    for tr, te in GroupKFold(nfold).split(g, g, g):
        zte = train_encode([data[i] for i in tr], [data[i] for i in te], layer, metric, K, epochs)
        for k, i in enumerate(te):
            Z[i] = zte[k]
    return Z


def agg(v):
    v = np.array(v); v = v[np.isfinite(v)]
    return {"mean": round(float(v.mean()), 3), "ci": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]}


def paired(real, null, rng):
    r = np.array(real); n = np.array(null); m = np.isfinite(r) & np.isfinite(n); d = r[m] - n[m]
    bs = [float(np.mean(rng.choice(d, len(d), True))) for _ in range(2000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    return {"mean_diff": round(float(d.mean()), 3), "ci": [round(float(lo), 3), round(float(hi), 3)], "sig": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--structure", required=True, choices=["total_order", "cyclic", "grid2d"])
    ap.add_argument("--family", required=True); ap.add_argument("--scheme", default="readout")
    ap.add_argument("--n-items", type=int, default=16); ap.add_argument("--difficulty", default="all")
    ap.add_argument("--epochs", type=int, default=350); ap.add_argument("--out", default=None)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    data, metric = load(a.acts, a.model, a.family, a.structure, a.scheme, a.n_items, a.difficulty, is_null=False)
    if len(data) < 20:
        print(f"too few stimuli ({len(data)}) for {a.family}/{a.structure}"); return
    K = a.n_items; layer = pick_layer(data, data[0]["x"].shape[1])
    Zr = oof_embed(data, layer, metric, K, a.epochs)
    dperm = [{"x": d["x"], "gc": d["gc"][rng.permutation(len(d["gc"]))].copy(), "gp": d["gp"]} for d in data]
    Zn = oof_embed(dperm, layer, metric, K, a.epochs)
    br, pr, cr, bn, pn, cn = [], [], [], [], [], []
    for i, d in enumerate(data):
        b, p, c = desc_stim(Zr[i], d["gp"], rng); br.append(b); pr.append(p); cr.append(c)
        b, p, c = desc_stim(Zn[i], d["gp"], rng); bn.append(b); pn.append(p); cn.append(c)
    res = {"structure": a.structure, "family": a.family, "scheme": a.scheme, "n": len(data), "layer": int(layer),
           "REAL": {"betti1": agg(br), "PRdim": agg(pr), "curv": agg(cr)},
           "NULL": {"betti1": agg(bn), "PRdim": agg(pn), "curv": agg(cn)},
           "REAL_minus_NULL": {"betti1": paired(br, bn, rng), "PRdim": paired(pr, pn, rng), "curv": paired(cr, cn, rng)}}
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1); print(f"wrote -> {a.out}")


if __name__ == "__main__":
    main()
