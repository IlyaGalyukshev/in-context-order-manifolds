#!/usr/bin/env python
"""M5 — Dirichlet energy of resting activations over the stimulus relation graph (Park bridge).

Does the coherent relation graph embed SMOOTHLY in the residual stream — related entities (graph
edges) placed close? E(H)=Σ_edges‖h_i−h_j‖² normalized. Two references per layer:
  * vs within-stimulus PERMUTATION null (shuffle which entity sits at which node) — a low ratio
    that beats its own shuffle means the relation structure IS folded into geometry;
  * real vs TWIN (each uses its own edges) — does the coherent graph embed smoother than the
    incoherent-cycle twin.
Uses the per-entity k-mean vectors (mean_<scheme> from extract_repeat --store, or a plain
[N,L+1,D] scheme array); edges come from the stimulus cards (entity,entity_b) joined by stimulus_id.

  python scripts/probe_dirichlet.py --acts results/e1_20260820/acts --model qwen3-4b \
      --stimuli data/byN_all/stimuli.jsonl --stimuli-null data/byN_all/stimuli_null.jsonl \
      --family s0_zib --n-items 12 --scheme card_mean --json out/dir.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icom.probes import dirichlet_energy, interior_mask


def _edges_by_id(stimuli_path):
    """{stimulus_id: (latent_order, [(i,j) edge index pairs from cards])}."""
    out = {}
    for line in open(stimuli_path):
        s = json.loads(line)
        lo = s["latent_order"]; pos = {e: k for k, e in enumerate(lo)}
        ed = [(pos[c["entity"]], pos[c["entity_b"]]) for c in s["cards"]
              if c.get("entity_b") and c["entity"] in pos and c["entity_b"] in pos]
        out[s["stimulus_id"]] = (lo, ed)
    return out


def _load_means(acts, model, family, condition, scheme, is_null, n_items):
    """Per-stimulus {sid, X:[N,L,D], ranks, N} using the k-mean (mean_<scheme>) or a scheme array."""
    recs = []
    for f in sorted((Path(acts) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if m["family"] != family or m["condition"] != condition:
            continue
        if bool(m.get("is_null", False)) != is_null or int(m["n_items"]) != int(n_items):
            continue
        key = f"mean_{scheme}" if f"mean_{scheme}" in z.files else (scheme if scheme in z.files else None)
        if key is None:
            continue
        X = z[key].astype(np.float64)
        if X.ndim == 4:                                        # reads mode [N,k,L,D] -> mean over k
            X = X.mean(axis=1)
        recs.append({"sid": f.stem.split("_")[0], "X": X,
                     "ranks": z["ranks"].astype(int), "N": int(m["n_items"])})
    return recs


def _profile(rec, edges, interior, normalize=True):
    """Dirichlet energy per layer over `edges`, optionally restricting to interior entities
    (edges with both endpoints interior). Returns [L] or None."""
    X = rec["X"]
    if interior:
        mask = interior_mask(rec["ranks"], rec["N"])
        keep = np.where(mask)[0]
        remap = {o: n for n, o in enumerate(keep)}
        edges = [(remap[i], remap[j]) for (i, j) in edges if i in remap and j in remap]
        X = X[keep]
    if len(edges) < 2:
        return None
    L = X.shape[1]
    return np.array([dirichlet_energy(X[:, layer, :], edges, normalize) for layer in range(L)])


def _cell(recs, edge_map, interior, n_perm, seed):
    """Mean per-layer energy + mean per-stimulus perm-null energy. Returns (energy[L], null[L], n)."""
    rng = np.random.default_rng(seed)
    en, nu = [], []
    for r in recs:
        if r["sid"] not in edge_map:
            continue
        _, edges = edge_map[r["sid"]]
        prof = _profile(r, edges, interior)
        if prof is None:
            continue
        en.append(prof)
        # perm-null: shuffle entity->node assignment (permute X rows) a few times, average energy
        pn = []
        for _ in range(n_perm):
            rr = {"X": r["X"][rng.permutation(len(r["ranks"]))], "ranks": r["ranks"], "N": r["N"]}
            p = _profile(rr, edges, interior)
            if p is not None:
                pn.append(p)
        if pn:
            nu.append(np.mean(pn, axis=0))
    if not en:
        return None, None, 0
    return np.mean(en, axis=0), (np.mean(nu, axis=0) if nu else None), len(en)


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def run(acts, model, stimuli, stimuli_null, family, condition, scheme, n_items, interior,
        n_perm, n_boot, seed):
    emap = _edges_by_id(stimuli); emap.update(_edges_by_id(stimuli_null))
    real = _load_means(acts, model, family, condition, scheme, False, n_items)
    twin = _load_means(acts, model, family, condition, scheme, True, n_items)
    if not real:
        return None
    e_real, e_null, n = _cell(real, emap, interior, n_perm, seed)
    e_twin, _, nt = _cell(twin, emap, interior, 1, seed) if twin else (None, None, 0)
    if e_real is None:
        return None
    # peak = layer of MAX smoothness advantage (null - real): real most below its shuffle
    gap = (e_null - e_real) if e_null is not None else None
    peak = int(np.nanargmax(gap)) if gap is not None and np.isfinite(gap).any() else int(np.nanargmin(e_real))
    # bootstrap CI over stimuli on real energy + (null-real) at peak
    rng = np.random.default_rng(seed); idx = np.arange(len(real)); gg = []
    for _ in range(n_boot):
        rb = [real[i] for i in rng.choice(idx, len(idx), replace=True)]
        er, en, _ = _cell(rb, emap, interior, 1, seed)
        if er is not None and en is not None:
            gg.append(float(en[peak] - er[peak]))
    gg = np.array([x for x in gg if x == x])
    ci = [_r(np.percentile(gg, 2.5)), _r(np.percentile(gg, 97.5))] if len(gg) > 2 else [None, None]
    return dict(model=model, family=family, n_items=n_items, scheme=scheme, interior=interior,
                n_real=n, n_twin=nt, peak_layer=peak,
                energy_real=_r(e_real[peak]), energy_null=_r(e_null[peak]) if e_null is not None else None,
                smoothness_gap=_r(gap[peak]) if gap is not None else None, gap_ci=ci,
                sig_vs_shuffle=bool(ci[0] is not None and ci[0] > 0),
                energy_twin=_r(e_twin[peak]) if e_twin is not None else None,
                real_below_twin=_r((e_twin[peak] - e_real[peak])) if e_twin is not None else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--stimuli", required=True); ap.add_argument("--stimuli-null", required=True)
    ap.add_argument("--families", default="s0_zib"); ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--scheme", default="card_mean"); ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--interior", action="store_true", default=True)
    ap.add_argument("--n-perm", type=int, default=20); ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = []
    for family in args.families.split(","):
        r = run(args.acts, args.model, args.stimuli, args.stimuli_null, family, args.condition,
                args.scheme, args.n_items, args.interior, args.n_perm, args.n_boot, args.seed)
        if r is None:
            print(f"{args.model} {family} N{args.n_items}: no data", flush=True); continue
        results.append(r)
        print(f"{r['model']:12s} {family} N{r['n_items']} {r['scheme']} | peak L{r['peak_layer']} "
              f"E_real={r['energy_real']} E_shuffle={r['energy_null']} smooth_gap={r['smoothness_gap']}"
              f"{r['gap_ci']} {'SIG' if r['sig_vs_shuffle'] else 'ns'} | "
              f"E_twin={r['energy_twin']} real_below_twin={r['real_below_twin']}", flush=True)
    if args.json and results:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
