"""Load per-stimulus activation records written by extract_activations.py.

Each npz holds {scheme: [N, L+1, D] fp16} plus `ranks` [N] (1..N by latent
rank), `slots`, `entities` (json), `meta` (json: family, condition, n_items,
content_key, model). We read a (model, family, condition[, N]) slice for one
pooling scheme and expose per-stimulus records so probes can operate either
pooled-across-stimuli (per layer) or per-stimulus (RSA, intrinsic dim).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _scheme_key(z, scheme):
    """Canonical scheme key ('name'/'readout'/...) or the legacy 'loc_<scheme>' layout."""
    if scheme in z.files:
        return scheme
    if ("loc_" + scheme) in z.files:
        return "loc_" + scheme
    return None


def load_records(acts, model, family, condition, scheme, N=None, structure=None, is_null=False,
                 with_extras=False):
    """List of {X:[N,L,D] float32, ranks:[N] int, N:int, content_key:str}.
    `structure` (e.g. 'total_order', 'cyclic', 'grid2d', 'partial_order') filters on
    meta.structure so cyclic and total-order cells are not mixed. `is_null` selects the
    coherence-null twins (when real + null share one acts dir). Reads canonical or the
    legacy 'loc_<scheme>' array layout. `with_extras` additionally attaches the structure
    coordinates written by extract_activations (coord_x/coord_y for grids, cyclic_pos for
    rings, within_rank/chain_of for partial orders) when present — ordered to match `ranks`
    — so the form litmus can read them alongside the activations."""
    recs = []
    for f in sorted((Path(acts) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        key = _scheme_key(z, scheme)
        if m["family"] != family or m["condition"] != condition or key is None:
            continue
        if bool(m.get("is_null", False)) != is_null:
            continue
        if N is not None and int(m["n_items"]) != int(N):
            continue
        if structure is not None and m.get("structure", "total_order") != structure:
            continue
        rec = {"X": z[key].astype(np.float32), "ranks": z["ranks"].astype(int),
               "N": int(m["n_items"]), "content_key": m["content_key"]}
        if with_extras:
            for fld in ("coord_x", "coord_y", "cyclic_pos", "within_rank", "chain_of"):
                if fld in z.files:
                    rec[fld] = z[fld].astype(float)
        recs.append(rec)
    return recs


def n_layers(recs):
    return recs[0]["X"].shape[1] if recs else 0


def interior_mask(ranks, N):
    """Interior = ranks 3..N-2 (identical role/frequency; the confound-clean set)."""
    ranks = np.asarray(ranks)
    return (ranks >= 3) & (ranks <= N - 2)


def stack_layer(recs, layer, interior_only=False):
    """Pool one layer across stimuli -> (X [Σn, D], y [Σn] normalized rank in
    [0,1] per stimulus, groups [Σn] stimulus id). y is normalized per stimulus
    so mixed-N pooling stays comparable; groups feed GroupKFold."""
    Xs, ys, gs = [], [], []
    for gi, r in enumerate(recs):
        ranks, N = r["ranks"], r["N"]
        mask = interior_mask(ranks, N) if interior_only else np.ones(len(ranks), bool)
        # drop entities whose pooled vector is non-finite (fp16 overflow / empty
        # span) so one bad row can't crash PCA downstream.
        finite = np.isfinite(r["X"][:, layer, :]).all(axis=1)
        mask = mask & finite
        if mask.sum() < 1 or N < 2:
            continue
        Xs.append(r["X"][mask, layer, :])
        ys.append((ranks[mask] - 1) / (N - 1))          # per-stimulus [0,1]
        gs.append(np.full(int(mask.sum()), gi))
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def stack_all_layers(recs, interior_only=False):
    """Like stack_layer but returns EVERY layer over a SHARED row set (entities
    finite across all layers) so one permutation can be evaluated consistently
    across layers — required for a max-over-layers (family-wise) null.
    Returns (Xlayers: list[L] of [n, D], y [n], g [n]) or None."""
    if not recs:
        return None
    L = recs[0]["X"].shape[1]
    Xs = [[] for _ in range(L)]
    ys, gs = [], []
    for gi, r in enumerate(recs):
        ranks, N = r["ranks"], r["N"]
        mask = interior_mask(ranks, N) if interior_only else np.ones(len(ranks), bool)
        mask = mask & np.isfinite(r["X"]).all(axis=(1, 2))   # finite at ALL layers
        if mask.sum() < 1 or N < 2:
            continue
        for layer in range(L):
            Xs[layer].append(r["X"][mask, layer, :])
        ys.append((ranks[mask] - 1) / (N - 1))
        gs.append(np.full(int(mask.sum()), gi))
    if not ys:
        return None
    return [np.concatenate(x) for x in Xs], np.concatenate(ys), np.concatenate(gs)
