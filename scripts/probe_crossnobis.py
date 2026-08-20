#!/usr/bin/env python
"""M2 — whitened cross-validated RDM (crossnobis) probe over repeat-read acts (from extract_repeat.py).

Per stimulus, per layer: build an UNBIASED crossnobis RDM from the k reads of the interior entities
(noise-normalized, cross-validated across reads) and take whitened-RSA to the ideal RDM (line for
total_order, ring for cyclic). The headline is the real − twin increment: how much ordinal structure
survives above the incoherent-cycle twin, now measured by a principled unsupervised distance instead
of the raw-cosine / remove-top-k hack. Held-out layer selection is MAIN (argmax = optimistic bound);
bootstrap-CI over stimuli; within-stimulus rank-permutation null.

  python scripts/probe_crossnobis.py --acts results/repeat_smoke --model qwen3-4b \
      --family s0_zib --condition shuffle --scheme readout --ideal line \
      --n-boot 1000 --n-perm 200 --json out/crossnobis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icom.probes import interior_mask
from icom.probes.crossnobis import crossnobis_rdm, line_rdm, ring_rdm, whitened_rsa


def load_repeat(acts, model, family, condition, scheme, structure=None, is_null=False, n_items=None):
    """Per-stimulus records for one scheme, from EITHER storage mode written by extract_repeat.py:
      * reads : {mode:'reads', X:[N,k,L+1,D] f32}   (raw repeat-reads)
      * rdm   : {mode:'rdm',  RDM:[N,N,L+1] f32}     (crossnobis RDM precomputed at extraction)
    plus ranks[N], N. The whitened-RSA is identical either way — the RDM store just moves the
    crossnobis computation from probe-time to extraction-time (disk-compaction for scale)."""
    recs = []
    for f in sorted((Path(acts) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if m["family"] != family or m["condition"] != condition:
            continue
        if bool(m.get("is_null", False)) != is_null:
            continue
        if structure is not None and m.get("structure", "total_order") != structure:
            continue
        if n_items is not None and int(m["n_items"]) != int(n_items):
            continue
        base = {"ranks": z["ranks"].astype(int), "N": int(m["n_items"])}
        if f"rdm_{scheme}" in z.files:
            recs.append({"mode": "rdm", "RDM": z[f"rdm_{scheme}"].astype(np.float64), **base})
        elif scheme in z.files:
            recs.append({"mode": "reads", "X": z[scheme].astype(np.float32), **base})
    return recs


def _interior_rdm(rec, layer, n_splits, seed):
    """(crossnobis RDM [n_int, n_int], ranks [n_int], N) at one layer, interior entities only, or
    None. Computes crossnobis from raw reads, or slices the precomputed RDM — same object either way."""
    mask = interior_mask(rec["ranks"], rec["N"])
    if rec["mode"] == "rdm":
        R = rec["RDM"][:, :, layer]
        fin = np.isfinite(R).all(axis=1) & mask                # keep interior rows with a finite RDM row
        idx = np.where(fin)[0]
        if len(idx) < 4:
            return None
        return R[np.ix_(idx, idx)], rec["ranks"][idx], rec["N"]
    reads = rec["X"][mask][:, :, layer, :]                     # [n_int, k, D]
    ranks = rec["ranks"][mask]
    finite = np.isfinite(reads).all(axis=(1, 2))
    reads, ranks = reads[finite], ranks[finite]
    if len(ranks) < 4 or reads.shape[1] < 2:
        return None
    return crossnobis_rdm(reads, n_splits=n_splits, seed=seed), ranks, rec["N"]


def _stim_rsa(rec, layer, ideal, n_splits, seed):
    ir = _interior_rdm(rec, layer, n_splits, seed)
    if ir is None:
        return float("nan")
    rdm, ranks, N = ir
    ideal_rdm = ring_rdm(ranks, N) if ideal == "ring" else line_rdm(ranks)
    return whitened_rsa(rdm, ideal_rdm)


def _layer_mean(recs, layer, ideal, n_splits, seed):
    v = [_stim_rsa(r, layer, ideal, n_splits, seed) for r in recs]
    v = [x for x in v if x == x]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def _peak_rdms(recs, layer, n_splits, seed):
    """Precompute (interior crossnobis RDM, ranks, N) per stimulus at `layer` — invariant to the
    rank-label permutation and to stimulus resampling, so the null/bootstrap reuse these (cheap)."""
    out = []
    for r in recs:
        ir = _interior_rdm(r, layer, n_splits, seed)
        if ir is not None:
            out.append(ir)
    return out


def _rsa_from(rdms, ideal, ranks_list=None):
    """Mean whitened-RSA over precomputed RDMs; `ranks_list` overrides each stimulus's ranks
    (for the permutation null)."""
    v = []
    for i, (rdm, ranks, N) in enumerate(rdms):
        rk = ranks if ranks_list is None else ranks_list[i]
        idl = ring_rdm(rk, N) if ideal == "ring" else line_rdm(rk)
        x = whitened_rsa(rdm, idl)
        if x == x:
            v.append(x)
    return float(np.mean(v)) if v else float("nan")


def _held_out_peak(recs, L, ideal, n_splits, seed):
    """Leak-free peak RSA: pick the argmax layer on one half of stimuli, score it on the other
    (both directions, averaged). Returns (heldout_rsa, peak_layer_on_full, argmax_rsa_full)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))
    half = len(idx) // 2
    A, B = [recs[i] for i in idx[:half]], [recs[i] for i in idx[half:]]
    full = np.array([_layer_mean(recs, l, ideal, n_splits, seed)[0] for l in range(L)])
    ho = []
    for tr, te in ((A, B), (B, A)):
        prof = np.array([_layer_mean(tr, l, ideal, n_splits, seed)[0] for l in range(L)])
        if np.isfinite(prof).any():
            lpk = int(np.nanargmax(prof))
            ho.append(_layer_mean(te, lpk, ideal, n_splits, seed)[0])
    heldout = float(np.nanmean(ho)) if ho else float("nan")
    peak = int(np.nanargmax(full)) if np.isfinite(full).any() else 0
    return heldout, peak, (float(full[peak]) if np.isfinite(full).any() else float("nan"))


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def run_cell(acts, model, family, condition, scheme, ideal, n_splits, n_boot, n_perm, seed, n_items=None):
    real = load_repeat(acts, model, family, condition, scheme, is_null=False, n_items=n_items)
    twin = load_repeat(acts, model, family, condition, scheme, is_null=True, n_items=n_items)
    if not real:
        return None
    r0 = real[0]
    L = r0["RDM"].shape[2] if r0["mode"] == "rdm" else r0["X"].shape[2]
    ho, peak, argmax = _held_out_peak(real, L, ideal, n_splits, seed)
    # precompute the peak-layer crossnobis RDMs ONCE (invariant to rank-perm / stimulus-resample)
    rdms_real = _peak_rdms(real, peak, n_splits, seed)
    rdms_twin = _peak_rdms(twin, peak, n_splits, seed) if twin else []
    rsa_real = _rsa_from(rdms_real, ideal)
    rsa_twin = _rsa_from(rdms_twin, ideal) if rdms_twin else float("nan")

    # within-stimulus rank-permutation null at the peak layer (reuses precomputed RDMs)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = [rng.permutation(rk) for (_, rk, _) in rdms_real]
        null.append(_rsa_from(rdms_real, ideal, ranks_list=perm))
    null = np.array([x for x in null if x == x])
    p = (1 + int((null >= rsa_real).sum())) / (1 + len(null)) if len(null) else None
    null95 = float(np.nanpercentile(null, 95)) if len(null) else float("nan")

    # bootstrap CI over stimuli on real, twin, and the increment (resample precomputed RDMs)
    rr, tt, gg = [], [], []
    ir = np.arange(len(rdms_real)); it = np.arange(len(rdms_twin)) if rdms_twin else np.array([])
    for _ in range(n_boot):
        vr = _rsa_from([rdms_real[i] for i in rng.choice(ir, len(ir), replace=True)], ideal)
        rr.append(vr)
        if rdms_twin:
            vt = _rsa_from([rdms_twin[i] for i in rng.choice(it, len(it), replace=True)], ideal)
            tt.append(vt); gg.append(vr - vt)
    def _ci(a):
        a = np.array([x for x in a if x == x])
        return [None, None] if len(a) < 2 else [_r(np.percentile(a, 2.5)), _r(np.percentile(a, 97.5))]

    gap = rsa_real - rsa_twin if twin else float("nan")
    gci = _ci(gg)
    return dict(model=model, family=family, condition=condition, scheme=scheme, ideal=ideal,
                n_real=len(real), n_twin=len(twin), peak_layer=peak, peak_frac=round(peak / max(1, L - 1), 3),
                rsa_real=_r(rsa_real), rsa_real_ci=_ci(rr), rsa_heldout=_r(ho), rsa_argmax=_r(argmax),
                rsa_twin=_r(rsa_twin), rsa_twin_ci=_ci(tt),
                increment=_r(gap), increment_ci=gci,
                sig_vs_twin=bool(gci[0] is not None and gci[0] > 0),
                null95=_r(null95), p=_r(p, 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--families", default="s0_zib")
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--scheme", default="readout")
    ap.add_argument("--ideal", default="line", choices=["line", "ring"])
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-items", type=int, default=None, help="filter to one N (for the N-curve)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = []
    for family in args.families.split(","):
        try:
            r = run_cell(args.acts, args.model, family, args.condition, args.scheme, args.ideal,
                         args.n_splits, args.n_boot, args.n_perm, args.seed, n_items=args.n_items)
        except Exception as e:
            print(f"{args.model} {family}/{args.scheme}: ERROR {e}", flush=True)
            continue
        if r is None:
            print(f"{args.model} {family}/{args.scheme}: no data", flush=True)
            continue
        r["n_items"] = args.n_items
        results.append(r)
        print(f"{r['model']:12s} {family}/{args.scheme} N={args.n_items} ideal={args.ideal} | peak L{r['peak_layer']}"
              f"({r['peak_frac']}) rsa_real={r['rsa_real']}(heldout={r['rsa_heldout']},argmax={r['rsa_argmax']}) "
              f"twin={r['rsa_twin']} incr={r['increment']}{r['increment_ci']} "
              f"{'SIG' if r['sig_vs_twin'] else 'ns'} (p_null={r['p']})", flush=True)

    if args.json and results:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
