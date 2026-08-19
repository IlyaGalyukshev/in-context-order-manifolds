#!/usr/bin/env python
"""P0.2 / E7 — per-item representation↔behavior coupling (CPU; original goal #2).

Does the geometry of a single stimulus predict whether the model answers a specific pair correctly?
Join: battery pairwise rows (correct, rank_distance) ⋈ questions.jsonl (the asked pair a,b + gold)
⋈ resting acts (per-entity OOF rank prediction). The predictor is the decoded rank MARGIN between
the pair — a signed, leakage-safe OOF ridge readout (GroupKFold over stimuli) — aligned to the gold
direction. Test: logistic  correct ~ signed_margin (+ rank_distance control); β>0 with CI>0 means the
resting geometry is behaviorally load-bearing at the item level. Cluster-bootstrap CI over stimuli.

  python scripts/probe_coupling.py --acts results/camp_gemmaform_20260819 --model gemma-4-12b-it \
      --battery results/gate_v2crit/battery_gemma-4-12b-it.jsonl \
      --questions data/bcs_v2_crit/questions.jsonl --family s0_quomp --condition shuffle \
      --q-family pairwise --json out/coupling.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icom.probes import interior_mask, reduce
from icom.probes.linear import cv_predict, cv_spearman


def _stim_id(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[0] if "_" in stem else stem            # {id}.npz or {id}_fam_cond.npz


def load_acts(acts, model, family, condition):
    """Per-stimulus {sid, entities[list], ranks[N], X[N,L+1,D]} for the real (non-null) cell."""
    recs = []
    d = Path(acts) / model
    if not d.exists():
        d = Path(acts) / "acts" / model                           # some runs nest under acts/
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if m["family"] != family or m["condition"] != condition:
            continue
        if bool(m.get("is_null", False)):
            continue
        scheme = "readout" if "readout" in z.files else ("loc_readout" if "loc_readout" in z.files else None)
        if scheme is None:
            continue
        recs.append({"sid": _stim_id(f), "entities": json.loads(str(z["entities"])),
                     "ranks": z["ranks"].astype(int), "X": z[scheme].astype(np.float32)})
    return recs


def oof_rank_by_entity(recs, scheme_layer=None):
    """OOF ridge rank prediction per (sid, entity), interior only, at the best rank-decodable layer.
    Returns ({(sid,entity): pred}, layer, cv_rho)."""
    L = recs[0]["X"].shape[1]
    # stack interior with bookkeeping of (sid, entity)
    def stack(layer):
        Xs, ys, gs, keys = [], [], [], []
        for gi, r in enumerate(recs):
            mask = interior_mask(r["ranks"], r["N"] if "N" in r else len(r["ranks"]))
            fin = np.isfinite(r["X"][:, layer, :]).all(axis=1)
            mask = mask & fin
            if mask.sum() < 2:
                continue
            idx = np.where(mask)[0]
            Xs.append(r["X"][idx, layer, :])
            rk = r["ranks"][idx]; ys.append((rk - 1) / (len(r["ranks"]) - 1))
            gs.append(np.full(len(idx), gi))
            keys += [(r["sid"], r["entities"][i]) for i in idx]
        if not Xs:
            return None
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), keys
    if scheme_layer is None:                                       # pick best rank-decode layer
        best, bl = -9, L // 2
        for layer in range(1, L, max(1, L // 12)):
            s = stack(layer)
            if s is None:
                continue
            rho = cv_spearman(reduce(s[0]), s[1], s[2])
            if rho == rho and rho > best:
                best, bl = rho, layer
        scheme_layer = bl
    s = stack(scheme_layer)
    if s is None:
        return {}, scheme_layer, float("nan")
    X, y, g, keys = s
    pred = cv_predict(reduce(X), y, g)
    rho = cv_spearman(reduce(X), y, g)
    return ({keys[i]: float(pred[i]) for i in range(len(keys))} if pred is not None else {}), scheme_layer, float(rho)


def _logit_beta(x, ctrl, yb):
    """β on x (standardized) in logistic correct ~ x + ctrl, via sklearn. Returns coef of x."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.column_stack([x, ctrl]))
    if len(np.unique(yb)) < 2:
        return float("nan")
    lr = LogisticRegression(max_iter=1000).fit(Z, yb)
    return float(lr.coef_[0, 0])


def run(acts, model, battery, questions, family, condition, q_family, layer, n_boot, seed):
    recs = load_acts(acts, model, family, condition)
    if not recs:
        return {"error": "no acts"}
    predmap, lyr, cv_rho = oof_rank_by_entity(recs, layer)
    q = {}
    for line in open(questions):
        d = json.loads(line)
        if d.get("family") == q_family and d.get("target_entities"):
            q[d["qid"]] = (d["target_entities"], d.get("answer_key"))
    rows = []
    for line in open(battery):
        b = json.loads(line)
        if b.get("q_family") != q_family or b.get("family") != family or b.get("condition") != condition:
            continue
        if b.get("correct") is None or b["qid"] not in q:
            continue
        (a, bb), gold = q[b["qid"]]; sid = b["stimulus_id"]
        ka, kb = (sid, a), (sid, bb)
        if ka not in predmap or kb not in predmap:
            continue
        # signed margin aligned to gold: positive when the geometry ranks the gold-earlier entity lower
        earlier = gold if gold in (a, bb) else a
        later = bb if earlier == a else a
        signed = predmap[(sid, later)] - predmap[(sid, earlier)]   # >0 ⇒ geometry agrees with gold order
        rows.append((signed, b.get("rank_distance") or 0, int(bool(b["correct"])), sid))
    if len(rows) < 20:
        return {"error": f"too few joined rows ({len(rows)})", "layer": lyr, "cv_rho": round(cv_rho, 3)}
    signed = np.array([r[0] for r in rows]); rd = np.array([r[1] for r in rows], float)
    yb = np.array([r[2] for r in rows]); sids = np.array([r[3] for r in rows])
    beta = _logit_beta(signed, rd, yb)
    # cluster-bootstrap over stimuli for CI on beta
    rng = np.random.default_rng(seed); uids = np.unique(sids); bs = []
    for _ in range(n_boot):
        pick = rng.choice(uids, len(uids), replace=True)
        m = np.concatenate([np.where(sids == u)[0] for u in pick])
        if len(np.unique(yb[m])) < 2:
            continue
        bs.append(_logit_beta(signed[m], rd[m], yb[m]))
    bs = np.array([x for x in bs if x == x])
    ci = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)] if len(bs) > 2 else [None, None]
    return dict(model=model, family=family, condition=condition, q_family=q_family, layer=lyr,
                cv_rho=round(cv_rho, 3), n_pairs=len(rows), n_stim=int(len(uids)),
                accuracy=round(float(yb.mean()), 3), beta_margin=round(beta, 3), beta_ci=ci,
                sig=bool(ci[0] is not None and ci[0] > 0),
                corr_margin_correct=round(float(np.corrcoef(signed, yb)[0, 1]), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--battery", required=True); ap.add_argument("--questions", required=True)
    ap.add_argument("--families", default="s0_quomp"); ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--q-family", default="pairwise"); ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = []
    for family in args.families.split(","):
        r = run(args.acts, args.model, args.battery, args.questions, family, args.condition,
                args.q_family, args.layer, args.n_boot, args.seed)
        results.append(r)
        if "error" in r:
            print(f"{args.model} {family}/{args.q_family}: {r['error']}", flush=True)
        else:
            print(f"{r['model']:14s} {family}/{r['q_family']} L{r['layer']}(cv_rho={r['cv_rho']}) | "
                  f"n_pairs={r['n_pairs']} acc={r['accuracy']} | beta_margin={r['beta_margin']}{r['beta_ci']} "
                  f"{'SIG' if r['sig'] else 'ns'} (corr={r['corr_margin_correct']})", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
