#!/usr/bin/env python
"""Proper cross-stimulus rank decode from ROLE/POSITION features. Unlike the
stock audit_confounds.py, this (a) restricts to TOTAL-ORDER shuffle stimuli,
(b) can filter difficulty, and (c) FORCES the decode to run by dropping
zero-variance columns (the stock guard skips the decode entirely when
mention/subj are constant, which hides the position leak).

Features per entity (cross-stimulus): continuous mean CHAR position of its
mentions in the prompt (normalized by prompt length), continuous mean CARD
presentation-slot (normalized), mention_count, subj_frac. Target: normalized
rank. v1 baseline (role features decode rank) = 0.318."""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

STIM = "/Users/galyukshev/Desktop/claude/manifolds_research/workspace_mirror/data/bcs_pilot/stimuli.jsonl"


def feats(stim):
    ents = stim["latent_order"]; n = len(ents)
    rank = {e: i + 1 for i, e in enumerate(ents)}
    prompt = stim["prompt"]; L = len(prompt)
    cardpos = defaultdict(list); first = defaultdict(int); cnt = defaultdict(int)
    for c in stim["cards"]:
        cardpos[c["entity"]].append(c["presentation_slot"])
        cardpos[c["entity_b"]].append(c["presentation_slot"])
        first[c["entity"]] += 1; cnt[c["entity"]] += 1; cnt[c["entity_b"]] += 1
    rows = []
    for e in ents:
        charpos = [m.start() for m in re.finditer(r"\b" + re.escape(e) + r"\b", prompt)]
        rows.append([
            np.mean(charpos) / L,                       # mean char position (with roster)
            np.mean(cardpos[e]) / len(stim["cards"]),   # mean card slot (no roster)
            cnt[e],                                     # mention count
            first[e] / max(cnt[e], 1),                  # subj frac
            rank[e],
        ])
    return np.array(rows, float)


def decode(rows_list, label):
    X = []; y = []; g = []
    for gi, arr in enumerate(rows_list):
        r = arr[:, -1]
        yy = (r - r.min()) / (r.max() - r.min())
        for i in range(len(arr)):
            X.append(arr[i, :-1]); y.append(yy[i]); g.append(gi)
    X = np.array(X); y = np.array(y); g = np.array(g)
    keep = np.std(X, 0) > 1e-12
    Xk = X[:, keep]
    names = np.array(["char", "card", "ment", "subj"])[keep]
    oob = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(Xk, y, g):
        oob[te] = Ridge(alpha=1.0).fit(Xk[tr], y[tr]).predict(Xk[te])
    rho = abs(spearmanr(oob, y)[0])
    # single-feature char-only decode (the real position leak)
    ci = list(names).index("char") if "char" in names else None
    rho_char = None
    if ci is not None:
        oc = np.full(len(y), np.nan)
        Xc = Xk[:, [ci]]
        for tr, te in GroupKFold(5).split(Xc, y, g):
            oc[te] = Ridge(alpha=1.0).fit(Xc[tr], y[tr]).predict(Xc[te])
        rho_char = abs(spearmanr(oc, y)[0])
    print(f"  {label:18s} stimuli={len(rows_list):5d}  kept={list(names)}  "
          f"decode_spearman={rho:.3f}  char_only={rho_char:.3f}")
    return rho


def main():
    stims = [json.loads(l) for l in open(STIM)]
    ts = [s for s in stims if s.get("structure") is None and s["condition"] == "shuffle"]
    print("=== CROSS-STIMULUS rank decode from role/position features (v1 baseline=0.318) ===")
    for fam in ("ALL", "s0_zib", "s1_size", "s1_loud"):
        pool = ts if fam == "ALL" else [s for s in ts if s["family"] == fam]
        print(f"-- family={fam}")
        for diff in ("both", "easy", "hard"):
            sub = pool if diff == "both" else [s for s in pool if s["difficulty"] == diff]
            decode([feats(s) for s in sub], f"{diff}")

    # per-rank normalized position shape (endpoint check), pooled by N
    print("\n=== per-rank mean normalized CHAR position (endpoint/systematic shape) ===")
    for N in (7, 9, 12, 16):
        pool = [s for s in ts if s["n_items"] == N]
        acc = defaultdict(list)
        for s in pool:
            arr = feats(s)
            r = arr[:, -1].astype(int)
            # normalize char position within stimulus to 0..1 across its entities
            cp = arr[:, 0]
            cpn = (cp - cp.min()) / (cp.max() - cp.min() + 1e-12)
            for i in range(len(r)):
                acc[r[i]].append(cpn[i])
        means = [np.mean(acc[k]) for k in range(1, N + 1)]
        # correlation of rank vs mean-position-shape
        rr = np.arange(1, N + 1)
        rho = spearmanr(rr, means)[0]
        print(f"  N={N:2d}: " + " ".join(f"{m:.2f}" for m in means) + f"   spearman(rank,meanpos)={rho:+.3f}")


if __name__ == "__main__":
    main()
