#!/usr/bin/env python
"""Data-level confound audit (NO activations, CPU, runnable now).

For a stimuli.jsonl, compute per-entity ROLE features that must NOT encode rank
if the dataset is clean:
  mention_count  — # cards the entity appears in
  subj_frac      — fraction of its cards where it is named first
  mean_pos       — mean card position of its mentions (normalized)
Then cross-stimulus, try to decode latent rank from ONLY these role features.

  v1 (linear chain): role features predict rank (endpoint/role confound) -> HIGH.
  v2 (BCS):          role features are rank-balanced by construction    -> ~NULL.

This proves at the DATA level that the redesign removed the artifact the
interior-only activation control exposed — without needing GPU.

  python scripts/audit_confounds.py --stimuli <path> [--family relational --condition shuffle]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


def role_features(stim):
    ents = stim["latent_order"]
    rank = {e: i + 1 for i, e in enumerate(ents)}
    cnt = defaultdict(int); first = defaultdict(int); pos = defaultdict(list)
    for si, c in enumerate(stim["cards"]):
        for role, e in (("first", c["entity"]), ("second", c.get("entity_b"))):
            if e is None:
                continue
            cnt[e] += 1; pos[e].append(si)
            if role == "first":
                first[e] += 1
    rows = []
    for e in ents:
        m = cnt[e] or 1
        rows.append((cnt[e], first[e] / m, np.mean(pos[e]) / max(len(stim["cards"]) - 1, 1), rank[e]))
    return rows  # (mention_count, subj_frac, mean_pos, rank)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--family", default=None)
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--n-max", type=int, default=None, help="filter n_items")
    args = ap.parse_args()

    X, y, groups = [], [], []
    per_feat = defaultdict(list)  # per-stimulus corr of each feature with rank
    gi = 0
    for l in open(args.stimuli):
        s = json.loads(l)
        if s.get("condition") != args.condition:
            continue
        if args.family and s.get("family") != args.family:
            continue
        if args.n_max and s.get("n_items") != args.n_max:
            continue
        feats = role_features(s)
        arr = np.array(feats, float)
        ranks = arr[:, 3]
        for j, name in enumerate(("mention_count", "subj_frac", "mean_pos")):
            if np.std(arr[:, j]) > 0:
                per_feat[name].append(abs(spearmanr(arr[:, j], ranks)[0]))
            else:
                per_feat[name].append(0.0)
        yy = (ranks - ranks.min()) / (ranks.max() - ranks.min())
        for r in range(len(feats)):
            X.append(arr[r, :3]); y.append(yy[r]); groups.append(gi)
        gi += 1

    X = np.array(X); y = np.array(y); groups = np.array(groups)
    print(f"stimuli={gi}  items={len(y)}  (family={args.family} condition={args.condition})")
    print("--- per-feature |spearman(feature, rank)| averaged over stimuli (want ~0):")
    for name, v in per_feat.items():
        print(f"    {name:14s} {np.mean(v):.3f}")

    # cross-stimulus decode of rank from role features (want ~0 for clean data)
    if gi >= 5 and np.std(X, 0).min() > 0:
        oob = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, groups):
            oob[te] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
        rho = abs(spearmanr(oob, y)[0])
        print(f"--- CROSS-STIMULUS rank decodable from ROLE FEATURES: spearman={rho:.3f}")
        print("    (v1 linear chain expected HIGH = confound; v2 BCS expected ~0 = clean)")
    else:
        print("--- role features constant across ranks (perfectly balanced) => rank not decodable")


if __name__ == "__main__":
    main()
