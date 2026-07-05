#!/usr/bin/env python
"""Pilot Track-A first look: does per-stimulus manifold quality predict
per-stimulus behavioral accuracy WITHIN family × condition cells?

Per stimulus: quality = q_content_partial at (--scheme, --variant) and the
family's best mid-layer band (mean over --layers); behavior = mean pairwise
score, reconstruction tau, adjacency accuracy. Reports within-cell Spearman
correlations (quality is |q| — PC1 sign is arbitrary). This is a LOOK, not
the preregistered mixed-effects analysis (Stage 2); shuffle cells are the
ones that matter.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--battery-dir", required=True)
    ap.add_argument("--models", default="qwen3-4b,olmo3-7b-inst")
    ap.add_argument("--scheme", default="last_token")
    ap.add_argument("--variant", default="posproj")
    ap.add_argument("--layers", default="10-24", help="layer band lo-hi (inclusive)")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.layers.split("-"))
    geo = pd.read_parquet(args.geometry)
    geo = geo[(geo["scheme"] == args.scheme) & (geo["variant"] == args.variant)
              & geo["layer"].between(lo, hi)]
    qual = (geo.assign(abs_q=geo["q_content_partial"].abs())
            .groupby(["model", "stimulus_id", "family", "condition"])["abs_q"]
            .mean().reset_index())

    bats = []
    for m in args.models.split(","):
        bats.append(pd.read_parquet(f"{args.battery_dir}/battery_{m}.parquet"))
    bat = pd.concat(bats)
    beh = bat.groupby(["model", "stimulus_id"]).apply(
        lambda g: pd.Series({
            "pairwise": g.loc[g.q_family == "pairwise", "score"].mean(),
            "recon_tau": g.loc[g.q_family == "reconstruction", "tau"].mean(),
            "adjacency": g.loc[g.q_family == "adjacency", "score"].mean(),
            "parse_fail": g["parse_failed"].mean(),
        }), include_groups=False).reset_index()

    df = qual.merge(beh, on=["model", "stimulus_id"])
    print(f"joined {len(df)} stimulus-model rows "
          f"(scheme={args.scheme}, variant={args.variant}, L{lo}-{hi})")
    print("\n=== within-cell Spearman(|q_content_partial|, behavior) ===")
    for (m, fam, cond), g in df.groupby(["model", "family", "condition"]):
        line = f"{m:14s} {fam:10s} {cond:8s} n={len(g):3d} "
        for metric in ("pairwise", "recon_tau", "adjacency"):
            gg = g.dropna(subset=[metric])
            if len(gg) >= 10 and gg[metric].std() > 0 and gg["abs_q"].std() > 0:
                rho, p = spearmanr(gg["abs_q"], gg[metric])
                line += f" {metric}: ρ={rho:+.2f} (p={p:.3f})"
            else:
                line += f" {metric}: --"
        print(line)

    print("\n=== behavioral means per cell (for the G0 shuffle-vs-forward gap) ===")
    cell = df.groupby(["model", "family", "condition"])[
        ["pairwise", "recon_tau", "adjacency", "parse_fail"]].mean().round(3)
    print(cell.to_string())


if __name__ == "__main__":
    main()
