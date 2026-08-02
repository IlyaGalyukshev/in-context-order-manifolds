#!/usr/bin/env python
"""Build auxiliary SOURCE-AXIS contrast sets for the pretrained-axis-reuse test (G10).

These are NOT BCS stimuli — they are small contrastive-pair sets used to extract a
pretrained direction (CAA/RepE style: a direction = mean(act[positive] − act[negative]))
that the in-context BCS order manifold is then aligned to / causally tested against.

Three axes, each written as JSONL of {axis, positive, negative, meta}:
  - magnitude: MATCHED number pairs (large vs small) so a difference-of-means
    direction averages out digit identity (avoids the per-digit landmine; the
    magnitude signal is the *comparative*, not the digit string).
  - space:     left-vs-right placements of the same entities (Tehenan-style).
  - size:      curated small↔big antonym words (Grand-style semantic magnitude).

  python scripts/build_source_axes.py --out docs/samples/source_axes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20260724

# small↔big antonym seed list (clear, imageable size contrasts; Grand et al. axis)
SIZE_PAIRS = [
    ("ant", "whale"), ("pebble", "boulder"), ("mouse", "elephant"), ("seed", "tree"),
    ("drop", "ocean"), ("crumb", "loaf"), ("coin", "wheel"), ("button", "plate"),
    ("marble", "planet"), ("dot", "mountain"), ("flea", "horse"), ("grain", "dune"),
    ("thimble", "bucket"), ("twig", "log"), ("spark", "bonfire"), ("cell", "body"),
    ("atom", "galaxy"), ("nail", "beam"), ("berry", "melon"), ("chick", "ostrich"),
]


def magnitude_pairs(rng, n=60, lo=1, hi=999):
    """Matched (large, small) integer pairs spanning the range; the difference of
    their activations estimates a magnitude direction without digit-identity bias."""
    out = []
    for _ in range(n):
        a, b = rng.integers(lo, hi + 1), rng.integers(lo, hi + 1)
        while a == b:
            b = rng.integers(lo, hi + 1)
        large, small = (int(a), int(b)) if a > b else (int(b), int(a))
        out.append({
            "axis": "magnitude",
            "positive": f"The value is {large}.",
            "negative": f"The value is {small}.",
            "meta": {"large": large, "small": small,
                     "comparative": f"Which number is larger: {small} or {large}?",
                     "comparative_key": str(large)},
        })
    return out


def space_pairs(rng, entities, n=60):
    """Left-vs-right placements of the SAME two entities (order swapped) so the
    contrast isolates the left/right spatial axis (Tehenan-style)."""
    out = []
    for _ in range(n):
        i, j = (int(x) for x in rng.choice(len(entities), size=2, replace=False))
        a, b = entities[i], entities[j]
        out.append({
            "axis": "space",
            "positive": f"The {a} is to the left of the {b}.",
            "negative": f"The {a} is to the right of the {b}.",
            "meta": {"a": a, "b": b},
        })
    return out


def size_pairs():
    return [{"axis": "size",
             "positive": f"The {big} is very big.",
             "negative": f"The {small} is very small.",
             "meta": {"big": big, "small": small}} for small, big in SIZE_PAIRS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/samples/source_axes")
    ap.add_argument("--pool", default="data/pools/entities_v1.json")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    ents = json.load(open(args.pool))["names"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    sets = {"magnitude": magnitude_pairs(rng, args.n),
            "space": space_pairs(rng, ents, args.n),
            "size": size_pairs()}
    for name, rows in sets.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} contrast pairs -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
