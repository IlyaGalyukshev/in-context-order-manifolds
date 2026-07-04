#!/usr/bin/env python
"""Build and commit the nonce-entity pool: syllable generator + wordfreq screen
+ tokenizer screen on every roster tokenizer. Run on the GPU worker (needs
transformers + the HF cache); the output artifact is committed so
generate_dataset.py never needs transformers.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from icom.generator.entities import build_vocabulary, check_tokenization


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--tokenizer-ids", default="Qwen/Qwen3-1.7B,allenai/Olmo-3-1025-7B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    vocab = build_vocabulary(seed=args.seed, size=args.size)
    report = {}
    for tid in args.tokenizer_ids.split(","):
        before = len(vocab)
        vocab = check_tokenization(vocab, AutoTokenizer.from_pretrained(tid.strip()))
        report[tid] = {"before": before, "after": len(vocab)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"version": "v1", "date": str(date.today()), "seed": args.seed,
                 "requested": args.size, "tokenizer_screen": report,
                 "wordfreq_screen": "zipf>1.5 excluded"},
        "names": vocab,
    }, indent=2))
    print(f"wrote {len(vocab)} names → {out}")


if __name__ == "__main__":
    main()
