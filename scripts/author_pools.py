#!/usr/bin/env python
"""Author the event-predicate pool via OpenRouter (offline step; output committed).

Loop: author (claude-sonnet-5) proposes → cheap local leak screen → supervisor
(gpt-5.5) adversarially reviews → dedup → repeat until target size or call cap.
Then a tokenizer screen (all roster tokenizers) and the final pool JSON.

Writes:
  <out>/event_predicates_v1.json   — the pool + full provenance metadata
  <out>/review_log_v1.json        — every supervisor verdict (for eyeballing)

Run inside the worker container:
  python scripts/author_pools.py --config configs/generation.yaml \
      --out /workspace/manifolds/data/pools --target 400
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml

from icom.generator import llm_assist as la


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/generation.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--max-author-calls", type=int, default=25)
    ap.add_argument("--tokenizer-ids", default="Qwen/Qwen3-1.7B,allenai/Olmo-3-1025-7B")
    ap.add_argument("--max-pred-tokens", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["llm_assist"]
    target = args.target or cfg["predicate_pool_size"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kept: list[str] = []
    all_verdicts: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def add_usage(u: dict) -> None:
        usage_total["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += u.get("completion_tokens", 0)

    calls = 0
    while len(kept) < target and calls < args.max_author_calls:
        theme = la.AUTHOR_THEMES[calls % len(la.AUTHOR_THEMES)]
        calls += 1
        cands, u = la.author_event_predicates(60, cfg["author_model"], existing=kept, theme=theme)
        add_usage(u)
        fresh = [c for c in dict.fromkeys(cands) if c not in kept]
        pre = la.cheap_local_screen(fresh)
        print(f"[round {calls}] authored={len(cands)} fresh={len(fresh)} after_local_screen={len(pre)}", flush=True)
        if not pre:
            continue
        approved, verdicts, u = la.review_pool(pre, cfg["reviewer_model"])
        add_usage(u)
        all_verdicts.extend(verdicts)
        kept.extend(p for p in approved if p not in kept)
        rej = len(pre) - len(approved)
        print(f"[round {calls}] supervisor kept {len(approved)}, rejected {rej} → pool={len(kept)}", flush=True)

    # tokenizer screen on all roster tokenizers
    from transformers import AutoTokenizer

    survivors = kept
    tok_report = {}
    for tid in args.tokenizer_ids.split(","):
        tok = AutoTokenizer.from_pretrained(tid.strip())
        before = len(survivors)
        survivors = [
            p for p in survivors
            if len(tok(" " + p, add_special_tokens=False)["input_ids"]) <= args.max_pred_tokens
        ]
        tok_report[tid] = {"before": before, "after": len(survivors)}
        print(f"[tokscreen] {tid}: {before} → {len(survivors)}", flush=True)

    pool = {
        "meta": {
            "version": "v1",
            "date": str(date.today()),
            "author_model": cfg["author_model"],
            "reviewer_model": cfg["reviewer_model"],
            "prompt_version": "author-v1/reviewer-v1",
            "target": target,
            "authored_rounds": calls,
            "supervisor_verdicts": len(all_verdicts),
            "tokenizer_screen": tok_report,
            "usage_tokens": usage_total,
        },
        "predicates": sorted(survivors),
    }
    (out / "event_predicates_v1.json").write_text(json.dumps(pool, indent=2))
    (out / "review_log_v1.json").write_text(json.dumps(all_verdicts, indent=2))
    print(f"\nFINAL pool={len(survivors)} predicates → {out}/event_predicates_v1.json")
    print(f"usage: {usage_total}")


if __name__ == "__main__":
    main()
