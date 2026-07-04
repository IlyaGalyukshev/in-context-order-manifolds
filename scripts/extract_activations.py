#!/usr/bin/env python
"""Step 3/4: one hooked forward pass per stimulus → pooled acts/<model>/<id>.npz.

Idempotent per stimulus (skips existing npz). Logs span sanity for the first
few stimuli — eyeball those before trusting a full run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--model", required=True, help="short name from models.yaml")
    ap.add_argument("--stimuli", required=True, help="stimuli.jsonl")
    ap.add_argument("--out", required=True, help="acts root dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from icom.extraction.hooks import extract_pooled

    roster = {}
    mcfg = yaml.safe_load(open(args.models_config))
    for sec in ("models", "confirmatory", "exploratory"):
        roster.update(mcfg.get(sec) or {})
    spec = roster[args.model]
    is_instruct = spec.get("role", "instruct") != "base"

    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli = [json.loads(l) for l in open(args.stimuli)]
    if args.limit:
        stimuli = stimuli[: args.limit]

    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=torch.float16, attn_implementation="eager",
        device_map=args.device)
    model.eval()

    done = skipped = 0
    t0 = time.monotonic()
    for i, st in enumerate(stimuli):
        path = out_dir / f"{st['stimulus_id']}.npz"
        if path.exists():
            skipped += 1
            continue
        rec = extract_pooled(model, tok, st, is_instruct, device=args.device)
        np.savez_compressed(
            path,
            ranks=rec["ranks"], slots=rec["slots"],
            entities=json.dumps(st["latent_order"]),
            meta=json.dumps({"family": st["family"], "condition": st["condition"],
                             "n_items": st["n_items"], "content_key": st["content_key"],
                             "n_tokens": rec["n_tokens"], "model": args.model}),
            **rec["pooled"],
        )
        done += 1
        if done <= 3:
            sizes = list(rec["span_sizes"].items())[:3]
            print(f"[sanity {st['family']}/{st['condition']}] n_tokens={rec['n_tokens']} "
                  f"schemes={list(rec['pooled'].keys())} spans={sizes}", flush=True)
        if done % 50 == 0:
            print(f"[{done}/{len(stimuli)}] {(time.monotonic()-t0)/done:.2f}s/stim", flush=True)
    print(f"DONE model={args.model} done={done} skipped={skipped} "
          f"total_s={time.monotonic()-t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
