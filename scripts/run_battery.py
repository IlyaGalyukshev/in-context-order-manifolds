#!/usr/bin/env python
"""Step 2/4: behavioral battery → results/battery_<model>.jsonl (+parquet).

Idempotent per stimulus (resumes from existing jsonl). Logs sample Q/A pairs
every --sample-every stimuli — eyeball those during the run, per workflow
rules. Summary stats printed at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=25)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from icom.battery.client import BatteryRunner
    from icom.battery.scoring import score_row

    roster = {}
    for sec in ("models", "confirmatory", "exploratory"):
        roster.update(yaml.safe_load(open(args.models_config)).get(sec) or {})
    spec = roster[args.model]

    stimuli = [json.loads(l) for l in open(args.stimuli)]
    if args.limit:
        stimuli = stimuli[: args.limit]
    questions = defaultdict(list)
    for l in open(args.questions):
        q = json.loads(l)
        questions[q["stimulus_content_key"]].append(q)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"battery_{args.model}.jsonl"
    done_stims = set()
    if out_path.exists():
        for l in open(out_path):
            done_stims.add(json.loads(l)["stimulus_id"])
        # only stimuli with a full battery count as done
        counts = defaultdict(int)
        for l in open(out_path):
            counts[json.loads(l)["stimulus_id"]] += 1
        done_stims = {s for s, c in counts.items()
                      if c >= len(questions[next(st["content_key"] for st in stimuli
                                                 if st["stimulus_id"] == s)])}

    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=torch.float16, attn_implementation="eager",
        device_map=args.device)
    model.eval()
    runner = BatteryRunner(model, tok, is_instruct=spec.get("role", "instruct") != "base",
                           batch_size=args.batch_size, device=args.device)

    t0, n_done = time.monotonic(), 0
    with open(out_path, "a") as f:
        for st in stimuli:
            if st["stimulus_id"] in done_stims:
                continue
            qs = questions[st["content_key"]]
            raws = runner.run_stimulus(st, qs)
            raw_by_qid = {r["qid"]: r for r in raws}
            for q in qs:
                raw = raw_by_qid[q["qid"]]
                scored = score_row(q, raw["completion"], st["latent_order"],
                                   raw["logit_margin"])
                f.write(json.dumps({
                    "stimulus_id": st["stimulus_id"], "content_key": st["content_key"],
                    "model": args.model, "family": st["family"],
                    "condition": st["condition"], "n_items": st["n_items"],
                    "qid": q["qid"], "q_family": q["family"],
                    "rank_distance": q.get("rank_distance"),
                    "is_endpoint": q.get("is_endpoint"),
                    "span_location": q.get("span_location"),
                    "completion": raw["completion"],
                    "logit_margin": raw["logit_margin"], **scored,
                }) + "\n")
            f.flush()
            n_done += 1
            if n_done % args.sample_every == 1:
                ex = raws[0]
                q0 = next(q for q in qs if q["qid"] == ex["qid"])
                print(f"[sample {st['family']}/{st['condition']}] Q: {q0['text'][:80]}\n"
                      f"  A: {ex['completion'][:100]!r}", flush=True)
            if n_done % 25 == 0:
                dt = (time.monotonic() - t0) / n_done
                print(f"[{n_done} stimuli] {dt:.1f}s/stim, ETA "
                      f"{(len(stimuli) - len(done_stims) - n_done) * dt / 60:.0f}min", flush=True)

    # summary
    import pandas as pd
    df = pd.read_json(out_path, lines=True)
    df.to_parquet(out_dir / f"battery_{args.model}.parquet")
    print("\n=== summary (mean score / parse-fail rate) ===")
    g = df.groupby(["family", "condition", "q_family"]).agg(
        score=("score", "mean"), pf=("parse_failed", "mean"), n=("score", "size"))
    print(g.to_string())
    print(f"DONE battery model={args.model} rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
