#!/usr/bin/env python
"""M1 repeat-read extraction: k re-presentations per stimulus → pooled [N, k, L+1, D] fp16 per
locus, the input crossnobis / whitened-cvRDM (probe_crossnobis.py) needs. Idempotent per stimulus.
Mirrors extract_activations.py (same model load, meta, coords, is_null tagging) but adds the k-axis.

  python scripts/extract_repeat.py --model qwen3-4b --stimuli data/loci_qwen/stimuli.jsonl \
      --out results/repeat_smoke --k 8 --loci readout,card_mean --limit 24
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k", type=int, default=8, help="repeat reads per stimulus")
    ap.add_argument("--loci", default="readout,card_mean",
                    help="comma list of schemes to pool/store (readout/card_mean/last_token/name)")
    ap.add_argument("--store", default="rdm+mean", choices=["reads", "rdm", "rdm+mean"],
                    help="reads = raw [N,k,L,D] (8x disk, dev/smoke); rdm = crossnobis RDM [N,N,L] "
                         "on the fly (whitened-RSA at scale); rdm+mean = + k-mean [N,L,D] (decode/cPCA)")
    ap.add_argument("--n-splits", type=int, default=20, help="crossnobis CV splits when store!=reads")
    ap.add_argument("--probe", action="store_true",
                    help="M1(b): neutral-probe locus — read each entity via k 'Consider the {e}.' "
                         "paraphrases (KV-cached from the card block) instead of roster re-presentations")
    ap.add_argument("--prefix", default="none", choices=["none", "order", "mention"],
                    help="E5 task-expectation prefix prepended to the card block")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from icom.extraction.repeat import extract_pooled_repeat, extract_probe_repeat

    roster = {}
    mcfg = yaml.safe_load(open(args.models_config))
    for sec in ("models", "confirmatory", "exploratory"):
        roster.update(mcfg.get(sec) or {})
    spec = roster[args.model]
    is_instruct = spec.get("role", "instruct") != "base"
    loci = set(args.loci.split(",")) if args.loci else None

    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli = [json.loads(l) for l in open(args.stimuli)]
    import random
    random.Random(20260724).shuffle(stimuli)                   # unbiased partial-run sample
    if args.limit:
        stimuli = stimuli[: args.limit]

    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=torch.float16, attn_implementation="eager",
        device_map=args.device).eval()

    done = skipped = 0
    t0 = time.monotonic()
    for st in stimuli:
        path = out_dir / f"{st['stimulus_id']}.npz"
        if path.exists():
            skipped += 1
            continue
        prefix = {"none": "", "order": "After reading, you will be asked about the ORDER of these entities.",
                  "mention": "After reading, you will be asked WHICH entities were mentioned."}[args.prefix]
        if args.probe:
            rec = extract_probe_repeat(model, tok, st, is_instruct, k=args.k, device=args.device)
        else:
            rec = extract_pooled_repeat(model, tok, st, is_instruct, k=args.k,
                                        device=args.device, root_seed=args.seed, loci=loci, prefix=prefix)
        extra = {}
        if "coord_x" in st:
            extra["coord_x"] = np.array([st["coord_x"][e] for e in st["latent_order"]])
            extra["coord_y"] = np.array([st["coord_y"][e] for e in st["latent_order"]])
        if "cyclic_pos" in st:
            extra["cyclic_pos"] = np.array([st["cyclic_pos"][e] for e in st["latent_order"]])
        # --store compaction: raw reads (8x disk) vs on-the-fly crossnobis RDM [N,N,L] (+k-mean)
        arrays = {}
        if args.store == "reads":
            arrays.update(rec["pooled"])                       # {scheme: [N,k,L+1,D]}
        else:
            from icom.probes.crossnobis import crossnobis_rdm
            for s, arr in rec["pooled"].items():               # arr [N,k,L+1,D]
                Lp = arr.shape[2]
                rdm = np.stack([crossnobis_rdm(arr[:, :, l, :].astype(np.float64),
                                               n_splits=args.n_splits, seed=0) for l in range(Lp)],
                               axis=2).astype(np.float32)       # [N,N,L+1]
                arrays[f"rdm_{s}"] = rdm
                if args.store == "rdm+mean":
                    arrays[f"mean_{s}"] = arr.mean(axis=1).astype(np.float16)  # [N,L+1,D]
        np.savez_compressed(
            path,
            ranks=rec["ranks"], entities=json.dumps(st["latent_order"]),
            meta=json.dumps({"family": st["family"], "condition": st["condition"],
                             "n_items": st["n_items"], "content_key": st["content_key"],
                             "structure": st.get("structure", "total_order"),
                             "is_null": bool(st.get("is_null", False) or st.get("incoherent", False)),
                             "difficulty": st.get("difficulty"), "n_reads": rec["n_reads"],
                             "declared": st.get("declared"), "prefix": args.prefix,   # E2 declared / E5 prefix
                             "store": args.store, "model": args.model}),
            **arrays, **extra,
        )
        done += 1
        if done <= 3:
            sh = {s: v.shape for s, v in arrays.items() if s.startswith(("rdm_", "mean_")) or s in loci}
            print(f"[sanity {st['family']}/{st['condition']} null={st.get('incoherent', False)}] "
                  f"k={rec['n_reads']} store={args.store} shapes={sh}", flush=True)
        if done % 25 == 0:
            print(f"[{done}/{len(stimuli)}] {(time.monotonic()-t0)/done:.2f}s/stim", flush=True)
    print(f"DONE model={args.model} done={done} skipped={skipped} "
          f"total_s={time.monotonic()-t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
