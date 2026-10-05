#!/usr/bin/env python
"""Query-local assembly: is the order assembled for an ORDER question about a pair, beyond the pair
merely being named?

For every interior pair (X, Y) of a stimulus we append one of two questions with X/Y order
randomised — ORDER: "which is earlier in the <rel>-order: the X or the Y?"; NON-ORDER: "did both the
X and the Y appear above?" — and read the hidden state of the final prompt token (after the chat
template's generation prompt) at a few depths. A logistic decoder (PCA-64) predicts "is X earlier",
grouped K-fold over stimuli, separately per question type and for real vs twin stimuli. The layer is
cross-fitted (picked on half of the stimuli, scored on the other). The order − non-order AUC gap,
with a stimulus bootstrap, is the query-local signature; mentioning the pair alone is the non-order arm.

  python scripts/query_pair_decode.py --model-path Qwen/Qwen3-4B --tag qwen3-4b \
      --stimuli <core/stimuli.jsonl> --stimuli-null <core/stimuli_null.jsonl> \
      --families s0_zib,s1_loud --n-items 16 --out results/query_local
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

RELWORD = {"s0_zib": "zib", "s0_quomp": "quomp", "s1_size": "size", "s1_loud": "loudness", "s1_heat": "heat"}


def fmt(tok, prompt, question):
    user = prompt + "\n\n" + question
    if tok.chat_template is None:
        return user + "\nAnswer:"
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template([{"role": "user", "content": user}], enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": user}], **kw)


@torch.no_grad()
def final_states(model, tok, texts, layers, device, bs):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1 if tok.padding_side == "right" else torch.full((enc["input_ids"].shape[0],), enc["input_ids"].shape[1] - 1, device=device)
        rows = torch.arange(enc["input_ids"].shape[0], device=device)
        out.append(torch.stack([hs[l][rows, last].float().cpu() for l in layers], 1).numpy().astype(np.float16))
    return np.concatenate(out, 0)                                      # [n, len(layers), D]


def extract(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16, attn_implementation="eager",
                                                 device_map=args.device, local_files_only=True).eval()
    cfg = getattr(model.config, "text_config", model.config)
    nl = cfg.num_hidden_layers
    layers = sorted({int(round(f * nl)) for f in (0.3, 0.4, 0.5, 0.6, 0.7)})
    out = Path(args.out) / args.tag; out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for path, is_null in ((args.stimuli, False), (args.stimuli_null, True)):
        S = [json.loads(l) for l in open(path)]
        S = [s for s in S if s["family"] in args.families.split(",") and int(s["n_items"]) == args.n_items][: args.limit]
        for s in S:
            fn = out / f"{s['stimulus_id']}{'_null' if is_null else ''}.npz"
            if fn.exists():
                continue
            er = s["entity_ranks"]; N = int(s["n_items"]); rel = RELWORD.get(s["family"], "")
            inter = [e for e in s["latent_order"] if 3 <= er[e] <= N - 2]
            pairs = [(a, b) for i, a in enumerate(inter) for b in inter[i + 1:]]
            recs = {"order": [], "nonorder": []}
            for a, b in pairs:
                x, y = (a, b) if rng.random() < 0.5 else (b, a)
                recs["order"].append((x, y, f"By the relations above, which is earlier in the {rel}-order: the {x} or the {y}? Reply with only one entity name."))
                recs["nonorder"].append((x, y, f"Did both the {x} and the {y} appear in the text above? Reply with only yes or no."))
            arrays = {}
            for qt, rr in recs.items():
                arrays[qt] = final_states(model, tok, [fmt(tok, s["prompt"], q) for _, _, q in rr], layers, args.device, args.batch)
                arrays[qt + "_xearlier"] = np.array([int(er[x] < er[y]) for x, y, _ in rr])
                arrays[qt + "_dist"] = np.array([abs(er[x] - er[y]) for x, y, _ in rr])
            np.savez_compressed(fn, **arrays, layers=np.array(layers),
                                meta=json.dumps({"family": s["family"], "n_items": N, "is_null": is_null,
                                                 "content_key": s["content_key"]}))
        print(f"{args.tag}: extracted {len(S)} {'twin' if is_null else 'real'} stimuli", flush=True)


def decode(args):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    d = Path(args.out) / args.tag
    res = {}
    for is_null in (False, True):
        files = sorted(f for f in d.glob("*.npz") if f.stem.endswith("_null") == is_null)
        if not files:
            continue
        Z = [np.load(f, allow_pickle=False) for f in files]
        half = np.array([int(hashlib.md5(json.loads(str(z["meta"]))["content_key"].encode()).hexdigest(), 16) % 2 for z in Z])
        nL = len(Z[0]["layers"])
        for qt in ("order", "nonorder"):
            # per-stimulus OOF scores for every layer → AUC by layer, cross-fitted layer choice
            X = [np.concatenate([z[qt][:, li, :] for z in Z]).astype(np.float32) for li in range(nL)]
            y = np.concatenate([z[qt + "_xearlier"] for z in Z]); g = np.concatenate([np.full(len(z[qt + "_xearlier"]), i) for i, z in enumerate(Z)])
            oof = np.zeros((nL, len(y)))
            for li in range(nL):
                for tr, te in GroupKFold(5).split(X[li], y, g):
                    p = PCA(n_components=min(64, len(tr) - 1), random_state=0).fit(X[li][tr])
                    clf = LogisticRegression(max_iter=2000).fit(p.transform(X[li][tr]), y[tr])
                    oof[li, te] = clf.decision_function(p.transform(X[li][te]))
            hs = half[g]
            pick = {h: int(np.argmax([roc_auc_score(y[hs == h], oof[li, hs == h]) for li in range(nL)])) for h in (0, 1)}
            score = np.where(hs == 0, oof[pick[1]], oof[pick[0]])      # each half scored at the other's best layer
            res[(qt, is_null)] = (y, score, g)
            print(f"{args.tag} {'twin' if is_null else 'real'} {qt:8s} layers {[int(Z[0]['layers'][pick[h]]) for h in (0, 1)]}  "
                  f"AUC={roc_auc_score(y, score):.3f}", flush=True)
    # bootstrap over stimuli for order − non-order (real) and coherence real − twin (order)
    rng = np.random.default_rng(args.seed)

    def auc_boot(y, s, g, B=1000):
        from sklearn.metrics import roc_auc_score as A
        u = np.unique(g); out = []
        for _ in range(B):
            pick = rng.choice(u, len(u), replace=True); m = np.concatenate([np.where(g == k)[0] for k in pick])
            if len(np.unique(y[m])) == 2:
                out.append(A(y[m], s[m]))
        return np.array(out)
    summ = {}
    for key, (y, s, g) in res.items():
        b = auc_boot(y, s, g)
        from sklearn.metrics import roc_auc_score as A
        summ[f"{key[0]}_{'twin' if key[1] else 'real'}"] = dict(auc=float(A(y, s)), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    if ("order", False) in res and ("nonorder", False) in res:
        (y1, s1, g1), (y2, s2, g2) = res[("order", False)], res[("nonorder", False)]
        u = np.unique(g1); diffs = []
        from sklearn.metrics import roc_auc_score as A
        for _ in range(1000):
            pick = rng.choice(u, len(u), replace=True)
            m1 = np.concatenate([np.where(g1 == k)[0] for k in pick]); m2 = np.concatenate([np.where(g2 == k)[0] for k in pick])
            if len(np.unique(y1[m1])) == 2 and len(np.unique(y2[m2])) == 2:
                diffs.append(A(y1[m1], s1[m1]) - A(y2[m2], s2[m2]))
        diffs = np.array(diffs)
        summ["order_minus_nonorder"] = dict(diff=float(A(y1, s1) - A(y2, s2)), ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                            p_gt0=float((diffs > 0).mean()))
    print(json.dumps({"tag": args.tag, **summ}, indent=1), flush=True)
    json.dump(summ, open(Path(args.out) / f"{args.tag}_summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path"); ap.add_argument("--tag", required=True)
    ap.add_argument("--stimuli"); ap.add_argument("--stimuli-null")
    ap.add_argument("--families", default="s0_zib,s1_loud"); ap.add_argument("--n-items", type=int, default=16)
    ap.add_argument("--limit", type=int, default=1000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--decode-only", action="store_true")
    args = ap.parse_args()
    if not args.decode_only:
        extract(args)
    decode(args)


if __name__ == "__main__":
    main()
