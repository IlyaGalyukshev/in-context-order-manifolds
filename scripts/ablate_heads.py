#!/usr/bin/env python
"""E6 — induction-head ablation: is the coherence increment carried by induction / previous-token
heads (the mechanism the ICLR-2026 critique uses to explain Park-style 'manifolds'), or a separate
integration circuit?

Identify the top induction heads with the standard repeated-random-sequence diagnostic, zero their
contribution during the BCS forward (a forward-pre-hook on each layer's o_proj that nulls the ablated
heads' slices), and re-measure the interior rank decode of card_mean for real vs twin. Fork (both
publishable, direct dialogue with the critique):
  * ablation cuts the TWIN decode (local chaining) but the real−twin GAP survives → integration is
    NOT on induction — coherence is a distinct circuit;
  * the GAP dies with the heads → integration rides induction.
A random-head ablation of the same count is the control.

  python scripts/ablate_heads.py --model qwen3-4b --stimuli data/e1hard_20260820/stimuli.jsonl \
      --stimuli-null data/e1hard_20260820/stimuli_null.jsonl --family s0_zib --n-items 12 \
      --topk 12 --json out/ablate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from icom.probes import cv_spearman, interior_mask, reduce


def get_decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "model.model.language_model.layers", "model.model.layers"):
        obj = model; ok = True
        for p in path.split("."):
            if not hasattr(obj, p):
                ok = False; break
            obj = getattr(obj, p)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 8:
            return m
    raise RuntimeError("no decoder layers")


@torch.no_grad()
def induction_scores(model, tok, device, seq_len=40, seed=0):
    """[n_layers, n_heads] induction score: mean attention from a token in the 2nd copy back to the
    token that FOLLOWED its match in the 1st copy (offset seq_len-1)."""
    rng = np.random.default_rng(seed)
    vocab = min(tok.vocab_size, 20000)
    seq = rng.integers(5, vocab, size=seq_len).tolist()
    ids = torch.tensor([seq + seq], device=device)
    out = model(ids, output_attentions=True)
    att = out.attentions                                        # tuple[L] of [1, H, T, T]
    L, H = len(att), att[0].shape[1]
    scores = np.zeros((L, H))
    off = seq_len - 1
    for li in range(L):
        A = att[li][0].float().cpu().numpy()                    # [H, T, T]
        idx = np.arange(seq_len + 1, 2 * seq_len)               # positions in the 2nd copy
        tgt = idx - off                                         # induction target
        scores[li] = A[:, idx, tgt].mean(axis=1)
    return scores


@torch.no_grad()
def prev_token_scores(model, tok, device, seq_len=60, seed=0):
    """[n_layers, n_heads] PREV-TOKEN score: mean attention from token i to token i−1 on a random
    sequence (the R13 second shoulder — previous-token heads are the other mechanism the induction
    critique invokes; the induction diagnostic looks BACK to the match, this looks one step back)."""
    rng = np.random.default_rng(seed)
    vocab = min(tok.vocab_size, 20000)
    seq = rng.integers(5, vocab, size=seq_len).tolist()
    ids = torch.tensor([seq], device=device)
    out = model(ids, output_attentions=True)
    att = out.attentions
    L, H = len(att), att[0].shape[1]
    scores = np.zeros((L, H))
    idx = np.arange(1, seq_len)
    for li in range(L):
        A = att[li][0].float().cpu().numpy()                    # [H, T, T]
        scores[li] = A[:, idx, idx - 1].mean(axis=1)            # attention i → i−1
    return scores


@torch.no_grad()
def induction_copy_acc(model, tok, device, seq_len=40, seed=1):
    """MANIPULATION CHECK: fraction of 2nd-copy tokens the model predicts correctly on a repeated random
    sequence (induction copying). Called with the ablation hooks active/inactive — if the ablated heads
    carry induction, this drops under induction-ablation. (o_proj-slice ablation leaves attention weights
    intact, so the FUNCTIONAL copy behavior — not the raw attention score — is the valid check.)"""
    rng = np.random.default_rng(seed)
    vocab = min(tok.vocab_size, 20000)
    seq = rng.integers(5, vocab, size=seq_len).tolist()
    ids = torch.tensor([seq + seq], device=device)
    pred = model(ids).logits[0][:-1].argmax(-1).cpu().numpy()   # next-token prediction
    tgt = np.array((seq + seq)[1:])
    reg = slice(seq_len - 1, 2 * seq_len - 1)                   # 2nd-copy region (induction-predictable)
    return float((pred[reg] == tgt[reg]).mean())


def make_ablation(layers, head_dim, ablate_set):
    """Register o_proj forward-pre-hooks that zero the ablated (layer,head) slices when state['on'].
    Returns (state, handles)."""
    state = {"on": False}
    handles = []
    for li, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        oproj = getattr(attn, "o_proj", None) if attn is not None else None
        if oproj is None:
            continue
        heads = [h for (l, h) in ablate_set if l == li]
        if not heads:
            continue

        def hook(mod, args, heads=heads):
            if not state["on"]:
                return None
            x = args[0].clone()
            for h in heads:
                x[..., h * head_dim:(h + 1) * head_dim] = 0.0
            return (x,) + args[1:]
        handles.append(oproj.register_forward_pre_hook(hook))
    return state, handles


@torch.no_grad()
def decode_cell(model, tok, stimuli, is_instruct, layer, family, n_items, device, state):
    """cv_spearman of interior rank at `layer`/card_mean over these stimuli (ablation per state)."""
    from icom.extraction.hooks import extract_pooled
    Xs, ys, gs = [], [], []
    for gi, st in enumerate(stimuli):
        if st.get("family") != family or int(st["n_items"]) != int(n_items):
            continue
        rec = extract_pooled(model, tok, st, is_instruct, device=device)
        if "card_mean" not in rec["pooled"]:
            continue
        X = rec["pooled"]["card_mean"][:, layer, :].astype(np.float32)
        ranks = rec["ranks"]
        mask = interior_mask(ranks, len(ranks)) & np.isfinite(X).all(axis=1)
        if mask.sum() < 2:
            continue
        Xs.append(X[mask]); ys.append((ranks[mask] - 1) / (len(ranks) - 1))
        gs.append(np.full(int(mask.sum()), gi))
    if not Xs:
        return float("nan")
    return float(cv_spearman(reduce(np.concatenate(Xs)), np.concatenate(ys), np.concatenate(gs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stimuli", required=True); ap.add_argument("--stimuli-null", required=True)
    ap.add_argument("--family", default="s0_zib"); ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--layer", type=int, default=None, help="decode layer; default ~50% depth")
    ap.add_argument("--topk", type=int, default=12); ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    roster = {}
    for sec in ("models", "confirmatory", "exploratory"):
        roster.update(yaml.safe_load(open(args.models_config)).get(sec) or {})
    spec = roster[args.model]; is_instruct = spec.get("role", "instruct") != "base"
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=torch.float16, attn_implementation="eager", device_map=args.device).eval()
    layers = get_decoder_layers(model); n_layers = len(layers)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    layer = args.layer if args.layer is not None else max(2, n_layers // 2)

    scores = induction_scores(model, tok, args.device, seed=args.seed)
    flat = [(li, h, scores[li, h]) for li in range(scores.shape[0]) for h in range(scores.shape[1])]
    flat.sort(key=lambda t: -t[2])
    top = set((li, h) for li, h, _ in flat[:args.topk])
    pscores = prev_token_scores(model, tok, args.device, seed=args.seed)          # R13 prev-token shoulder
    pflat = sorted([(li, h, pscores[li, h]) for li in range(pscores.shape[0]) for h in range(pscores.shape[1])],
                   key=lambda t: -t[2])
    ptop = set((li, h) for li, h, _ in pflat[:args.topk])
    rng = np.random.default_rng(args.seed)
    allhh = [(li, h) for li in range(scores.shape[0]) for h in range(scores.shape[1])]
    rand = set(allhh[i] for i in rng.choice(len(allhh), args.topk, replace=False))
    print(f"top induction heads: {[(li, h, round(float(s),3)) for li,h,s in flat[:5]]}", flush=True)
    print(f"top prev-token heads: {[(li, h, round(float(s),3)) for li,h,s in pflat[:5]]}", flush=True)

    def _load(path):                                          # filter to (family, N) BEFORE limiting —
        S = [json.loads(l) for l in open(path)]               # else --limit can grab only the wrong-N block
        S = [s for s in S if s.get("family") == args.family and int(s["n_items"]) == args.n_items]
        return S[: args.limit]
    real, twin = _load(args.stimuli), _load(args.stimuli_null)
    print(f"stimuli: real={len(real)} twin={len(twin)} (family={args.family} N={args.n_items})", flush=True)

    def gap(state, tag):
        state["on"] = tag != "intact"
        r = decode_cell(model, tok, real, is_instruct, layer, args.family, args.n_items, args.device, state)
        t = decode_cell(model, tok, twin, is_instruct, layer, args.family, args.n_items, args.device, state)
        state["on"] = False
        return dict(tag=tag, real=round(r, 3), twin=round(t, 3), gap=round(r - t, 3))

    copy0 = induction_copy_acc(model, tok, args.device)                            # intact copy accuracy
    st_ind, h_ind = make_ablation(layers, head_dim, top)
    intact = gap(st_ind, "intact")
    abl_ind = gap(st_ind, "ablate_induction")
    st_ind["on"] = True; copy_ind = induction_copy_acc(model, tok, args.device); st_ind["on"] = False
    for h in h_ind:
        h.remove()
    st_prev, h_prev = make_ablation(layers, head_dim, ptop)                         # R13: prev-token shoulder
    abl_prev = gap(st_prev, "ablate_prevtoken")
    st_prev["on"] = True; copy_prev = induction_copy_acc(model, tok, args.device); st_prev["on"] = False
    for h in h_prev:
        h.remove()
    st_rnd, h_rnd = make_ablation(layers, head_dim, rand)
    abl_rnd = gap(st_rnd, "ablate_random")
    for h in h_rnd:
        h.remove()

    # MANIPULATION CHECK: induction-copy accuracy must fall under induction-ablation (heads are load-bearing)
    manip = dict(copy_intact=round(copy0, 3), copy_ablate_induction=round(copy_ind, 3),
                 copy_ablate_prevtoken=round(copy_prev, 3),
                 induction_ablation_worked=bool(copy_ind < copy0 - 0.05))
    res = dict(model=args.model, family=args.family, n_items=args.n_items, layer=layer,
               topk=args.topk, conditions=[intact, abl_ind, abl_prev, abl_rnd], manipulation=manip)
    for c in res["conditions"]:
        print(f"{args.model} {args.family} N{args.n_items} L{layer} {c['tag']:16s} | "
              f"real={c['real']} twin={c['twin']} gap={c['gap']}", flush=True)
    print(f"MANIP-CHECK copy-acc: intact={manip['copy_intact']} abl-ind={manip['copy_ablate_induction']} "
          f"abl-prev={manip['copy_ablate_prevtoken']} | induction-ablation worked={manip['induction_ablation_worked']}", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))
        print(f"wrote -> {args.json}")


if __name__ == "__main__":
    main()
