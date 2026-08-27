#!/usr/bin/env python
"""E9(b): entity-substitution activation patch (CIK-protocol) — is an entity's IN-CONTEXT rank
causally carried by its mid-depth residual-stream activation?

For an interior entity A, OVERWRITE A's residual stream (at A's query mention tokens, mid layer L)
with a donor activation captured from the block, then read the model's ANSWERED position for A:
  * donor = B (another interior entity, large true-rank gap): does A's answer move TOWARD B's rank?
  * donor = C (a random third interior entity): matched control for "any patch perturbs the answer".
  * donor = None: baseline.
Decisive read: patchB moves A's answer toward rank_B significantly more than the patchC control
=> the entity's rank is a causally-used mid-depth code, not an epiphenomenal trace. Mirror-image of
steer_rank.py (axis-add lever); this is the representation-swap lever (CIK entity-substitution patch).

Reuses steer_rank.py conventions (resolve_model / get_decoder_layers / mention_token_ids / chat).

Usage:
  python scripts/patch_entity.py --stimuli <stimuli.jsonl> --model gemma-4-12b-it \
      --families s0_zib --scheme readout --n-stim 16 --n-pairs 3 --out patch.parquet
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steer_rank import resolve_model, get_decoder_layers, mention_token_ids  # noqa: E402


def _ranks_of(s):
    """entity -> true rank (1..N), from entity_ranks if present else latent_order position."""
    if s.get("entity_ranks"):
        return {e: int(r) for e, r in s["entity_ranks"].items()}
    return {e: i + 1 for i, e in enumerate(s["latent_order"])}


def _pairs_for(s, n_pairs, rng):
    """Up to n_pairs (A, B, C): A,B interior (rank 3..N-2) with a LARGE true-rank gap (paired from the
    extremes inward); C a random interior entity != A,B (matched-perturbation control)."""
    order = s["latent_order"]; N = len(order)
    rk = _ranks_of(s)
    interior = sorted([e for e in order if 3 <= rk[e] <= N - 2], key=lambda e: rk[e])
    out = []
    for i in range(min(n_pairs, len(interior) // 2)):
        A, B = interior[i], interior[-1 - i]                 # widest remaining gap
        cands = [e for e in interior if e not in (A, B)]
        if not cands:
            continue
        C = cands[int(rng.integers(len(cands)))]
        out.append((A, B, C, rk[A], rk[B], rk[C]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", default=None, help="unused (interface parity with the geometry tools)")
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--model", default="gemma-4-12b-it")
    ap.add_argument("--families", default="s0_zib")
    ap.add_argument("--scheme", default="readout",
                    help="patch/read locus: readout (roster last mention) | card_mean (in-card mentions, "
                         "where the steering axis lives) | name (all mentions)")
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--declared", default="__any__",
                    help="R14 filter: '__any__' all, 'none' derived(D1), 'list' declared(D2)")
    ap.add_argument("--patch-layers", default="", help="comma model-layer idxs; default ~40/55/70% depth")
    ap.add_argument("--n-stim", type=int, default=16)
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    spec = resolve_model(args.models_config, args.model)
    which = {"readout": "last", "card_mean": "cards", "name": "all"}.get(args.scheme, "all")
    is_qwen = "qwen" in args.model.lower()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=torch.float16, attn_implementation=spec.get("attn_implementation", "eager"),
        device_map="cuda:0").eval()
    layers = get_decoder_layers(model); n_layers = len(layers)
    patch_layers = ([int(x) for x in args.patch_layers.split(",")] if args.patch_layers
                    else [max(2, int(n_layers * f)) for f in (0.40, 0.55, 0.70)])
    rng = np.random.default_rng(args.seed)
    stims = [json.loads(l) for l in open(args.stimuli)]

    def chat(text, gen):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=gen,
                                       **({"enable_thinking": False} if is_qwen else {}))

    # single mutable state read by the patch hook (overwrite, not add)
    state = {"donor": None, "pos": None}

    def hook(mod, inp, out):
        if state["donor"] is None or not state["pos"]:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= max(state["pos"]):                  # skip KV-cached single-token gen steps
            return out
        h[0, state["pos"], :] = state["donor"].to(h.dtype)   # REPLACE A's stream with the donor entity's
        return (h,) + out[1:] if isinstance(out, tuple) else h

    want_decl = (None if args.declared == "none" else args.declared) if args.declared != "__any__" else "__any__"
    rows = []
    for family in args.families.split(","):
        pool = [s for s in stims if s.get("family") == family and s.get("condition") == args.condition
                and s.get("structure", "total_order") == "total_order"
                and not bool(s.get("incoherent", False))
                and (want_decl == "__any__" or s.get("declared") == want_decl)   # R14: D2-declared vs D1-derived
                ][: (2 if args.smoke else args.n_stim)]
        for s in pool:
            block = chat(s["prompt"], gen=False)
            enc_b = tok(block, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            with torch.no_grad():
                hs = model(**enc_b, output_hidden_states=True).hidden_states  # tuple[L+1] of [1,T,D]
            for (A, B, C, rA, rB, rC) in _pairs_for(s, args.n_pairs, rng):
                posB = mention_token_ids(block, B, tok, which)
                posC = mention_token_ids(block, C, tok, which)
                if not posB or not posC:
                    continue
                q = (f"{s['prompt']}\n\nCounting from the earliest as position 1, what position is "
                     f"the {A}? Reply with only the number. No explanation.")
                qtext = chat(q, gen=True)
                qposA = mention_token_ids(qtext, A, tok, which)
                if not qposA:
                    continue
                enc_q = tok(qtext, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                if max(qposA) >= enc_q["input_ids"].shape[1]:
                    continue
                for L in patch_layers:
                    donors = {"baseline": None,
                              "patchB": hs[L][0, posB, :].mean(0).detach().clone(),
                              "patchC": hs[L][0, posC, :].mean(0).detach().clone()}
                    handle = layers[L - 1].register_forward_hook(hook)   # edits hidden_states[L]
                    try:
                        for cond, donor in donors.items():
                            state.update(donor=donor, pos=qposA)
                            with torch.no_grad():
                                g = model.generate(**enc_q, max_new_tokens=8, do_sample=False,
                                                   pad_token_id=tok.eos_token_id)
                            state.update(donor=None, pos=None)
                            ans = tok.decode(g[0, enc_q["input_ids"].shape[1]:], skip_special_tokens=True)
                            mm = re.search(r"\d{1,3}", ans)
                            rows.append(dict(model=args.model, family=family, scheme=args.scheme,
                                             patch_layer=L, stim=s["stimulus_id"], A=A, B=B, C=C,
                                             true_rank_A=rA, true_rank_B=rB, true_rank_C=rC, cond=cond,
                                             answered=int(mm.group()) if mm else None, raw=ans.strip()[:20]))
                    finally:
                        handle.remove(); state.update(donor=None, pos=None)
        print(f"[{family}] {len(pool)} stimuli patched", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)

    # ---- decisive read: does patchB move A's answer TOWARD rank_B more than the patchC control? ----
    def _toward_rates(g):
        piv = g.pivot_table(index=["stim", "A", "B"], columns="cond", values="answered", aggfunc="first")
        for c in ("baseline", "patchB", "patchC"):
            if c not in piv.columns:
                piv[c] = np.nan
        meta = g.drop_duplicates(["stim", "A", "B"]).set_index(["stim", "A", "B"])[["true_rank_A", "true_rank_B"]]
        piv = piv.join(meta)
        # keep only pairs where ALL three conditions answered → toward_B and toward_C are ALIGNED
        # (same rows), so the paired bootstrap on their difference is valid.
        ok = piv["baseline"].notna() & piv["patchB"].notna() & piv["patchC"].notna()
        p = piv[ok]
        exp = np.sign(p["true_rank_B"] - p["true_rank_A"])                    # expected shift direction
        toward_B = (np.sign(p["patchB"] - p["baseline"]) == exp).to_numpy(dtype=float)
        toward_C = (np.sign(p["patchC"] - p["baseline"]) == exp).to_numpy(dtype=float)
        return piv, toward_B, toward_C

    print("=== E9b entity-substitution patch: toward-B rate vs toward-C control (per patch layer) ===")
    for (fam, L), g in df.groupby(["family", "patch_layer"]):
        piv, tb, tc = _toward_rates(g)          # tb, tc are aligned (same pairs), so len(tb)==len(tc)
        n = len(tb)
        if n < 2:
            print(f"{fam:8s} {args.scheme:8s} L{L:<2d} | insufficient paired data (n={n})"); continue
        rateB, rateC = float(tb.mean()), float(tc.mean())
        # paired bootstrap CI on (toward_B_rate - toward_C_rate): resample the SAME pair indices
        bb = np.empty(1000)
        for i in range(1000):
            idx = rng.integers(0, n, n)
            bb[i] = tb[idx].mean() - tc[idx].mean()
        lo, hi = float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))
        mb = float(np.nanmean(piv["patchB"])); mc = float(np.nanmean(piv["patchC"]))
        m0 = float(np.nanmean(piv["baseline"]))
        sig = "SIG" if lo > 0 else "ns"
        print(f"{fam:8s} {args.scheme:8s} L{L:<2d} | towardB={rateB:.2f} towardC={rateC:.2f} "
              f"Δ={rateB - rateC:+.2f} [{lo:+.2f},{hi:+.2f}] {sig} | "
              f"ans base={m0:.1f} patchB={mb:.1f} patchC={mc:.1f} (n={n})", flush=True)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
