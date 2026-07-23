#!/usr/bin/env python
"""Clean deployment test (Beat 4) via attention knockout.

The manifold forms under FULL attention (prompt forward), but the QUERY tokens
are blocked from attending to the source cards — EXCEPT the target entity's
name tokens, which carry the manifold. The model must then answer "what position
is X?" from X's representation alone, not by re-reading tags/relations. We read
the answer as a distribution over position tokens 1..N at a single forward
(no generation, so the 4D mask stays simple).

Decisive:
  full attention: expected answered position tracks true rank  (sanity: model can answer).
  knockout, no steer: can it still answer from X's representation alone?
  knockout + steer X's manifold coordinate (along vs matched random, dose):
    answer moves dose-dependently  => manifold is DEPLOYED (causal readout).
    answer flat  => even as the only route, the manifold is not read => inert.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PEAK = {("qwen3-4b", "relational"): 22, ("qwen3-4b", "tagged"): 20,
        ("olmo3-7b-inst", "relational"): 18, ("olmo3-7b-inst", "tagged"): 15}
HF = {"qwen3-4b": "Qwen/Qwen3-4B", "olmo3-7b-inst": "allenai/Olmo-3-7B-Instruct"}


def token_span(full, sub, tok):
    """char span of `sub` in `full` -> token index range [lo, hi)."""
    c0 = full.index(sub); c1 = c0 + len(sub)
    offs = tok(full, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    idx = [i for i, (s, e) in enumerate(offs) if s < c1 and e > c0 and e > s]
    return min(idx), max(idx) + 1, offs


def name_ids(full, entity, tok, offs, lo, hi):
    ids = []
    for m in re.finditer(rf"\b[Tt]he {re.escape(entity)}\b", full):
        a, b = m.start() + 4, m.end()
        ids += [i for i, (s, e) in enumerate(offs) if s < b and e > a and e > s and lo <= i < hi]
    return sorted(set(ids))


def fit_dir(acts, model, family, layer):
    Xs, ys = [], []
    for f in sorted((Path(acts) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        if json.loads(str(z["meta"]))["family"] == family and json.loads(str(z["meta"]))["condition"] == "shuffle":
            Xs.append(z["name"][:, layer, :].astype(np.float32))
            r = z["ranks"]; ys.append((r - r.min()) / (r.max() - r.min()))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    sc = StandardScaler().fit(X); pc = PCA(64, random_state=0).fit(sc.transform(X))
    rg = Ridge(alpha=10.0).fit(pc.transform(sc.transform(X)), y)
    grad = (pc.components_.T @ rg.coef_) / sc.scale_
    v = (grad / np.linalg.norm(grad)).astype(np.float32)
    return v, float(np.std(X @ v))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--families", default="relational,tagged")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-stim", type=int, default=16)
    ap.add_argument("--alphas", default="-8,-4,0,4,8")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[args.model])
    model = AutoModelForCausalLM.from_pretrained(HF[args.model], dtype=torch.float16,
                                                 attn_implementation="eager", device_map="cuda:0").eval()
    layers = model.model.layers
    dt = next(model.parameters()).dtype
    minval = torch.finfo(dt).min
    alphas = [float(a) for a in args.alphas.split(",")]
    stims = [json.loads(l) for l in open(args.stimuli)]
    rng = np.random.default_rng(0)
    state = {"vec": None, "pos": None, "scale": 0.0}

    def hook(mod, inp, out):
        if state["vec"] is None or not state["pos"]:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= max(state["pos"]):
            return out
        h[0, state["pos"], :] += torch.tensor(state["vec"], device=h.device, dtype=h.dtype) * state["scale"]
        return (h,) + out[1:] if isinstance(out, tuple) else h

    rows = []
    for family in args.families.split(","):
        Ls = PEAK[(args.model, family)]
        v_along, spread = fit_dir(args.acts, args.model, family, Ls)
        v_rand = rng.standard_normal(v_along.shape).astype(np.float32); v_rand /= np.linalg.norm(v_rand)
        handle = layers[Ls - 1].register_forward_hook(hook)
        pool = [s for s in stims if s["family"] == family and s["condition"] == "shuffle"][
            : (2 if args.smoke else args.n_stim)]
        for s in pool:
            N = len(s["latent_order"])
            X = s["latent_order"][N // 2]      # mid-rank target (steered)
            Y = s["latent_order"][0]           # earliest reference (rank 1)
            # forced-choice pairwise, read first token of X vs Y after "Answer: the"
            prompt = (s["prompt"] + f"\n\nWhich acted earlier: the {X} or the {Y}? Answer: the")
            full = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
                **({"enable_thinking": False} if "qwen" in args.model.lower() else {}))
            enc = tok(full, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            S = enc["input_ids"].shape[1]
            lo, hi, offs = token_span(full, s["prompt"], tok)         # card-block token span
            xpos = name_ids(full, X, tok, offs, lo, hi)
            ypos = name_ids(full, Y, tok, offs, lo, hi)
            qspan = list(range(hi, S))
            keep = set(xpos) | set(ypos)                              # both candidates readable
            block_cols = [j for j in range(lo, hi) if j not in keep]
            x_id = tok(" " + X, add_special_tokens=False)["input_ids"][0]
            y_id = tok(" " + Y, add_special_tokens=False)["input_ids"][0]
            base = torch.triu(torch.full((S, S), minval, device="cuda:0", dtype=dt), 1)
            knock = base.clone()
            for i in qspan:
                knock[i, block_cols] = minval
            for cond, mask4d in (("full", base[None, None]), ("knock", knock[None, None])):
                combos = [("none", None, 0.0)] if cond == "full" else \
                    [("along", v_along, a) for a in alphas] + [("random", v_rand, a) for a in alphas]
                for dname, vec, a in combos:
                    state.update(vec=vec, pos=xpos, scale=a * spread)
                    with torch.no_grad():
                        lg = model(input_ids=enc["input_ids"], attention_mask=mask4d).logits[0, -1].float()
                    # margin toward X (says X earlier). X is truly later than Y,
                    # so a correct model is negative; steering X earlier should raise it.
                    margin_X = float(lg[x_id] - lg[y_id])
                    rows.append(dict(model=args.model, family=family, stim=s["stimulus_id"],
                                     target=X, ref=Y, cond=cond, direction=dname, alpha=a,
                                     margin_x=round(margin_X, 3)))
                    state.update(vec=None)
        handle.remove()

    import pandas as pd
    df = pd.DataFrame(rows); df.to_parquet(args.out)
    print("=== FULL attention sanity: margin toward X (X is truly LATER than Y -> expect < 0) ===")
    for fam, g in df[df.cond == "full"].groupby("family"):
        print(f"  {fam:10s} mean margin_X={g.margin_x.mean():+.2f}  (frac correct 'Y earlier': {(g.margin_x<0).mean():.2f})")
    print("=== KNOCKOUT dose-response: margin toward X vs alpha (v increases rank; -alpha = X earlier) ===")
    for (fam, d), g in df[df.cond == "knock"].groupby(["family", "direction"]):
        line = f"  {fam:10s} {d:6s} | "
        for a in alphas:
            line += f"a{a:+.0f}:m={g[g.alpha==a]['margin_x'].mean():+.2f} "
        print(line)
    if args.smoke:
        print("\n=== smoke raw ===")
        for r in rows:
            print(f"  {r['family'][:3]} {r['cond']:5s} {r['direction']:6s} a{r['alpha']:+.0f} margin_X={r['margin_x']:+.2f}")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
