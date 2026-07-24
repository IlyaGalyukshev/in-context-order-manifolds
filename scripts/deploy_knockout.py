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
    ap.add_argument("--peak-layer", type=int, default=0, help="override manifold peak layer")
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
        Ls = args.peak_layer or PEAK[(args.model, family)]
        v_along, spread = fit_dir(args.acts, args.model, family, Ls)
        v_rand = rng.standard_normal(v_along.shape).astype(np.float32); v_rand /= np.linalg.norm(v_rand)
        handle = layers[Ls - 1].register_forward_hook(hook)
        pool = [s for s in stims if s["family"] == family and s["condition"] == "shuffle"][
            : (2 if args.smoke else args.n_stim)]
        for si, s in enumerate(pool):
            N = len(s["latent_order"])
            xr = N // 2                        # X at a mid rank (steerable both ways)
            d = max(2, N // 3)                 # separation, N-safe
            yr = xr + d if (si % 2 == 0 and xr + d < N) else xr - d
            if not (0 <= yr < N) or yr == xr:  # fallback to the valid side
                yr = xr + d if xr + d < N else xr - d
            X = s["latent_order"][xr]
            Y = s["latent_order"][yr]
            x_earlier = xr < yr                # ground truth for THIS pair (varies!)
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
                    # correct iff the model prefers the truly-earlier one
                    correct = (margin_X > 0) == x_earlier
                    rows.append(dict(model=args.model, family=family, stim=s["stimulus_id"],
                                     target=X, ref=Y, x_earlier=bool(x_earlier), cond=cond,
                                     direction=dname, alpha=a, margin_x=round(margin_X, 3),
                                     correct=bool(correct)))
                    state.update(vec=None)
        handle.remove()

    import pandas as pd
    df = pd.DataFrame(rows); df.to_parquet(args.out)
    print("=== BASELINE accuracy (balanced pairs, chance=0.50) — must be >0.5 and <1 to be a real task ===")
    for (fam, cond), g in df[df.direction.isin(["none"]) | (df.alpha == 0)].groupby(["family", "cond"]):
        gg = g[(g.cond == "full") | ((g.cond == "knock") & (g.direction == "along"))]  # a=0 same for along/random
        print(f"  {fam:10s} {cond:5s} acc={g.correct.mean():.2f} (n={len(g)})  mean|margin|={g.margin_x.abs().mean():.2f}")
    print("=== KNOCKOUT dose-response: margin toward X vs alpha (v increases rank => expect margin DOWN with alpha) ===")
    slopes = {}
    for (fam, d), g in df[df.cond == "knock"].groupby(["family", "direction"]):
        line = f"  {fam:10s} {d:6s} | "
        for a in alphas:
            line += f"a{a:+.0f}:m={g[g.alpha==a]['margin_x'].mean():+.2f} "
        # per-stimulus slope of margin vs alpha
        sl = []
        for st, gs in g.groupby("stim"):
            gs = gs.sort_values("alpha")
            sl.append(np.polyfit(gs.alpha, gs.margin_x, 1)[0])
        slopes[(fam, d)] = np.array(sl)
        print(line + f" | mean slope={np.mean(sl):+.3f}")
    print("=== DEPLOYMENT TEST: along vs random steering slope (paired over stimuli) ===")
    from scipy.stats import wilcoxon
    for fam in df.family.unique():
        if (fam, "along") in slopes and (fam, "random") in slopes:
            a_sl, r_sl = slopes[(fam, "along")], slopes[(fam, "random")]
            try:
                p = wilcoxon(a_sl, r_sl).pvalue
            except Exception:
                p = float("nan")
            print(f"  {fam:10s} along slope={a_sl.mean():+.3f} vs random={r_sl.mean():+.3f} "
                  f"(|along|-|random|={np.abs(a_sl).mean()-np.abs(r_sl).mean():+.3f}, paired p={p:.3f})")
    if args.smoke:
        print("\n=== smoke raw ===")
        for r in rows:
            print(f"  {r['family'][:3]} xE={int(r['x_earlier'])} {r['cond']:5s} {r['direction']:6s} "
                  f"a{r['alpha']:+.0f} m={r['margin_x']:+.2f} ok={int(r['correct'])}")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
