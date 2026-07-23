#!/usr/bin/env python
"""Track C: causal test of the order manifold's inertness.

Learn the manifold's rank axis from the probe, add it to ONE item's residual
stream at run time, and measure (a) the item's internally-decoded rank and
(b) the model's answered rank, as a function of dose alpha, vs a matched-norm
random direction.

Decisive reads:
  along-manifold moves INTERNAL decoded rank, dose-dependent, >> random
    => the direction is causal for the representation (manifold is real).
  along-manifold moves BEHAVIOUR (answered rank) weakly / << its internal
    effect => the manifold is behaviourally INERT (encoded, not deployed).
  along-manifold moves behaviour ~ proportionally => it IS deployed.

Steer at the manifold-peak layer; read internal rank at the LAST layer (so a
non-trivial effect requires the perturbation to PROPAGATE, not just be re-read).
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


def name_token_ids(prompt, entity, tok):
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    ids = []
    for m in re.finditer(rf"\b[Tt]he {re.escape(entity)}\b", prompt):
        lo, hi = m.start() + 4, m.end()
        ids += [i for i, (s, e) in enumerate(offs) if s < hi and e > lo and e > s]
    return sorted(set(ids))


def fit_probe(acts_dir, model, family, condition, layer, pca=64):
    """Return (rank direction in raw D-space, unit; probe fn on raw x -> norm rank)."""
    Xs, ys = [], []
    for f in sorted((Path(acts_dir) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["family"] == family and meta["condition"] == condition:
            Xs.append(z["name"][:, layer, :].astype(np.float32))
            r = z["ranks"]; ys.append((r - r.min()) / (r.max() - r.min()))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    sc = StandardScaler().fit(X)
    pc = PCA(n_components=pca, random_state=0).fit(sc.transform(X))
    rg = Ridge(alpha=10.0).fit(pc.transform(sc.transform(X)), y)
    # d(decoded rank)/d(x) in raw space = (1/scale) * components^T * coef
    grad = (pc.components_.T @ rg.coef_) / sc.scale_
    v = grad / (np.linalg.norm(grad) + 1e-9)
    def probe(xraw):  # xraw [n, D] -> decoded norm rank
        return rg.predict(pc.transform(sc.transform(xraw)))
    return v.astype(np.float32), probe, float(np.std(sc.transform(X) @ (grad / np.linalg.norm(grad)) ))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--families", default="relational,tagged")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-stim", type=int, default=16)
    ap.add_argument("--alphas", default="-8,-4,-2,0,2,4,8")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[args.model])
    model = AutoModelForCausalLM.from_pretrained(HF[args.model], dtype=torch.float16,
                                                 attn_implementation="eager", device_map="cuda:0").eval()
    layers = model.model.layers
    n_layers = len(layers)
    alphas = [float(a) for a in args.alphas.split(",")]
    stims = [json.loads(l) for l in open(args.stimuli)]
    rng = np.random.default_rng(0)

    # steering hook state
    state = {"vec": None, "pos": None, "scale": 0.0}

    def hook(mod, inp, out):
        if state["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        v = torch.tensor(state["vec"], device=h.device, dtype=h.dtype) * state["scale"]
        h[0, state["pos"], :] += v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    rows = []
    for family in args.families.split(","):
        Ls = PEAK[(args.model, family)]
        v_along, _, spread = fit_probe(args.acts, args.model, family, "shuffle", Ls)
        _, probe_last, _ = fit_probe(args.acts, args.model, family, "shuffle", n_layers)  # last-layer readout
        v_rand = rng.standard_normal(v_along.shape).astype(np.float32)
        v_rand /= np.linalg.norm(v_rand)
        pool = [s for s in stims if s["family"] == family and s["condition"] == "shuffle"]
        pool = pool[: (2 if args.smoke else args.n_stim)]
        handle = layers[Ls].register_forward_hook(hook)
        for s in pool:
            N = len(s["latent_order"])
            target = s["latent_order"][N // 2]           # mid-rank item (can move both ways)
            true_rank = N // 2 + 1
            block = tok.apply_chat_template([{"role": "user", "content": s["prompt"]}],
                                            tokenize=False, add_generation_prompt=False,
                                            **({"enable_thinking": False} if "qwen" in args.model.lower() else {}))
            pos = name_token_ids(block, target, tok)
            # behaviour prompt: rank question
            q = (f"{s['prompt']}\n\nCounting from the earliest as position 1, what position is "
                 f"the {target}? Reply with only the number. No explanation.")
            qtext = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                            add_generation_prompt=True,
                                            **({"enable_thinking": False} if "qwen" in args.model.lower() else {}))
            qpos = name_token_ids(qtext, target, tok)  # steer the same entity in the Q prompt too
            for direction, vec in (("along", v_along), ("random", v_rand)):
                for a in alphas:
                    # (a) internal: forward the card block, read last-layer name pooled -> probe
                    enc = tok(block, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                    state.update(vec=vec, pos=pos, scale=a * spread)
                    with torch.no_grad():
                        hs = model(**enc, output_hidden_states=True).hidden_states[n_layers][0]
                    dec = float(probe_last(hs[pos].float().mean(0, keepdim=True).cpu().numpy())[0])
                    # (b) behaviour: generate the rank answer with same steering on the Q prompt
                    encq = tok(qtext, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                    state.update(vec=vec, pos=qpos, scale=a * spread)
                    with torch.no_grad():
                        g = model.generate(**encq, max_new_tokens=8, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                    ans = tok.decode(g[0, encq["input_ids"].shape[1]:], skip_special_tokens=True)
                    m = re.search(r"\d{1,3}", ans)
                    rows.append(dict(model=args.model, family=family, stim=s["stimulus_id"],
                                     target=target, true_rank=true_rank, direction=direction,
                                     alpha=a, decoded_rank=round(dec, 3),
                                     answered=int(m.group()) if m else None, raw=ans.strip()[:20]))
                    state.update(vec=None)
        handle.remove()

    import pandas as pd
    pd.DataFrame(rows).to_parquet(args.out)
    df = pd.DataFrame(rows)
    print("=== dose-response: mean over stimuli (decoded internal rank / answered rank) ===")
    for (fam, d), g in df.groupby(["family", "direction"]):
        line = f"{fam:10s} {d:6s} | "
        for a in alphas:
            ga = g[g.alpha == a]
            dec = ga["decoded_rank"].mean()
            ans = ga["answered"].dropna()
            line += f"a{a:+.0f}:dec={dec:.2f},ans={ans.mean():.1f}({len(ans)}) "
        print(line)
    if args.smoke:
        print("\n=== smoke raw ===")
        for r in rows:
            print(f"  {r['family']:10s} {r['direction']:6s} a{r['alpha']:+.0f} "
                  f"true={r['true_rank']} dec={r['decoded_rank']:.2f} ans={r['answered']} raw={r['raw']!r}")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
