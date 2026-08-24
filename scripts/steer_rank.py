#!/usr/bin/env python
"""Track C: causal test of the order axis — is it USED as a lever, or an epiphenomenal trace?

Learn the ridge rank-axis at a (layer, locus), add it to ONE mid-rank entity's residual stream
at run time, and measure the shift in (a) the internally-decoded rank (propagated to a peak
layer) and (b) the model's ANSWERED rank — as a function of dose alpha — against a MATCHED-NORM
OFF-AXIS control (a random direction orthogonalised to the rank axis, same norm). The matched-
norm control is mandatory: without it, steering itself becomes confirmation bias.

Decisive reads (both publishable):
  along-axis moves the answered rank predictably, off-axis (matched norm) only noises
    => the axis is CAUSALLY USED (not an epiphenomenon); the confirmation-bias charge is
       causally dismissed.
  along-axis does not move behaviour => the axis is a fingerprint, order is computed off-axis
    => strengthens query-local ("no map even as a lever").
Sweep layer x locus (--steer-layers x --scheme): readout may carry the axis as a trace while
the lever lives on in-card mention tokens.

Reuses the repo conventions: models.yaml for the model spec, and the same acts (from
extract_activations.py, all layers) that the geometry probes read. One parameterized tool.

Usage:
  python scripts/steer_rank.py --acts <dir> --stimuli <stimuli.jsonl> --model gemma-4-12b-it \
      --families s0_zib --scheme readout --out steer.parquet
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def resolve_model(models_config, name):
    roster = {}
    mcfg = yaml.safe_load(open(models_config))
    for sec in ("models", "confirmatory", "exploratory"):
        roster.update(mcfg.get(sec) or {})
    return roster[name]


def get_decoder_layers(model):
    """The decoder-layer ModuleList, robust across architectures (Gemma-4 unified/multimodal
    nests it under .language_model; Qwen/OLMo expose model.model.layers)."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "model.model.language_model.layers", "model.model.layers"):
        obj = model; ok = True
        for p in path.split("."):
            if not hasattr(obj, p):
                ok = False; break
            obj = getattr(obj, p)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    for _, mod in model.named_modules():   # fallback: the deepest big ModuleList of blocks
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 8:
            return mod
    raise RuntimeError("could not locate decoder layers")


def mention_token_ids(prompt, entity, tok, which="all"):
    """token indices of 'The <entity>' mentions. which='all' (name locus, every mention),
    'last' (readout locus, the roster mention after all cards), or 'cards' (in-card locus: every
    mention EXCEPT the trailing roster one — the card-token locus where the steering axis lives)."""
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    spans = []
    for m in re.finditer(rf"\b[Tt]he {re.escape(entity)}\b", prompt):
        lo, hi = m.start() + 4, m.end()
        spans.append([i for i, (s, e) in enumerate(offs) if s < hi and e > lo and e > s])
    if not spans:
        return []
    if which == "last":
        sel = spans[-1]
    elif which == "cards":
        sel = [i for sp in (spans[:-1] if len(spans) > 1 else spans) for i in sp]  # drop roster mention
    else:
        sel = [i for sp in spans for i in sp]
    return sorted(set(sel))


def fit_axis(acts_dir, model, family, condition, scheme, layer, pca=64):
    """Ridge rank-axis at (scheme, layer) in raw D-space (unit) + a decode fn + natural spread."""
    Xs, ys = [], []
    for f in sorted((Path(acts_dir) / model).glob("*.npz")):
        z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"]))
        if m.get("family") == family and m.get("condition") == condition and scheme in z.files \
                and not bool(m.get("is_null", False)):
            Xs.append(z[scheme][:, layer, :].astype(np.float32))
            r = z["ranks"]; ys.append((r - r.min()) / (r.max() - r.min()))
    if not Xs:
        return None
    X = np.concatenate(Xs); y = np.concatenate(ys)
    sc = StandardScaler().fit(X)
    pc = PCA(n_components=min(pca, X.shape[0] - 1), random_state=0).fit(sc.transform(X))
    rg = Ridge(alpha=10.0).fit(pc.transform(sc.transform(X)), y)
    grad = (pc.components_.T @ rg.coef_) / sc.scale_
    v = grad / (np.linalg.norm(grad) + 1e-9)

    def decode(xraw):
        return rg.predict(pc.transform(sc.transform(xraw)))
    from scipy.stats import spearmanr
    fit_q = abs(spearmanr(X @ v, y)[0] or 0)
    return v.astype(np.float32), decode, float(np.std(X @ v)), fit_q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True); ap.add_argument("--stimuli", required=True)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--model", default="gemma-4-12b-it")
    ap.add_argument("--families", default="s0_zib")
    ap.add_argument("--scheme", default="readout", help="locus to steer/read: name|readout")
    ap.add_argument("--condition", default="shuffle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-stim", type=int, default=16)
    ap.add_argument("--alphas", default="-8,-4,-2,0,2,4,8")
    ap.add_argument("--steer-layers", default="", help="comma ints (model-layer idx); default 40/55/70% depth")
    ap.add_argument("--peak-layer", type=int, default=None, help="read propagation here; default auto-best")
    ap.add_argument("--n-offaxis", type=int, default=1,
                    help="number of matched-norm off-axis controls (null distribution of the "
                         "off-axis effect; the along-axis effect is read against this null + a "
                         "bootstrap CI on the along-minus-offaxis slope).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    spec = resolve_model(args.models_config, args.model)
    which = "last" if args.scheme == "readout" else "all"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    is_qwen = "qwen" in args.model.lower()
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=torch.float16, attn_implementation=spec.get("attn_implementation", "eager"),
        device_map="cuda:0").eval()
    layers = get_decoder_layers(model); n_layers = len(layers)
    alphas = [float(a) for a in args.alphas.split(",")]
    stims = [json.loads(l) for l in open(args.stimuli)]
    rng = np.random.default_rng(args.seed)

    state = {"vec": None, "pos": None, "scale": 0.0}

    def hook(mod, inp, out):
        if state["vec"] is None or not state["pos"]:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= max(state["pos"]):   # skip KV-cached single-token gen steps
            return out
        v = torch.tensor(state["vec"], device=h.device, dtype=h.dtype) * state["scale"]
        h[0, state["pos"], :] += v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    steer_layers = ([int(x) for x in args.steer_layers.split(",")] if args.steer_layers
                    else [max(2, int(n_layers * f)) for f in (0.40, 0.55, 0.70)])

    def chat(text, gen):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=gen,
                                       **({"enable_thinking": False} if is_qwen else {}))

    rows = []
    for family in args.families.split(","):
        # peak layer for reading propagation (max rank-decodability at this locus)
        if args.peak_layer is not None:
            LP = args.peak_layer
        else:
            best = (0, -9)
            for L in range(1, n_layers, max(1, n_layers // 12)):
                r = fit_axis(args.acts, args.model, family, args.condition, args.scheme, L)
                if r and r[3] > best[1]:
                    best = (L, r[3])
            LP = best[0]
        fp = fit_axis(args.acts, args.model, family, args.condition, args.scheme, LP)
        if fp is None:
            print(f"{family}: no acts for scheme {args.scheme}"); continue
        _, decode_peak, _, peak_q = fp
        print(f"[{family}/{args.scheme}] peak layer L{LP} (fit rho={peak_q:.2f})", flush=True)
        pool = [s for s in stims if s.get("family") == family and s.get("condition") == args.condition
                and s.get("structure", "total_order") == "total_order"][: (2 if args.smoke else args.n_stim)]
        for Ls in steer_layers:
            fa = fit_axis(args.acts, args.model, family, args.condition, args.scheme, Ls)
            if fa is None:
                continue
            v_along, _, spread, _ = fa
            # matched-norm OFF-AXIS null: n_offaxis random directions each orthogonalised to the
            # rank axis, unit norm. The off-axis effect distribution IS the null the along-axis
            # effect is judged against (one direction was confirmation-bias-prone).
            offdirs = []
            for _k in range(max(1, args.n_offaxis)):
                vr = rng.standard_normal(v_along.shape).astype(np.float32)
                vr -= (vr @ v_along) * v_along
                vr /= (np.linalg.norm(vr) + 1e-9)
                offdirs.append(vr)
            directions = [("along", v_along)] + [(f"offaxis{_k}", vr) for _k, vr in enumerate(offdirs)]
            handle = layers[Ls - 1].register_forward_hook(hook)   # affect hidden_states[Ls]
            for s in pool:
                N = len(s["latent_order"]); target = s["latent_order"][N // 2]; true_rank = N // 2 + 1
                block = chat(s["prompt"], gen=False); pos = mention_token_ids(block, target, tok, which)
                q = (f"{s['prompt']}\n\nCounting from the earliest as position 1, what position is "
                     f"the {target}? Reply with only the number. No explanation.")
                qtext = chat(q, gen=True); qpos = mention_token_ids(qtext, target, tok, which)
                if not pos or not qpos:
                    continue
                for direction, vec in directions:
                    for a in alphas:
                        enc = tok(block, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                        state.update(vec=vec, pos=pos, scale=a * spread)
                        with torch.no_grad():
                            allh = model(**enc, output_hidden_states=True).hidden_states
                        dec = float(decode_peak(allh[LP][0][pos].float().mean(0, keepdim=True).cpu().numpy())[0])
                        encq = tok(qtext, return_tensors="pt", add_special_tokens=False).to("cuda:0")
                        state.update(vec=vec, pos=qpos, scale=a * spread)
                        with torch.no_grad():
                            g = model.generate(**encq, max_new_tokens=8, do_sample=False,
                                               pad_token_id=tok.eos_token_id)
                        ans = tok.decode(g[0, encq["input_ids"].shape[1]:], skip_special_tokens=True)
                        mm = re.search(r"\d{1,3}", ans)
                        rows.append(dict(model=args.model, family=family, scheme=args.scheme, steer_layer=Ls,
                                         peak_layer=LP, stim=s["stimulus_id"], target=target, true_rank=true_rank,
                                         direction=direction, alpha=a, decoded_rank=round(dec, 3),
                                         answered=int(mm.group()) if mm else None, raw=ans.strip()[:20]))
                        state.update(vec=None)
            handle.remove()

    import pandas as pd
    df = pd.DataFrame(rows); df.to_parquet(args.out)
    # collapse offaxis0/1/2.. -> a single "offaxis" null band for the human summary; the parquet
    # keeps every direction so the gather step can build the null distribution + CI.
    df["dir_base"] = df["direction"].str.replace(r"\d+$", "", regex=True)
    print("=== dose-response: propagated decoded rank / answered rank "
          f"(offaxis = mean over {max(1, args.n_offaxis)} matched-norm controls) ===")
    for (fam, sl, d), gdf in df.groupby(["family", "steer_layer", "dir_base"]):
        line = f"{fam:8s} {args.scheme:8s} L{sl:<2d} {d:7s} | "
        for a in alphas:
            ga = gdf[gdf.alpha == a]; ans = ga["answered"].dropna()
            line += f"a{a:+.0f}:dec={ga['decoded_rank'].mean():.2f},ans={(ans.mean() if len(ans) else float('nan')):.1f} "
        print(line, flush=True)
    # along-vs-null: answered-rank slope (rank per unit alpha) for along vs each off-axis dir.
    def _slope(gdf):
        g = gdf.dropna(subset=["answered"])
        if g["alpha"].nunique() < 2 or len(g) < 3:
            return float("nan")
        return float(np.polyfit(g["alpha"].to_numpy(float), g["answered"].to_numpy(float), 1)[0])
    print("=== answered-rank slope: along vs off-axis null (per steer layer) ===")
    for (fam, sl), gdf in df.groupby(["family", "steer_layer"]):
        al = _slope(gdf[gdf.direction == "along"])
        offs = [_slope(gdf[gdf.direction == dd]) for dd in sorted(gdf.direction.unique()) if dd.startswith("offaxis")]
        offs = [o for o in offs if o == o]
        if offs:
            om, osd = float(np.mean(offs)), float(np.std(offs))
            hi = (1 + sum(abs(o) >= abs(al) for o in offs)) / (1 + len(offs))  # |off| >= |along| rate
            print(f"{fam:8s} {args.scheme:8s} L{sl:<2d} | along_slope={al:+.3f}  "
                  f"offaxis_null={om:+.3f}±{osd:.3f} (n={len(offs)})  p(|off|>=|along|)={hi:.3f}", flush=True)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
