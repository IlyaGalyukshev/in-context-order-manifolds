#!/usr/bin/env python
"""V100 admission test for roster models — run BEFORE any model enters the grid.

Per model:
  1. fp16 load (eager attention) on one GPU — wall-clock load time from NFS
  2. 20 fixed prompts → last-token logits + greedy generations (saved for eyeballing)
  3. thinking-suppression check (Qwen3: enable_thinking=False must kill <think>)
  4. hidden-states hook check: n_layers+1 tensors of [1, seq, d]
  5. fp32 reload (device_map=auto across visible GPUs) → same prompts
  6. KL(fp16 || fp32) on next-token distributions + top-1 agreement

Verdict: PASS if mean KL < 0.02 nats, top-1 agreement >= 0.9, no <think> leakage,
hidden-states shapes correct. Everything is written to --out (JSON + samples.txt)
so a human can eyeball the generations — never trust the numbers alone.

Usage (inside the worker container):
  python scripts/smoke_v100.py --models qwen3-1.7b,olmo3-7b-inst \
      --config configs/models.yaml --out /workspace/manifolds/results/smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

PROMPTS = [
    # plain factual / completion (work for base and instruct alike)
    "The capital of France is",
    "Water boils at a temperature of",
    "The third planet from the Sun is called",
    "Two plus two equals",
    "The opposite of hot is",
    "A triangle has this many sides:",
    "The chemical symbol for gold is",
    "In one word, the color of grass is",
    "The first month of the year is",
    "Seven minus three equals",
    # tiny ordering probes (eyeball material for our actual task)
    "Events: B happened first, then C, then A. The earliest event was",
    "List in order: 3, 1, 2 sorted ascending is",
    "Monday, Tuesday, Wednesday. The day after Tuesday is",
    "The glemb wobbled before the drane fizzed. What happened first? The",
    "Tag 3 came before Tag 7. The earlier tag was Tag",
    # nonce tokenization eyeball
    "The snorvic blethered twice and the plimth",
    "A quonch, a fennel and a gondrel walked into",
    # short instructions (instruct models should comply; base will ramble — fine)
    "Answer with one word only. What is the largest ocean?",
    "Answer yes or no. Is five greater than three?",
    "Complete the sequence: alpha, beta, gamma,",
]

NONCE_WORDS = ["glemb", "drane", "quonch", "fennel", "snorvic", "plimth", "gondrel", "festrick"]


def load_roster(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path))
    roster = {}
    for section in ("models", "confirmatory", "exploratory"):
        roster.update(cfg.get(section) or {})
    return roster


def format_prompt(tok, prompt: str, is_instruct: bool, enable_thinking_flag: bool):
    if not is_instruct or tok.chat_template is None:
        return prompt
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking_flag:
        kwargs["enable_thinking"] = False
    try:
        return tok.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
    except TypeError:  # template without enable_thinking kwarg (e.g. OLMo)
        kwargs.pop("enable_thinking", None)
        return tok.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


@torch.no_grad()
def run_pass(model, tok, prompts: list[str], device: str, generate: bool):
    """Return (last-token logits [n_prompts, vocab] on CPU fp32, generations)."""
    logits_rows, gens = [], []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        out = model(**enc)
        logits_rows.append(out.logits[0, -1, :].float().cpu())
        if generate:
            g = model.generate(
                **enc, max_new_tokens=40, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            gens.append(tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=False))
    return torch.stack(logits_rows), gens


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def smoke_model(name: str, spec: dict, out_dir: Path) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = spec["hf_id"]
    is_qwen3 = "Qwen3" in hf_id
    rec: dict = {"model": name, "hf_id": hf_id}
    print(f"\n=== {name} ({hf_id}) ===", flush=True)

    tok = AutoTokenizer.from_pretrained(hf_id)
    # Chat-format unless the roster explicitly marks the model as a base model.
    # Substring checks on hf_id are NOT reliable (Qwen/Qwen3-1.7B is a chat
    # model with no "Instruct" suffix) — this exact bug was caught in pilot.
    is_instruct = spec.get("role", "instruct") != "base" and tok.chat_template is not None
    rec["chat_formatted"] = is_instruct
    rec["nonce_token_counts"] = {
        w: len(tok(w, add_special_tokens=False)["input_ids"]) for w in NONCE_WORDS
    }

    prompts = [format_prompt(tok, p, is_instruct, enable_thinking_flag=is_qwen3) for p in PROMPTS]

    # --- fp16 pass ---
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, attn_implementation="eager", device_map="cuda:0"
    )
    rec["load_s_fp16"] = round(time.monotonic() - t0, 1)
    rec["n_layers"] = model.config.num_hidden_layers
    rec["d_model"] = model.config.hidden_size

    # hidden-states hook check
    enc = tok(prompts[0], return_tensors="pt").to("cuda:0")
    hs = model(**enc, output_hidden_states=True).hidden_states
    rec["hidden_states_ok"] = (
        len(hs) == model.config.num_hidden_layers + 1
        and all(h.shape == (1, enc["input_ids"].shape[1], model.config.hidden_size) for h in hs)
    )

    logits16, gens = run_pass(model, tok, prompts, "cuda:0", generate=True)
    rec["think_leakage"] = sum("<think>" in g for g in gens) if is_qwen3 else None
    free(model)

    # --- fp32 pass (reference) ---
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, attn_implementation="eager", device_map="auto"
    )
    rec["load_s_fp32"] = round(time.monotonic() - t0, 1)
    logits32, _ = run_pass(model, tok, prompts, model.device, generate=False)
    free(model)

    # --- compare ---
    p16 = torch.log_softmax(logits16, -1)
    p32 = torch.log_softmax(logits32, -1)
    kl = torch.sum(p16.exp() * (p16 - p32), dim=-1)  # KL(fp16 || fp32) per prompt, nats
    rec["kl_mean"] = round(kl.mean().item(), 5)
    rec["kl_max"] = round(kl.max().item(), 5)
    rec["top1_agree"] = round((logits16.argmax(-1) == logits32.argmax(-1)).float().mean().item(), 3)

    rec["verdict"] = (
        "PASS"
        if rec["kl_mean"] < 0.02
        and rec["top1_agree"] >= 0.9
        and rec["hidden_states_ok"]
        and not rec["think_leakage"]
        else "FAIL"
    )

    # samples for human eyeballing — mandatory review artifact
    with open(out_dir / f"samples_{name}.txt", "w") as f:
        for p, g in zip(PROMPTS, gens):
            f.write(f"PROMPT: {p}\nGEN   : {g}\n{'-' * 70}\n")
    print(json.dumps(rec, indent=2), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="comma-separated short names from models.yaml")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    roster = load_roster(args.config)

    results = []
    for name in args.models.split(","):
        try:
            results.append(smoke_model(name.strip(), roster[name.strip()], out_dir))
        except Exception as e:  # keep going: one bad model must not sink the matrix
            results.append({"model": name, "verdict": "ERROR", "error": f"{type(e).__name__}: {e}"})
            print(f"ERROR on {name}: {e}", flush=True)

    with open(out_dir / "smoke_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['model']:16s} {r.get('verdict'):6s} kl={r.get('kl_mean')} top1={r.get('top1_agree')} "
              f"load16={r.get('load_s_fp16')}s think_leak={r.get('think_leakage')}")


if __name__ == "__main__":
    main()
