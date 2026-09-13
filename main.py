#!/usr/bin/env python3
"""Cluster smoke-test: is a model_registry model correctly mounted and loadable here?

Run by the `test-hf-model` job (args: python3 /work/main.py). Self-contained on purpose —
the test preset uploads only `*.py`, so this imports nothing from the repo. It verifies, in order:
  1. environment (torch / transformers / CUDA),
  2. the model directory is mounted (finds config.json under MODEL_PATH),
  3. transformers knows the architecture (loads the config),
  4. the tokenizer loads and tokenizes,
  5. (best-effort) the weights load and one forward runs — skipped gracefully if the flavour
     lacks the RAM/VRAM (e.g. a 31B on 2cpu-32ram), so the plumbing test still passes.

Exit code: 0 if mount + config + tokenizer are OK (weights are a bonus); 1 only if the model
can't be found or the config/tokenizer fail (a real plumbing/compat problem).
"""
import os
import sys
import glob
import traceback
from datetime import datetime

MODEL_PATH = os.environ.get("MODEL_PATH", "/hf_models")
MODEL_ID = os.environ.get("MODEL_ID", "(from model_registry)")


def log(msg=""):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def banner(msg):
    log("=" * 70)
    log(msg)
    log("=" * 70)


def find_model_dir(root):
    """Return the directory that holds config.json (the model root), searching root and subdirs."""
    if os.path.isfile(os.path.join(root, "config.json")):
        return root
    hits = sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True))
    return os.path.dirname(hits[0]) if hits else None


def main():
    banner(f"SMOKE TEST | model_id={MODEL_ID} | MODEL_PATH={MODEL_PATH}")

    # 1. environment ---------------------------------------------------------
    import torch
    import transformers
    log(f"python        {sys.version.split()[0]}")
    log(f"torch         {torch.__version__}")
    log(f"transformers  {transformers.__version__}")
    log(f"cuda avail    {torch.cuda.is_available()} | n_gpu {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log(f"  GPU[{i}]     {p.name}  {p.total_memory/1e9:.0f} GB")

    # 2. mount ---------------------------------------------------------------
    banner("STEP 1/4 — model mount")
    if not os.path.isdir(MODEL_PATH):
        log(f"!! MODEL_PATH {MODEL_PATH} does not exist — model_registry not mounted"); sys.exit(1)
    log(f"contents of {MODEL_PATH}:")
    for name in sorted(os.listdir(MODEL_PATH))[:20]:
        log(f"  - {name}")
    model_dir = find_model_dir(MODEL_PATH)
    if not model_dir:
        log(f"!! no config.json found under {MODEL_PATH} — model files missing"); sys.exit(1)
    log(f"model dir     {model_dir}")
    weight_files = glob.glob(os.path.join(model_dir, "*.safetensors")) + \
        glob.glob(os.path.join(model_dir, "*.bin"))
    total_gb = sum(os.path.getsize(f) for f in weight_files) / 1e9
    log(f"weight shards {len(weight_files)}  (~{total_gb:.1f} GB on disk)")

    # 3. config --------------------------------------------------------------
    banner("STEP 2/4 — config (does transformers know this architecture?)")
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    tcfg = getattr(cfg, "text_config", cfg)   # VLMs nest the LM config
    log(f"model_type    {getattr(cfg, 'model_type', '?')}")
    log(f"architectures {getattr(cfg, 'architectures', '?')}")
    log(f"hidden_size   {getattr(tcfg, 'hidden_size', '?')}")
    log(f"num_layers    {getattr(tcfg, 'num_hidden_layers', '?')}")
    log(f"vocab_size    {getattr(tcfg, 'vocab_size', '?')}")

    # 4. tokenizer -----------------------------------------------------------
    banner("STEP 3/4 — tokenizer")
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True,
                                        local_files_only=True)
    sample = "The zib comes before the quomp. Which is earlier?"
    ids = tok(sample, return_tensors="pt")["input_ids"]
    log(f"tokenizer     {type(tok).__name__} | is_fast={tok.is_fast}")
    log(f"sample        {sample!r} -> {ids.shape[1]} tokens: {ids[0, :12].tolist()}...")

    # 5. weights + forward (best-effort) ------------------------------------
    banner("STEP 4/4 — weights + one forward (best-effort; may skip on small flavour)")
    try:
        from transformers import AutoModelForCausalLM
        has_cuda = torch.cuda.is_available()
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float16 if has_cuda else torch.float32,
            low_cpu_mem_usage=True, device_map="auto" if has_cuda else None,
            trust_remote_code=True, local_files_only=True,
        ).eval()
        dev = model.get_input_embeddings().weight.device
        with torch.no_grad():
            out = model(input_ids=ids.to(dev), output_hidden_states=True)
        log(f"OK — forward ran: logits {tuple(out.logits.shape)}, "
            f"{len(out.hidden_states)} hidden-state layers")
        log("FULL PASS — model loads and runs here.")
    except Exception as e:
        log(f"weights step SKIPPED/FAILED: {type(e).__name__}: {str(e)[:200]}")
        log("(expected on a CPU/low-RAM flavour for a large model — mount + config + tokenizer PASSED)")
        traceback.print_exc()

    banner("SMOKE TEST DONE — mount + config + tokenizer OK")


if __name__ == "__main__":
    main()
