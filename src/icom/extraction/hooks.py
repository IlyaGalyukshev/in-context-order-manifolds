"""Residual-stream capture: one forward pass per stimulus, pooled on the fly.

Hard rule: full [L × T × D] tensors never touch the disk — only pooled
[N × L × D] fp16 per scheme.

The extraction prompt for instruct models is the chat-template-wrapped card
block (matching the battery's shared prefix as closely as possible); base
models get the raw block. Spans are located in the final formatted string.
"""

from __future__ import annotations

import numpy as np
import torch

from icom.extraction.pooling import build_spans, pool_all


def format_extraction_prompt(tok, card_block: str, is_instruct: bool) -> str:
    if not is_instruct or tok.chat_template is None:
        return card_block
    kwargs = dict(tokenize=False, add_generation_prompt=False)
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": card_block}], enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": card_block}], **kwargs)


@torch.no_grad()
def extract_pooled(model, tok, stimulus: dict, is_instruct: bool, device: str = "cuda:0") -> dict:
    prompt = format_extraction_prompt(tok, stimulus["prompt"], is_instruct)
    enc = tok(prompt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offsets = [tuple(x) for x in enc.pop("offset_mapping")[0].tolist()]
    spans = build_spans(prompt, stimulus, offsets)

    out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
    hidden = torch.stack(out.hidden_states, dim=0)[:, 0].float().cpu().numpy()  # [L+1, T, D]

    pooled = pool_all(hidden, spans, stimulus["latent_order"])
    slot_of = {}
    for c in stimulus["cards"]:
        slot_of.setdefault(c["entity"], c["presentation_slot"])
        if c.get("entity_b"):
            slot_of.setdefault(c["entity_b"], c["presentation_slot"])
    return {
        "pooled": pooled,  # {scheme: [N, L+1, D] fp16}
        "ranks": np.array([i + 1 for i in range(len(stimulus["latent_order"]))]),
        "slots": np.array([slot_of[e] for e in stimulus["latent_order"]]),
        "n_tokens": hidden.shape[1],
        "span_sizes": {e: {s: len(ix) for s, ix in d.items()} for e, d in spans.items()},
    }
