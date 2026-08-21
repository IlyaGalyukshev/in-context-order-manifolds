"""M1 — repeat-read extraction. k re-presentations of the same stimulus with a fresh
(card-order, roster-order) permutation each time (seeded, deterministic) → k INDEPENDENT reads
per entity at every locus (readout moves with the roster; the in-card loci move with the shuffled
card context). These k reads are what crossnobis / whitened-cvRDM (M2) needs to estimate noise and
cross-validate. Reuses the single-read span+pool core (build_spans / pool_all); the raw token
tensor is still never written to disk — pooled per entity per read on the fly.

Correctness-first: this does k independent forwards. The KV-cache branch (forward the card prefix
once, copy the cache per roster branch) is a later speed-only optimization for the readout/probe
loci; it does NOT give independent in-card reads (the prefix is shared), so the general estimator
that yields independence at every locus is the k re-presentation loop here.
"""

from __future__ import annotations

import numpy as np

from icom.utils.seeding import rng_for
# torch + the hooks/pooling core are imported lazily inside extract_pooled_repeat so the pure
# prompt-rebuild helper (rebuild_prompt) is importable/testable without a torch install.


def rebuild_prompt(st: dict, card_perm, roster_perm) -> str:
    """Card-block string with cards reordered by `card_perm` and (if the stimulus carries a roster)
    the roster reordered by `roster_perm`. Only positions change — every card's text and every
    entity mention are preserved, so build_spans still locates them and pool_all is unchanged."""
    prompt, cards, ents = st["prompt"], st["cards"], st["latent_order"]
    positions = [prompt.find(c["text"]) for c in cards]
    assert min(positions) >= 0, "card text not found while rebuilding prompt"
    preamble = prompt[: min(positions)]                        # preamble + its separator, or ""
    body = preamble + "\n".join(cards[i]["text"] for i in card_perm)
    if roster_perm is not None:
        roster = "Entities: " + ", ".join(f"the {ents[j]}" for j in roster_perm) + "."
        body = body + "\n\n" + roster
    return body


PROBES = ["Consider the {e}.", "Think about the {e}.", "Recall the {e}.", "Note the {e}.",
          "Focus on the {e}.", "Regarding the {e}.", "As for the {e}.", "Take the {e}."]


def extract_probe_repeat(model, tok, st: dict, is_instruct: bool, k: int = 8,
                         device: str = "cuda:0") -> dict:
    """M1(b) — NEUTRAL-PROBE read locus: forward the CARD BLOCK once (KV-cached), then for each
    entity × k neutral-probe paraphrases ("Consider the {e}." …) continue from a COPY of that cache
    and read the entity's token in the probe. Gives a content-neutral, comparable evoked read per
    entity (the read every declared/derived condition and every assembly-ladder rung can share).
    Returns {pooled:{'probe':[N,k,L+1,D] fp16}, ranks, entities, n_reads}."""
    import re
    from copy import deepcopy

    import torch
    from icom.extraction.hooks import format_extraction_prompt

    ents = st["latent_order"]; N = len(ents)
    prompt, cards = st["prompt"], st["cards"]
    rs = prompt.rfind("\n\nEntities:")                         # drop the roster; cards are the context
    card_block = prompt[:rs] if rs > 0 else prompt
    prefix = format_extraction_prompt(tok, card_block, is_instruct)
    torch.set_grad_enabled(False)
    penc = tok(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    pout = model(**penc, use_cache=True)
    cache = pout.past_key_values                               # DynamicCache (mutated in place → copy per branch)

    per = {e: [] for e in ents}
    for e in ents:
        for p in PROBES[:k]:
            sent = "\n\n" + p.format(e=e)
            enc = tok(sent, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
            offs = [tuple(x) for x in enc.pop("offset_mapping")[0].tolist()]
            m = list(re.finditer(rf"\b[Tt]he {re.escape(e)}\b", sent))[-1]  # the probe mention of e
            toks = [i for i, (s, en) in enumerate(offs) if s < m.end() and en > m.start() + 4 and en > s]
            out = model(input_ids=enc["input_ids"].to(device), past_key_values=deepcopy(cache),
                        use_cache=True, output_hidden_states=True)
            hid = torch.stack(out.hidden_states, dim=0)[:, 0].float().cpu().numpy()   # [L+1, sent_len, D]
            per[e].append(hid[:, toks, :].mean(axis=1))        # [L+1, D]
    probe = np.stack([np.stack(per[e]) for e in ents]).astype(np.float16)             # [N, k, L+1, D]
    return {"pooled": {"probe": probe}, "ranks": np.array([i + 1 for i in range(N)]),
            "entities": ents, "n_reads": k}


def extract_pooled_repeat(model, tok, st: dict, is_instruct: bool, k: int = 8,
                          device: str = "cuda:0", root_seed: int = 20260724, loci=None) -> dict:
    """{pooled:{scheme:[N,k,L+1,D] fp16}, ranks:[N], entities:list, n_reads:k}. `loci` (set of
    scheme names) restricts what is pooled/stored (readout/card_mean/last_token/name)."""
    import torch
    from icom.extraction.hooks import format_extraction_prompt
    from icom.extraction.pooling import build_spans, pool_all

    ents = st["latent_order"]
    N, ncards = len(ents), len(st["cards"])
    has_roster = st.get("readout_order") is not None
    per_read = []
    torch.set_grad_enabled(False)
    for r in range(k):
        rng = rng_for(root_seed, "repeat", st["stimulus_id"], r)
        card_perm = rng.permutation(ncards)
        roster_perm = rng.permutation(N) if has_roster else None
        raw = rebuild_prompt(st, card_perm, roster_perm)
        fmt = format_extraction_prompt(tok, raw, is_instruct)
        enc = tok(fmt, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
        offsets = [tuple(x) for x in enc.pop("offset_mapping")[0].tolist()]
        spans = build_spans(fmt, st, offsets)
        out = model(**{kk: v.to(device) for kk, v in enc.items()}, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)[:, 0].float().cpu().numpy()   # [L+1,T,D]
        pooled = pool_all(hidden, spans, ents)                 # {scheme:[N,L+1,D]}
        if loci:
            pooled = {s: v for s, v in pooled.items() if s in loci}
        per_read.append(pooled)
    schemes = set.intersection(*(set(p.keys()) for p in per_read))
    pooled_k = {s: np.stack([per_read[r][s] for r in range(k)], axis=1).astype(np.float16)
                for s in schemes}                              # [N, k, L+1, D]
    return {"pooled": pooled_k,
            "ranks": np.array([i + 1 for i in range(N)]),
            "entities": ents, "n_reads": k}
