"""Residual-stream capture.

One forward pass per stimulus caches every layer's residual stream, pools
per-card token spans on the fly (see pooling.py), and returns
[n_items × n_layers × D] fp16. Card token spans come from offset mapping at
tokenization time and are stored with the stimulus.

Attention entropy (the dispersion covariate) requires eager attention
(output_attentions=True); it is computed on a configurable subsample of
stimuli, not everywhere — it roughly doubles pass cost.

V100 notes: load fp16 (no bf16 on SM 7.0), attn_implementation="eager".
"""

from __future__ import annotations


def extract_pooled(model, tokenizer, stimulus, pooling: str, with_attention_entropy: bool = False):
    raise NotImplementedError
