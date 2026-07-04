"""HF battery runner with manual prefix KV-cache reuse.

Modern vLLM dropped Volta (SM 7.0), so the battery runs on plain transformers:
encode the shared card-block prefix ONCE per stimulus, then answer every
question from the cached past_key_values — recovering the prefix-caching win
vLLM used to provide. Questions are batched (expand the cached KV along the
batch dim) with per-family max_new_tokens, temperature=0, and logprobs
captured where logit scoring applies (pairwise yes/no).

Qwen3-family models run with enable_thinking=False in the chat template so
short constrained answers are not preceded by reasoning chains.
"""

from __future__ import annotations

from icom.generator.schemas import Question, Stimulus


class BatteryRunner:
    def __init__(self, model, tokenizer, batch_size: int = 16, enable_thinking: bool = False):
        raise NotImplementedError

    def run_stimulus(self, stimulus: Stimulus, questions: list[Question]) -> list[dict]:
        """Prefix-encode once, then answer all questions from cached KV.
        Returns raw completions + logprobs; scoring happens in scoring.py."""
        raise NotImplementedError
