"""Prefix-cache-friendly vLLM client.

All questions for a stimulus share the card-block prefix; issue them grouped
by stimulus (and sorted by prefix) so vLLM's --enable-prefix-caching gets
near-100% hits. Requests carry temperature=0, max_tokens per question family,
and logprobs where logit scoring applies (pairwise yes/no).
"""

from __future__ import annotations

from icom.generator.schemas import Question, Stimulus


class BatteryClient:
    def __init__(self, base_url: str, model: str, max_concurrency: int = 32):
        raise NotImplementedError

    def run_stimulus(self, stimulus: Stimulus, questions: list[Question]) -> list[dict]:
        """Return raw completions + logprobs; scoring happens in scoring.py."""
        raise NotImplementedError
